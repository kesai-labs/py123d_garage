# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""AdamW with gradient-norm clipping as one Triton pass per model over flat parameter, gradient and moment buffers."""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import Tensor
from typing_extensions import override

from py123d_garage.training.clip_adamw_kernel import clip_adamw_kernel

# Inductor's compiled graphs read the parameters with vectorized loads, so every parameter starts on a 256-byte boundary.
_ALIGNMENT_BYTES = 256
_BLOCK_SIZE = 2048
_NUM_WARPS = 8
# clip_grad_norm_'s guard against a zero norm.
_NORM_EPS = 1e-6


def _aligned_offsets(parameters: Sequence[Tensor]) -> tuple[list[int], int]:
    """
    Lay out parameters in one flat buffer with each starting on an aligned boundary.

    Args:
        parameters: the parameters to lay out, in order

    Returns:
        each parameter's element offset and the buffer's total size in elements
    """
    offsets: list[int] = []
    offset = 0
    for parameter in parameters:
        offsets.append(offset)
        alignment = _ALIGNMENT_BYTES // parameter.element_size()
        offset += -(-parameter.numel() // alignment) * alignment
    return offsets, offset


class FlatAdamW(torch.optim.Optimizer):
    """
    AdamW with gradient-norm clipping as one Triton pass per model over flat parameter, gradient and moment buffers.

    Parameters and gradients become views into one buffer per model, and torch's LR schedulers still apply.
    """

    def __init__(
        self,
        model_param_groups: Sequence[Sequence[dict[str, Any]]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_grad_norm: float | None = None,
        gradient_pool: torch.cuda.MemPool | None = None,
    ) -> None:
        """
        Initializes the optimizer and moves the parameters into the flat buffers.

        Args:
            model_param_groups: per model, its param groups (params and weight_decay) as torch.optim.AdamW takes them.
            lr: learning rate; the schedulers move it per group.
            betas: running-average factors of the gradient and of its square.
            eps: added to the root of the second moment.
            max_grad_norm: every model's gradient is scaled to at most this norm before the update; None clips nothing.
            gradient_pool: the memory pool the gradient buffers are allocated from; None uses the caching allocator.
        """
        groups: list[dict[str, Any]] = [
            {**group, "model": model_index}
            for model_index, groups_of_model in enumerate(model_param_groups)
            for group in groups_of_model
        ]
        super().__init__(groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0})
        self._max_grad_norm = max_grad_norm
        self._num_steps = 0
        self._graphed = False
        self._flat_params: list[Tensor] = []
        self._flat_grads: list[Tensor] = []
        self._exp_avgs: list[Tensor] = []
        self._exp_avg_sqs: list[Tensor] = []
        self._first_parameters: list[Tensor] = []
        # Per param group: the model's index and the group's element range in the model's buffers.
        self._group_ranges: list[tuple[int, int, int]] = []
        for model_index in range(len(model_param_groups)):
            groups_of_model = [group for group in self.param_groups if group["model"] == model_index]
            parameters = [cast(Tensor, parameter) for group in groups_of_model for parameter in group["params"]]
            dtypes = {parameter.dtype for parameter in parameters}
            assert len(dtypes) == 1, f"one buffer per model needs one parameter dtype, got {dtypes}"
            offsets, total = _aligned_offsets(parameters)
            flat_params = torch.zeros(total, dtype=dtypes.pop(), device=parameters[0].device)
            with contextlib.nullcontext() if gradient_pool is None else torch.cuda.use_mem_pool(gradient_pool):
                flat_grads = torch.zeros_like(flat_params)
            for parameter, offset in zip(parameters, offsets, strict=True):
                view = flat_params[offset : offset + parameter.numel()].as_strided(parameter.shape, parameter.stride())
                view.copy_(parameter.detach())
                parameter.data = view
                parameter.grad = flat_grads[offset : offset + parameter.numel()].as_strided(
                    parameter.shape,
                    parameter.stride(),
                )
            start = 0
            for group in groups_of_model:
                size = _aligned_offsets([cast(Tensor, parameter) for parameter in group["params"]])[1]
                self._group_ranges.append((model_index, start, start + size))
                start += size
            self._flat_params.append(flat_params)
            self._flat_grads.append(flat_grads)
            self._exp_avgs.append(torch.zeros_like(flat_params))
            self._exp_avg_sqs.append(torch.zeros_like(flat_params))
            self._first_parameters.append(parameters[0])
        # Per group: learning rate, step size and root of the second-moment bias correction.
        self._hyperparameters = torch.zeros(len(self.param_groups), 3, device=self._flat_params[0].device)
        # Compiles the kernel eagerly, outside any graph capture.
        self._launch(0, 0, 0, torch.ones((), device=self._hyperparameters.device))

    @property
    def flat_gradients(self) -> list[Tensor]:
        """Every model's gradient buffer, which its parameters' .grad are views into."""
        return list(self._flat_grads)

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradient buffers, keeping every parameter's .grad a view into them."""
        for flat_grads in self._flat_grads:
            flat_grads.zero_()

    def _clip_scale(self, flat_grads: Tensor) -> Tensor:
        """
        Compute the factor clip_grad_norm_ would scale one model's gradients by.

        Args:
            flat_grads: the model's gradient buffer

        Returns:
            the scale as a scalar device tensor the kernel reads
        """
        if self._max_grad_norm is None:
            return torch.ones((), dtype=flat_grads.dtype, device=flat_grads.device)
        return torch.clamp(self._max_grad_norm / (torch.dot(flat_grads, flat_grads).sqrt() + _NORM_EPS), max=1.0)

    def _launch(self, group_index: int, start: int, end: int, scale: Tensor) -> None:
        """
        Launch the update kernel on one param group's range of its model's buffers.

        Args:
            group_index: the param group to update
            start: the group's first element in the model's buffers
            end: the element after the group's last
            scale: the model's clip scale
        """
        group = self.param_groups[group_index]
        model_index = self._group_ranges[group_index][0]
        beta1, beta2 = cast(tuple[float, float], group["betas"])
        num_elements = end - start
        clip_adamw_kernel()[(max(1, -(-num_elements // _BLOCK_SIZE)),)](
            self._flat_params[model_index][start:end],
            self._flat_grads[model_index][start:end],
            self._exp_avgs[model_index][start:end],
            self._exp_avg_sqs[model_index][start:end],
            scale,
            self._hyperparameters[group_index],
            num_elements,
            beta1,
            beta2,
            cast(float, group["eps"]),
            cast(float, group["weight_decay"]),
            BLOCK=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )

    def _update(self, model_index: int) -> None:
        """
        Clip and update one model's parameters with the loaded hyperparameters.

        Args:
            model_index: the model to update
        """
        scale = self._clip_scale(self._flat_grads[model_index])
        for group_index, (group_model, start, end) in enumerate(self._group_ranges):
            if group_model == model_index and start != end:
                self._launch(group_index, start, end, scale)

    def clear_moments(self, parameters: Sequence[Tensor]) -> None:
        """
        Zero the moments of these parameters in place, so a re-initialized module starts its averages afresh.

        The bias corrections follow the shared step count, so the first updates after a late clear exceed Adam's.

        Args:
            parameters: the parameters whose moments to clear
        """
        for parameter in parameters:
            (model_index,) = [
                index
                for index, flat_params in enumerate(self._flat_params)
                if parameter.untyped_storage().data_ptr() == flat_params.untyped_storage().data_ptr()
            ]
            start = parameter.storage_offset() - self._flat_params[model_index].storage_offset()
            self._exp_avgs[model_index][start : start + parameter.numel()].zero_()
            self._exp_avg_sqs[model_index][start : start + parameter.numel()].zero_()

    def load_hyperparameters(self) -> None:
        """Write every group's learning rate and the coming step's bias corrections into the tensor the kernel reads."""
        step = self._num_steps + 1
        rows: list[list[float]] = []
        for group in self.param_groups:
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            rows.append([lr, lr / (1.0 - beta1**step), math.sqrt(1.0 - beta2**step)])
        # Pinned through the caching host allocator, which frees the block only once the copy is done.
        self._hyperparameters.copy_(torch.tensor(rows, dtype=torch.float32).pin_memory(), non_blocking=True)

    def graph_updates(self) -> list[Callable[[], None]]:
        """
        Hand out each model's clip and update, to record into a CUDA graph right after the model's backward.

        Afterwards load_hyperparameters must run before every replay, and step only counts the steps.

        Returns:
            one update callable per model
        """
        self._graphed = True
        return [functools.partial(self._update, model_index) for model_index in range(len(self._flat_params))]

    # Torch's stubs overload step on the closure; one signature cannot satisfy both overloads.
    @override
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Clip every model's gradient and update each group, or only count the step once graph_updates is recorded.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for first_parameter, flat_params in zip(self._first_parameters, self._flat_params, strict=True):
            assert first_parameter.untyped_storage().data_ptr() == flat_params.untyped_storage().data_ptr(), (
                "the parameters left their flat buffer; build FlatAdamW after moving the model"
            )
        if not self._graphed:
            self.load_hyperparameters()
            for model_index in range(len(self._flat_params)):
                self._update(model_index)
        self._num_steps += 1
        return loss

    @override
    def state_dict(self) -> dict[str, Any]:
        """Save the moments per model buffer, the shared step count and the groups without their params."""
        return {
            "state": {
                "num_steps": self._num_steps,
                "exp_avgs": [exp_avg.clone() for exp_avg in self._exp_avgs],
                "exp_avg_sqs": [exp_avg_sq.clone() for exp_avg_sq in self._exp_avg_sqs],
            },
            "param_groups": [
                {key: value for key, value in group.items() if key != "params"} for group in self.param_groups
            ],
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Copy the moments into the buffers and restore every group's hyperparameters."""
        state = cast(dict[str, Any], state_dict["state"])
        self._num_steps = cast(int, state["num_steps"])
        for exp_avg, saved in zip(self._exp_avgs, cast(list[Tensor], state["exp_avgs"]), strict=True):
            exp_avg.copy_(saved)
        for exp_avg_sq, saved in zip(self._exp_avg_sqs, cast(list[Tensor], state["exp_avg_sqs"]), strict=True):
            exp_avg_sq.copy_(saved)
        saved_groups = cast(list[dict[str, Any]], state_dict["param_groups"])
        for group, saved_group in zip(self.param_groups, saved_groups, strict=True):
            group.update({key: value for key, value in saved_group.items() if key != "params"})
