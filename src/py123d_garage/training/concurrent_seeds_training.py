# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""One seed's policy forward and loss, and the CUDA graph that runs every seed's step concurrently."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.datatypes.tensor import TensorBundle

LOG = logging.getLogger(__name__)

# Eager passes before the recording: cuDNN's algorithm search, the allocator and the NCCL communicators warm up.
_WARMUP_PASSES = 2

SeedInputs = tuple[AbstractFeatures, AbstractLabels, NavigationConditioning]


def to_channels_last(tensor: Tensor) -> Tensor:
    """
    Convert a 4D (N, C, H, W) tensor to channels-last memory format, leaving other tensors as they are.

    Args:
        tensor: the tensor to convert

    Returns:
        the channels-last tensor, or the input when it is not 4D
    """
    if tensor.dim() != 4:
        return tensor
    return tensor.to(memory_format=torch.channels_last)


def policy_forward_and_loss(
    policy: AnyPolicy,
    features: AbstractFeatures,
    labels: AbstractLabels,
    navigation: NavigationConditioning,
    loss_weights: Mapping[str, float | Tensor],
    channels_last: bool,
) -> tuple[Tensor, dict[str, Tensor], AbstractPredictions]:
    """
    Run one policy's forward and loss on already augmented features.

    Args:
        policy: the policy to run
        features: the batch's features
        labels: the batch's labels
        navigation: the batch's navigation conditioning
        loss_weights: the weight of every loss by name
        channels_last: convert the features and labels to channels-last first

    Returns:
        the weighted total loss, the unweighted losses by name and the predictions
    """
    if channels_last:
        features = features.apply(to_channels_last)
        labels = labels.apply(to_channels_last)
    predictions = policy.forward(features, navigation)
    losses: dict[str, Tensor] = policy.compute_loss(labels, predictions)
    total_loss = torch.zeros((), device=next(iter(losses.values())).device)
    for name, value in losses.items():
        total_loss = total_loss + loss_weights[name] * value.float().reshape(())
    return total_loss, losses, predictions


def _flat_grad_buffer(policy: AnyPolicy) -> Tensor:
    """
    Make one zeroed buffer holding every trainable parameter's gradient, setting each .grad to a view into it.

    Args:
        policy: the policy whose gradients the buffer holds

    Returns:
        the flat gradient buffer
    """
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    dtypes = {parameter.dtype for parameter in parameters}
    assert len(dtypes) == 1, f"one gradient buffer needs one parameter dtype, got {dtypes}"
    flat = torch.zeros(
        sum(parameter.numel() for parameter in parameters),
        dtype=dtypes.pop(),
        device=parameters[0].device,
    )
    offset = 0
    for parameter in parameters:
        # The same strides as the parameter, so autograd accumulates into the view in place.
        parameter.grad = flat[offset : offset + parameter.numel()].as_strided(parameter.shape, parameter.stride())
        offset += parameter.numel()
    return flat


def _assert_grads_alias(policy: AnyPolicy, flat: Tensor) -> None:
    """
    Check that every trainable parameter's gradient is still a view into the flat buffer.

    Args:
        policy: the policy to check
        flat: the policy's flat gradient buffer
    """
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no gradient"
            assert parameter.grad.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr(), (
                f"autograd replaced the gradient view of {name}"
            )


def _copy_gradients(gradient_views: Sequence[tuple[Tensor, Tensor]], zero_missing: bool) -> None:
    """
    Copy the gradients autograd handed over into their views of the flat buffer with one batched copy.

    Args:
        gradient_views: each parameter with its gradient view into the flat buffer
        zero_missing: zero the views of the parameters that got no gradient, for a buffer not zeroed before the step
    """
    produced = [(view, parameter.grad) for parameter, view in gradient_views if parameter.grad is not None]
    torch._foreach_copy_([view for view, _ in produced], [gradient for _, gradient in produced])  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
    missing = [view for parameter, view in gradient_views if parameter.grad is None]
    if zero_missing and missing:
        torch._foreach_zero_(missing)  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
    for parameter, view in gradient_views:
        parameter.grad = view


def _static_copy(tensor: Tensor) -> Tensor:
    """
    Copy a tensor into a new buffer the graph can record on.

    Args:
        tensor: the tensor to copy

    Returns:
        the detached copy
    """
    return tensor.detach().clone()


def _copy_into(static_bundle: TensorBundle, bundle: TensorBundle) -> None:
    """
    Copy a bundle's tensors into the recorded static bundle, checking names, shapes and dtypes match.

    Args:
        static_bundle: the bundle the graph recorded on
        bundle: the new values
    """
    static_tensors = dict(static_bundle.named_tensors())
    tensors = dict(bundle.named_tensors())
    assert static_tensors.keys() == tensors.keys(), (
        f"the batch's tensors {sorted(tensors)} differ from the recorded ones {sorted(static_tensors)}"
    )
    for name, tensor in tensors.items():
        static_tensor = static_tensors[name]
        assert static_tensor.shape == tensor.shape, (
            f"{name}: recorded {tuple(static_tensor.shape)}, got {tuple(tensor.shape)}"
        )
        assert static_tensor.dtype == tensor.dtype, f"{name}: recorded {static_tensor.dtype}, got {tensor.dtype}"
        static_tensor.copy_(tensor)


class CapturedSeedsStep:
    """
    One CUDA graph running every seed's forward, loss, backward, gradient all-reduce and optional update concurrently.

    The static inputs are copied in before every replay and the static outputs read after it.
    """

    def __init__(
        self,
        policies: Sequence[AnyPolicy],
        loss_weights: Mapping[str, Tensor],
        channels_last: bool,
        flat_grads: Sequence[Tensor] | None = None,
        seed_updates: Sequence[Callable[[], None]] | None = None,
        copy_gradients: bool = False,
        skip_gradient_zeroing: bool = False,
    ) -> None:
        """
        Initialize the step's static state; the graph is recorded by capture.

        Args:
            policies: one policy per seed, on the device.
            loss_weights: the weight of every loss, as tensors the recording reads.
            channels_last: convert the features and labels to channels-last before the forward.
            flat_grads: every policy's gradient buffer, its parameters' .grad views into it; None builds them.
            seed_updates: per seed, its optimizer update to record after its backward and all-reduce; None records none.
            copy_gradients: copy the gradients into the flat buffer with one batched copy, not one kernel per parameter.
            skip_gradient_zeroing: with copy_gradients, zero only the views of the parameters without a gradient instead
                of the whole buffer; the rest of the buffer (alignment gaps) must stay zero.
        """
        self._policies = list(policies)
        self._loss_weights = loss_weights
        self._channels_last = channels_last
        self._seed_updates = list(seed_updates) if seed_updates is not None else None
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._flat_grads = (
            list(flat_grads) if flat_grads is not None else [_flat_grad_buffer(policy) for policy in self._policies]
        )
        self._gradient_views: list[list[tuple[Tensor, Tensor]]] | None = None
        self._skip_gradient_zeroing = copy_gradients and skip_gradient_zeroing
        if copy_gradients:
            for policy, flat in zip(self._policies, self._flat_grads, strict=True):
                _assert_grads_alias(policy, flat)
            self._gradient_views = [
                [
                    (parameter, parameter.grad)
                    for parameter in policy.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                for policy in self._policies
            ]
        self._streams = [torch.cuda.Stream() for _ in self._policies]
        self._graph = torch.cuda.CUDAGraph()
        self._inputs: list[SeedInputs] = []
        self.totals: list[Tensor] = []
        self.losses: list[dict[str, Tensor]] = []
        self.predictions: list[AbstractPredictions] = []

    @property
    def labels(self) -> list[AbstractLabels]:
        """Every seed's static labels, the ones its static predictions belong to."""
        return [labels for _, labels, _ in self._inputs]

    def capture(self, inputs: Sequence[SeedInputs]) -> None:
        """
        Record the graph on static copies of the inputs, after eager warm-up passes whose BatchNorm updates are undone.

        Args:
            inputs: one seed's features, labels and navigation per seed
        """
        self._inputs = [
            (features.apply(_static_copy), labels.apply(_static_copy), navigation.apply(_static_copy))
            for features, labels, navigation in inputs
        ]
        saved_buffers = [{name: buffer.clone() for name, buffer in policy.named_buffers()} for policy in self._policies]
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())  # pyright: ignore[reportUnknownMemberType]
        with torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_PASSES):
                self._run(with_updates=False)
        torch.cuda.current_stream().wait_stream(side_stream)  # pyright: ignore[reportUnknownMemberType]
        torch.cuda.synchronize()
        for policy, buffers in zip(self._policies, saved_buffers, strict=True):
            for name, buffer in policy.named_buffers():
                buffer.copy_(buffers[name])
        for policy, flat in zip(self._policies, self._flat_grads, strict=True):
            _assert_grads_alias(policy, flat)
        # thread_local: the loader's pin-memory thread keeps making CUDA calls while this thread records.
        with torch.cuda.graph(self._graph, stream=side_stream, capture_error_mode="thread_local"):
            self._run(with_updates=True)
        LOG.info(f"Recorded the {len(self._policies)} seeds' step into one CUDA graph")

    def _run(self, with_updates: bool) -> None:
        """
        Run every seed's step on its own stream.

        Args:
            with_updates: also run the optimizer updates, which the warm-up passes skip
        """
        current_stream = torch.cuda.current_stream()
        self.totals, self.losses, self.predictions = [], [], []
        for seed_index, (policy, stream, flat, (features, labels, navigation)) in enumerate(
            zip(
                self._policies,
                self._streams,
                self._flat_grads,
                self._inputs,
                strict=True,
            ),
        ):
            stream.wait_stream(current_stream)  # pyright: ignore[reportUnknownMemberType]
            with torch.cuda.stream(stream):
                if not self._skip_gradient_zeroing:
                    flat.zero_()
                if self._gradient_views is not None:
                    for parameter, _ in self._gradient_views[seed_index]:
                        parameter.grad = None
                total_loss, losses, predictions = policy_forward_and_loss(
                    policy,
                    features,
                    labels,
                    navigation,
                    self._loss_weights,
                    self._channels_last,
                )
                total_loss.backward()  # pyright: ignore[reportUnknownMemberType]
                if self._gradient_views is not None:
                    _copy_gradients(self._gradient_views[seed_index], self._skip_gradient_zeroing)
                if self._world_size > 1:
                    # DDP's order: divide, then sum across ranks.
                    flat.div_(self._world_size)
                    dist.all_reduce(flat)  # pyright: ignore[reportUnknownMemberType]
                if with_updates and self._seed_updates is not None:
                    self._seed_updates[seed_index]()
            self.totals.append(total_loss.detach())
            self.losses.append({name: value.detach() for name, value in losses.items()})
            self.predictions.append(predictions.apply(Tensor.detach))
        for stream in self._streams:
            current_stream.wait_stream(stream)  # pyright: ignore[reportUnknownMemberType]

    def copy_inputs(
        self,
        seed_index: int,
        features: AbstractFeatures,
        labels: AbstractLabels,
        navigation: NavigationConditioning,
    ) -> None:
        """
        Copy one seed's inputs of this step into its static buffers; names and shapes must match the recording.

        Args:
            seed_index: the seed whose inputs these are
            features: the seed's features
            labels: the seed's labels
            navigation: the seed's navigation conditioning
        """
        for static_bundle, bundle in zip(self._inputs[seed_index], (features, labels, navigation), strict=True):
            _copy_into(static_bundle, bundle)

    def replay(self) -> None:
        """Run the recorded step on the current static inputs."""
        self._graph.replay()
