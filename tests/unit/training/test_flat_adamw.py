# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""FlatAdamW against torch's AdamW with clip_grad_norm_: the same updates under moving learning rates, aligned views, a checkpoint round trip."""

from __future__ import annotations

import copy
import io
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from py123d_garage.training.clip_adamw_kernel import clip_adamw_kernel
from py123d_garage.training.flat_adamw import FlatAdamW

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlatAdamW runs a Triton kernel on the GPU")

_LR = 1e-3
_WEIGHT_DECAY = 0.01
_MAX_NORM = 1.0


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5),
        nn.LayerNorm(5),
    )
    return model.cuda().to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]


def _groups(model: nn.Module) -> list[dict[str, object]]:
    return [
        {
            "params": [parameter for parameter in model.parameters() if parameter.ndim >= 2],
            "weight_decay": _WEIGHT_DECAY,
        },
        {"params": [parameter for parameter in model.parameters() if parameter.ndim < 2], "weight_decay": 0.0},
    ]


def _set_gradients(models: list[nn.Module], references: list[nn.Module], scale: float) -> None:
    """The same random gradients into the flat views and onto the reference parameters."""
    for model, reference in zip(models, references, strict=True):
        for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
            gradient = torch.randn_like(parameter) * scale
            assert parameter.grad is not None
            parameter.grad.copy_(gradient)
            reference_parameter.grad = gradient.clone()


def test_matches_torch_adamw_with_clipping() -> None:
    """Clipped and unclipped steps, a learning-rate change half way, two models with their own clipping norms."""
    models = [_model(0), _model(1)]
    references = copy.deepcopy(models)
    optimizer = FlatAdamW([_groups(model) for model in models], lr=_LR, max_grad_norm=_MAX_NORM)
    reference_optimizer = torch.optim.AdamW(
        [group for reference in references for group in _groups(reference)],
        lr=_LR,
        fused=True,
    )
    for step, scale in enumerate((1.0, 1e-3, 1.0, 10.0, 1e-3)):
        if step == 2:
            for group in [*optimizer.param_groups, *reference_optimizer.param_groups]:
                group["lr"] = _LR / 2
        torch.manual_seed(step)
        _set_gradients(models, references, scale)
        for reference in references:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), _MAX_NORM)
        reference_optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
    for model, reference in zip(models, references, strict=True):
        for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, reference_parameter, atol=1e-5, rtol=1e-5)


def test_parameters_and_gradients_are_aligned_views_of_the_buffers() -> None:
    model = _model(0)
    optimizer = FlatAdamW([_groups(model)], lr=_LR)
    (flat_grads,) = optimizer.flat_gradients
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.untyped_storage().data_ptr() == flat_grads.untyped_storage().data_ptr()
        assert parameter.data_ptr() % 256 == 0
        assert parameter.grad.data_ptr() % 256 == 0
    model(torch.randn(2, 3, 4, 4, device="cuda").to(memory_format=torch.channels_last)).sum().backward()
    assert flat_grads.abs().sum() > 0
    first_parameter = next(model.parameters())
    assert first_parameter.grad is not None
    assert first_parameter.grad.untyped_storage().data_ptr() == flat_grads.untyped_storage().data_ptr()
    optimizer.zero_grad()
    assert flat_grads.abs().sum() == 0
    assert first_parameter.grad is not None


def test_state_dict_round_trip() -> None:
    """A restored optimizer continues exactly where the saved one was, through a CPU checkpoint like split_checkpoint's."""
    models = [_model(0), _model(1)]
    optimizer = FlatAdamW([_groups(model) for model in models], lr=_LR, max_grad_norm=_MAX_NORM)
    for step in range(3):
        torch.manual_seed(step)
        _set_gradients(models, copy.deepcopy(models), 1.0)
        optimizer.step()
    for group in optimizer.param_groups:
        group["lr"] = _LR / 4
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=False)

    restored_models = copy.deepcopy(models)
    restored = FlatAdamW([_groups(model) for model in restored_models], lr=_LR, max_grad_norm=_MAX_NORM)
    restored.load_state_dict(saved)
    assert all(group["lr"] == _LR / 4 for group in restored.param_groups)
    torch.manual_seed(7)
    _set_gradients(models, copy.deepcopy(models), 1.0)
    for restored_model, model in zip(restored_models, models, strict=True):
        for restored_parameter, parameter in zip(restored_model.parameters(), model.parameters(), strict=True):
            assert restored_parameter.grad is not None
            assert parameter.grad is not None
            restored_parameter.grad.copy_(parameter.grad)
    optimizer.step()
    restored.step()
    for model, restored_model in zip(models, restored_models, strict=True):
        for parameter, restored_parameter in zip(model.parameters(), restored_model.parameters(), strict=True):
            torch.testing.assert_close(parameter, restored_parameter, atol=0.0, rtol=0.0)


def test_group_weight_decay_and_learning_rate_apply_per_group() -> None:
    """A zero gradient leaves undecayed parameters alone and shrinks decayed ones by lr * weight_decay."""
    model = _model(0)
    before: list[Tensor] = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = FlatAdamW([_groups(model)], lr=_LR)
    optimizer.step()
    for parameter, old in zip(model.parameters(), before, strict=True):
        expected = old * (1.0 - _LR * _WEIGHT_DECAY) if parameter.ndim >= 2 else old
        torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=1e-6)


def _reference_step(references: list[nn.Module], reference_optimizer: torch.optim.Optimizer) -> None:
    for reference in references:
        torch.nn.utils.clip_grad_norm_(reference.parameters(), _MAX_NORM)
    reference_optimizer.step()


def _assert_models_match(models: list[nn.Module], references: list[nn.Module]) -> None:
    for model, reference in zip(models, references, strict=True):
        for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, reference_parameter, atol=1e-5, rtol=1e-5)


def test_recorded_updates_replay_with_the_current_learning_rate() -> None:
    """The updates recorded into a CUDA graph follow load_hyperparameters: a moving learning rate and the step count."""
    models = [_model(0), _model(1)]
    references = copy.deepcopy(models)
    optimizer = FlatAdamW([_groups(model) for model in models], lr=_LR, max_grad_norm=_MAX_NORM)
    reference_optimizer = torch.optim.AdamW(
        [group for reference in references for group in _groups(reference)],
        lr=_LR,
        fused=True,
    )
    updates = optimizer.graph_updates()
    graph = torch.cuda.CUDAGraph()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=side_stream):
        for update in updates:
            update()
    torch.cuda.current_stream().wait_stream(side_stream)
    for step, scale in enumerate((1.0, 1e-3, 10.0, 1.0)):
        if step == 2:
            for group in [*optimizer.param_groups, *reference_optimizer.param_groups]:
                group["lr"] = _LR / 2
        torch.manual_seed(step)
        _set_gradients(models, references, scale)
        _reference_step(references, reference_optimizer)
        optimizer.load_hyperparameters()
        graph.replay()
        optimizer.step()
    torch.cuda.synchronize()
    assert optimizer._num_steps == 4
    _assert_models_match(models, references)


def test_clear_moments_zeroes_only_the_given_parameters() -> None:
    """The second model's linear weight loses both moments; every other element keeps them."""
    models = [_model(0), _model(1)]
    optimizer = FlatAdamW([_groups(model) for model in models], lr=_LR, max_grad_norm=_MAX_NORM)
    for step in range(2):
        torch.manual_seed(step)
        _set_gradients(models, copy.deepcopy(models), 1.0)
        optimizer.step()
    before = optimizer.state_dict()["state"]
    cleared = cast(nn.Linear, cast(nn.Sequential, models[1])[3]).weight
    optimizer.clear_moments([cleared])
    after = optimizer.state_dict()["state"]
    start = cleared.storage_offset()
    for name in ("exp_avgs", "exp_avg_sqs"):
        torch.testing.assert_close(after[name][0], before[name][0], atol=0.0, rtol=0.0)
        assert before[name][1][start : start + cleared.numel()].abs().sum() > 0
        assert after[name][1][start : start + cleared.numel()].abs().sum() == 0
        kept = torch.ones_like(after[name][1], dtype=torch.bool)
        kept[start : start + cleared.numel()] = False
        torch.testing.assert_close(after[name][1][kept], before[name][1][kept], atol=0.0, rtol=0.0)


def test_kernel_reaches_triton_unwrapped() -> None:
    """Triton resolves tl through the kernel's own globals and closure, which a jaxtyped wrapper would hide."""
    assert not hasattr(clip_adamw_kernel().fn, "__wrapped__")
