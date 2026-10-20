# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Every kernel patch of the registry on a real sub-module: it keeps the state_dict keys, changes the module, changes
nothing more when applied twice, gives the plain module's outputs and gradients under bf16 autocast as closely as the
plain bf16 path gives fp32's, and on a device the registry rules out falls back bit-identically to the plain module.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser import registry
from py123d_garage_cuda.transfuser.registry import PATCHES

from py123d_garage.common.config_help import import_string
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, TransfuserConfig
from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import AdaptiveNormTransformerDecoder
from py123d_garage.policy.transfuser.network.encoder.transfuser import TransfuserBackbone
from py123d_garage.training.kernel_verification import StepOutcome, compare, exact_fp32
from tests.cuda_unit.transfuser.test_planner import _planner_masks
from tests.unit.policy.transfuser.test_transfuser_forward import _small_geometry, config  # noqa: F401

_CUDA = "py123d_garage_cuda.transfuser"
_FACTOR = 2.0
_FLOOR = 1e-3


@dataclass(frozen=True)
class Case:
    """A patch's real sub-module and the inputs it runs on."""

    module: Callable[[TransfuserConfig], nn.Module]
    inputs: Callable[[], tuple[torch.Tensor, ...]]
    dropout: float


def _backbone(architecture: str) -> Callable[[TransfuserConfig], nn.Module]:
    def build(transfuser_config: TransfuserConfig) -> nn.Module:
        backbone_config = BackboneConfig(
            image_encoder_pretrained=False,
            image_architecture=architecture,
            lidar_architecture="resnet34",
        )
        torch.manual_seed(0)
        small = replace(_small_geometry(transfuser_config), backbone_config=backbone_config)
        return TransfuserBackbone(small, False)

    return build


def _backbone_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    return torch.rand(2, 3, 64, 128) * 255.0, torch.rand(2, 1, 64, 64)


def _decoder(transfuser_config: TransfuserConfig) -> nn.Module:
    del transfuser_config
    torch.manual_seed(0)
    decoder = AdaptiveNormTransformerDecoder(256, 8, 2)
    for layer in decoder.layers:
        nn.init.normal_(cast(nn.Linear, layer.modulation).weight, 0.0, 0.02)
        nn.init.normal_(cast(nn.Linear, layer.modulation).bias, 0.0, 0.1)
    return decoder


def _decoder_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    self_mask, cross_mask = _planner_masks(64, 64, 73, 3)
    return torch.randn(4, 192, 256), torch.randn(4, 73, 256), torch.randn(4, 256), self_mask.cpu(), cross_mask.cpu()


CASES: dict[str, Case] = {
    f"{_CUDA}.fusion:patch_fusion_upsample_add": Case(_backbone("resnet34"), _backbone_inputs, 0.1),
    f"{_CUDA}.linear_convs:patch_patchify_stems": Case(_backbone("convnext_tiny"), _backbone_inputs, 0.1),
    f"{_CUDA}.linear_convs:patch_pointwise_convs": Case(_backbone("resnet34"), _backbone_inputs, 0.1),
    f"{_CUDA}.adaptive_norm:patch_planner_adaptive_norm": Case(_decoder, _decoder_inputs, 0.1),
    f"{_CUDA}.convnext:patch_depthwise_convs": Case(_backbone("convnext_tiny"), _backbone_inputs, 0.1),
    f"{_CUDA}.planner:patch_planner_attention": Case(_decoder, _decoder_inputs, 0.1),
    f"{_CUDA}.planner:patch_planner_swiglu": Case(_decoder, _decoder_inputs, 0.1),
    f"{_CUDA}.planner:patch_planner_feed_forward": Case(_decoder, _decoder_inputs, 0.1),
    f"{_CUDA}.regnet:patch_grouped_convs": Case(_backbone("regnety_032"), _backbone_inputs, 0.1),
    f"{_CUDA}.regnet:patch_regnet_blocks": Case(_backbone("regnety_032"), _backbone_inputs, 0.1),
    f"{_CUDA}.resnet:fuse_stem": Case(_backbone("resnet34"), _backbone_inputs, 0.1),
    f"{_CUDA}.resnet:pin_conv_engines": Case(_backbone("resnet34"), _backbone_inputs, 0.1),
}


def test_every_registered_patch_has_a_case() -> None:
    assert set(CASES) == set(PATCHES)


def _description(module: nn.Module) -> list[tuple[str, str, dict[str, Any]]]:
    """Every sub-module's class and hook attributes, which a patch changes and a second application must not."""
    return [
        (name, type(child).__name__, {key: value for key, value in vars(child).items() if key.startswith("fused_")})
        for name, child in module.named_modules()
    ]


def _step(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    context: Callable[[], AbstractContextManager[None]],
) -> StepOutcome:
    """One forward and backward under a fixed seed on the outputs' squared mean; the outputs join the gradients as out/i."""
    for parameter in module.parameters():
        parameter.grad = None
    torch.manual_seed(2)
    with context():
        outputs = module(*inputs)
    outputs = [out for out in (outputs if isinstance(outputs, tuple) else (outputs,)) if out is not None]
    loss = torch.stack([out.float().square().mean() for out in outputs]).sum()
    loss.backward()
    gradients = {name: p.grad.detach().float().clone() for name, p in module.named_parameters() if p.grad is not None}
    gradients |= {f"out/{index}": out.detach().float().clone() for index, out in enumerate(outputs)}
    return StepOutcome(float(loss.item()), gradients, set())


def _autocast() -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], torch.autocast("cuda", dtype=torch.bfloat16))


def _set_dropout(module: nn.Module, p: float) -> None:
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = p
        elif isinstance(child, nn.MultiheadAttention) or isinstance(vars(child).get("dropout"), float):
            setattr(child, "dropout", p)  # noqa: B010


@pytest.mark.parametrize("patch", sorted(CASES))
def test_patch_keeps_state_changes_module_once_and_matches_plain(
    patch: str,
    config: TransfuserConfig,  # noqa: F811
) -> None:
    case = CASES[patch]
    plain = case.module(config).cuda().train()
    _set_dropout(plain, 0.0)
    inputs = tuple(tensor.cuda() for tensor in case.inputs())
    patched = copy.deepcopy(plain)
    import_string(patch)(patched)
    assert list(patched.state_dict()) == list(plain.state_dict())
    assert _description(patched) != _description(plain), "the patch changed nothing"
    once = _description(patched)
    import_string(patch)(patched)
    assert _description(patched) == once, "a second application changed the module further"

    with exact_fp32():
        reference = _step(plain, inputs, nullcontext)
    plain_outcome = _step(plain, inputs, _autocast)
    patched_outcome = _step(patched, inputs, _autocast)
    comparison = compare(patched_outcome, plain_outcome, reference, 1e-2, _FACTOR, _FLOOR)
    assert comparison.failures == [], comparison.failures


@pytest.mark.parametrize("patch", sorted(name for name, ops in PATCHES.items() if ops))
def test_patch_falls_back_bit_identically_where_the_device_is_ruled_out(
    patch: str,
    config: TransfuserConfig,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[patch]
    plain = case.module(config).cuda().train()
    _set_dropout(plain, case.dropout)
    inputs = tuple(tensor.cuda() for tensor in case.inputs())
    patched = copy.deepcopy(plain)
    import_string(patch)(patched)
    monkeypatch.setattr(registry, "unsupported_reason", lambda *_, **__: "ruled out by the test")
    plain_outcome = _step(plain, inputs, _autocast)
    patched_outcome = _step(patched, inputs, _autocast)
    assert patched_outcome.loss == plain_outcome.loss
    for name, gradient in plain_outcome.gradients.items():
        assert torch.equal(patched_outcome.gradients[name], gradient), name


@pytest.mark.parametrize("patch", sorted(name for name, ops in PATCHES.items() if ops))
def test_patch_falls_back_bit_identically_on_the_cpu(
    patch: str,
    config: TransfuserConfig,  # noqa: F811
) -> None:
    case = CASES[patch]
    plain = case.module(config).train()
    _set_dropout(plain, case.dropout)
    inputs = case.inputs()
    patched = copy.deepcopy(plain)
    import_string(patch)(patched)
    for op in PATCHES[patch]:
        assert "CUDA devices only" in (registry.unsupported_reason(op, torch.device("cpu")) or "")
    plain_outcome = _step(plain, inputs, nullcontext)
    patched_outcome = _step(patched, inputs, nullcontext)
    assert patched_outcome.loss == plain_outcome.loss
    for name, gradient in plain_outcome.gradients.items():
        assert torch.equal(patched_outcome.gradients[name], gradient), name
