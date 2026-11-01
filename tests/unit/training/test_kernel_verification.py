# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The kernel check finds every kernel_patches list of a config, zeroes and restores dropouts, compares the patched steps
with the plain ones per module by the median over batches, names the patch at fault and leaves the policy as it found
it.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, PlanningConfig, TransfuserConfig
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.concurrent_seeds_training import SeedInputs
from py123d_garage.training.kernel_verification import (
    BatchErrors,
    StepOutcome,
    batch_errors,
    compare,
    kernel_patches_of,
    module_of,
    patches_not_running,
    verify_kernel_patches,
    with_kernel_patches,
    zero_dropouts,
)
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel check runs on the GPU")
_FUSE_QKV = "py123d_garage.policy.transfuser.network.encoder.transfuser:fuse_fusion_qkv"
_PERTURB = "tests.unit.training.test_kernel_verification:perturb_backbone"


@dataclass
class _Leaf:
    kernel_patches: list[str] = field(default_factory=list)
    name: str = ""


@dataclass
class _Tree:
    leaves: list[_Leaf] = field(default_factory=list)
    optional: _Leaf | None = None
    kernel_patches: list[str] = field(default_factory=list)


def test_kernel_patches_of_walks_nested_dataclasses_lists_and_optionals() -> None:
    tree = _Tree(
        leaves=[_Leaf(["a:b", "c:d"]), _Leaf([])],
        optional=_Leaf(["c:d", "e:f"]),
        kernel_patches=["g:h"],
    )
    assert kernel_patches_of(tree) == ["a:b", "c:d", "e:f", "g:h"]
    assert kernel_patches_of(_Tree(optional=None)) == []


def test_with_kernel_patches_filters_every_list_on_a_copy() -> None:
    tree = _Tree(leaves=[_Leaf(["a:b", "c:d"])], optional=_Leaf(["c:d"]), kernel_patches=["g:h"])
    cleared = with_kernel_patches(tree, lambda _: False)
    only_cd = with_kernel_patches(tree, lambda patch: patch == "c:d")
    assert kernel_patches_of(cleared) == []
    assert only_cd.leaves[0].kernel_patches == ["c:d"]
    assert only_cd.optional is not None
    assert only_cd.optional.kernel_patches == ["c:d"]
    assert only_cd.kernel_patches == []
    assert kernel_patches_of(tree) == ["a:b", "c:d", "g:h"]


def test_kernel_patches_of_reads_the_training_config() -> None:
    transfuser = TransfuserConfig(
        backbone_config=BackboneConfig(kernel_patches=["x:backbone"]),
        planning_config=PlanningConfig(kernel_patches=["x:planner", "x:backbone"]),
    )
    training_config = TrainingConfig(policy_config=PolicyConfig(transfuser_config=transfuser))
    assert kernel_patches_of(training_config.policy_config) == ["x:backbone", "x:planner"]


class _FloatDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = 0.1
        self.drop_prob = 0.2
        self.p = 0.3


def test_patches_not_running_names_every_requested_patch_whose_kernels_did_not_run() -> None:
    entries = [
        {"patch": "a:runs", "status": "runs"},
        {"patch": "b:torch", "status": "torch only"},
        {"patch": "c:unknown", "status": "unknown to the kernel registry"},
        {
            "patch": "d:shape",
            "status": "does not run: pinned_conv: shape (8, 64) is not in the engine table (12 calls)",
        },
        {"patch": "e:replaced", "status": "does not run: planner_swiglu_linear: never called, f:ffn replaces it"},
        {"patch": "f:hopper", "status": "falls back: swiglu is built on sm_90a instructions; GPU is sm_120"},
        {"patch": "g:partly", "status": "partly runs: fusion_upsample_add: map and grid differ (1 calls)"},
    ]
    assert patches_not_running(entries) == [
        "d:shape: pinned_conv: shape (8, 64) is not in the engine table (12 calls)",
        "e:replaced: planner_swiglu_linear: never called, f:ffn replaces it",
        "f:hopper: swiglu is built on sm_90a instructions; GPU is sm_120",
        "g:partly: fusion_upsample_add: map and grid differ (1 calls)",
    ]


def test_zero_dropouts_zeroes_modules_and_float_attributes_and_restores() -> None:
    module = nn.Sequential(nn.Dropout(0.3), nn.MultiheadAttention(8, 2, dropout=0.2), _FloatDropout())
    dropout = module[0]
    attention = module[1]
    floats = module[2]
    assert isinstance(dropout, nn.Dropout)
    assert isinstance(attention, nn.MultiheadAttention)
    assert isinstance(floats, _FloatDropout)
    restore = zero_dropouts(module)
    assert dropout.p == 0.0
    assert attention.dropout == 0.0
    assert floats.dropout == 0.0
    assert floats.drop_prob == 0.0
    assert floats.p == 0.3
    restore()
    assert dropout.p == 0.3
    assert attention.dropout == 0.2
    assert floats.dropout == 0.1
    assert floats.drop_prob == 0.2


def _outcome(loss: float, gradients: dict[str, list[float]]) -> StepOutcome:
    return StepOutcome(loss, {name: torch.tensor(values) for name, values in gradients.items()}, set())


def test_batch_errors_take_the_norm_over_each_module_and_name_the_problems() -> None:
    assert module_of("center_net_decoder.wh_head.2.weight") == "center_net_decoder.wh_head"
    reference = _outcome(1.0, {"head.a.weight": [1.0, 0.0], "head.a.bias": [0.0], "head.b": [2.0], "other.c": [1.0]})
    plain = _outcome(1.001, {"head.a.weight": [1.03, 0.0], "head.a.bias": [0.04], "head.b": [2.0], "other.c": [1.0]})
    patched = _outcome(1.002, {"head.a.weight": [1.0, 0.06], "head.a.bias": [0.08], "head.b": [2.0], "extra": [1.0]})

    errors = batch_errors(patched, plain, reference)

    assert errors.plain["head.a"] == pytest.approx(0.05)
    assert errors.patched["head.a"] == pytest.approx(0.1)
    assert errors.patched["head.b"] == errors.plain["head.b"] == 0.0
    assert errors.problems == [
        "extra: a gradient only with the patches",
        "other.c: a gradient only with no patches and fp32",
    ]
    same = _outcome(1.0, {"head.a.weight": [1.0, 0.0]})
    broken = batch_errors(_outcome(math.nan, {"head.a.weight": [math.inf, 0.0]}), same, same)
    assert broken.problems == ["loss nan with the patches", "head.a.weight: a non-finite gradient with the patches"]


def _batch(ratios: dict[str, float], patched_loss: float = 1.0) -> BatchErrors:
    """A batch whose every module lies ratio times as far from fp32 with the patches as without."""
    patched = {module: 0.01 * ratio for module, ratio in ratios.items()}
    return BatchErrors(patched_loss, 1.0, patched, dict.fromkeys(ratios, 0.01), [])


def test_compare_decides_on_the_median_over_batches_not_on_one_batch() -> None:
    # One batch's rounding can put a small module 20 times as far from fp32; the median over the batches does not.
    ratios = {"heads.noisy": 1.1, "heads.drift": 2.5, "heads.wrong": 3.5, "backbone.encoder": 1.0}
    batches = [_batch(ratios | {"heads.noisy": 20.0})] + [_batch(ratios) for _ in range(4)]

    comparison = compare(batches, 1e-2, 2.0, 3.0)

    assert comparison.module_ratios == pytest.approx(ratios)
    assert [failure.split(":")[0] for failure in comparison.failures] == ["heads.wrong"]
    assert "over kernel_check_refuse_factor 3" in comparison.failures[0]
    assert [warning.split(":")[0] for warning in comparison.warnings] == ["heads.drift"]
    assert (comparison.worst_module, comparison.worst_ratio) == ("heads.wrong", pytest.approx(3.5))


def test_compare_fails_on_any_batch_whose_loss_is_off_and_on_any_batch_problem() -> None:
    off = compare([_batch({"a.b": 1.0}), _batch({"a.b": 1.0}, patched_loss=1.02)], 1e-2, 2.0, 3.0)
    assert off.failures == [
        "batch 1: loss 1.02 with the patches, 1 without: relative error 0.02 over kernel_check_loss_rtol 0.01",
    ]
    assert off.loss_error == pytest.approx(0.02)
    broken = _batch({"a.b": 1.0})
    broken.problems.append("a.b.weight: a non-finite gradient with the patches")
    assert compare([broken], 1e-2, 2.0, 3.0).failures == ["batch 0: a.b.weight: a non-finite gradient with the patches"]
    assert compare([BatchErrors(1.0, 1.0, {"a.b": 0.0}, {"a.b": 0.0}, [])], 1e-2, 2.0, 3.0).failures == []
    assert compare([BatchErrors(1.0, 1.0, {"a.b": 0.1}, {"a.b": 0.0}, [])], 1e-2, 2.0, 3.0).worst_ratio == math.inf


def perturb_backbone(backbone: nn.Module) -> None:
    """A patch at fault: scales every output of the backbone by five percent."""

    def scale(module: nn.Module, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(value * 1.05 if isinstance(value, torch.Tensor) else value for value in output)

    backbone.register_forward_hook(scale)


def _policy_and_config(
    base_config: TransfuserConfig,
    patches: list[str],
    **training_fields: Any,
) -> tuple[TransfuserPolicy, TrainingConfig]:
    transfuser = TransfuserConfig(
        trajectory_offsets_us=base_config.trajectory_offsets_us,
        required_target_point_distances_m=base_config.required_target_point_distances_m,
        backbone_config=BackboneConfig(image_encoder_pretrained=False, kernel_patches=patches),
        planning_config=base_config.planning_config,
    )
    training_config = TrainingConfig(
        policy_config=PolicyConfig(transfuser_config=transfuser),
        offline_data_sources=[],
        compile_mode=None,
        **training_fields,
    )
    torch.manual_seed(0)
    policy = TransfuserPolicy(training_config.policy_config)
    policy.initialize()
    policy.prepare_training()
    policy = policy.to(memory_format=torch.channels_last).cuda().train()  # pyright: ignore[reportCallIssue]
    return policy, training_config


def _autocast() -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], torch.autocast("cuda", dtype=torch.bfloat16))


def _inputs(training_config: TrainingConfig, seed: int = 1) -> SeedInputs:
    transfuser = training_config.policy_config.transfuser_config
    assert transfuser is not None
    torch.manual_seed(seed)
    return (
        _features(transfuser, 2).to("cuda"),
        _labels(transfuser, batch_size=2).to("cuda"),
        _navigation(transfuser, 2).to("cuda"),
    )


def _batches(training_config: TrainingConfig) -> list[SeedInputs]:
    return [_inputs(training_config, seed) for seed in (1, 2, 3)]


def _check(
    policy: TransfuserPolicy,
    training_config: TrainingConfig,
    report_file: Path,
    inputs: list[SeedInputs] | None = None,
) -> dict[str, Any]:
    batches = _batches(training_config) if inputs is None else inputs
    return verify_kernel_patches(
        policy,
        training_config,
        batches,
        policy.loss_weights(0),
        _autocast,
        torch.device("cuda", 0),
        report_file,
    )


@_needs_gpu
def test_a_torch_only_patch_verifies_and_leaves_the_policy_as_found(config: TransfuserConfig, tmp_path: Path) -> None:  # noqa: F811
    policy, training_config = _policy_and_config(config, [_FUSE_QKV])
    inputs = _batches(training_config)
    buffers = {name: buffer.clone() for name, buffer in policy.named_buffers()}
    cuda_rng, cpu_rng = torch.cuda.get_rng_state(), torch.random.get_rng_state()
    report_file = tmp_path / "kernel_report.json"

    report = _check(policy, training_config, report_file, inputs)

    assert report["status"] == "verified"
    assert report["failures"] == []
    assert json.loads(report_file.read_text())["patches"][0]["patch"] == _FUSE_QKV
    assert report["warnings"] == []
    assert report["batches"] == 3
    assert report["gradients"]["worst_ratio"] < training_config.kernel_check_warn_factor
    for name, buffer in policy.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(module.p == 0.1 for module in policy.modules() if isinstance(module, nn.Dropout) and module.p)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)


@_needs_gpu
def test_a_patch_at_fault_is_named_and_refused(config: TransfuserConfig, tmp_path: Path) -> None:  # noqa: F811
    policy, training_config = _policy_and_config(config, [_FUSE_QKV, _PERTURB])
    report_file = tmp_path / "kernel_report.json"

    with pytest.raises(
        RuntimeError,
        match="patches at fault on their own: \\['" + _PERTURB.replace(".", "\\.") + "'\\]",
    ):
        _check(policy, training_config, report_file)

    report = json.loads(report_file.read_text())
    assert report["status"] == "mismatch"
    assert report["failures"]
    assert {entry["patch"]: bool(entry["failures"]) for entry in report["attribution"]} == {
        _FUSE_QKV: False,
        _PERTURB: True,
    }


@_needs_gpu
def test_a_mismatch_only_warns_with_refuse_off(
    config: TransfuserConfig,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy, training_config = _policy_and_config(config, [_PERTURB], kernel_check_refuse=False)
    with caplog.at_level(logging.WARNING):
        report = _check(policy, training_config, tmp_path / "kernel_report.json")
    assert report["status"] == "mismatch"
    assert "training anyway" in caplog.text
