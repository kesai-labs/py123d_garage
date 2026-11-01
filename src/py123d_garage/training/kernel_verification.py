# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Checks before training that a policy's kernel patches give the loss and parameter gradients of the plain torch policy
within bf16 rounding over several batches, and writes the run's kernel report.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.metadata
import json
import logging
import math
import operator
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import torch
import torch.version
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.common.config_help import build_from_string
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.training.concurrent_seeds_training import SeedInputs, policy_forward_and_loss

LOG = logging.getLogger(__name__)

KERNEL_PATCHES_FIELD = "kernel_patches"
KERNEL_OPS_NAMESPACE = "py123d_garage_cuda"
_SEED = 0
_DROPOUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
_DROPOUT_ATTRIBUTES = ("dropout", "drop_prob")

ConfigT = TypeVar("ConfigT")


def _kernel_patches_fields(config: object) -> list[tuple[object, str]]:
    """
    Finds every kernel_patches list under a config.

    Args:
        config: a config dataclass, or a value inside one.

    Returns:
        the dataclass instance and field name of every list, in config order.
    """
    found: list[tuple[object, str]] = []
    if is_dataclass(config) and not isinstance(config, type):
        for field in fields(config):
            value = getattr(config, field.name)
            if field.name == KERNEL_PATCHES_FIELD and isinstance(value, list):
                found.append((config, field.name))
            else:
                found.extend(_kernel_patches_fields(value))
    elif isinstance(config, (list, tuple)):
        for item in cast("list[object]", config):
            found.extend(_kernel_patches_fields(item))
    return found


def kernel_patches_of(config: object) -> list[str]:
    """
    Lists every kernel patch a config names in its kernel_patches fields, nested dataclasses and lists included.

    Args:
        config: the config dataclass.

    Returns:
        the "module:function" patches in config order, each once.
    """
    patches: list[str] = []
    for container, name in _kernel_patches_fields(config):
        for patch in cast("list[str]", getattr(container, name)):
            if patch not in patches:
                patches.append(patch)
    return patches


def with_kernel_patches(config: ConfigT, kept: Callable[[str], bool]) -> ConfigT:
    """
    Copies a config keeping only the kernel patches kept accepts.

    Args:
        config: the config dataclass.
        kept: decides per patch.

    Returns:
        the deep copy.
    """
    copied = copy.deepcopy(config)
    for container, name in _kernel_patches_fields(copied):
        setattr(container, name, [patch for patch in cast("list[str]", getattr(container, name)) if kept(patch)])
    return copied


def zero_dropouts(module: nn.Module) -> Callable[[], None]:
    """
    Sets every dropout probability inside module to zero: the dropout modules' p, and the float dropout and drop_prob
    attributes of the other modules, such as nn.MultiheadAttention and timm's DropPath.

    Args:
        module: the module to change in place.

    Returns:
        the function that restores the probabilities.
    """
    saved: list[tuple[nn.Module, str, float]] = []
    for child in module.modules():
        names = ("p",) if isinstance(child, _DROPOUT_MODULES) else _DROPOUT_ATTRIBUTES
        for name in names:
            value = child.__dict__.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value:
                saved.append((child, name, float(value)))
                setattr(child, name, 0.0)

    def restore() -> None:
        """Puts the probabilities back."""
        for child, name, value in saved:
            setattr(child, name, value)

    return restore


class _KernelOpsRecorder(TorchDispatchMode):
    """Records the names of the py123d_garage_cuda ops dispatched while it is active."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.ops: set[str] = set()

    @override
    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Note the op when it is one of the package's, then run it."""
        if func.namespace == KERNEL_OPS_NAMESPACE:
            self.ops.add(func.name().split("::")[-1])
        return func(*args, **(kwargs or {}))


@dataclass
class StepOutcome:
    """One forward and backward of a policy: the loss, every parameter gradient and the kernel ops that ran."""

    loss: float
    gradients: dict[str, Tensor]
    kernel_ops: set[str]


@contextmanager
def exact_fp32() -> Iterator[None]:
    """Runs fp32 matrix products and convolutions without TF32 tensor cores, for a reference step."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def run_step(
    policy: AnyPolicy,
    inputs: SeedInputs,
    loss_weights: Mapping[str, float],
    channels_last: bool,
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
) -> StepOutcome:
    """
    Runs one forward and backward of a policy on a seed's inputs under a fixed seed, leaving the generators as found.

    Args:
        policy: the policy, on device and in training mode
        inputs: the seed's features, labels and navigation on device
        loss_weights: the weight of every loss by name
        channels_last: convert the features and labels to channels-last first
        train_step_context: the trainer's context around a training step, e.g. its autocast; nullcontext runs in fp32
        device: the device the policy runs on

    Returns:
        the outcome; the gradients are fp32 copies on device
    """
    features, labels, navigation = inputs
    for parameter in policy.parameters():
        parameter.grad = None
    recorder = _KernelOpsRecorder()
    with torch.random.fork_rng(devices=[device]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(_SEED)  # pyright: ignore[reportUnknownMemberType]
        with recorder, train_step_context():
            total_loss, _, _ = policy_forward_and_loss(
                policy,
                features,
                labels,
                navigation,
                loss_weights,
                channels_last,
            )
        with recorder:
            total_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.cuda.synchronize(device)
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }
    return StepOutcome(float(total_loss.item()), gradients, recorder.ops)


def run_step_without_trace(
    policy: AnyPolicy,
    inputs: SeedInputs,
    loss_weights: Mapping[str, float],
    channels_last: bool,
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
) -> StepOutcome:
    """
    run_step on a policy that trains afterwards: with its dropouts at zero for the run, its buffers and dropout
    probabilities restored and its gradients cleared.

    Args:
        policy: the policy, on device and in training mode
        inputs: the seed's features, labels and navigation on device
        loss_weights: the weight of every loss by name
        channels_last: convert the features and labels to channels-last first
        train_step_context: the trainer's context around a training step, e.g. its autocast
        device: the device the policy runs on

    Returns:
        the outcome
    """
    # The check runs before fit, when the constructor still has every module in train mode; the trainer's train() is
    # what puts a frozen backbone into eval mode (TransfuserPolicy.train), as it is in the rebuilt copies.
    policy.train()
    saved_buffers = {name: buffer.clone() for name, buffer in policy.named_buffers()}
    restore_dropouts = zero_dropouts(policy)
    try:
        return run_step(policy, inputs, loss_weights, channels_last, train_step_context, device)
    finally:
        restore_dropouts()
        with torch.no_grad():
            for name, buffer in policy.named_buffers():
                buffer.copy_(saved_buffers[name])
        for parameter in policy.parameters():
            parameter.grad = None


def build_policy(
    policy_config: PolicyConfig,
    state_dict: Mapping[str, Tensor],
    device: torch.device,
    channels_last: bool,
) -> AnyPolicy:
    """
    Builds a policy from a config as the training does, loads it with a trained policy's weights and zeroes its
    dropouts.

    Args:
        policy_config: the policy config, with the kernel patches to apply
        state_dict: the weights to load, strictly
        device: the device the policy goes to
        channels_last: put the policy in channels-last memory format, as the training does

    Returns:
        the policy, in training mode
    """
    with torch.random.fork_rng(devices=[device]):  # pyright: ignore[reportUnknownMemberType]
        policy = cast(AnyPolicy, build_from_string(policy_config, AbstractPolicy))
        policy.build_model(from_local_checkpoint=True)
        policy.prepare_training()
    policy.load_state_dict(state_dict)
    if channels_last:
        policy = cast(AnyPolicy, policy.to(memory_format=torch.channels_last))  # pyright: ignore[reportCallIssue]
    policy = policy.to(device).train()
    zero_dropouts(policy)
    return policy


# A module is a parameter name's first parts, such as backbone.image_encoder or center_net_decoder.wh_head. A module's
# gradient holds enough elements that bf16 rounding averages out; a single small tensor's can lie 20 times as far from
# fp32 as the plain step's on one batch for rounding alone.
_MODULE_DEPTH: Final[int] = 2


def module_of(parameter: str) -> str:
    """
    Names the module a parameter belongs to for the check.

    Args:
        parameter: the parameter's name in the policy.

    Returns:
        the first _MODULE_DEPTH parts of the name.
    """
    return ".".join(parameter.split(".")[:_MODULE_DEPTH])


@dataclass
class BatchErrors:
    """One batch of the check: the patched and plain losses and each module's gradient distance from the fp32 step."""

    patched_loss: float
    plain_loss: float
    patched: dict[str, float]
    plain: dict[str, float]
    problems: list[str]


def batch_errors(patched: StepOutcome, plain: StepOutcome, reference: StepOutcome) -> BatchErrors:
    """
    Reduces one batch's three steps to each module's distance from the fp32 step: the norm over every gradient of the
    module's parameters.

    Args:
        patched: the patched policy's step
        plain: the plain policy's step under the same precision
        reference: the plain policy's step in fp32

    Returns:
        the batch's errors; problems names a non-finite patched loss or gradient and a gradient missing from a step
    """
    problems: list[str] = []
    if not math.isfinite(patched.loss):
        problems.append(f"loss {patched.loss} with the patches")
    outcomes = {"the patches": patched, "no patches": plain, "fp32": reference}
    patched_squares: defaultdict[str, float] = defaultdict(float)
    plain_squares: defaultdict[str, float] = defaultdict(float)
    for name in sorted(set(patched.gradients) | set(plain.gradients) | set(reference.gradients)):
        present = [label for label, outcome in outcomes.items() if name in outcome.gradients]
        if len(present) < len(outcomes):
            problems.append(f"{name}: a gradient only with {' and '.join(present)}")
            continue
        if not bool(torch.isfinite(patched.gradients[name]).all().item()):
            problems.append(f"{name}: a non-finite gradient with the patches")
            continue
        module = module_of(name)
        patched_squares[module] += float((patched.gradients[name] - reference.gradients[name]).square().sum().item())
        plain_squares[module] += float((plain.gradients[name] - reference.gradients[name]).square().sum().item())
    return BatchErrors(
        patched.loss,
        plain.loss,
        {module: math.sqrt(value) for module, value in patched_squares.items()},
        {module: math.sqrt(value) for module, value in plain_squares.items()},
        problems,
    )


@dataclass
class Comparison:
    """How the patched steps stand against the plain bf16 steps over the check's batches, both measured from fp32."""

    loss_error: float
    module_ratios: dict[str, float]
    worst_module: str | None
    worst_ratio: float
    failures: list[str]
    warnings: list[str]


def _error_ratio(patched_error: float, plain_error: float) -> float:
    """
    Divides a module's patched error by its plain error; two zero errors are the same step.

    Args:
        patched_error: the patched gradient's distance from fp32
        plain_error: the plain gradient's distance from fp32

    Returns:
        the ratio, infinite for a patched error where the plain step has none
    """
    if plain_error > 0.0:
        return patched_error / plain_error
    return 1.0 if patched_error == 0.0 else math.inf


def compare(batches: list[BatchErrors], loss_rtol: float, warn_factor: float, refuse_factor: float) -> Comparison:
    """
    Compares the patched steps with the plain ones over the check's batches. Fails on a patched loss further than
    loss_rtol from the plain loss on any batch, on any batch's problem, and on a module whose patched gradient lies more
    than refuse_factor times as far from the fp32 gradient as the plain one, by the median over the batches; warns above
    warn_factor.

    Args:
        batches: every batch's errors
        loss_rtol: the loss tolerance, relative to the plain loss
        warn_factor: the median ratio of patched to plain error above which a module warns
        refuse_factor: the median ratio above which a module fails

    Returns:
        the comparison; failures is empty where the steps agree
    """
    failures: list[str] = []
    warnings: list[str] = []
    loss_error = 0.0
    for index, batch in enumerate(batches):
        error = abs(batch.patched_loss - batch.plain_loss) / max(abs(batch.plain_loss), 1e-30)
        loss_error = max(loss_error, error)
        if error > loss_rtol:
            failures.append(
                f"batch {index}: loss {batch.patched_loss:.6g} with the patches, {batch.plain_loss:.6g} without: "
                f"relative error {error:.3g} over kernel_check_loss_rtol {loss_rtol:.3g}",
            )
        failures.extend(f"batch {index}: {problem}" for problem in batch.problems)
    modules = sorted({module for batch in batches for module in batch.patched})
    module_ratios = {
        module: statistics.median(
            _error_ratio(batch.patched[module], batch.plain[module]) for batch in batches if module in batch.patched
        )
        for module in modules
    }
    worst_module, worst_ratio = max(module_ratios.items(), key=lambda item: item[1], default=(None, 0.0))
    for module, ratio in sorted(module_ratios.items(), key=lambda item: -item[1]):
        message = (
            f"{module}: the patched gradient lies {ratio:.3g} times as far from fp32 as the plain one, median of "
            f"{len(batches)} batches"
        )
        if ratio > refuse_factor:
            failures.append(f"{message}, over kernel_check_refuse_factor {refuse_factor:.3g}")
        elif ratio > warn_factor:
            warnings.append(f"{message}, over kernel_check_warn_factor {warn_factor:.3g}")
    return Comparison(loss_error, module_ratios, worst_module, worst_ratio, failures, warnings)


def _registry() -> Any | None:
    """
    Imports the kernel registry of py123d_garage_cuda.

    Returns:
        the registry module, or None without the package
    """
    try:
        return importlib.import_module(f"{KERNEL_OPS_NAMESPACE}.transfuser.registry")
    except ImportError:
        return None


RULED_OUT = "falls back: "
NOT_RUNNING = "does not run: "
PARTLY_RUNNING = "partly runs: "
_SHOWN_REASONS = 3


def patches_not_running(entries: list[dict[str, Any]]) -> list[str]:
    """
    Names the requested patches whose kernels fell back to torch on any call of the check, each with why: the device,
    the build or its libraries rule them out, the batch falls outside what they handle, or another patch replaces them.

    Args:
        entries: the kernel report's patch entries.

    Returns:
        "patch: reason" per patch, in report order; torch-only patches and patches unknown to the registry are none.
    """
    return [
        f"{entry['patch']}: {entry['status'].split(': ', 1)[1]}"
        for entry in entries
        if entry["status"].startswith((RULED_OUT, NOT_RUNNING, PARTLY_RUNNING))
    ]


def _fallback_text(family: str, counts: dict[str, int]) -> str:
    """
    Describes the recorded fallbacks of one kernel family, the most frequent reason first.

    Args:
        family: the kernel family.
        counts: the number of calls per reason.

    Returns:
        "family: reason (n calls)" for the most frequent reasons.
    """
    reasons = sorted(counts.items(), key=lambda item: -item[1])
    shown = "; ".join(f"{family}: {reason} ({count} calls)" for reason, count in reasons[:_SHOWN_REASONS])
    if len(reasons) <= _SHOWN_REASONS:
        return shown
    return f"{shown}; and {len(reasons) - _SHOWN_REASONS} more reasons in the kernel report"


def _patch_report(
    patch: str,
    patches: list[str],
    registry: Any | None,
    device: torch.device,
    ran: set[str],
    fallbacks: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """
    Describes one patch: its kernel families per the registry, which of their ops ran, why the others did not, and the
    calls that fell back to torch.

    Args:
        patch: the "module:function" patch
        patches: every patch the policy config requests
        registry: the kernel registry, or None
        device: the device the step ran on
        ran: the kernel ops the patched step dispatched
        fallbacks: per kernel family, the calls that fell back to torch for each reason

    Returns:
        the patch's report entry
    """
    if registry is None or patch not in registry.PATCHES:
        return {"patch": patch, "ops": None, "ran": None, "status": "unknown to the kernel registry", "fallbacks": {}}
    families: list[str] = list(registry.PATCHES[patch])
    if not families:
        return {"patch": patch, "ops": [], "ran": [], "status": "torch only", "fallbacks": {}}
    ran_ops = sorted(op for op in ran if any(op in registry.OPS[family].ops for family in families))
    recorded = {family: fallbacks[family] for family in families if family in fallbacks}
    entry: dict[str, Any] = {"patch": patch, "ops": families, "ran": ran_ops, "fallbacks": recorded}
    reasons = [registry.unsupported_reason(family, device) for family in families]
    if any(reasons):
        return entry | {"status": RULED_OUT + "; ".join(reason for reason in reasons if reason)}
    idle = [family for family in families if not any(op in registry.OPS[family].ops for op in ran)]
    if not idle:
        if not recorded:
            return entry | {"status": "runs"}
        return entry | {
            "status": PARTLY_RUNNING + "; ".join(_fallback_text(family, recorded[family]) for family in recorded),
        }
    replacing = [other for other, replaced in registry.REPLACES.items() if replaced == patch and other in patches]
    why = [
        _fallback_text(family, recorded[family])
        if family in recorded
        else f"{family}: never called, {replacing[0]} replaces it"
        if replacing
        else f"{family}: never called on this batch"
        for family in idle
    ]
    return entry | {"status": NOT_RUNNING + "; ".join(why)}


def _environment_report(device: torch.device) -> dict[str, Any]:
    """
    Describes the software and the device of the check.

    Args:
        device: the device the step ran on

    Returns:
        the package, torch, CUDA and cuDNN versions and the device
    """
    try:
        package_version: str | None = importlib.metadata.version("py123d-garage-cuda")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    properties = cast(Any, torch.cuda.get_device_properties)(device)
    return {
        "py123d_garage_cuda": package_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": {
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "shared_memory_per_block_optin": properties.shared_memory_per_block_optin,
        },
    }


def _measure(
    policy: AnyPolicy,
    plain_policy: AnyPolicy,
    inputs: Sequence[SeedInputs],
    loss_weights: Mapping[str, float],
    channels_last: bool,
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
    restore: bool,
) -> tuple[list[BatchErrors], set[str]]:
    """
    Runs the check's three steps on every batch: the patched policy and the plain one in the step's precision, and the
    plain one in fp32.

    Args:
        policy: the patched policy
        plain_policy: the plain torch policy with the same weights
        inputs: every batch's features, labels and navigation on device
        loss_weights: the weight of every loss by name
        channels_last: convert the features and labels to channels-last first
        train_step_context: the trainer's context around a training step, e.g. its autocast
        device: the device the policies run on
        restore: leave the patched policy as found, for a policy that trains afterwards

    Returns:
        every batch's errors and the kernel ops the patched steps dispatched
    """
    errors: list[BatchErrors] = []
    kernel_ops: set[str] = set()
    for batch_inputs in inputs:
        step = run_step_without_trace if restore else run_step
        patched = step(policy, batch_inputs, loss_weights, channels_last, train_step_context, device)
        plain = run_step(plain_policy, batch_inputs, loss_weights, channels_last, train_step_context, device)
        with exact_fp32():
            reference = run_step(plain_policy, batch_inputs, loss_weights, channels_last, nullcontext, device)
        errors.append(batch_errors(patched, plain, reference))
        kernel_ops |= patched.kernel_ops
    return errors, kernel_ops


def verify_kernel_patches(
    policy: AnyPolicy,
    training_config: TrainingConfig,
    inputs: Sequence[SeedInputs],
    loss_weights: Mapping[str, float],
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
    report_file: Path | None,
) -> dict[str, Any]:
    """
    Checks a patched policy against the plain torch policy built from the same config and weights: one forward and
    backward each on every batch, the same seed and no dropout, plus one of the plain policy in fp32 as the reference.
    The loss of every batch, and every module's gradient by the median over the batches, must lie within the training
    config's kernel_check tolerances; see compare. On a failure, finds the patches at fault by rebuilding with one patch
    at a time, and refuses to train unless kernel_check_refuse is off. A requested patch whose kernels fall back to
    torch on some call refuses too, with kernel_check_require_patches on; otherwise such a patch is a fast path, and the
    report says where it ran. Writes the kernel report.

    Args:
        policy: policy 0 with its kernel patches applied, on device
        training_config: the training config, for the policy config and the tolerances
        inputs: seed 0's features, labels and navigation of every batch, on device
        loss_weights: the weight of every loss by name
        train_step_context: the trainer's context around a training step, e.g. its autocast
        device: the device the policy runs on
        report_file: where the kernel report goes; None writes none

    Returns:
        the kernel report

    Raises:
        RuntimeError: on a failure with kernel_check_refuse on, or on a patch whose kernels do not run with
            kernel_check_require_patches on
    """
    report = _check(policy, training_config, inputs, loss_weights, train_step_context, device)
    if report_file is not None:
        report_file.write_text(json.dumps(report, indent=2) + "\n")
    _act_on(report, training_config, report_file)
    return report


def _check(
    policy: AnyPolicy,
    training_config: TrainingConfig,
    inputs: Sequence[SeedInputs],
    loss_weights: Mapping[str, float],
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
) -> dict[str, Any]:
    """
    Runs the check of verify_kernel_patches on the given batches.

    Args:
        policy: policy 0 with its kernel patches applied, on device
        training_config: the training config, for the policy config and the tolerances
        inputs: every batch's features, labels and navigation on device
        loss_weights: the weight of every loss by name
        train_step_context: the trainer's context around a training step, e.g. its autocast
        device: the device the policy runs on

    Returns:
        the kernel report
    """
    policy_config = training_config.policy_config
    channels_last = training_config.channels_last
    tolerances = (
        training_config.kernel_check_loss_rtol,
        training_config.kernel_check_warn_factor,
        training_config.kernel_check_refuse_factor,
    )
    patches = kernel_patches_of(policy_config)
    registry = _registry()
    without_registry: dict[str, dict[str, int]] = {}
    recording = cast(
        "AbstractContextManager[dict[str, dict[str, int]]]",
        registry.recording_fallbacks() if registry is not None else nullcontext(without_registry),
    )
    state_dict = {name: tensor.detach() for name, tensor in policy.state_dict().items()}
    plain_policy = build_policy(with_kernel_patches(policy_config, lambda _: False), state_dict, device, channels_last)
    measure = functools.partial(
        _measure,
        plain_policy=plain_policy,
        inputs=inputs,
        loss_weights=loss_weights,
        channels_last=channels_last,
        train_step_context=train_step_context,
        device=device,
    )
    with recording as fallbacks:
        errors, kernel_ops = measure(policy, restore=True)
    comparison = compare(errors, *tolerances)
    attribution: list[dict[str, Any]] = []
    if comparison.failures:
        for patch in patches:
            only_this = with_kernel_patches(policy_config, functools.partial(operator.eq, patch))
            single = build_policy(only_this, state_dict, device, channels_last)
            failures = compare(measure(single, restore=False)[0], *tolerances).failures
            attribution.append({"patch": patch, "failures": failures})
            del single
    del plain_policy
    torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "status": "verified" if not comparison.failures else "mismatch",
        **_environment_report(device),
        "patches": [_patch_report(patch, patches, registry, device, kernel_ops, fallbacks) for patch in patches],
        "kernel_ops_run": sorted(kernel_ops),
        "batches": len(inputs),
        "loss": {"max_relative_error": comparison.loss_error, "rtol": training_config.kernel_check_loss_rtol},
        "gradients": {
            "worst_module": comparison.worst_module,
            "worst_ratio": comparison.worst_ratio,
            "warn_factor": training_config.kernel_check_warn_factor,
            "refuse_factor": training_config.kernel_check_refuse_factor,
            "module_ratios": comparison.module_ratios,
        },
        "failures": comparison.failures,
        "warnings": comparison.warnings,
        "attribution": attribution,
    }
    report["not_running"] = patches_not_running(report["patches"])
    return report


def _act_on(report: dict[str, Any], training_config: TrainingConfig, report_file: Path | None) -> None:
    """
    Logs a kernel report and refuses to train where the training config asks to.

    Args:
        report: the kernel report
        training_config: the training config, for kernel_check_refuse and kernel_check_require_patches
        report_file: where the report went, for the messages; None for none

    Raises:
        RuntimeError: on a failure with kernel_check_refuse on, or on a patch whose kernels do not run with
            kernel_check_require_patches on
    """
    see = f"; see {report_file}" if report_file is not None else ""
    summary = ", ".join(f"{entry['patch'].split(':')[-1]}: {entry['status']}" for entry in report["patches"])
    gradients = report["gradients"]
    LOG.info(
        f"Kernel check over {report['batches']} batches: loss relative error up to "
        f"{report['loss']['max_relative_error']:.3g}; the worst module, {gradients['worst_module']}, lies "
        f"{gradients['worst_ratio']:.3g} times as far from fp32 as without the patches; {summary}",
    )
    for warning in report["warnings"]:
        LOG.warning(f"Kernel check: {warning}")
    if report["not_running"]:
        message = "Requested kernel patches fall back to torch on some calls: " + "; ".join(report["not_running"])
        if training_config.kernel_check_require_patches:
            raise RuntimeError(
                f"{message}. Drop them from kernel_patches, fix their kernels, or set "
                f"kernel_check_require_patches=false to run them as fast paths{see}",
            )
        LOG.info(f"{message}; those calls run the torch path")
    if not report["failures"]:
        return
    at_fault = [entry["patch"] for entry in report["attribution"] if entry["failures"]]
    message = (
        f"The kernel patches change the training step: {len(report['failures'])} failures, the first "
        f"{report['failures'][0]}; patches at fault on their own: {at_fault or 'none, only together'}{see}"
    )
    if training_config.kernel_check_refuse:
        raise RuntimeError(message)
    LOG.warning(f"{message}; kernel_check_refuse is off, training anyway")
