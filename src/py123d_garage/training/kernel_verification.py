# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Checks before training that a policy's kernel patches give the loss and parameter gradients of the plain torch policy,
and writes the run's kernel report.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.metadata
import json
import logging
import operator
from collections.abc import Callable, Iterator, Mapping
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


@dataclass
class Comparison:
    """How a patched bf16 step stands against the plain bf16 step, both measured from the plain fp32 step."""

    loss_error: float
    worst_parameter: str | None
    worst_margin: float
    worst_patched_error: float
    worst_plain_error: float
    failures: list[str]


def _norm(tensor: Tensor) -> float:
    """
    Computes a tensor's Euclidean norm.

    Args:
        tensor: the tensor

    Returns:
        the norm
    """
    return float(tensor.square().sum().sqrt().item())


# A gradient smaller than the plain bf16 step's own error from fp32 is rounding noise in both bf16 steps, such as the
# gradient of a key norm's bias, which softmax cancels; the ratio of two noise norms says nothing about the kernels,
# so there only a gap of this many times the plain error fails.
_UNRESOLVED_GRAD_FACTOR: Final[float] = 10.0


def compare(
    patched: StepOutcome,
    plain: StepOutcome,
    reference: StepOutcome,
    loss_rtol: float,
    grad_factor: float,
    grad_floor: float,
) -> Comparison:
    """
    Compares a patched step with the plain one: the patched loss within loss_rtol of the plain loss, and every patched
    gradient at most grad_factor times as far from the fp32 reference gradient as the plain gradient is, plus grad_floor
    times the norm of that parameter's reference gradient. A parameter whose reference gradient is smaller than the plain
    gradient's error is unresolved in bf16 and gets _UNRESOLVED_GRAD_FACTOR instead of grad_factor.

    Args:
        patched: the patched policy's step
        plain: the plain policy's step under the same precision
        reference: the plain policy's step in fp32
        loss_rtol: the loss tolerance, relative to the plain loss
        grad_factor: how many times further from the reference a patched gradient may lie than the plain one
        grad_floor: the error allowed on top, as a fraction of the parameter's reference gradient norm

    Returns:
        the comparison; failures is empty where the steps agree
    """
    failures: list[str] = []
    loss_error = abs(patched.loss - plain.loss) / max(abs(plain.loss), 1e-30)
    if loss_error > loss_rtol:
        failures.append(
            f"loss {patched.loss:.6g} with the patches, {plain.loss:.6g} without: relative error {loss_error:.3g} "
            f"over kernel_check_loss_rtol {loss_rtol:.3g}",
        )
    worst_parameter: str | None = None
    worst_margin = worst_patched_error = worst_plain_error = 0.0
    outcomes = {"the patches": patched, "no patches": plain, "fp32": reference}
    for name in sorted(set(patched.gradients) | set(plain.gradients) | set(reference.gradients)):
        present = [label for label, outcome in outcomes.items() if name in outcome.gradients]
        if len(present) < len(outcomes):
            failures.append(f"{name}: a gradient only with {' and '.join(present)}")
            continue
        patched_error = _norm(patched.gradients[name] - reference.gradients[name])
        plain_error = _norm(plain.gradients[name] - reference.gradients[name])
        reference_norm = _norm(reference.gradients[name])
        factor = grad_factor if reference_norm >= plain_error else max(grad_factor, _UNRESOLVED_GRAD_FACTOR)
        margin = patched_error / max(factor * plain_error + grad_floor * reference_norm, 1e-30)
        if margin > worst_margin:
            worst_parameter, worst_margin = name, margin
            worst_patched_error, worst_plain_error = patched_error, plain_error
        if margin > 1.0:
            failures.append(
                f"{name}: {patched_error:.3g} from the fp32 gradient with the patches, {plain_error:.3g} without; "
                f"allowed {'kernel_check_grad_factor' if factor == grad_factor else 'an unresolved gradient'} "
                f"{factor:.3g} x {plain_error:.3g} + kernel_check_grad_floor "
                f"{grad_floor:.3g} x {reference_norm:.3g}",
            )
    return Comparison(loss_error, worst_parameter, worst_margin, worst_patched_error, worst_plain_error, failures)


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


def verify_kernel_patches(
    policy: AnyPolicy,
    training_config: TrainingConfig,
    inputs: SeedInputs,
    loss_weights: Mapping[str, float],
    train_step_context: Callable[[], AbstractContextManager[None]],
    device: torch.device,
    report_file: Path | None,
) -> dict[str, Any]:
    """
    Checks a patched policy against the plain torch policy built from the same config and weights: one forward and
    backward each on the same inputs, the same seed and no dropout, plus one of the plain policy in fp32 as the
    reference. The loss and every parameter gradient must lie within the training config's kernel_check tolerances.
    On a mismatch, finds the patches at fault by rebuilding with one patch at a time, and refuses to train unless
    kernel_check_refuse is off. A requested patch whose kernels fall back to torch on any call refuses too, unless
    kernel_check_require_patches is off: the device, the build or its libraries rule them out, the batch falls outside
    what they handle, or another patch replaces them. Writes the kernel report.

    Args:
        policy: policy 0 with its kernel patches applied, on device
        training_config: the training config, for the policy config and the tolerances
        inputs: seed 0's features, labels and navigation on device
        loss_weights: the weight of every loss by name
        train_step_context: the trainer's context around a training step, e.g. its autocast
        device: the device the policy runs on
        report_file: where the kernel report goes; None writes none

    Returns:
        the kernel report

    Raises:
        RuntimeError: on a mismatch with kernel_check_refuse on, or on a patch whose kernels do not run with
            kernel_check_require_patches on
    """
    policy_config = training_config.policy_config
    channels_last = training_config.channels_last
    tolerances = (
        training_config.kernel_check_loss_rtol,
        training_config.kernel_check_grad_factor,
        training_config.kernel_check_grad_floor,
    )
    patches = kernel_patches_of(policy_config)
    registry = _registry()
    without_registry: dict[str, dict[str, int]] = {}
    recording = cast(
        "AbstractContextManager[dict[str, dict[str, int]]]",
        registry.recording_fallbacks() if registry is not None else nullcontext(without_registry),
    )
    with recording as fallbacks:
        patched = run_step_without_trace(policy, inputs, loss_weights, channels_last, train_step_context, device)
    state_dict = {name: tensor.detach() for name, tensor in policy.state_dict().items()}
    unpatched = build_policy(with_kernel_patches(policy_config, lambda _: False), state_dict, device, channels_last)
    plain = run_step(unpatched, inputs, loss_weights, channels_last, train_step_context, device)
    with exact_fp32():
        reference = run_step(unpatched, inputs, loss_weights, channels_last, nullcontext, device)
    del unpatched
    comparison = compare(patched, plain, reference, *tolerances)
    attribution: list[dict[str, Any]] = []
    if comparison.failures:
        for patch in patches:
            only_this = with_kernel_patches(policy_config, functools.partial(operator.eq, patch))
            single = build_policy(only_this, state_dict, device, channels_last)
            outcome = run_step(single, inputs, loss_weights, channels_last, train_step_context, device)
            del single
            attribution.append({"patch": patch, "failures": compare(outcome, plain, reference, *tolerances).failures})
    torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "status": "verified" if not comparison.failures else "mismatch",
        **_environment_report(device),
        "patches": [
            _patch_report(patch, patches, registry, device, patched.kernel_ops, fallbacks) for patch in patches
        ],
        "kernel_ops_run": sorted(patched.kernel_ops),
        "loss": {
            "patched": patched.loss,
            "plain": plain.loss,
            "fp32": reference.loss,
            "relative_error": comparison.loss_error,
            "rtol": training_config.kernel_check_loss_rtol,
        },
        "gradients": {
            "parameters": len(reference.gradients),
            "worst_parameter": comparison.worst_parameter,
            "worst_margin": comparison.worst_margin,
            "worst_patched_error": comparison.worst_patched_error,
            "worst_plain_error": comparison.worst_plain_error,
            "factor": training_config.kernel_check_grad_factor,
            "floor": training_config.kernel_check_grad_floor,
        },
        "failures": comparison.failures,
        "attribution": attribution,
    }
    report["not_running"] = patches_not_running(report["patches"])
    if report_file is not None:
        report_file.write_text(json.dumps(report, indent=2) + "\n")
    summary = ", ".join(f"{entry['patch'].split(':')[-1]}: {entry['status']}" for entry in report["patches"])
    LOG.info(
        f"Kernel check: loss relative error {comparison.loss_error:.3g}; the gradient of {comparison.worst_parameter} "
        f"lies {comparison.worst_patched_error:.3g} from fp32 with the patches and {comparison.worst_plain_error:.3g} "
        f"without, {comparison.worst_margin:.2f} of its allowance; {summary}",
    )
    if report["not_running"]:
        message = (
            "Requested kernel patches fall back to torch: "
            + "; ".join(report["not_running"])
            + ". Drop them from kernel_patches, fix their kernels, or set kernel_check_require_patches=false to "
            "train on the torch path" + (f"; see {report_file}" if report_file is not None else "")
        )
        if training_config.kernel_check_require_patches:
            raise RuntimeError(message)
        LOG.warning(f"{message}; kernel_check_require_patches is off, training on the torch path")
    if not comparison.failures:
        return report
    at_fault = [entry["patch"] for entry in attribution if entry["failures"]]
    message = (
        f"The kernel patches change the training step: {len(comparison.failures)} mismatches, the first "
        f"{comparison.failures[0]}; patches at fault on their own: {at_fault or 'none, only together'}"
        + (f"; see {report_file}" if report_file is not None else "")
    )
    if training_config.kernel_check_refuse:
        raise RuntimeError(message)
    LOG.warning(f"{message}; kernel_check_refuse is off, training anyway")
    return report
