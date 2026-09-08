"""The constrained numeric aliases, as enforced by the runtime type checker."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from py123d_garage.common.runtime_typing import typechecker
from py123d_garage.datatypes.numerics import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    WrappedAngleRadArray,
    WrappedAngleRadTensor,
    WrappedSE2Array,
    _is_wrapped_angle_rad,
)


@typechecker
def _positive_int(value: PositiveInt) -> PositiveInt:
    return value


@typechecker
def _positive_float(value: PositiveFloat) -> PositiveFloat:
    return value


@typechecker
def _non_negative_int(value: NonNegativeInt) -> NonNegativeInt:
    return value


@typechecker
def _non_negative_float(value: NonNegativeFloat) -> NonNegativeFloat:
    return value


def test_positive_aliases_accept_values_above_zero() -> None:
    assert _positive_int(1) == 1
    assert _positive_float(0.5) == 0.5


def test_non_negative_aliases_accept_zero() -> None:
    assert _non_negative_int(0) == 0
    assert _non_negative_float(0.0) == 0.0


@pytest.mark.parametrize("value", [0, -1])
def test_positive_int_rejects_zero_and_below(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _positive_int(value)


@pytest.mark.parametrize("value", [0.0, -0.5])
def test_positive_float_rejects_zero_and_below(value: float) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _positive_float(value)


def test_non_negative_aliases_reject_negatives() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _non_negative_int(-1)
    with pytest.raises(BeartypeCallHintParamViolation):
        _non_negative_float(-0.5)


def test_float_aliases_accept_ints() -> None:
    """The PEP 484 numeric tower is on, so int-valued configs reach float parameters."""
    assert _positive_float(2) == 2
    assert _non_negative_float(0) == 0


def test_int_aliases_reject_floats() -> None:
    """The tower widens int to float, never the reverse."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _positive_int(1.5)  # pyright: ignore[reportArgumentType]


@typechecker
def _wrapped_yaw(yaw_wrapped_rad: WrappedAngleRadArray) -> WrappedAngleRadArray:
    return yaw_wrapped_rad


@typechecker
def _wrapped_se2(pose_se2_array: WrappedSE2Array) -> WrappedSE2Array:
    return pose_se2_array


def test_wrapped_yaw_array_accepts_the_range() -> None:
    yaw = np.array([-np.pi, -1.0, 0.0, 3.14], dtype=np.float64)
    assert _wrapped_yaw(yaw) is yaw


def test_wrapped_yaw_array_rejects_unwrapped_values() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _wrapped_yaw(np.array([0.0, 5.027], dtype=np.float64))


def test_wrapped_yaw_array_rejects_positive_pi() -> None:
    """The range is half-open, so +pi is out and -pi is in."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _wrapped_yaw(np.array([np.pi], dtype=np.float64))
    assert _wrapped_yaw(np.array([-np.pi], dtype=np.float64)) is not None


def test_wrapped_se2_array_only_constrains_the_yaw_column() -> None:
    """x and y are free; only column 2 is an angle."""
    assert _wrapped_se2(np.array([[1e6, -1e6, 0.5]], dtype=np.float64)) is not None
    with pytest.raises(BeartypeCallHintParamViolation):
        _wrapped_se2(np.array([[0.0, 0.0, 5.027]], dtype=np.float64))


@typechecker
def _wrapped_yaw_tensor(yaw_wrapped_rad: WrappedAngleRadTensor) -> WrappedAngleRadTensor:
    return yaw_wrapped_rad


def test_wrapped_yaw_tensor_accepts_and_rejects() -> None:
    assert _wrapped_yaw_tensor(torch.tensor([0.0, -3.0])) is not None
    with pytest.raises(BeartypeCallHintParamViolation):
        _wrapped_yaw_tensor(torch.tensor([5.027]))


def test_wrapped_yaw_tensor_rejects_a_numpy_array() -> None:
    """The two aliases are not interchangeable; each names its own container."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _wrapped_yaw_tensor(np.array([0.0]))  # pyright: ignore[reportArgumentType]


def test_predicate_agrees_across_numpy_and_torch() -> None:
    """One predicate backs both aliases, so they cannot drift apart."""
    for values in ([0.0, -3.0], [5.027], [np.pi], [-np.pi]):
        assert _is_wrapped_angle_rad(np.array(values)) == _is_wrapped_angle_rad(torch.tensor(values))
