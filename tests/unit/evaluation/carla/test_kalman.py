"""Unit tests for the filter localizing the CARLA agent on the noisy GNSS."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from py123d_garage.evaluation.carla.help.kalman import GnssKalmanFilter, _bicycle_model_forward

_GNSS_NOISE_M = 0.556
_SPEED_MPS = 8.0
_DT_S = 0.05
_NUM_STEPS = 200
_SETTLED = slice(_NUM_STEPS // 2, None)


def _drive(steer: float) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Coasts through an arc at constant speed, filtering its noisy fixes.

    The ego holds one steering angle and no throttle, so the truth is exactly what
    the filter's own vehicle model predicts and only the GNSS noise is left to fight.

    Args:
        steer: the steering command held for the whole drive, zero for a straight line.

    Returns:
        the true, the measured and the filtered (x, y) positions, one row per step.
    """
    generator = np.random.default_rng(0)
    kalman_filter = GnssKalmanFilter(_DT_S)
    state = np.array([100.0, -250.0, 0.3, _SPEED_MPS])
    truths, measured, filtered = [], [], []
    for _ in range(_NUM_STEPS):
        state = _bicycle_model_forward(state, _DT_S, steer, throttle=0.0, brake=0.0)
        noisy = state[:2] + generator.normal(scale=_GNSS_NOISE_M, size=2)
        truths.append(state[:2].copy())
        measured.append(noisy)
        filtered.append(kalman_filter.step(noisy, state[2], state[3], steer, throttle=0.0, brake=0.0))
    return np.array(truths)[_SETTLED], np.array(measured)[_SETTLED], np.array(filtered)[_SETTLED]


def _error_m(truths: npt.NDArray[np.float64], estimates: npt.NDArray[np.float64]) -> float:
    """
    The RMS position error of paired estimates.

    Args:
        truths: the true positions.
        estimates: the estimated positions.

    Returns:
        the error in meters.
    """
    return float(np.sqrt(np.mean(np.sum((estimates - truths) ** 2, axis=1))))


def test_filter_cuts_the_gnss_noise_on_a_straight_line() -> None:
    truths, measured, filtered = _drive(steer=0.0)
    assert _error_m(truths, filtered) < 0.4 * _error_m(truths, measured)


def test_filter_cuts_the_gnss_noise_through_a_turn() -> None:
    truths, measured, filtered = _drive(steer=0.18)
    assert _error_m(truths, filtered) < 0.4 * _error_m(truths, measured)


def test_consecutive_estimates_move_like_the_vehicle_not_like_the_noise() -> None:
    _, measured, filtered = _drive(steer=0.0)
    hop_m = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
    noisy_hop_m = np.linalg.norm(np.diff(measured, axis=0), axis=1)
    assert np.std(hop_m) < 0.1 * np.std(noisy_hop_m)
    assert np.mean(hop_m) == pytest.approx(_SPEED_MPS * _DT_S, abs=0.01)
