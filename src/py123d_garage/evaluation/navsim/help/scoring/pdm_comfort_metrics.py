# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3Metadata
from scipy.signal import savgol_filter

from py123d_garage.evaluation.navsim.help.utils.pdm_array_representation import (
    state_array_to_center_state_array,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import StateIndex

# TODO: Refactor & add to config

# (1) ego_jerk_metric,
MAX_ABS_MAG_JERK: float = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
MAX_ABS_LAT_ACCEL: float = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
MAX_LON_ACCEL: float = 2.40  # [m/s^2]
MIN_LON_ACCEL: float = -4.05  # [m/s^2]

# (4) ego_yaw_acceleration_metric
MAX_ABS_YAW_ACCEL: float = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
MAX_ABS_LON_JERK: float = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
MAX_ABS_YAW_RATE: float = 0.95  # [rad/s]


def _extract_ego_acceleration(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    acceleration_coordinate: str,
    metadata: EgoStateSE3Metadata,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> jt.Float64[npt.NDArray[np.float64], "batch times"]:
    """
    Extract acceleration of ego pose in simulation history over batch-dim.

    Args:
        states: array representation of ego state values
        acceleration_coordinate: string of axis to extract
        metadata: metadata of vehicle
        decimals: decimal precision, defaults to 8
        poly_order: polynomial order, defaults to 2
        window_length: window size for extraction, defaults to 8

    Returns:
        array containing acceleration values

    Raises:
        ValueError: when coordinate not available
    """

    _n_batch, n_time, _n_states = states.shape
    if acceleration_coordinate in {"x", "y"}:
        center_states = state_array_to_center_state_array(states, metadata)
        coordinate_index = StateIndex.ACCELERATION_X if acceleration_coordinate == "x" else StateIndex.ACCELERATION_Y
        acceleration: npt.NDArray[np.float64] = center_states[
            ...,
            coordinate_index,
        ]

    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude",
        )

    acceleration = cast(
        npt.NDArray[np.float64],
        savgol_filter(
            acceleration,
            polyorder=poly_order,
            window_length=min(window_length, n_time),
            axis=-1,
        ),
    )
    return np.round(acceleration, decimals=decimals)


def _extract_ego_jerk(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    acceleration_coordinate: str,
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> jt.Float64[npt.NDArray[np.float64], "batch times"]:
    """
    Extract jerk of ego pose in simulation history over batch-dim.

    Args:
        states: array representation of ego state values
        acceleration_coordinate: string of axis to extract
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle
        decimals: decimal precision, defaults to 8
        deriv_order: order of derivative, defaults to 1
        poly_order: polynomial order, defaults to 2
        window_length: window size for extraction, defaults to 15

    Returns:
        array containing jerk values
    """
    _n_batch, n_time, _n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(
        states,
        acceleration_coordinate=acceleration_coordinate,
        metadata=metadata,
    )
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    return np.round(jerk, decimals=decimals)


def _extract_ego_yaw_rate(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> jt.Float64[npt.NDArray[np.float64], "batch times"]:
    """
    Extract yaw-rate of simulation history over batch-dim.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        deriv_order: order of derivative, defaults to 1
        poly_order: polynomial order, defaults to 2
        decimals: decimal precision, defaults to 8
        window_length: window size for extraction, defaults to 15

    Returns:
        array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )
    return np.round(ego_yaw_rate, decimals=decimals)


def _phase_unwrap(
    headings: jt.Float64[npt.NDArray[np.float64], "batch times"],
) -> jt.Float64[npt.NDArray[np.float64], "batch times"]:
    """
    Phase-unwraps heading angles so successive differences stay within pi radians.

    Args:
        headings: An array of headings (radians)

    Returns:
        The phase-unwrapped equivalent headings.
    """
    # Each step's adjustment is its heading difference rounded to a whole number of turns.
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi),
        axis=-1,
    )
    return headings - two_pi * adjustments


def _approximate_derivatives(
    y: jt.Float64[npt.NDArray[np.float64], "batch times"],
    x: jt.Float64[npt.NDArray[np.float64], " times"],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> jt.Float64[npt.NDArray[np.float64], "batch times"]:
    """
    Approximates the n-th derivative of y over equally spaced x with a Savitzky-Golay filter.

    Args:
        y: The dependent variable.
        x: The independent variable, strictly increasing and equally spaced.
        window_length: Window length of the filter, capped at the length of x.
        poly_order: Degree of the filter polynomial, less than window_length.
        deriv_order: Order of the derivative.
        axis: Axis of y along which the filter is applied.

    Returns:
        Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float64] = cast(
        npt.NDArray[np.float64],
        savgol_filter(
            y,
            polyorder=poly_order,
            window_length=window_length,
            deriv=deriv_order,
            delta=dx,
            axis=axis,
        ),
    )
    return derivative


def _within_bound(
    metric: jt.Float64[npt.NDArray[np.float64], "batch times"],
    min_bound: float | None = None,
    max_bound: float | None = None,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Determines whether values in the batch-dim are within bounds.

    Args:
        metric: metric values
        min_bound: minimum bound, defaults to None
        max_bound: maximum bound, defaults to None

    Returns:
        array of booleans whether metric values are within bounds
    """
    min_bound = min_bound or float(-np.inf)
    max_bound = max_bound or float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_lon_acceleration(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute longitudinal acceleration over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        longitudinal acceleration within bound
    """
    lon_acceleration = _extract_ego_acceleration(
        states,
        acceleration_coordinate="x",
        metadata=metadata,
    )
    return _within_bound(
        lon_acceleration,
        min_bound=MIN_LON_ACCEL,
        max_bound=MAX_LON_ACCEL,
    )


def _compute_lat_acceleration(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute lateral acceleration over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        lateral acceleration within bound
    """
    lat_acceleration = _extract_ego_acceleration(
        states,
        acceleration_coordinate="y",
        metadata=metadata,
    )
    return _within_bound(
        lat_acceleration,
        min_bound=-MAX_ABS_LAT_ACCEL,
        max_bound=MAX_ABS_LAT_ACCEL,
    )


def _compute_jerk_metric(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute absolute jerk over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        absolute jerk within bound
    """
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        metadata=metadata,
    )
    return _within_bound(
        jerk_metric,
        min_bound=-MAX_ABS_MAG_JERK,
        max_bound=MAX_ABS_MAG_JERK,
    )


def _compute_lon_jerk_metric(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute longitudinal jerk over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        longitudinal jerk within bound
    """
    lon_jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        metadata=metadata,
    )
    return _within_bound(
        lon_jerk_metric,
        min_bound=-MAX_ABS_LON_JERK,
        max_bound=MAX_ABS_LON_JERK,
    )


def _compute_yaw_accel(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute acceleration of yaw-angle over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        acceleration of yaw-angle within bound
    """
    yaw_accel_metric = _extract_ego_yaw_rate(
        states,
        time_steps_s,
        deriv_order=2,
        poly_order=3,
    )
    return _within_bound(
        yaw_accel_metric,
        min_bound=-MAX_ABS_YAW_ACCEL,
        max_bound=MAX_ABS_YAW_ACCEL,
    )


def _compute_yaw_rate(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_steps_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], " batch"]:
    """
    Compute velocity of yaw-angle over batch-dim of simulated proposals.

    Args:
        states: array representation of ego state values
        time_steps_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        velocity of yaw-angle within bound
    """
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s)
    return _within_bound(
        yaw_rate_metric,
        min_bound=-MAX_ABS_YAW_RATE,
        max_bound=MAX_ABS_YAW_RATE,
    )


def ego_is_comfortable(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    time_point_s: jt.Float64[npt.NDArray[np.float64], " times"],
    metadata: EgoStateSE3Metadata,
) -> jt.Bool[npt.NDArray[np.bool_], "batch 6"]:
    """
    Accumulates all within-bound comfort metrics into a per-proposal, per-metric array.

    Args:
        states: array representation of ego state values
        time_point_s: time steps [s] of time dim
        metadata: metadata of vehicle

    Returns:
        boolean array (n_batch, n_metrics) flagging which comfort metrics are within bound
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == len(StateIndex)

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]
    results: npt.NDArray[np.bool_] = np.zeros(
        (n_batch, len(comfort_metric_functions)),
        dtype=np.bool_,
    )
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s, metadata)

    return results
