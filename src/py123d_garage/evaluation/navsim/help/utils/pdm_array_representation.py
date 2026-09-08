# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

# TODO: Move & rename this file for common usage (not specific for PDM)

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import shapely
from py123d.datatypes import EgoStateSE2, EgoStateSE3Metadata

from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    BBCoordsIndex,
    PointIndex,
    StateIndex,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_geometry_utils import (
    get_acceleration_shifted,
    get_velocity_shifted,
    se2_array_translate_longitudinally,
    translate_lon_and_lat,
)


def ego_state_to_state_array(
    ego_state: EgoStateSE2,
) -> jt.Float64[npt.NDArray[np.float64], " 11"]:
    """
    Converts an EgoStateSE2 into an array representation (drops timestamp and metadata).

    The returned array follows StateIndex layout; STEERING_RATE and ANGULAR_ACCELERATION
    are not represented in py123d's EgoStateSE2 and stay zero.

    Args:
        ego_state: EgoStateSE2 instance

    Returns:
        array filled with ego state values (from the rear axle)
    """
    state_array = np.zeros(len(StateIndex), dtype=np.float64)

    state_array[StateIndex.STATE_SE2] = ego_state.rear_axle_se2.array

    if ego_state.dynamic_state_se2 is not None:
        state_array[StateIndex.VELOCITY_2D] = ego_state.dynamic_state_se2.velocity_2d.array
        state_array[StateIndex.ACCELERATION_2D] = ego_state.dynamic_state_se2.acceleration_2d.array
        state_array[StateIndex.ANGULAR_VELOCITY] = ego_state.dynamic_state_se2.angular_velocity

    if ego_state.tire_steering_angle is not None:
        state_array[StateIndex.STEERING_ANGLE] = ego_state.tire_steering_angle

    return state_array


def state_array_to_coords_array(
    states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    metadata: EgoStateSE3Metadata,
) -> jt.Float64[npt.NDArray[np.float64], "batch times 5 2"]:
    """
    Converts multi-dim array representation of ego states to bounding box coordinates.

    Args:
        states: array representation of ego states (n_batch, n_time, len(StateIndex))
        metadata: vehicle metadata

    Returns:
        multi-dim array of bounding box coordinates
    """
    n_batch, n_time, _ = states.shape

    half_length = metadata.half_length
    half_width = metadata.half_width
    rear_axle_to_center = metadata.rear_axle_to_center_longitudinal

    headings = states[..., StateIndex.HEADING]
    cos, sin = np.cos(headings), np.sin(headings)

    # calculate ego center from rear axle
    rear_axle_to_center_translate = np.stack(
        [rear_axle_to_center * cos, rear_axle_to_center * sin],
        axis=-1,
    )

    ego_centers: npt.NDArray[np.float64] = states[..., StateIndex.POINT] + rear_axle_to_center_translate

    coords_array: npt.NDArray[np.float64] = np.zeros(
        (n_batch, n_time, len(BBCoordsIndex), 2),
        dtype=np.float64,
    )

    coords_array[:, :, BBCoordsIndex.CENTER] = ego_centers
    coords_array[:, :, BBCoordsIndex.FRONT_LEFT] = translate_lon_and_lat(
        ego_centers,
        headings,
        half_length,
        half_width,
    )
    coords_array[:, :, BBCoordsIndex.FRONT_RIGHT] = translate_lon_and_lat(
        ego_centers,
        headings,
        half_length,
        -half_width,
    )
    coords_array[:, :, BBCoordsIndex.REAR_LEFT] = translate_lon_and_lat(
        ego_centers,
        headings,
        -half_length,
        half_width,
    )
    coords_array[:, :, BBCoordsIndex.REAR_RIGHT] = translate_lon_and_lat(
        ego_centers,
        headings,
        -half_length,
        -half_width,
    )

    return coords_array


def coords_array_to_polygon_array(
    coords: jt.Float64[npt.NDArray[np.float64], "... 5 2"],
) -> npt.NDArray[np.object_]:
    """
    Converts multi-dim array of bounding box coords to shapely polygons.

    Args:
        coords: bounding box coords (including corners and center)

    Returns:
        array of shapely polygons
    """
    # create coords copy and use center point for closed exterior
    coords_exterior: npt.NDArray[np.float64] = coords.copy()
    coords_exterior[..., BBCoordsIndex.CENTER, :] = coords_exterior[
        ...,
        BBCoordsIndex.FRONT_LEFT,
        :,
    ]

    # load new coordinates into polygon array
    return shapely.creation.polygons(coords_exterior)  # type: ignore[return-value]


def state_array_to_center_state_array(
    state_array: jt.Float64[npt.NDArray[np.float64], "*batch 11"],
    metadata: EgoStateSE3Metadata,
) -> jt.Float64[npt.NDArray[np.float64], "*batch 11"]:
    """
    Converts a rear-axle-referenced state array to a center-referenced one.

    Args:
        state_array: array representation of ego states (..., len(StateIndex))
        metadata: vehicle metadata

    Returns:
        center-referenced state array, same shape as input
    """
    assert state_array.shape[-1] == len(StateIndex)

    center_states = np.zeros(state_array.shape, dtype=np.float64)

    # coordinates
    center_states[..., StateIndex.STATE_SE2] = se2_array_translate_longitudinally(
        state_array[..., StateIndex.STATE_SE2],
        metadata.rear_axle_to_center_longitudinal,
    )

    # velocity & acceleration
    displacement = np.zeros((1, 2), dtype=np.float64)
    displacement[..., PointIndex.X] = metadata.rear_axle_to_center_longitudinal

    center_states[..., StateIndex.VELOCITY_2D] = get_velocity_shifted(
        displacement,
        state_array[..., StateIndex.VELOCITY_2D],
        state_array[..., StateIndex.ANGULAR_VELOCITY],
    )
    center_states[..., StateIndex.ACCELERATION_2D] = get_acceleration_shifted(
        displacement,
        state_array[..., StateIndex.ACCELERATION_2D],
        state_array[..., StateIndex.ANGULAR_VELOCITY],
        state_array[..., StateIndex.ANGULAR_ACCELERATION],
    )

    # rest is copied
    center_states[..., StateIndex.STEERING_ANGLE] = state_array[
        ...,
        StateIndex.STEERING_ANGLE,
    ]
    center_states[..., StateIndex.STEERING_RATE] = state_array[
        ...,
        StateIndex.STEERING_RATE,
    ]
    center_states[..., StateIndex.ANGULAR_VELOCITY] = state_array[
        ...,
        StateIndex.ANGULAR_VELOCITY,
    ]
    center_states[..., StateIndex.ANGULAR_ACCELERATION] = state_array[
        ...,
        StateIndex.ANGULAR_ACCELERATION,
    ]

    return center_states
