# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

# TODO: Move & rename this file for common usage (not specific for PDM)


from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from py123d_garage.evaluation.navsim.help.utils.pdm_enums import PointIndex, SE2Index


def normalize_angle(
    angle: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    Map an angle into the range [-π, π].

    Args:
        angle: any angle as float or array of angles

    Returns:
        normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def translate_lon_and_lat(
    centers: jt.Float64[npt.NDArray[np.float64], "*batch 2"],
    headings: jt.Float64[npt.NDArray[np.float64], " *batch"],
    lon: float,
    lat: float,
) -> jt.Float64[npt.NDArray[np.float64], "*batch 2"]:
    """
    Translate the position component of a centers point array.

    Args:
        centers: array to be translated
        headings: array with heading angles
        lon: [m] distance by which a point should be translated in longitudinal direction
        lat: [m] distance by which a point should be translated in lateral direction

    Returns:
        array of translated coordinates
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    )
    return centers + translation


def se2_array_translate_longitudinally(
    se2_array: jt.Float64[npt.NDArray[np.float64], "*batch 3"],
    distance: float,
) -> jt.Float64[npt.NDArray[np.float64], "*batch 3"]:
    """
    Translates an SE2 array along the heading axis by distance.

    Args:
        se2_array: array of SE2 states with (x,y,θ) in last dim
        distance: distance to translate [m]

    Returns:
        Translated SE2 coords array.
    """
    assert se2_array.shape[-1] == len(SE2Index)
    translate_se2 = np.zeros(se2_array.shape, dtype=np.float64)
    translate_se2[..., SE2Index.X] = se2_array[..., SE2Index.X] + np.cos(se2_array[..., SE2Index.HEADING]) * distance
    translate_se2[..., SE2Index.Y] = se2_array[..., SE2Index.Y] + np.sin(se2_array[..., SE2Index.HEADING]) * distance
    translate_se2[..., SE2Index.HEADING] = se2_array[..., SE2Index.HEADING]
    return translate_se2


def get_velocity_shifted(
    displacement: jt.Float64[npt.NDArray[np.float64], "... 2"],
    ref_velocity_2d: jt.Float64[npt.NDArray[np.float64], "*batch 2"],
    ref_angular_vel: jt.Float64[npt.NDArray[np.float64], " *batch"],
) -> jt.Float64[npt.NDArray[np.float64], "*batch 2"]:
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.

    Args:
        displacement: [m] The displacement vector from the reference to the query point
        ref_velocity_2d: [m/s] The velocity vector at the reference point
        ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis

    Returns:
        [m/s] The velocity vector at the given displacement.
    """
    assert displacement.shape[-1] == len(PointIndex)
    assert ref_velocity_2d.shape[-1] == len(PointIndex)
    assert ref_velocity_2d.shape[:-1] == ref_angular_vel.shape
    velocity_shift_term = np.zeros(
        ref_velocity_2d.shape,
        dtype=ref_velocity_2d.dtype,
    )
    velocity_shift_term[..., PointIndex.X] = -displacement[..., PointIndex.Y] * ref_angular_vel
    velocity_shift_term[..., PointIndex.Y] = displacement[..., PointIndex.X] * ref_angular_vel
    return ref_velocity_2d + velocity_shift_term


def get_acceleration_shifted(
    displacement: jt.Float64[npt.NDArray[np.float64], "... 2"],
    ref_accel_2d: jt.Float64[npt.NDArray[np.float64], "*batch 2"],
    ref_angular_vel: jt.Float64[npt.NDArray[np.float64], " *batch"],
    ref_angular_accel: jt.Float64[npt.NDArray[np.float64], " *batch"],
) -> jt.Float64[npt.NDArray[np.float64], "*batch 2"]:
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.

    Args:
        displacement: [m] The displacement vector from the reference to the query point
        ref_accel_2d: [m/s^2] The acceleration vector at the reference point
        ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
        ref_angular_accel: [rad/s^2] The angular acceleration of the body around the vertical axis

    Returns:
        [m/s^2] The acceleration vector at the given displacement.
    """
    assert displacement.shape[-1] == len(PointIndex)
    assert ref_accel_2d.shape[-1] == len(PointIndex)
    assert ref_accel_2d.shape[:-1] == ref_angular_vel.shape
    assert ref_accel_2d.shape[:-1] == ref_angular_accel.shape
    centripetal_acceleration_term = displacement * ref_angular_vel[..., None] ** 2
    angular_acceleration_term = displacement * ref_angular_accel[..., None]
    return ref_accel_2d + centripetal_acceleration_term + angular_acceleration_term
