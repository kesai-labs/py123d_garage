# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Ego poses, interpolated and sampled around an origin timestamp."""

from __future__ import annotations

from typing import Any, cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.modalities.sync_utils import get_modality_table
from py123d.datatypes import EgoStateSE3, ModalityType
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import EulerAnglesIndex, PoseSE2Index, PoseSE3Index
from py123d.geometry.transform import (
    abs_to_rel_se2_array,
)
from py123d.geometry.utils.rotation_utils import (
    get_euler_array_from_quaternion_array,
    get_rotation_matrices_from_quaternion_array,
    multiply_quaternion_arrays,
)

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt
from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
    TrajectoryXY,
)


def interpolate_ego_se2_at(
    scene_api: SceneAPI,
    timestamps_us: jt.Int64[npt.NDArray[np.int64], " num_poses"],
    interpolation_margin_us: NonNegativeInt = 500_000,
    boundary_tolerance_us: NonNegativeInt = 0,
) -> jt.Float64[npt.NDArray[np.float64], "num_poses 3"]:
    """
    Ego rear-axle SE2 poses interpolated at the given timestamps from the recorded states.

    Args:
        scene_api: scene interface providing the recorded ego states.
        timestamps_us: absolute timestamps to interpolate at, in any order.
        interpolation_margin_us: extra window of recorded states loaded on both sides
            so the boundary timestamps can be interpolated, defaults to 500_000.
        boundary_tolerance_us: how far a timestamp may fall outside the recorded ego range and still clamp to
            the boundary pose instead of raising; 0 is strict.

    Returns:
        interpolated (x, y, yaw) poses, one per timestamp.

    Raises:
        ValueError: if fewer than two ego states cover the requested range, or a
            timestamp falls outside it by more than the boundary tolerance.
    """
    window_start_us = int(timestamps_us.min()) - interpolation_margin_us
    window_end_us = int(timestamps_us.max()) + interpolation_margin_us
    poses_se2: list[npt.NDArray[np.float64]] = []
    ego_timestamps_us: list[int] = []
    for ego_state_se3 in scene_api.get_modality_between_timestamps(  # pyright: ignore[reportUnknownMemberType]
        start_timestamp=window_start_us,
        end_timestamp=window_end_us,
        modality_type="ego_state_se3",
        inclusive="both",
    ):
        assert isinstance(ego_state_se3, EgoStateSE3), f"Expected EgoStateSE3, got {type(ego_state_se3)}"
        poses_se2.append(ego_state_se3.rear_axle_se2.array)
        ego_timestamps_us.append(ego_state_se3.timestamp.time_us)
    if len(poses_se2) < 2:
        raise ValueError(
            f"scene {scene_api.scene_uuid} of log {scene_api.log_name} records "
            f"{len(poses_se2)} ego states in [{window_start_us}, {window_end_us}] µs; "
            "interpolation needs at least two.",
        )
    trajectory = TrajectorySE2(
        pose_se2_array=np.array(poses_se2, dtype=np.float64),
        timestamps_us=np.array(ego_timestamps_us, dtype=np.int64),
    )
    return trajectory.interpolate(timestamps_us, boundary_tolerance_us=boundary_tolerance_us)


def sample_ego_se2(
    scene_api: SceneAPI,
    offsets_us: list[PositiveInt],
    relative_to_anchor: bool = False,
    interpolation_margin_us: NonNegativeInt = 500_000,
) -> TrajectorySE2:
    """
    Loads the logged ego poses and resamples them onto the given grid.

    Args:
        scene_api: API providing access to the logged ego states.
        offsets_us: times past the anchor the returned trajectory holds a pose at.
        relative_to_anchor: if True, return poses relative to the anchor pose, defaults to False.
        interpolation_margin_us: extra horizon of logged states to load so the final
            pose can be interpolated, defaults to 500_000.

    Returns:
        future ego trajectory resampled to the target grid, as SE2.
    """

    initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state should be available for label computation!"

    # Whole-column reads: a per-state read builds an EgoStateSE3 and converts it alone, ~20x slower.
    initial_timestamp_us = initial_ego_state_se3.timestamp.time_us
    end_timestamp_us = initial_timestamp_us + offsets_us[-1] + interpolation_margin_us
    ego_key = ModalityType.EGO_STATE_SE3.serialize()
    log_dir = cast(ArrowSceneAPI, scene_api)._log_dir  # pyright: ignore[reportPrivateUsage]
    ego_table = get_modality_table(log_dir, ego_key)
    ego_metadata = scene_api.get_modality_metadata(ModalityType.EGO_STATE_SE3)
    assert ego_table is not None, f"{log_dir} records no ego state"
    assert isinstance(ego_metadata, EgoStateSE3Metadata), f"{log_dir} has no ego state metadata"
    all_timestamps_us = cast(
        npt.NDArray[np.int64],
        ego_table[f"{ego_key}.timestamp_us"].to_numpy(),  # pyright: ignore[reportUnknownMemberType]
    )
    first_row = int(np.searchsorted(all_timestamps_us, initial_timestamp_us, side="left"))
    end_row = int(np.searchsorted(all_timestamps_us, end_timestamp_us, side="right"))
    pose_column = cast(
        "pa.FixedSizeListArray[Any, Any]",
        ego_table[f"{ego_key}.imu_se3"].slice(first_row, end_row - first_row).combine_chunks(),
    )
    imu_se3_array = cast(
        npt.NDArray[np.float64],
        pose_column.flatten().to_numpy(zero_copy_only=False),  # pyright: ignore[reportUnknownMemberType]
    ).reshape(end_row - first_row, len(PoseSE3Index))

    rear_axle_in_imu_se3 = ego_metadata.rear_axle_to_imu_se3.array
    imu_rotations = get_rotation_matrices_from_quaternion_array(imu_se3_array[:, PoseSE3Index.QUATERNION])
    rear_axle_xyz = (
        np.einsum("nij,j->ni", imu_rotations, rear_axle_in_imu_se3[PoseSE3Index.XYZ])
        + imu_se3_array[:, PoseSE3Index.XYZ]
    )
    rear_axle_quaternions = multiply_quaternion_arrays(
        imu_se3_array[:, PoseSE3Index.QUATERNION],
        rear_axle_in_imu_se3[PoseSE3Index.QUATERNION],
    )
    rear_axle_se2_array = np.empty((end_row - first_row, len(PoseSE2Index)), dtype=np.float64)
    rear_axle_se2_array[:, PoseSE2Index.XY] = rear_axle_xyz[:, :2]
    rear_axle_euler_angles = get_euler_array_from_quaternion_array(rear_axle_quaternions)
    rear_axle_se2_array[:, PoseSE2Index.YAW] = rear_axle_euler_angles[:, EulerAnglesIndex.YAW]
    full_trajectory = TrajectorySE2(
        pose_se2_array=rear_axle_se2_array,
        timestamps_us=all_timestamps_us[first_row:end_row].astype(np.int64),
    )
    sampling_timestamps_us = initial_timestamp_us + np.array(offsets_us, dtype=np.int64)

    # Ego sampling jitter can leave the last recorded state up to one period short of the last grid point.
    ego_period_us = (
        int(np.median(np.diff(full_trajectory.timestamps_us))) if len(full_trajectory.timestamps_us) > 1 else 0
    )
    resampled_se2_array = full_trajectory.interpolate(
        sampling_timestamps_us,
        boundary_tolerance_us=ego_period_us,
    )

    if relative_to_anchor:
        resampled_se2_array = abs_to_rel_se2_array(
            origin=initial_ego_state_se3.rear_axle_se2,
            pose_se2_array=resampled_se2_array,
        )

    return TrajectorySE2(
        pose_se2_array=resampled_se2_array,
        timestamps_us=sampling_timestamps_us,
    )


def sample_ego_xy(
    scene_api: SceneAPI,
    offsets_us: list[PositiveInt],
    relative_to_anchor: bool = False,
    interpolation_margin_us: NonNegativeInt = 500_000,
) -> TrajectoryXY:
    """
    Loads the logged ego positions and resamples them onto the given grid.

    Args:
        scene_api: API providing access to the logged ego states.
        offsets_us: times past the anchor the returned trajectory holds a pose at.
        relative_to_anchor: if True, return positions relative to the anchor pose, defaults to False.
        interpolation_margin_us: extra horizon of logged states to load so the final
            position can be interpolated, defaults to 500_000.

    Returns:
        future ego trajectory resampled to the target grid, as xy.
    """
    trajectory = sample_ego_se2(
        scene_api=scene_api,
        offsets_us=offsets_us,
        relative_to_anchor=relative_to_anchor,
        interpolation_margin_us=interpolation_margin_us,
    )
    return TrajectoryXY(
        position_xy_array=trajectory.pose_se2_array[:, :2],
        timestamps_us=trajectory.timestamps_us,
    )
