# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The planner adapter the Odyssey runtime loads, driving every tick with one garage policy."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any, Final, cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from odyssey_bridge import ipc_common
from odyssey_bridge.planners.base import RIG_CAMERAS, NavsimPlanner
from odyssey_bridge.route_sidecar import StaticRouteCenterline
from py123d.datatypes import (
    BaseModality,
    BaseModalityMetadata,
    Camera,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LogMetadata,
    ModalityType,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import get_modality_key
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_EGO_STATE_SE3_METADATA
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.common.config_help import load_evaluation_config
from py123d_garage.config.schema.evaluation.odyssey_config import OdysseyBenchmarkConfig
from py123d_garage.evaluation.alpasim.help.plan import CachedPlan, make_plan, rig_to_local_xy
from py123d_garage.evaluation.help.policy_building import build_evaluation_policy
from py123d_garage.py123d_help.misc import provided_route_source_info
from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

# Odyssey's world cadence; the runtime calls the planner once per tick.
_TICK_US: Final[int] = 100_000
# Odyssey reconstructs nuPlan logs, so the policy reads its nuPlan cameras.
_DATASET: Final[str] = "nuplan"
# The route centerline Odyssey samples ahead of the ego is 1 m apart.
_ROUTE_RESOLUTION_M: Final[float] = 1.0


class GaragePlanner(NavsimPlanner):
    """Plans with a garage policy every replan interval and serves the stored plan on the ticks between."""

    TAG = "py123d_garage"
    PLAN_DT = _TICK_US / 1e6
    SD_ROUTE = "features"

    @override
    def build(self) -> None:
        """Builds the policy from the checkpoint and its config.yaml, and the route reader of the scene's route file."""
        config = load_evaluation_config(
            OdysseyBenchmarkConfig,
            "evaluate_odyssey",
            args=[f"policy_config.evaluation_checkpoint_files=[{self.ckpt_path}]", *self.overrides],
        )
        policy = build_evaluation_policy(config.policy_config)
        policy.verify_contract(benchmark_config=config)
        self._policy: AnyPolicy = policy.to(torch.device(self.device)).eval()
        self._replan_interval_us = config.replan_interval_us
        self._num_poses = (policy.policy_config.trajectory_horizon_us - config.replan_interval_us) // _TICK_US + 1

        declared_cameras = self.cameras or {}
        policy_cameras = {
            camera_id.name.removeprefix("P") for camera_id in policy.policy_config.required_cameras[_DATASET]
        }
        if {name.upper() for name in declared_cameras} != policy_cameras:
            raise ValueError(
                f"the agent config declares the cameras {sorted(name.upper() for name in declared_cameras)}, "
                f"but the policy reads {sorted(policy_cameras)}",
            )
        self._sensor_config = SimpleNamespace(
            lidar_pc=False,
            **{name: declared_cameras.get(name, False) for name in RIG_CAMERAS},
        )
        self._route_builder = StaticRouteCenterline(None, sidecar_dir=self.route_file, strict=True)
        self._scene_api: OnlineSceneAPI | None = None
        self._camera_metadatas: dict[str, PinholeCameraMetadata] = {}
        self._plan: CachedPlan | None = None

    @property
    @override
    def uses_driving_command(self) -> bool:
        """The policy reads target points from the route, never the driving command."""
        return False

    @override
    def infer(self, frames: list[dict[str, Any]]) -> jt.Float64[npt.NDArray[np.float64], "num_poses 3"]:
        """
        Observes the newest frame and serves the stored plan, replanning once it is replan_interval_us old.

        Args:
            frames: the frame dicts the runtime keeps, oldest first.

        Returns:
            the planned (x, y, heading) poses in the current ego frame, one per tick from the next tick on.

        Raises:
            RouteExhausted: once the ego has passed the end of the route, which ends the episode.
        """
        frame = frames[-1]
        if ipc_common.inject_ay_enabled():
            frame = ipc_common.inject_ay([copy.deepcopy(frame)])[0]
        step = int(frame["frame_idx"])
        time_us = step * _TICK_US
        # A strict route builder raises instead of returning None.
        route_centerline, route_mask = cast("tuple[torch.Tensor, torch.Tensor]", self._route_builder.build(frame))
        # The worker sets input_builder to its in-memory camera bank's build method; the bank holds the decoded images.
        images: dict[str, dict[int, npt.NDArray[np.uint8]]] = cast("Any", self.input_builder).__self__.images
        ego_state = _ego_state(frame, time_us)

        if step == 0:
            self._camera_metadatas = {
                name: _camera_metadata(name, frame["cams"][name], images[name][step]) for name in images
            }
            ego_metadatas: dict[str, BaseModalityMetadata] = {
                get_modality_key(ModalityType.EGO_STATE_SE3): NUPLAN_EGO_STATE_SE3_METADATA,
            }
            self._scene_api = OnlineSceneAPI(
                log_metadata=LogMetadata(dataset=_DATASET, split="", log_name="", location=None),
                modality_metadatas=ego_metadatas
                | {
                    get_modality_key(ModalityType.CAMERA, metadata.camera_id): metadata
                    for metadata in self._camera_metadatas.values()
                },
                iteration_interval_us=_TICK_US,
                num_history_iterations=math.ceil(self._policy.policy_config.required_history_duration_us / _TICK_US),
            )
            # The episode opens with no past ego states, so the first one fills the history.
            for past_step in range(1, self._scene_api.num_history_iterations + 1):
                past_us = time_us - past_step * _TICK_US
                self._scene_api.append(
                    past_us,
                    {get_modality_key(ModalityType.EGO_STATE_SE3): _ego_state(frame, past_us)},
                )
            self._plan = None
        assert self._scene_api is not None, "an episode starts at step 0"

        modalities: dict[str, BaseModality] = {get_modality_key(ModalityType.EGO_STATE_SE3): ego_state}
        for name, metadata in self._camera_metadatas.items():
            modalities[get_modality_key(ModalityType.CAMERA, metadata.camera_id)] = Camera(
                metadata=metadata,
                image=images[name][step],
                camera_to_global_se3=rel_to_abs_se3(
                    origin=ego_state.rear_axle_se3,
                    pose_se3=metadata.camera_to_imu_se3,
                ),
                timestamp=Timestamp.from_us(time_us),
            )
        self._scene_api.append(time_us, modalities)

        ego_pose = ego_state.rear_axle_se3
        ego_pose_local = np.array([ego_pose.x, ego_pose.y, ego_pose.z, ego_pose.yaw], dtype=np.float64)
        if self._plan is None or time_us - self._plan.created_time_us >= self._replan_interval_us:
            route_xy = route_centerline[0, route_mask[0], :2].numpy().astype(np.float64)
            self._plan = self._replan(
                self._scene_api,
                rig_to_local_xy(route_xy, ego_pose_local),
                ego_pose_local,
                time_us,
            )
        return _plan_in_ego_frame(self._plan, ego_pose_local, time_us, self._num_poses)

    def _replan(
        self,
        scene_api: OnlineSceneAPI,
        route_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
        ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
        time_us: int,
    ) -> CachedPlan:
        """
        Runs the policy with target points on the route ahead of the ego.

        Args:
            scene_api: the episode's live scene.
            route_xy: the route from the ego's projection onto it on, in the global frame.
            ego_pose_local: the (x, y, z, yaw) of the ego in the global frame.
            time_us: the current simulation time.

        Returns:
            the plan, fixed in the global frame.
        """
        scene_api.set_route(
            RouteMetadata(
                resolution_m=_ROUTE_RESOLUTION_M,
                total_arc_m=(len(route_xy) - 1) * _ROUTE_RESOLUTION_M,
                polyline_x=route_xy[:, 0].tolist(),
                polyline_y=route_xy[:, 1].tolist(),
                polyline_z=[float(ego_pose_local[2])] * len(route_xy),
                cache_source_info=provided_route_source_info("odyssey route centerline"),
                source="provided",
            )
            if len(route_xy) >= 2
            else None,
        )
        snapshot = scene_api.snapshot()
        policy_config = self._policy.policy_config
        features = self._policy.build_features(snapshot, {}).apply(lambda tensor: tensor.unsqueeze(0))
        # The points past the route's end read as no intent instead of being made up.
        navigation = NavigationConditioning.from_scene(
            snapshot,
            policy_config.required_target_point_distances_m,
            nan_past_route_end=True,
        ).apply(lambda tensor: tensor.unsqueeze(0))
        device = torch.device(self.device)
        with torch.no_grad():
            predictions = self._policy.forward(features.to(device), navigation.to(device))
        if self.model_observer is not None:
            self.model_observer(
                dict(features.named_tensors()),
                dict(navigation.named_tensors()),
                dict(predictions.named_tensors()),
            )
        poses_rig = predictions.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        return make_plan(poses_rig, ego_pose_local, time_us, policy_config.trajectory_offsets_us)


def _ego_state(frame: dict[str, Any], timestamp_us: int) -> EgoStateSE3:
    """
    Turns a frame's ego pose and motion into an EgoStateSE3.

    Args:
        frame: the runtime's frame dict.
        timestamp_us: the time to stamp the state with.

    Returns:
        the ego state.
    """
    velocity_x, velocity_y, acceleration_x, acceleration_y = (float(value) for value in frame["ego_dynamic_state"])
    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=PoseSE3.from_transformation_matrix(np.asarray(frame["ego2global"], dtype=np.float64)),
        metadata=NUPLAN_EGO_STATE_SE3_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(x=velocity_x, y=velocity_y, z=0.0),
            acceleration=Vector3D(x=acceleration_x, y=acceleration_y, z=0.0),
            angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        ),
    )


def _camera_metadata(
    name: str,
    calibration: dict[str, Any],
    image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
) -> PinholeCameraMetadata:
    """
    Builds a camera's metadata from the calibration the runtime renders it with.

    Args:
        name: the rig camera's name, e.g. CAM_F0.
        calibration: the camera's entry in the frame's cams.
        image: a rendered frame of the camera.

    Returns:
        the undistorted pinhole camera metadata, mounted relative to the rear axle.
    """
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    x, y, z = (float(value) for value in calibration["sensor2ego_translation"])
    qw, qx, qy, qz = (float(value) for value in calibration["sensor2ego_rotation"])
    return PinholeCameraMetadata(
        camera_name=name,
        camera_id=CameraID[f"P{name}"],
        intrinsics=PinholeIntrinsics(
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
        ),
        distortion=None,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        camera_to_imu_se3=PoseSE3(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz),
        is_undistorted=True,
    )


def _plan_in_ego_frame(
    plan: CachedPlan,
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
    time_us: int,
    num_poses: int,
) -> jt.Float64[npt.NDArray[np.float64], "num_poses 3"]:
    """
    Samples the plan on the ticks after now, in the current ego frame.

    Args:
        plan: the plan, fixed in the global frame.
        ego_pose_local: the (x, y, z, yaw) of the ego in the global frame.
        time_us: the current simulation time.
        num_poses: how many ticks to sample.

    Returns:
        the (x, y, heading) poses relative to the ego, x forward and y left.
    """
    sample_times_s = (time_us - plan.created_time_us) / 1e6 + np.arange(1, num_poses + 1) * _TICK_US / 1e6
    delta_x = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 0]) - ego_pose_local[0]
    delta_y = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 1]) - ego_pose_local[1]
    headings = np.interp(sample_times_s, plan.times_s, np.unwrap(plan.yaws)) - ego_pose_local[3]
    cos_yaw, sin_yaw = math.cos(ego_pose_local[3]), math.sin(ego_pose_local[3])
    return np.stack(
        [
            cos_yaw * delta_x + sin_yaw * delta_y,
            -sin_yaw * delta_x + cos_yaw * delta_y,
            np.arctan2(np.sin(headings), np.cos(headings)),
        ],
        axis=1,
    )
