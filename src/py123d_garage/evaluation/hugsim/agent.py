# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The agent HUGSIM's closed_loop.py launches for an episode, driving every simulation step with one garage policy."""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Final

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from agents.client import serve
from py123d.datatypes import (
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
from py123d.datatypes.modalities.base_modality import get_modality_key
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_EGO_STATE_SE3_METADATA

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.common.config_help import load_evaluation_config
from py123d_garage.config.schema.evaluation.hugsim_config import HugsimBenchmarkConfig
from py123d_garage.evaluation.alpasim.help.plan import CachedPlan, make_plan
from py123d_garage.evaluation.help.policy_building import build_evaluation_policy
from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

# HUGSIM renders one virtual rig on every dataset, and its front camera has about the field of view of nuPlan's.
_DATASET: Final[str] = "nuplan"
_CAMERA_NAME: Final[str] = "CAM_FRONT"
_CAMERA_ID: Final[CameraID] = CameraID.PCAM_F0
# HUGSIM tracks and scores the plan as positions 0.5 s apart; its baselines plan 4 s.
_PLAN_INTERVAL_S: Final[float] = 0.5
_NUM_PLAN_POSES: Final[int] = 8


class GarageAgent:
    """Plans with a garage policy every replan interval and answers with the stored plan on the steps between."""

    def __init__(self, checkpoint_file: str, device: torch.device) -> None:
        """
        Builds the policy from the checkpoint and its config.yaml.

        Args:
            checkpoint_file: the policy checkpoint.
            device: the device the policy runs on.
        """
        config = load_evaluation_config(
            HugsimBenchmarkConfig,
            "evaluate_hugsim",
            args=[f"policy_config.evaluation_checkpoint_files=[{checkpoint_file}]"],
        )
        policy = build_evaluation_policy(config.policy_config)
        policy.verify_contract(benchmark_config=config)
        self._policy: AnyPolicy = policy.to(device).eval()
        self._device = device
        self._replan_interval_us = config.replan_interval_us
        self._scene_api: OnlineSceneAPI | None = None
        self._camera_metadata: PinholeCameraMetadata | None = None
        self._plan: CachedPlan | None = None

    def step(
        self,
        obs: dict[str, dict[str, npt.NDArray[Any]]],
        info: dict[str, Any],
    ) -> jt.Float64[npt.NDArray[np.float64], "num_poses 2"]:
        """
        Observes the newest step and answers with the stored plan, replanning once it is replan_interval_us old.

        Args:
            obs: the rendered images by channel and camera.
            info: the simulator's state of the step.

        Returns:
            the planned (right, forward) positions in the current ego frame, one every 0.5 s from 0.5 s on.
        """
        time_us = round(info["timestamp"] * 1e6)
        ego_state = _ego_state(info, time_us)
        image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"] = obs["rgb"][_CAMERA_NAME]
        if self._scene_api is None:
            step_us = round(info["dt"] * 1e6)
            self._camera_metadata = _camera_metadata(info["cam_params"][_CAMERA_NAME])
            self._scene_api = OnlineSceneAPI(
                log_metadata=LogMetadata(dataset=_DATASET, split="", log_name="", location=None),
                modality_metadatas={
                    get_modality_key(ModalityType.EGO_STATE_SE3): NUPLAN_EGO_STATE_SE3_METADATA,
                    get_modality_key(ModalityType.CAMERA, _CAMERA_ID): self._camera_metadata,
                },
                iteration_interval_us=step_us,
                num_history_iterations=math.ceil(self._policy.policy_config.required_history_duration_us / step_us),
            )
            # The episode opens with no past ego states, so the first one fills the history.
            for past_step in range(1, self._scene_api.num_history_iterations + 1):
                past_us = time_us - past_step * step_us
                self._scene_api.append(
                    past_us,
                    {get_modality_key(ModalityType.EGO_STATE_SE3): _ego_state(info, past_us)},
                )
        assert self._camera_metadata is not None, "the first step builds the camera metadata"

        self._scene_api.append(
            time_us,
            {
                get_modality_key(ModalityType.EGO_STATE_SE3): ego_state,
                get_modality_key(ModalityType.CAMERA, _CAMERA_ID): Camera(
                    metadata=self._camera_metadata,
                    image=image,
                    camera_to_global_se3=rel_to_abs_se3(
                        origin=ego_state.rear_axle_se3,
                        pose_se3=self._camera_metadata.camera_to_imu_se3,
                    ),
                    timestamp=Timestamp.from_us(time_us),
                ),
            },
        )
        ego_pose = ego_state.rear_axle_se3
        ego_pose_local = np.array([ego_pose.x, ego_pose.y, ego_pose.z, ego_pose.yaw], dtype=np.float64)
        if self._plan is None or time_us - self._plan.created_time_us >= self._replan_interval_us:
            self._plan = self._replan(self._scene_api, ego_pose_local, time_us)
        return _plan_in_ego_frame(self._plan, ego_pose_local, time_us)

    def _replan(
        self,
        scene_api: OnlineSceneAPI,
        ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
        time_us: int,
    ) -> CachedPlan:
        """
        Runs the policy without navigation intent, since HUGSIM serves no route.

        Args:
            scene_api: the episode's live scene.
            ego_pose_local: the (x, y, z, yaw) of the ego in the global frame.
            time_us: the current simulation time.

        Returns:
            the plan, fixed in the global frame.
        """
        policy_config = self._policy.policy_config
        features = self._policy.build_features(scene_api.snapshot(), {}).apply(lambda tensor: tensor.unsqueeze(0))
        num_target_points = len(policy_config.required_target_point_distances_m)
        navigation = NavigationConditioning(
            target_points=torch.full((1, num_target_points, 2), torch.nan),
            target_point_headings=torch.full((1, num_target_points), torch.nan),
        )
        with torch.no_grad():
            predictions = self._policy.forward(features.to(self._device), navigation.to(self._device))
        poses_rig = predictions.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        return make_plan(poses_rig, ego_pose_local, time_us, policy_config.trajectory_offsets_us)


def _ego_state(info: dict[str, Any], timestamp_us: int) -> EgoStateSE3:
    """
    Turns the simulator's ego box and speed into an EgoStateSE3.

    Args:
        info: the simulator's state of the step.
        timestamp_us: the time to stamp the state with.

    Returns:
        the ego state, its rear axle at the box's reference point.
    """
    x, y, z, _, _, _, yaw = (float(value) for value in info["ego_box"])
    ego_to_global = np.eye(4)
    ego_to_global[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    ego_to_global[:3, 3] = [x, y, z]
    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=PoseSE3.from_transformation_matrix(ego_to_global),
        metadata=NUPLAN_EGO_STATE_SE3_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(x=float(info["ego_velo"]), y=0.0, z=0.0),
            acceleration=Vector3D(x=float(info["accelerate"]), y=0.0, z=0.0),
            angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        ),
    )


def _camera_metadata(camera_params: dict[str, Any]) -> PinholeCameraMetadata:
    """
    Builds the front camera's metadata from the calibration HUGSIM renders it with.

    Args:
        camera_params: the camera's entry in the step's cam_params.

    Returns:
        the pinhole camera metadata, mounted relative to the vehicle frame of the rig.
    """
    intrinsic = camera_params["intrinsic"]
    width, height = int(intrinsic["W"]), int(intrinsic["H"])
    return PinholeCameraMetadata(
        camera_name=_CAMERA_NAME,
        camera_id=_CAMERA_ID,
        intrinsics=PinholeIntrinsics(
            fx=width / (2 * math.tan(intrinsic["fovx"] / 2)),
            fy=height / (2 * math.tan(intrinsic["fovy"] / 2)),
            cx=float(intrinsic["cx"]),
            cy=float(intrinsic["cy"]),
        ),
        distortion=None,
        width=width,
        height=height,
        camera_to_imu_se3=PoseSE3.from_transformation_matrix(np.linalg.inv(camera_params["v2c"])),
        is_undistorted=True,
    )


def _plan_in_ego_frame(
    plan: CachedPlan,
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
    time_us: int,
) -> jt.Float64[npt.NDArray[np.float64], "num_poses 2"]:
    """
    Samples the plan every 0.5 s after now, in the current ego frame.

    Args:
        plan: the plan, fixed in the global frame.
        ego_pose_local: the (x, y, z, yaw) of the ego in the global frame.
        time_us: the current simulation time.

    Returns:
        the (right, forward) positions relative to the ego.
    """
    sample_times_s = (time_us - plan.created_time_us) / 1e6 + np.arange(1, _NUM_PLAN_POSES + 1) * _PLAN_INTERVAL_S
    delta_x = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 0]) - ego_pose_local[0]
    delta_y = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 1]) - ego_pose_local[1]
    cos_yaw, sin_yaw = math.cos(ego_pose_local[3]), math.sin(ego_pose_local[3])
    forward = cos_yaw * delta_x + sin_yaw * delta_y
    left = -sin_yaw * delta_x + cos_yaw * delta_y
    return np.stack([-left, forward], axis=1)


def main() -> None:
    cuda_id, episode_dir = sys.argv[1:]
    agent = GarageAgent(os.environ["CHECKPOINT_FILE"], torch.device(f"cuda:{cuda_id}"))
    serve(episode_dir, agent.step, name="py123d_garage")


if __name__ == "__main__":
    main()
