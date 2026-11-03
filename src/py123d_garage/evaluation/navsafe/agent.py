# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The agent process NavSafe's py123d_garage adapter starts for an episode, answering every replan with a plan."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, Final

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import yaml
from py123d.datatypes import (
    Camera,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LogMetadata,
    ModalityType,
    Timestamp,
    camera_metadata_from_dict,
)
from py123d.datatypes.modalities.base_modality import get_modality_key
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_EGO_STATE_SE3_METADATA

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import AbstractFeatures, NavigationConditioning
from py123d_garage.common.config_help import load_evaluation_config
from py123d_garage.config.schema.evaluation.navsafe_config import NavsafeBenchmarkConfig
from py123d_garage.evaluation.alpasim.help.plan import make_plan
from py123d_garage.evaluation.help.policy_building import build_evaluation_policy
from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

# NavSafe reconstructs nuPlan logs and renders nuPlan's front camera.
_DATASET: Final[str] = "nuplan"
_CAMERA_ID: Final[CameraID] = CameraID.PCAM_F0
_FRAME_US: Final[int] = 100_000
# NavSafe tracks the plan as positions 0.5 s apart over 4 s.
_PLAN_INTERVAL_S: Final[float] = 0.5
_NUM_PLAN_POSES: Final[int] = 8


class GarageAgent:
    """Plans with a garage policy on every call, from the front camera and the ego state NavSafe serves."""

    def __init__(self, checkpoint_file: str, device: torch.device) -> None:
        """
        Builds the policy from the checkpoint, its config.yaml and the nuPlan sensor rig saved next to it.

        Args:
            checkpoint_file: the policy checkpoint.
            device: the device the policy runs on.
        """
        sensor_rig_file = Path(checkpoint_file).parent / f"sensor_rig_{_DATASET}.yaml"
        config = load_evaluation_config(
            NavsafeBenchmarkConfig,
            "evaluate_navsafe",
            args=[
                f"policy_config.evaluation_checkpoint_files=[{checkpoint_file}]",
                f"policy_config.evaluation_sensor_rig_file={sensor_rig_file}",
            ],
        )
        policy = build_evaluation_policy(config.policy_config)
        policy.verify_contract(benchmark_config=config)
        self._policy: AnyPolicy = policy.to(device).eval()
        self._device = device
        rig_modalities: dict[str, Any] = yaml.safe_load(sensor_rig_file.read_text())["modalities"]
        self._camera_metadata = camera_metadata_from_dict(
            rig_modalities[get_modality_key(ModalityType.CAMERA, _CAMERA_ID)],
        )
        self._scene_api: OnlineSceneAPI | None = None

    def plan(
        self,
        request: dict[str, Any],
        image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    ) -> jt.Float64[npt.NDArray[np.float64], "num_poses 2"]:
        """
        Observes the call's ego state and front camera, then plans without navigation intent.

        Args:
            request: the frame and the ego's box centre state in the world frame.
            image: the RGB front camera frame.

        Returns:
            the planned (left, forward) positions of the box centre relative to the current one, one every 0.5 s.
        """
        time_us = request["frame"] * _FRAME_US
        ego_state = _ego_state(request, time_us)
        if self._scene_api is None:
            self._scene_api = OnlineSceneAPI(
                log_metadata=LogMetadata(dataset=_DATASET, split="", log_name="", location=None),
                modality_metadatas={
                    get_modality_key(ModalityType.EGO_STATE_SE3): NUPLAN_EGO_STATE_SE3_METADATA,
                    get_modality_key(ModalityType.CAMERA, _CAMERA_ID): self._camera_metadata,
                },
                iteration_interval_us=_FRAME_US,
                num_history_iterations=math.ceil(self._policy.policy_config.required_history_duration_us / _FRAME_US),
            )
            # The episode opens with no past ego states, so the first one fills the history.
            for past_frame in range(1, self._scene_api.num_history_iterations + 1):
                past_us = time_us - past_frame * _FRAME_US
                self._scene_api.append(
                    past_us,
                    {get_modality_key(ModalityType.EGO_STATE_SE3): _ego_state(request, past_us)},
                )
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

        policy_config = self._policy.policy_config
        features: AbstractFeatures = self._policy.build_features(self._scene_api.snapshot(), {})
        features = features.apply(lambda tensor: tensor.unsqueeze(0))
        num_target_points = len(policy_config.required_target_point_distances_m)
        navigation = NavigationConditioning(
            target_points=torch.full((1, num_target_points, 2), torch.nan),
            target_point_headings=torch.full((1, num_target_points), torch.nan),
        )
        with torch.no_grad():
            predictions = self._policy.forward(features.to(self._device), navigation.to(self._device))
        poses_rig = predictions.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        rear_axle = ego_state.rear_axle_se3
        plan = make_plan(
            poses_rig,
            np.array([rear_axle.x, rear_axle.y, rear_axle.z, rear_axle.yaw], dtype=np.float64),
            time_us,
            policy_config.trajectory_offsets_us,
        )

        # NavSafe tracks the box centre, which sits ahead of the rear axle the policy plans.
        center = ego_state.center_se3
        cos_yaw, sin_yaw = math.cos(rear_axle.yaw), math.sin(rear_axle.yaw)
        center_offset_forward = cos_yaw * (center.x - rear_axle.x) + sin_yaw * (center.y - rear_axle.y)
        sample_times_s = np.arange(1, _NUM_PLAN_POSES + 1) * _PLAN_INTERVAL_S
        yaws = np.interp(sample_times_s, plan.times_s, np.unwrap(plan.yaws))
        delta_x = (
            np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 0])
            + center_offset_forward * np.cos(yaws)
            - center.x
        )
        delta_y = (
            np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 1])
            + center_offset_forward * np.sin(yaws)
            - center.y
        )
        cos_heading, sin_heading = math.cos(request["heading"]), math.sin(request["heading"])
        forward = cos_heading * delta_x + sin_heading * delta_y
        left = -sin_heading * delta_x + cos_heading * delta_y
        return np.stack([left, forward], axis=1)


def _ego_state(request: dict[str, Any], timestamp_us: int) -> EgoStateSE3:
    """
    Turns NavSafe's world-frame box centre state into an EgoStateSE3.

    Args:
        request: the ego's box centre state in the world frame.
        timestamp_us: the time to stamp the state with.

    Returns:
        the ego state, with the velocity and acceleration in the vehicle frame.
    """
    heading = request["heading"]
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)
    center_to_global = np.eye(4)
    center_to_global[:2, :2] = [[cos_heading, -sin_heading], [sin_heading, cos_heading]]
    center_to_global[:2, 3] = request["position_xy"]
    velocity_x, velocity_y = request["velocity_xy"]
    acceleration_x, acceleration_y = request["acceleration_xy"]
    return EgoStateSE3.from_center(
        center_se3=PoseSE3.from_transformation_matrix(center_to_global),
        metadata=NUPLAN_EGO_STATE_SE3_METADATA,
        timestamp=Timestamp.from_us(timestamp_us),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(
                x=cos_heading * velocity_x + sin_heading * velocity_y,
                y=-sin_heading * velocity_x + cos_heading * velocity_y,
                z=0.0,
            ),
            acceleration=Vector3D(
                x=cos_heading * acceleration_x + sin_heading * acceleration_y,
                y=-sin_heading * acceleration_x + cos_heading * acceleration_y,
                z=0.0,
            ),
            angular_velocity=Vector3D(x=0.0, y=0.0, z=request["yaw_rate"]),
        ),
    )


def _read_exactly(stream: BinaryIO, num_bytes: int) -> bytes:
    """
    Reads a payload that may arrive in several pipe chunks.

    Args:
        stream: the pipe.
        num_bytes: the payload's size.

    Returns:
        the payload.
    """
    chunks: list[bytes] = []
    while num_bytes > 0:
        chunk = stream.read(num_bytes)
        assert chunk, "the adapter closed the pipe inside a payload"
        chunks.append(chunk)
        num_bytes -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    # The protocol owns stdout; everything the policy prints goes to stderr.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    agent = GarageAgent(sys.argv[1], torch.device("cuda"))
    protocol_out.write("READY\n")
    protocol_out.flush()
    for line in sys.stdin.buffer:
        request = json.loads(line)
        height, width, channels = request["image_shape"]
        image = np.frombuffer(_read_exactly(sys.stdin.buffer, height * width * channels), dtype=np.uint8)
        trajectory = agent.plan(request, image.reshape(height, width, channels))
        protocol_out.write(json.dumps({"trajectory": trajectory.tolist()}) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    main()
