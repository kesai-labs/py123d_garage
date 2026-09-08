from __future__ import annotations

from typing import Any, ClassVar, cast

import carla
import numpy as np
import numpy.typing as npt
import torch
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from py123d.datatypes import (
    BaseModality,
    BaseModalityMetadata,
    Camera,
    DynamicStateSE3,
    EgoStateSE3,
    Lidar,
    LogMetadata,
    ModalityType,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import get_modality_key
from py123d.geometry import EulerAngles, PoseSE3, Quaternion, Vector3D

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.common.config_help import build_from_string, load_evaluation_config, run_dir
from py123d_garage.common.video_writer import VideoWriter
from py123d_garage.config.schema.evaluation.carla_config import (
    CarlaBenchmarkConfig,
)
from py123d_garage.datatypes.numerics import NonNegativeFloat
from py123d_garage.evaluation.carla.help.kalman import GnssKalmanFilter
from py123d_garage.evaluation.carla.help.localization import (
    find_gps_ref,
    gnss_to_carla,
    preprocess_compass,
)
from py123d_garage.evaluation.carla.help.pid import WaypointFollower
from py123d_garage.evaluation.carla.help.rig import CARLA_FPS, SensorRig
from py123d_garage.evaluation.carla.help.route import Route
from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

COMPASS_INDEX = 6


def get_entry_point() -> str:
    """
    The agent class the leaderboard instantiates from this module.

    Returns:
        the class name.
    """
    return "PolicyAgent"


class PolicyAgent(AutonomousAgent):
    """
    Leaderboard agent running a garage policy and tracking its predicted waypoints.

    --agent-config points to the config arguments the evaluation run saved: they
    name the policy, its checkpoint, and the sensor rig it was trained on.
    """

    _routes_driven: ClassVar[int] = 0
    _py123d_garage_video_writer: VideoWriter | None = None

    def setup(self, path_to_conf_file: str) -> None:
        """
        Loads the policy, mounts its rig, and prepares the route.

        Args:
            path_to_conf_file: path to the config arguments the evaluation run saved.
        """
        self.track = Track.SENSORS
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        del path_to_conf_file
        py123d_garage_config: CarlaBenchmarkConfig = load_evaluation_config(
            CarlaBenchmarkConfig,
            "evaluate_carla",
            args=[],
        )
        sensor_rig_file = py123d_garage_config.policy_config.evaluation_sensor_rig_file
        if sensor_rig_file is None:
            raise ValueError(
                "driving in CARLA needs policy_config.evaluation_sensor_rig_file, the rig the checkpoint was trained on",
            )
        self._sensor_rig = SensorRig(sensor_rig_file)

        py123d_garage_policy: AnyPolicy = build_from_string(py123d_garage_config.policy_config, AbstractPolicy)
        py123d_garage_policy.initialize(py123d_garage_config.policy_config.evaluation_checkpoint_file)
        self._py123d_garage_policy = py123d_garage_policy.to(self._device).eval()
        self._target_point_distances_m = py123d_garage_policy.policy_config.required_target_point_distances_m

        self._route = Route(
            self._world_route,
            extension_m=max(self._target_point_distances_m) + 2.0,
        )
        self._scene_api = OnlineSceneAPI(
            log_metadata=LogMetadata(
                dataset=py123d_garage_config.dataset,
                split="",
                log_name="",
                location=py123d_garage_config.location,
            ),
            modality_metadatas=self._modality_metadatas(),
            iteration_interval_us=round(1e6 / CARLA_FPS),
            num_history_iterations=py123d_garage_config.num_history_iterations,
        )
        self._scene_api.set_route(self._route.metadata)
        self._step = 0
        self._control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self._kalman_filter = GnssKalmanFilter(1.0 / CARLA_FPS) if py123d_garage_config.use_kalman_filter else None

        if py123d_garage_config.visualization_interval:
            self._py123d_garage_video_writer = VideoWriter(
                run_dir() / f"route_{PolicyAgent._routes_driven:02d}",
                py123d_garage_config.visualization_interval,
                fps=CARLA_FPS / py123d_garage_config.visualization_interval,
            )
        PolicyAgent._routes_driven += 1

        self._waypoint_follower = WaypointFollower(
            self._py123d_garage_policy.policy_config.trajectory_interval_s,
        )

        assert self._global_plan is not None
        assert self._global_plan_world_coord is not None
        self._lat_ref, self._lon_ref = find_gps_ref(
            self._global_plan,
            self._global_plan_world_coord,
        )

    def set_global_plan(
        self,
        global_plan_gps: list[tuple[dict[str, float], Any]],
        global_plan_world_coord: list[tuple[Any, Any]],
    ) -> None:
        """
        Keeps the dense route the base class is about to downsample away.

        The leaderboard samples its plan down to 200 m spacing, far coarser than
        the arc-length target points a py123d_garage_policy is conditioned on.

        Args:
            global_plan_gps: the route as ({'lat', 'lon', 'z'}, road option) tuples.
            global_plan_world_coord: the same route in CARLA world coordinates.
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._world_route = global_plan_world_coord

    def _modality_metadatas(self) -> dict[str, BaseModalityMetadata]:
        """The modality metadata the live scene serves, keyed as a 123D log keys them."""
        metadatas: dict[str, BaseModalityMetadata] = {
            get_modality_key(ModalityType.EGO_STATE_SE3): self._sensor_rig.ego_metadata,
            self._sensor_rig.lidar_modality_key: self._sensor_rig.lidar_metadata,
        }
        for camera_id, metadata in self._sensor_rig.camera_metadatas.items():
            metadatas[get_modality_key(ModalityType.CAMERA, camera_id)] = metadata
        return metadatas

    def sensors(self) -> list[dict[str, float | int | str]]:
        """
        The rig the policy was trained on, plus localization.

        Returns:
            the leaderboard sensor definitions.
        """
        return self._sensor_rig.leaderboard_sensors()

    def _record_tick(
        self,
        input_data: dict[str, tuple[int, Any]],
        ego_pose_se2: npt.NDArray[np.float64],
        speed: NonNegativeFloat,
        route_progress_m: NonNegativeFloat,
    ) -> None:
        """
        Turns this tick's sensor readings into 123D modalities and appends them to the scene.

        Args:
            input_data: sensor readings keyed by sensor id.
            ego_pose_se2: the localized ego rear-axle pose (x, y, yaw) in the ISO world frame.
            speed: current ego speed in m/s.
            route_progress_m: the ego's arc-length position on the route.
        """
        timestamp = Timestamp.from_us(round(self._step * 1e6 / CARLA_FPS))
        quaternion = Quaternion.from_euler_angles(
            EulerAngles(roll=0.0, pitch=0.0, yaw=float(ego_pose_se2[2])),
        )
        modalities: dict[str, BaseModality] = {
            get_modality_key(ModalityType.EGO_STATE_SE3): EgoStateSE3.from_rear_axle(
                rear_axle_se3=PoseSE3(
                    x=float(ego_pose_se2[0]),
                    y=float(ego_pose_se2[1]),
                    z=0.0,
                    qw=quaternion.qw,
                    qx=quaternion.qx,
                    qy=quaternion.qy,
                    qz=quaternion.qz,
                ),
                metadata=self._sensor_rig.ego_metadata,
                timestamp=timestamp,
                dynamic_state_se3=DynamicStateSE3(
                    velocity=Vector3D(x=speed, y=0.0, z=0.0),
                    acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
                    angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                ),
            ),
            self._sensor_rig.lidar_modality_key: Lidar(
                timestamp=timestamp,
                timestamp_end=timestamp,
                metadata=self._sensor_rig.lidar_metadata,
                point_cloud_3d=self._sensor_rig.merge_lidar_sweeps(
                    [
                        cast("npt.NDArray[np.float32]", input_data[sensor_id][1]).astype(np.float32)
                        for sensor_id in self._sensor_rig.lidar_sensor_ids
                    ],
                ),
            ),
        }
        for camera_id, metadata in self._sensor_rig.camera_metadatas.items():
            image = self._sensor_rig.crop_to_rig(
                camera_id,
                cast(
                    "npt.NDArray[np.uint8]",
                    input_data[self._sensor_rig.camera_sensor_id(camera_id)][1],
                ),
            )
            modalities[get_modality_key(ModalityType.CAMERA, camera_id)] = Camera(
                metadata=metadata,
                # CARLA renders BGRA; the logs store RGB.
                image=np.ascontiguousarray(image[:, :, 2::-1]),
                camera_to_global_se3=metadata.camera_to_imu_se3,
                timestamp=timestamp,
            )
        self._scene_api.append(timestamp.time_us, modalities, route_progress_m)

    @torch.inference_mode()
    def run_step(
        self,
        input_data: dict[str, tuple[int, Any]],
        timestamp: NonNegativeFloat,
    ) -> carla.VehicleControl:
        """
        Runs one control tick: localize, record, predict waypoints, track them.

        Args:
            input_data: sensor readings keyed by sensor id.
            timestamp: game time in seconds.

        Returns:
            the vehicle control to apply.
        """
        del timestamp
        self._step += 1
        speed = float(input_data["speed"][1]["speed"])
        carla_yaw = preprocess_compass(float(input_data["imu"][1][COMPASS_INDEX]))
        carla_position = gnss_to_carla(
            cast("npt.NDArray[np.float64]", input_data["gps"][1]).astype(np.float64),
            self._lat_ref,
            self._lon_ref,
        )[:2]
        if self._kalman_filter is not None:
            carla_position = self._kalman_filter.step(
                carla_position,
                carla_yaw,
                speed,
                self._control.steer,
                self._control.throttle,
                self._control.brake,
            )
        ego_pose_se2 = np.array([carla_position[0], -carla_position[1], -carla_yaw])
        self._record_tick(
            input_data,
            ego_pose_se2,
            speed,
            self._route.advance(ego_pose_se2[:2]),
        )

        scene_api = self._scene_api.snapshot()
        navigation = NavigationConditioning.from_scene(
            scene_api,
            self._target_point_distances_m,
        )
        features = self._py123d_garage_policy.build_features(scene_api, {}).apply(
            lambda tensor: tensor.unsqueeze(0),
        )
        batched_navigation = navigation.apply(lambda tensor: tensor.unsqueeze(0))
        predictions = self._py123d_garage_policy.forward(
            features.to(self._device),
            batched_navigation.to(self._device),
        )
        trajectory_se2 = predictions.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        # TransFuser++'s follower tracks the vehicle centre, which swings out of the rear axle's path as the car turns.
        rear_axle_to_center_m = self._sensor_rig.ego_metadata.rear_axle_to_center_longitudinal
        waypoints = trajectory_se2[:, :2] + rear_axle_to_center_m * np.stack(
            [np.cos(trajectory_se2[:, 2]) - 1.0, np.sin(trajectory_se2[:, 2])],
            axis=1,
        )

        if self._py123d_garage_video_writer is not None and self._py123d_garage_video_writer.is_record_tick(self._step):
            self._py123d_garage_video_writer.write(
                self._py123d_garage_policy.visualize_batch(
                    features,
                    None,
                    batched_navigation,
                    predictions,
                    scene_api,
                ),
            )

        steer, throttle, brake = self._waypoint_follower.step(waypoints, speed)
        self._control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        return self._control

    def destroy(self) -> None:
        """Finalizes the videos and releases the policy and its device memory."""
        if self._py123d_garage_video_writer is not None:
            self._py123d_garage_video_writer.close()
        # The leaderboard calls destroy() even when setup() failed before the policy was built.
        if hasattr(self, "_py123d_garage_policy"):
            del self._py123d_garage_policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
