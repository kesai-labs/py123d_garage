# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from NVlabs/alpasim (Apache License 2.0).

"""The AlpaSim egodriver gRPC service, driving every rollout with one garage policy."""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import cv2
import grpc
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc, sensorsim_pb2
from py123d.datatypes import (
    BaseCameraMetadata,
    BaseModalityMetadata,
    Camera,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    EgoStateSE3Metadata,
    LogMetadata,
    ModalityType,
    Timestamp,
)
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, get_modality_key
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import rel_to_abs_se3
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.common.config_help import run_dir
from py123d_garage.common.sensor.hood_swap import ego_hood_names, paste_hood
from py123d_garage.common.video_writer import VideoWriter
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt
from py123d_garage.evaluation.alpasim.help.config_help import build_policy
from py123d_garage.evaluation.alpasim.help.plan import (
    SERVE_HORIZON_US,
    SERVE_INTERVAL_US,
    CachedPlan,
    make_plan,
    rig_to_local_xy,
    sample_plan,
)
from py123d_garage.evaluation.alpasim.help.sensor_rig_help import (
    build_camera_metadatas,
    build_ego_metadata,
    camera_metadata_from_announcement,
    rig_camera_map,
    rig_camera_metadatas,
    rig_dataset,
)
from py123d_garage.py123d_help.misc import FRONT_CAMERAS, provided_route_source_info
from py123d_garage.py123d_help.scene_api.online_scene_api import OnlineSceneAPI

LOG = logging.getLogger(__name__)

_MIN_FALLBACK_SPEED_MPS = 2.0
# The rate every dataset serves ego states, so the past a policy reads is observed, not interpolated between plans.
_SCENE_INTERVAL_US = 100_000
# Plans per second when replan_interval_us is 0 and the runtime calls every 500 ms.
_VISUALIZATION_FPS = 2.0


class _Session:
    """The per-rollout state the driver keeps between gRPC calls."""

    def __init__(
        self,
        scene_api: OnlineSceneAPI,
        announced_cameras: dict[CameraID, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera],
        scene_id: str,
    ) -> None:
        """
        Starts a session with an empty plan history and no ego pose yet.

        Args:
            scene_api: the live scene the session's observations are appended to.
            announced_cameras: the runtime's camera announcements by 123D camera id.
            scene_id: the rollout's scene, naming its plan log.
        """
        self.scene_api = scene_api
        self.announced_cameras = announced_cameras
        self.scene_id = scene_id
        self.plan_records: list[dict[str, npt.NDArray[np.float64]]] = []
        self.camera_metadatas: dict[CameraID, BaseCameraMetadata] = {}
        # The latest (x, y, z, yaw) of the ego in the rollout's local frame.
        self.ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"] | None = None
        self.speed_mps: float = 0.0
        self.acceleration_mps2: float = 0.0
        self.route_polyline_rig_xy: jt.Float64[npt.NDArray[np.float64], "num_route_points 2"] | None = None
        # The route ahead of the ego at the last plan, in the local frame; route_memory merges the next one into it.
        self.route_xy_local: jt.Float64[npt.NDArray[np.float64], "num_route_points 2"] = np.zeros(
            (0, 2),
            dtype=np.float64,
        )
        self.cached_plan: CachedPlan | None = None
        self.num_plans: int = 0
        self.history_backfilled: bool = False


class GarageDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    """Serves plans for every concurrent rollout from one policy on one device."""

    def __init__(self, config: AlpasimBenchmarkConfig) -> None:
        """
        Reads the sensor rig and starts loading the policy in the background.

        Args:
            config: the benchmark config naming the policy, its rig and the serving options.
        """
        self._py123d_garage_config = config
        sensor_rig_file = self._py123d_garage_config.policy_config.evaluation_sensor_rig_file
        if sensor_rig_file is None:
            raise ValueError(
                "driving in AlpaSim needs policy_config.evaluation_sensor_rig_file, the rig the checkpoint was trained on",
            )
        self._sensor_rig_file = sensor_rig_file
        self._rig_camera_map = rig_camera_map(sensor_rig_file)
        self._camera_setup_dataset = rig_dataset(sensor_rig_file)
        # The cameras the policy reads and their rig calibrations, set once the policy has loaded.
        self._camera_id_by_alpasim_id: dict[str, CameraID] = {}
        self._rig_metadatas: dict[CameraID, BaseCameraMetadata] = {}
        # FRONT_CAMERAS lists left, front, right; the ego hood is pasted onto the front one.
        self._front_center_camera = FRONT_CAMERAS[self._camera_setup_dataset][1]
        for hood in (self._py123d_garage_config.ego_hood, self._py123d_garage_config.rendered_ego_hood):
            if hood is not None and hood not in ego_hood_names():
                raise ValueError(f"no ego hood {hood!r}; the hoods are {ego_hood_names()}")
        self._ego_metadata = build_ego_metadata(self._py123d_garage_config.ego_vehicle_config)
        self._py123d_garage_video_writer = (
            VideoWriter(
                run_dir() / "views",
                self._py123d_garage_config.visualization_interval,
                fps=(
                    1e6 / self._py123d_garage_config.replan_interval_us
                    if self._py123d_garage_config.replan_interval_us
                    else _VISUALIZATION_FPS
                )
                / self._py123d_garage_config.visualization_interval,
            )
            if self._py123d_garage_config.visualization_interval
            else None
        )
        self._plan_log_dir: Path | None = run_dir() / "plans" if self._py123d_garage_config.plan_log else None
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._py123d_garage_policy: AnyPolicy | None = None
        self._policy_error: BaseException | None = None
        self._policy_thread = threading.Thread(target=self._load_policy, daemon=True)
        self._policy_thread.start()

    def _load_policy(self) -> None:
        """Loads the policy off the serving thread so the port opens before the weights."""
        try:
            policy = build_policy(
                self._py123d_garage_config,
                torch.device(self._py123d_garage_config.device),
            )
            camera_id_by_alpasim_id = self._policy_cameras(policy)
            rig_metadatas = rig_camera_metadatas(self._sensor_rig_file, set(camera_id_by_alpasim_id.values()))
            with self._lock:
                self._camera_id_by_alpasim_id = camera_id_by_alpasim_id
                self._rig_metadatas = rig_metadatas
                self._py123d_garage_policy = policy
        except BaseException as error:
            with self._lock:
                self._policy_error = error
            LOG.exception("policy load failed")

    def _policy_cameras(self, policy: AnyPolicy) -> dict[str, CameraID]:
        """
        Maps the rig's AlpaSim camera names to the cameras the policy reads.

        Args:
            policy: the loaded policy.

        Returns:
            the 123D camera id of every AlpaSim camera the policy reads.

        Raises:
            ValueError: if the policy reads no camera of the rig's dataset, or one the rig lacks.
        """
        required = set(policy.policy_config.required_cameras.get(self._camera_setup_dataset, []))
        if not required:
            raise ValueError(
                f"the policy reads no camera of {self._camera_setup_dataset}, the dataset of the sensor rig "
                f"{self._sensor_rig_file}",
            )
        cameras = {name: camera_id for name, camera_id in self._rig_camera_map.items() if camera_id in required}
        missing = required - set(cameras.values())
        if missing:
            raise ValueError(
                f"the sensor rig {self._sensor_rig_file} has no {sorted(camera_id.name for camera_id in missing)}, "
                f"which the policy reads",
            )
        return cameras

    def _ready_policy(self, context: grpc.ServicerContext) -> AnyPolicy:
        """
        Returns the loaded policy, or aborts the call while it loads or after it failed.

        Args:
            context: the gRPC call to abort.

        Returns:
            the loaded policy.
        """
        with self._lock:
            policy = self._py123d_garage_policy
            error = self._policy_error
        if error is not None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"policy load failed: {error}")
        if policy is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "policy is still loading")
            raise AssertionError("unreachable")
        return policy

    @override
    def start_session(
        self,
        request: egodriver_pb2.DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        """Opens a rollout's scene with the announced camera calibration."""
        policy = self._ready_policy(context)
        iteration_interval_us = _SCENE_INTERVAL_US
        announced_cameras: dict[CameraID, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera] = {}
        for announced in request.rollout_spec.vehicle.available_cameras:
            announced_camera_id = self._camera_id_by_alpasim_id.get(announced.logical_id)
            if announced_camera_id is not None:
                announced_cameras[announced_camera_id] = announced
        # Without a camera the policy reads, every plan would be the straight-line fallback.
        unrendered = sorted(
            set(self._camera_id_by_alpasim_id)
            - {announced.logical_id for announced in request.rollout_spec.vehicle.available_cameras},
        )
        if unrendered:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"the runtime renders none of {unrendered}, which the policy reads; the sensor rig "
                f"{self._sensor_rig_file} or the camera config does not fit the track",
            )

        modality_metadatas: dict[str, BaseModalityMetadata] = {
            get_modality_key(ModalityType.EGO_STATE_SE3): self._ego_metadata,
        }
        for camera_id in self._camera_id_by_alpasim_id.values():
            metadata: BaseModalityMetadata | None = self._rig_metadatas.get(camera_id)
            announced = announced_cameras.get(camera_id)
            if announced is not None and announced.intrinsics.resolution_h > 0:
                metadata = (
                    camera_metadata_from_announcement(
                        announced,
                        camera_id,
                        int(announced.intrinsics.resolution_h),
                        int(announced.intrinsics.resolution_w),
                    )
                    or metadata
                )
            if metadata is not None:
                modality_metadatas[get_modality_key(ModalityType.CAMERA, camera_id)] = metadata
        scene_api = OnlineSceneAPI(
            log_metadata=LogMetadata(
                dataset=self._camera_setup_dataset,
                split="",
                log_name="",
                location=None,
            ),
            modality_metadatas=modality_metadatas,
            iteration_interval_us=iteration_interval_us,
            # Rounded up, so a history shorter than one iteration still keeps one past iteration.
            num_history_iterations=math.ceil(
                policy.policy_config.required_history_duration_us / iteration_interval_us,
            ),
        )
        with self._lock:
            self._sessions[request.session_uuid] = _Session(scene_api, announced_cameras, request.debug_info.scene_id)
        LOG.info(
            f"started session {request.session_uuid} with announced calibration for "
            f"{sorted(camera_id.name for camera_id in announced_cameras)}",
        )
        return common_pb2.SessionRequestStatus()

    @override
    def close_session(
        self,
        request: egodriver_pb2.DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """Closes a rollout, flushing its videos and plan log."""
        del context
        with self._lock:
            session = self._sessions.pop(request.session_uuid, None)
            if self._py123d_garage_video_writer is not None:
                self._py123d_garage_video_writer.close(f"policy_{request.session_uuid}_")
        if self._plan_log_dir is not None and session is not None and session.plan_records:
            self._plan_log_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._plan_log_dir / f"{session.scene_id}_{request.session_uuid}.npz",
                **{key: np.stack([record[key] for record in session.plan_records]) for key in session.plan_records[0]},  # pyright: ignore[reportArgumentType]
            )
        LOG.info(f"closed session {request.session_uuid}")
        return common_pb2.Empty()

    @override
    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """Appends one rendered camera frame to the rollout's scene."""
        camera_image = request.camera_image
        if camera_image.logical_id not in self._camera_id_by_alpasim_id:
            return common_pb2.Empty()
        image = _decode_image(camera_image.image_bytes)
        timestamp_us = int(camera_image.frame_end_us)
        session = self._get_session(request.session_uuid, context)
        camera_id = self._camera_id_by_alpasim_id[camera_image.logical_id]
        config = self._py123d_garage_config
        if config.ego_hood is not None and camera_id == self._front_center_camera:
            image = paste_hood(image, config.ego_hood, config.rendered_ego_hood, config.device)
        metadata = self._camera_metadata(session, camera_id, image)
        with self._lock:
            ego_pose_local = session.ego_pose_local
        camera = Camera(
            metadata=metadata,
            image=image,
            camera_to_global_se3=rel_to_abs_se3(
                origin=_ego_se3(ego_pose_local) if ego_pose_local is not None else PoseSE3.identity(),
                pose_se3=metadata.camera_to_imu_se3,
            ),
            timestamp=Timestamp.from_us(timestamp_us),
        )
        session.scene_api.append(
            timestamp_us,
            {get_modality_key(ModalityType.CAMERA, camera_id): camera},
        )
        return common_pb2.Empty()

    def _camera_metadata(
        self,
        session: _Session,
        camera_id: CameraID,
        image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    ) -> BaseCameraMetadata:
        """
        The session's calibration for one camera, preferring announced intrinsics over the rig file.

        Args:
            session: the rollout the calibration is cached in.
            camera_id: the 123D camera id.
            image: the decoded frame the calibration is scaled to.

        Returns:
            the camera metadata.
        """
        with self._lock:
            metadata = session.camera_metadatas.get(camera_id)
            if metadata is not None:
                return metadata
            announced = session.announced_cameras.get(camera_id)
            if announced is not None:
                metadata = camera_metadata_from_announcement(
                    announced,
                    camera_id,
                    int(image.shape[0]),
                    int(image.shape[1]),
                )
            if metadata is None:
                metadata = self._rig_metadatas.get(camera_id)
            if metadata is None:
                alpasim_id = next(
                    name for name, mapped_id in self._camera_id_by_alpasim_id.items() if mapped_id == camera_id
                )
                metadata = build_camera_metadatas({camera_id: image}, {alpasim_id: camera_id})[camera_id]
                LOG.warning(
                    f"camera {camera_id.name} has no announced or rig calibration; serving placeholder "
                    f"pinhole metadata with identity intrinsics and mount. A policy reading either "
                    f"gets wrong geometry.",
                )
            session.camera_metadatas[camera_id] = metadata
            return metadata

    @override
    def submit_egomotion_observation(
        self,
        request: egodriver_pb2.RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """Appends the observed ego poses to the rollout's scene and keeps the latest one."""
        if not request.trajectory.poses:
            return common_pb2.Empty()
        session = self._get_session(request.session_uuid, context)
        latest = request.trajectory.poses[-1]
        aligned = len(request.dynamic_states) == len(request.trajectory.poses)
        with self._lock:
            session.ego_pose_local = np.array(
                [latest.pose.vec.x, latest.pose.vec.y, latest.pose.vec.z, _yaw_from_quaternion(latest.pose.quat)],
                dtype=np.float64,
            )
            if aligned:
                dynamic_state = request.dynamic_states[-1]
                session.speed_mps = math.hypot(dynamic_state.linear_velocity.x, dynamic_state.linear_velocity.y)
                session.acceleration_mps2 = float(dynamic_state.linear_acceleration.x)
        for index, pose_at_time in enumerate(request.trajectory.poses):
            session.scene_api.append(
                int(pose_at_time.timestamp_us),
                {
                    get_modality_key(ModalityType.EGO_STATE_SE3): _ego_state_se3(
                        pose_at_time,
                        request.dynamic_states[index] if aligned else None,
                        self._ego_metadata,
                    ),
                },
            )
        return common_pb2.Empty()

    @override
    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """Keeps the route the runtime serves, in the rig frame."""
        session = self._get_session(request.session_uuid, context)
        polyline = np.array(
            [[waypoint.x, waypoint.y] for waypoint in request.route.waypoints],
            dtype=np.float64,
        ).reshape(-1, 2)
        polyline = polyline[~np.isnan(polyline).any(axis=1)]
        with self._lock:
            session.route_polyline_rig_xy = polyline
        return common_pb2.Empty()

    @override
    def submit_recording_ground_truth(
        self,
        request: egodriver_pb2.GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        """Ignores the recorded ground truth."""
        del request, context
        return common_pb2.Empty()

    @override
    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        """Serves the cached plan, replanning once it is replan_interval_us old."""
        policy = self._ready_policy(context)
        session = self._get_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        offsets_us = policy.policy_config.trajectory_offsets_us
        replan_interval_us = self._py123d_garage_config.replan_interval_us or offsets_us[0]
        with self._lock:
            ego_pose_local = session.ego_pose_local
            plan = session.cached_plan
            fallback_speed_mps = max(_MIN_FALLBACK_SPEED_MPS, session.speed_mps)
        if plan is None or time_now_us - plan.created_time_us >= replan_interval_us:
            fresh = self._replan(policy, session, request.session_uuid, ego_pose_local, time_now_us, offsets_us)
            if fresh is not None:
                plan = fresh
                with self._lock:
                    session.cached_plan = fresh
        trajectory = (
            _trajectory_from_plan(plan, ego_pose_local, time_now_us, fallback_speed_mps)
            if plan is not None and ego_pose_local is not None
            else _straight_line_trajectory(ego_pose_local, time_now_us, fallback_speed_mps)
        )
        return egodriver_pb2.DriveResponse(trajectory=trajectory)

    def _replan(
        self,
        policy: AnyPolicy,
        session: _Session,
        session_uuid: str,
        ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"] | None,
        time_now_us: NonNegativeInt,
        offsets_us: list[PositiveInt],
    ) -> CachedPlan | None:
        """
        Plans from the current pose once every camera and the ego pose have arrived.

        Args:
            policy: the loaded policy.
            session: the rollout to plan for.
            session_uuid: the rollout's id, for logs and videos.
            ego_pose_local: the latest (x, y, z, yaw) of the ego in the local frame.
            time_now_us: the runtime's current time.
            offsets_us: the policy's trajectory offsets.

        Returns:
            a fresh plan, or None to keep serving the previous one.
        """
        served = _served_cameras(session.scene_api, set(self._camera_id_by_alpasim_id.values()))
        if served != set(self._camera_id_by_alpasim_id.values()) or ego_pose_local is None:
            LOG.warning(
                f"session {session_uuid} has {sorted(camera_id.name for camera_id in served)} of "
                f"{sorted(camera_id.name for camera_id in self._camera_id_by_alpasim_id.values())} cameras and "
                f"{'an' if ego_pose_local is not None else 'no'} ego pose; not planning.",
            )
            return None
        poses_rig = self._plan(policy, session, session_uuid)
        return make_plan(poses_rig, ego_pose_local, time_now_us, offsets_us)

    @override
    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
        """Answers the runtime's version probe once the policy has loaded."""
        del request
        # The runtime's startup probe fails on the first non-OK reply.
        self._policy_thread.join()
        self._ready_policy(context)
        return common_pb2.VersionId(
            version_id=f"py123d-garage:{self._py123d_garage_config.policy_config.target}",
            git_hash=self._py123d_garage_config.git_hash,
            grpc_api_version=API_VERSION_MESSAGE,
        )

    @property
    def config(self) -> AlpasimBenchmarkConfig:
        """The benchmark config the driver serves."""
        return self._py123d_garage_config

    def close(self) -> None:
        """Finalizes the videos."""
        if self._py123d_garage_video_writer is not None:
            self._py123d_garage_video_writer.close()

    def _plan(
        self,
        policy: AnyPolicy,
        session: _Session,
        session_uuid: str,
    ) -> jt.Float64[npt.NDArray[np.float64], "num_poses 3"]:
        """
        Runs the policy on the session's scene and route.

        Args:
            policy: the loaded policy.
            session: the rollout to plan for.
            session_uuid: the rollout's id, for logs and videos.

        Returns:
            the planned (x, y, yaw) poses in the rig frame of the current ego pose.
        """
        with self._lock:
            ego_pose_local = session.ego_pose_local
            speed_mps = session.speed_mps
            acceleration_mps2 = session.acceleration_mps2
            route = session.route_polyline_rig_xy
        assert ego_pose_local is not None, "drive verifies the ego pose before planning"

        anchor_us = session.scene_api.newest_timestamp_us
        assert anchor_us is not None, "drive verifies the anchor cameras before planning"

        def _ego_state(timestamp_us: int) -> dict[str, BaseModality]:
            """
            Builds the current ego state as a modality bundle stamped at the given time.

            Args:
                timestamp_us: the time to stamp the state with.

            Returns:
                the ego state keyed by its modality key.
            """
            return {
                get_modality_key(ModalityType.EGO_STATE_SE3): EgoStateSE3.from_rear_axle(
                    rear_axle_se3=_ego_se3(ego_pose_local),
                    metadata=self._ego_metadata,
                    timestamp=Timestamp.from_us(timestamp_us),
                    dynamic_state_se3=DynamicStateSE3(
                        velocity=Vector3D(x=speed_mps, y=0.0, z=0.0),
                        acceleration=Vector3D(x=acceleration_mps2, y=0.0, z=0.0),
                        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                    ),
                ),
            }

        # A session opens with no past ego states; the rollout starts from rest, so the first pose fills the history.
        if not session.history_backfilled:
            interval_us = session.scene_api.iteration_interval_us
            for step in range(1, session.scene_api.num_history_iterations + 1):
                session.scene_api.append(anchor_us - step * interval_us, _ego_state(anchor_us - step * interval_us))
            session.history_backfilled = True
        session.scene_api.append(anchor_us, _ego_state(anchor_us))

        target_point_distances_m = policy.policy_config.required_target_point_distances_m
        served_xy = rig_to_local_xy(route, ego_pose_local) if route is not None else np.zeros((0, 2), dtype=np.float64)
        route_xy = _route_ahead(
            _merge_route(session.route_xy_local if self._py123d_garage_config.route_memory else None, served_xy),
            ego_pose_local[:2],
        )
        session.route_xy_local = route_xy
        # The target points walk from the ego onto the route; that first straight stretch stays out of the plots.
        walk_xy = np.vstack([ego_pose_local[None, :2], route_xy]) if len(route_xy) > 0 else route_xy
        resolution_m = self._py123d_garage_config.route_resolution_m
        if self._py123d_garage_config.extrapolate_route and target_point_distances_m:
            walk_xy = _extend_route(walk_xy, max(target_point_distances_m) + resolution_m)
        # No route clears the last one, so no target point comes from a stale route.
        session.scene_api.set_route(_resample_route(route_xy, resolution_m, float(ego_pose_local[2])))
        scene_api = session.scene_api.snapshot()
        navigation_scene_api = scene_api.snapshot()
        navigation_scene_api.set_route(_resample_route(walk_xy, resolution_m, float(ego_pose_local[2])))

        with self._inference_lock:
            features = policy.build_features(scene_api, {}).apply(lambda tensor: tensor.unsqueeze(0))
            # The points past the route's end read as no intent instead of being made up.
            navigation = NavigationConditioning.from_scene(
                navigation_scene_api,
                target_point_distances_m,
                nan_past_route_end=True,
            ).apply(lambda tensor: tensor.unsqueeze(0))
            device = next(policy.parameters()).device
            with torch.no_grad():
                predictions = policy.forward(features.to(device), navigation.to(device))
            if self._py123d_garage_video_writer is not None and self._py123d_garage_video_writer.is_record_tick(
                session.num_plans,
            ):
                views = policy.visualize_batch(features, None, navigation, predictions, scene_api)
                self._py123d_garage_video_writer.write(
                    {f"policy_{session_uuid}_{name}": image for name, image in views.items()},
                )
            session.num_plans += 1
        driven_poses = predictions.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        if self._plan_log_dir is not None:
            record = {
                "anchor_time_us": np.array(anchor_us, dtype=np.float64),
                "ego_pose_local": ego_pose_local,
                "speed_mps": np.array(speed_mps, dtype=np.float64),
                "acceleration_mps2": np.array(acceleration_mps2, dtype=np.float64),
                "target_points": navigation.target_points[0].cpu().numpy().astype(np.float64),
                "driven": driven_poses,
            }
            for name, output in predictions.plan_outputs.items():
                record[name] = output[0].float().cpu().numpy().astype(np.float64)
            session.plan_records.append(record)
        return driven_poses

    def _get_session(
        self,
        session_uuid: str,
        context: grpc.ServicerContext,
    ) -> _Session:
        """
        Looks up a session, aborting the call when it is unknown.

        Args:
            session_uuid: the rollout's id.
            context: the gRPC call to abort.

        Returns:
            the session.
        """
        with self._lock:
            session = self._sessions.get(session_uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_uuid}")
            raise AssertionError("unreachable")
        return session


def _served_cameras(scene_api: OnlineSceneAPI, camera_ids: set[CameraID]) -> set[CameraID]:
    """
    The cameras the scene's newest bundle carries.

    Args:
        scene_api: the session's live scene.
        camera_ids: the cameras to look for.

    Returns:
        the subset of camera_ids the newest bundle carries.
    """
    if scene_api.newest_timestamp_us is None:
        return set()
    snapshot = scene_api.snapshot()
    return {camera_id for camera_id in camera_ids if snapshot.get_camera_at_iteration(0, camera_id) is not None}


def _ego_se3(ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"]) -> PoseSE3:
    """
    Turns a flat ego pose into a yaw-only SE3 pose.

    Args:
        ego_pose_local: the (x, y, z, yaw) of the ego in the local frame.

    Returns:
        the pose.
    """
    x, y, z, yaw = ego_pose_local
    return PoseSE3(
        x=float(x),
        y=float(y),
        z=float(z),
        qw=math.cos(yaw / 2.0),
        qx=0.0,
        qy=0.0,
        qz=math.sin(yaw / 2.0),
    )


def _resample_route(
    polyline_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
    resolution_m: float,
    z: float,
) -> RouteMetadata | None:
    """
    Resamples a local-frame polyline into a route at a uniform resolution.

    Args:
        polyline_xy: the route polyline in the local frame.
        resolution_m: the spacing of the resampled vertices.
        z: the height given to every vertex.

    Returns:
        the route, or None when the polyline has fewer than two distinct points.
    """
    if len(polyline_xy) < 2:
        return None

    steps_m = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    vertices = polyline_xy[np.concatenate([[True], steps_m > 0.0])]
    if len(vertices) < 2:
        return None

    arc_m = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))])
    sample_arc_m = np.arange(int(arc_m[-1] / resolution_m) + 1, dtype=np.float64) * resolution_m
    polyline = np.stack(
        [np.interp(sample_arc_m, arc_m, vertices[:, axis]) for axis in range(2)],
        axis=1,
    )
    return RouteMetadata(
        resolution_m=resolution_m,
        total_arc_m=float(sample_arc_m[-1]),
        polyline_x=polyline[:, 0].tolist(),
        polyline_y=polyline[:, 1].tolist(),
        polyline_z=[z] * len(polyline),
        cache_source_info=provided_route_source_info("alpasim submit_route"),
        source="provided",
    )


def _project_onto_route(
    polyline_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
    point_xy: jt.Float64[npt.NDArray[np.float64], " 2"],
) -> tuple[int, float]:
    """
    Finds the polyline point nearest to a point.

    Args:
        polyline_xy: the polyline, with at least two points.
        point_xy: the point to project.

    Returns:
        the index of the nearest segment and the nearest point's fraction along it.
    """
    starts = polyline_xy[:-1]
    segments = np.diff(polyline_xy, axis=0)
    fractions = np.clip(
        ((point_xy - starts) * segments).sum(axis=1) / np.maximum((segments**2).sum(axis=1), 1e-12),
        0.0,
        1.0,
    )
    index = int(np.argmin(np.linalg.norm(starts + fractions[:, None] * segments - point_xy, axis=1)))
    return index, float(fractions[index])


def _merge_route(
    stored_xy: jt.Float64[npt.NDArray[np.float64], "num_stored 2"] | None,
    served_xy: jt.Float64[npt.NDArray[np.float64], "num_served 2"],
) -> jt.Float64[npt.NDArray[np.float64], "num_points 2"]:
    """
    Joins the stored route up to where the served one starts with the served one.

    Args:
        stored_xy: the route kept from the last plan, or None to use the served one alone.
        served_xy: the route the runtime serves now; when empty, the stored one is kept.

    Returns:
        the merged route in the local frame.
    """
    if stored_xy is None or len(stored_xy) < 2:
        return served_xy
    if len(served_xy) == 0:
        return stored_xy
    index, fraction = _project_onto_route(stored_xy, served_xy[0])
    if index == 0 and fraction == 0.0:
        return served_xy
    return np.vstack([stored_xy[: index + 1], served_xy])


def _route_ahead(
    route_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
    ego_xy: jt.Float64[npt.NDArray[np.float64], " 2"],
) -> jt.Float64[npt.NDArray[np.float64], "num_ahead 2"]:
    """
    Cuts the route behind its point nearest the ego; a route that starts ahead of the ego stays whole.

    Args:
        route_xy: the route in the local frame.
        ego_xy: the ego position in the local frame.

    Returns:
        the route from the nearest point on.
    """
    if len(route_xy) < 2:
        return route_xy
    index, fraction = _project_onto_route(route_xy, ego_xy)
    nearest_xy = route_xy[index] + fraction * (route_xy[index + 1] - route_xy[index])
    return np.vstack([nearest_xy[None], route_xy[index + 1 :]])


def _extend_route(
    polyline_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
    min_length_m: float,
    heading_lookback_m: float = 1.0,
) -> jt.Float64[npt.NDArray[np.float64], "num_extended 2"]:
    """
    Extends the route along its final heading so target points keep their trained distance near its end.

    Args:
        polyline_xy: the route in the local frame.
        min_length_m: the length the route is extended to.
        heading_lookback_m: how far back from the end the final heading is measured.

    Returns:
        the route, with one extra vertex when it was too short.
    """
    if len(polyline_xy) < 2:
        return polyline_xy
    lengths = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    shortfall_m = min_length_m - float(lengths.sum())
    if shortfall_m <= 0.0:
        return polyline_xy
    arc_from_end_m = np.concatenate([[0.0], np.cumsum(lengths[::-1])])[::-1]
    tail_start = int(np.searchsorted(-arc_from_end_m, -heading_lookback_m)) - 1
    tail_start = min(max(tail_start, 0), len(polyline_xy) - 2)
    heading = polyline_xy[-1] - polyline_xy[tail_start]
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm == 0.0:
        return polyline_xy
    return np.vstack([polyline_xy, polyline_xy[-1] + heading / heading_norm * shortfall_m])


def _decode_image(image_bytes: bytes) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Decodes an encoded camera frame into an RGB image.

    Args:
        image_bytes: the encoded frame.

    Returns:
        the RGB image.
    """
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"the runtime pushed {len(image_bytes)} bytes that decode to no image.")
    return np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)


def _yaw_from_quaternion(quaternion: Any) -> float:
    """
    Extracts the yaw of a quaternion message.

    Args:
        quaternion: a message with w, x, y and z fields.

    Returns:
        the yaw in radians.
    """
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _ego_state_se3(
    pose_at_time: common_pb2.PoseAtTime,
    dynamic_state: common_pb2.DynamicState | None,
    metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    """
    Turns one observed ego pose into an EgoStateSE3, stamped at its own observation time.

    Args:
        pose_at_time: the observed pose.
        dynamic_state: the observed velocity and acceleration, or None for a standstill.
        metadata: the ego vehicle metadata.

    Returns:
        the ego state.
    """
    pose = pose_at_time.pose
    return EgoStateSE3.from_rear_axle(
        rear_axle_se3=_ego_se3(
            np.array(
                [pose.vec.x, pose.vec.y, pose.vec.z, _yaw_from_quaternion(pose.quat)],
                dtype=np.float64,
            ),
        ),
        metadata=metadata,
        timestamp=Timestamp.from_us(int(pose_at_time.timestamp_us)),
        dynamic_state_se3=DynamicStateSE3(
            velocity=Vector3D(
                x=math.hypot(dynamic_state.linear_velocity.x, dynamic_state.linear_velocity.y)
                if dynamic_state is not None
                else 0.0,
                y=0.0,
                z=0.0,
            ),
            acceleration=Vector3D(
                x=float(dynamic_state.linear_acceleration.x) if dynamic_state is not None else 0.0,
                y=0.0,
                z=0.0,
            ),
            angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        ),
    )


def _pose_at_time(
    x: float,
    y: float,
    z: float,
    yaw: float,
    timestamp_us: int,
) -> common_pb2.PoseAtTime:
    """
    Builds a yaw-only pose message.

    Args:
        x: x position in the local frame.
        y: y position in the local frame.
        z: z position in the local frame.
        yaw: yaw in radians.
        timestamp_us: the pose's time.

    Returns:
        the pose message.
    """
    return common_pb2.PoseAtTime(
        timestamp_us=timestamp_us,
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=y, z=z),
            quat=common_pb2.Quat(w=math.cos(yaw / 2.0), x=0.0, y=0.0, z=math.sin(yaw / 2.0)),
        ),
    )


def _trajectory_from_plan(
    plan: CachedPlan,
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
    time_now_us: int,
    fallback_speed_mps: float,
) -> common_pb2.Trajectory:
    """
    Samples the plan from now on, on the 10 Hz grid of its own clock, so no served point moves between inferences.

    Args:
        plan: the cached plan.
        ego_pose_local: the (x, y, z, yaw) of the ego in the local frame.
        time_now_us: the runtime's current time.
        fallback_speed_mps: the speed of the straight line served when the plan has run out.

    Returns:
        the trajectory to serve.
    """
    samples = sample_plan(plan, time_now_us)
    if samples is None:
        return _straight_line_trajectory(ego_pose_local, time_now_us, fallback_speed_mps)
    z = float(ego_pose_local[2])
    trajectory = common_pb2.Trajectory()
    for timestamp_us, (x, y), yaw in zip(
        samples.timestamps_us.tolist(),
        samples.positions_xy.tolist(),
        samples.yaws.tolist(),
        strict=True,
    ):
        trajectory.poses.append(_pose_at_time(x, y, z, yaw, timestamp_us))
    return trajectory


def _straight_line_trajectory(
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"] | None,
    time_now_us: int,
    speed_mps: float,
) -> common_pb2.Trajectory:
    """
    Builds a constant-speed straight line along the ego's heading.

    Args:
        ego_pose_local: the (x, y, z, yaw) of the ego in the local frame, or None for the origin.
        time_now_us: the time of the first pose.
        speed_mps: the speed along the line.

    Returns:
        the trajectory to serve.
    """
    origin_x, origin_y, origin_z, origin_yaw = ego_pose_local if ego_pose_local is not None else (0.0, 0.0, 0.0, 0.0)
    num_poses = SERVE_HORIZON_US // SERVE_INTERVAL_US + 1

    trajectory = common_pb2.Trajectory()
    for index in range(num_poses):
        distance_m = speed_mps * index * SERVE_INTERVAL_US / 1e6
        trajectory.poses.append(
            _pose_at_time(
                origin_x + math.cos(origin_yaw) * distance_m,
                origin_y + math.sin(origin_yaw) * distance_m,
                origin_z,
                origin_yaw,
                time_now_us + index * SERVE_INTERVAL_US,
            ),
        )
    return trajectory
