from __future__ import annotations

import dataclasses
import logging
import math
import threading
from typing import Any

import cv2
import grpc
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
    LogMetadata,
    ModalityType,
    Timestamp,
)
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import get_modality_key
from py123d.geometry import PoseSE2Index, PoseSE3, Vector3D
from py123d.geometry.transform import rel_to_abs_se3
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.common.config_help import run_dir
from py123d_garage.common.video_writer import VideoWriter
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt
from py123d_garage.datatypes.trajectory import TrajectoryXY
from py123d_garage.evaluation.alpasim.help.config_help import build_policy
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
_SERVE_INTERVAL_US = 100_000
_SERVE_HORIZON_US = 5_000_000
_VISUALIZATION_FPS = 2.0


@dataclasses.dataclass(frozen=True)
class _CachedPlan:
    """One prediction fixed in the rollout's local frame; drive re-slices it every tick."""

    created_time_us: int
    times_s: npt.NDArray[np.float64]
    positions_xy: npt.NDArray[np.float64]
    yaws: npt.NDArray[np.float64]


class _Session:
    def __init__(
        self,
        scene_api: OnlineSceneAPI,
        announced_cameras: dict[CameraID, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera],
    ) -> None:
        self.scene_api = scene_api
        self.announced_cameras = announced_cameras
        self.camera_metadatas: dict[CameraID, BaseCameraMetadata] = {}
        self.ego_pose_local: npt.NDArray[np.float64] | None = None
        self.speed_mps: float = 0.0
        self.acceleration_mps2: float = 0.0
        self.route_polyline_rig_xy: npt.NDArray[np.float64] | None = None
        self.cached_plan: _CachedPlan | None = None
        self.num_plans: int = 0


class GarageDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    """Serves plans for every concurrent rollout from one policy on one device."""

    def __init__(self, config: AlpasimBenchmarkConfig) -> None:
        self._py123d_garage_config = config
        sensor_rig_file = self._py123d_garage_config.policy_config.evaluation_sensor_rig_file
        if sensor_rig_file is None:
            raise ValueError(
                "driving in AlpaSim needs policy_config.evaluation_sensor_rig_file, the rig the checkpoint was trained on",
            )
        camera_map = rig_camera_map(sensor_rig_file)
        self._camera_setup_dataset = rig_dataset(camera_map.values())
        front = set(FRONT_CAMERAS[self._camera_setup_dataset])
        self._camera_id_by_alpasim_id = {
            name: camera_id for name, camera_id in camera_map.items() if camera_id in front
        }
        self._rig_metadatas = rig_camera_metadatas(
            sensor_rig_file,
            set(self._camera_id_by_alpasim_id.values()),
        )
        self._ego_metadata = build_ego_metadata(self._py123d_garage_config.ego_vehicle_config)
        self._py123d_garage_video_writer = (
            VideoWriter(
                run_dir() / "views",
                self._py123d_garage_config.visualization_interval,
                fps=_VISUALIZATION_FPS / self._py123d_garage_config.visualization_interval,
            )
            if self._py123d_garage_config.visualization_interval
            else None
        )
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
            with self._lock:
                self._py123d_garage_policy = policy
        except BaseException as error:
            with self._lock:
                self._policy_error = error
            LOG.exception("policy load failed")

    def _ready_policy(self, context: grpc.ServicerContext) -> AnyPolicy:
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
        policy = self._ready_policy(context)
        interval_us = policy.policy_config.trajectory_interval_us
        announced_cameras: dict[CameraID, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera] = {}
        for announced in request.rollout_spec.vehicle.available_cameras:
            announced_camera_id = self._camera_id_by_alpasim_id.get(announced.logical_id)
            if announced_camera_id is not None:
                announced_cameras[announced_camera_id] = announced

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
            iteration_interval_us=interval_us,
            num_history_iterations=round(policy.policy_config.required_history_duration_us / interval_us),
        )
        with self._lock:
            self._sessions[request.session_uuid] = _Session(scene_api, announced_cameras)
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
        del context
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        LOG.info(f"closed session {request.session_uuid}")
        return common_pb2.Empty()

    @override
    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        camera_image = request.camera_image
        if camera_image.logical_id not in self._camera_id_by_alpasim_id:
            return common_pb2.Empty()
        image = _decode_image(camera_image.image_bytes)
        timestamp_us = int(camera_image.frame_end_us)
        session = self._get_session(request.session_uuid, context)
        camera_id = self._camera_id_by_alpasim_id[camera_image.logical_id]
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
        image: npt.NDArray[np.uint8],
    ) -> BaseCameraMetadata:
        """
        The session's calibration for one camera, scaled to the decoded frame; announced
        intrinsics win over the static rig file, which stays a fallback for local setups.
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
        if not request.trajectory.poses:
            return common_pb2.Empty()
        session = self._get_session(request.session_uuid, context)
        latest = request.trajectory.poses[-1]
        with self._lock:
            session.ego_pose_local = np.array(
                [latest.pose.vec.x, latest.pose.vec.y, latest.pose.vec.z, _yaw_from_quaternion(latest.pose.quat)],
                dtype=np.float64,
            )
            if len(request.dynamic_states) == len(request.trajectory.poses):
                dynamic_state = request.dynamic_states[-1]
                session.speed_mps = math.hypot(dynamic_state.linear_velocity.x, dynamic_state.linear_velocity.y)
                session.acceleration_mps2 = float(dynamic_state.linear_acceleration.x)
        return common_pb2.Empty()

    @override
    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
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
        del request, context
        return common_pb2.Empty()

    @override
    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        policy = self._ready_policy(context)
        session = self._get_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        interval_us = policy.policy_config.trajectory_interval_us
        with self._lock:
            ego_pose_local = session.ego_pose_local
            plan = session.cached_plan
            fallback_speed_mps = max(_MIN_FALLBACK_SPEED_MPS, session.speed_mps)
        if plan is None or time_now_us - plan.created_time_us >= interval_us:
            fresh = self._replan(policy, session, request.session_uuid, ego_pose_local, time_now_us, interval_us)
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
        ego_pose_local: npt.NDArray[np.float64] | None,
        time_now_us: NonNegativeInt,
        interval_us: PositiveInt,
    ) -> _CachedPlan | None:
        """A fresh plan anchored at the current pose, or None to keep serving the previous one."""
        served = _served_cameras(session.scene_api, set(self._camera_id_by_alpasim_id.values()))
        if served != set(self._camera_id_by_alpasim_id.values()) or ego_pose_local is None:
            LOG.warning(
                f"session {session_uuid} has {sorted(camera_id.name for camera_id in served)} of "
                f"{sorted(camera_id.name for camera_id in self._camera_id_by_alpasim_id.values())} cameras and "
                f"{'an' if ego_pose_local is not None else 'no'} ego pose; not planning.",
            )
            return None
        try:
            positions_rig = self._plan(policy, session, session_uuid)
        except Exception:
            LOG.exception(f"session {session_uuid} failed to plan.")
            return None
        return _make_plan(positions_rig, ego_pose_local, time_now_us, interval_us)

    @override
    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
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
        return self._py123d_garage_config

    def close(self) -> None:
        if self._py123d_garage_video_writer is not None:
            self._py123d_garage_video_writer.close()

    def _plan(self, policy: AnyPolicy, session: _Session, session_uuid: str) -> npt.NDArray[np.float64]:
        with self._lock:
            ego_pose_local = session.ego_pose_local
            speed_mps = session.speed_mps
            acceleration_mps2 = session.acceleration_mps2
            route = session.route_polyline_rig_xy
        assert ego_pose_local is not None, "drive verifies the ego pose before planning"

        anchor_us = session.scene_api.newest_timestamp_us
        assert anchor_us is not None, "drive verifies the anchor cameras before planning"
        session.scene_api.append(
            anchor_us,
            {
                get_modality_key(ModalityType.EGO_STATE_SE3): EgoStateSE3.from_rear_axle(
                    rear_axle_se3=_ego_se3(ego_pose_local),
                    metadata=self._ego_metadata,
                    timestamp=Timestamp.from_us(anchor_us),
                    dynamic_state_se3=DynamicStateSE3(
                        velocity=Vector3D(x=speed_mps, y=0.0, z=0.0),
                        acceleration=Vector3D(x=acceleration_mps2, y=0.0, z=0.0),
                        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                    ),
                ),
            },
        )

        target_point_distances_m = policy.policy_config.required_target_point_distances_m
        if route is not None and len(route) > 0:
            if float(np.linalg.norm(route[0])) > self._py123d_garage_config.route_resolution_m:
                route = np.vstack([np.zeros((1, 2)), route])
            if target_point_distances_m:
                route = _extend_route(route, max(target_point_distances_m) + 2.0)
        route_metadata = _resample_route(
            _rig_to_local_xy(route, ego_pose_local) if route is not None else np.zeros((0, 2), dtype=np.float64),
            self._py123d_garage_config.route_resolution_m,
            float(ego_pose_local[2]),
        )
        if route_metadata is not None:
            session.scene_api.set_route(route_metadata)
        scene_api = session.scene_api.snapshot()

        with self._inference_lock:
            features = policy.build_features(scene_api, {}).apply(lambda tensor: tensor.unsqueeze(0))
            navigation = NavigationConditioning.from_scene(
                scene_api,
                target_point_distances_m,
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
        return predictions.ego_trajectory_xy[0].float().cpu().numpy().astype(np.float64)

    def _get_session(
        self,
        session_uuid: str,
        context: grpc.ServicerContext,
    ) -> _Session:
        with self._lock:
            session = self._sessions.get(session_uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_uuid}")
            raise AssertionError("unreachable")
        return session


def _served_cameras(scene_api: OnlineSceneAPI, camera_ids: set[CameraID]) -> set[CameraID]:
    """The cameras the scene's newest bundle carries."""
    if scene_api.newest_timestamp_us is None:
        return set()
    snapshot = scene_api.snapshot()
    return {camera_id for camera_id in camera_ids if snapshot.get_camera_at_iteration(0, camera_id) is not None}


def _rig_to_local_xy(
    polyline_rig_xy: npt.NDArray[np.float64],
    ego_pose_local: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    x, y, _, yaw = ego_pose_local
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return polyline_rig_xy @ rotation.T + [x, y]


def _ego_se3(ego_pose_local: npt.NDArray[np.float64]) -> PoseSE3:
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
    polyline_xy: npt.NDArray[np.float64],
    resolution_m: float,
    z: float,
) -> RouteMetadata | None:
    """The local-frame polyline resampled at the resolution, or None when degenerate."""
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


def _extend_route(
    polyline_xy: npt.NDArray[np.float64],
    min_length_m: float,
    heading_lookback_m: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Extends the route along its final heading so target points keep their trained distance near the clip end."""
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


def _decode_image(image_bytes: bytes) -> npt.NDArray[np.uint8]:
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"the runtime pushed {len(image_bytes)} bytes that decode to no image.")
    return np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)


def _yaw_from_quaternion(quaternion: Any) -> float:
    siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_at_time(
    x: float,
    y: float,
    z: float,
    yaw: float,
    timestamp_us: int,
) -> common_pb2.PoseAtTime:
    return common_pb2.PoseAtTime(
        timestamp_us=timestamp_us,
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=y, z=z),
            quat=common_pb2.Quat(w=math.cos(yaw / 2.0), x=0.0, y=0.0, z=math.sin(yaw / 2.0)),
        ),
    )


def _make_plan(
    positions_rig: npt.NDArray[np.float64],
    ego_pose_local: npt.NDArray[np.float64],
    time_now_us: NonNegativeInt,
    interval_us: PositiveInt,
) -> _CachedPlan:
    """Fixes the rig-frame prediction in the local frame, anchored at the current pose."""
    origin_x, origin_y, _, origin_yaw = ego_pose_local
    positions_xy = np.vstack(
        [
            [[origin_x, origin_y]],
            _rig_to_local_xy(positions_rig, ego_pose_local),
        ],
    )
    poses_rig = TrajectoryXY(
        position_xy_array=positions_rig,
        timestamps_us=time_now_us + np.arange(1, len(positions_rig) + 1, dtype=np.int64) * interval_us,
    ).to_se2()
    yaws = np.concatenate(
        [[origin_yaw], origin_yaw + poses_rig.pose_se2_array[:, PoseSE2Index.YAW]],
    )
    times_s = np.arange(len(positions_xy), dtype=np.float64) * (interval_us / 1e6)
    return _CachedPlan(
        created_time_us=time_now_us,
        times_s=times_s,
        positions_xy=positions_xy,
        yaws=yaws,
    )


def _trajectory_from_plan(
    plan: _CachedPlan,
    ego_pose_local: npt.NDArray[np.float64],
    time_now_us: int,
    fallback_speed_mps: float,
) -> common_pb2.Trajectory:
    """
    Samples the fixed plan on the 10 Hz grid of its own clock, emitting only points at or
    after now: the trajectory shrinks between inferences, but no point moves.
    """
    serve_interval_s = _SERVE_INTERVAL_US / 1e6
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1e6)
    horizon_end_s = min(float(plan.times_s[-1]), elapsed_s + _SERVE_HORIZON_US / 1e6)
    first_step = math.ceil(elapsed_s / serve_interval_s - 1e-9)
    last_step = math.floor(horizon_end_s / serve_interval_s + 1e-9)
    if last_step - first_step < 1:
        return _straight_line_trajectory(ego_pose_local, time_now_us, fallback_speed_mps)

    sample_times_s = np.arange(first_step, last_step + 1, dtype=np.float64) * serve_interval_s
    xs = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 0])
    ys = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 1])
    yaws = np.interp(sample_times_s, plan.times_s, np.unwrap(plan.yaws))
    z = float(ego_pose_local[2])

    trajectory = common_pb2.Trajectory()
    for time_s, x, y, yaw in zip(sample_times_s, xs, ys, yaws, strict=True):
        trajectory.poses.append(
            _pose_at_time(
                float(x),
                float(y),
                z,
                float(yaw),
                plan.created_time_us + round(float(time_s) * 1e6),
            ),
        )
    return trajectory


def _straight_line_trajectory(
    ego_pose_local: npt.NDArray[np.float64] | None,
    time_now_us: int,
    speed_mps: float,
) -> common_pb2.Trajectory:
    origin_x, origin_y, origin_z, origin_yaw = ego_pose_local if ego_pose_local is not None else (0.0, 0.0, 0.0, 0.0)
    num_poses = _SERVE_HORIZON_US // _SERVE_INTERVAL_US + 1

    trajectory = common_pb2.Trajectory()
    for index in range(num_poses):
        distance_m = speed_mps * index * _SERVE_INTERVAL_US / 1e6
        trajectory.poses.append(
            _pose_at_time(
                origin_x + math.cos(origin_yaw) * distance_m,
                origin_y + math.sin(origin_yaw) * distance_m,
                origin_z,
                origin_yaw,
                time_now_us + index * _SERVE_INTERVAL_US,
            ),
        )
    return trajectory
