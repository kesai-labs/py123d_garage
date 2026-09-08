from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from alpasim_grpc.v0 import sensorsim_pb2
from py123d.datatypes import (
    BaseCameraMetadata,
    CameraID,
    FThetaCameraMetadata,
    PinholeCameraMetadata,
    camera_metadata_from_dict,
)
from py123d.datatypes.sensors.ftheta_camera import FThetaIntrinsics
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import PoseSE3

from py123d_garage.config.schema.evaluation.alpasim_config import EgoVehicleConfig
from py123d_garage.datatypes.numerics import PositiveInt
from py123d_garage.py123d_help.misc import FRONT_CAMERAS

LOG = logging.getLogger(__name__)


def rig_camera_map(sensor_rig_file: str) -> dict[str, CameraID]:
    """Maps the rig's camera names, which the simulator serves frames under, to 123D camera ids."""
    modalities = yaml.safe_load(Path(sensor_rig_file).read_text())["modalities"]
    return {
        modality["camera_name"]: CameraID[key.removeprefix("camera.").upper()]
        for key, modality in modalities.items()
        if key.startswith("camera.")
    }


def rig_dataset(camera_ids: Collection[CameraID]) -> str:
    """The dataset whose input-camera slots the rig's cameras fill, or overlap most."""
    overlaps = {dataset: len(set(cameras) & set(camera_ids)) for dataset, cameras in FRONT_CAMERAS.items()}
    for dataset, cameras in FRONT_CAMERAS.items():
        if set(cameras) <= set(camera_ids):
            return dataset
    best_dataset, best_overlap = max(overlaps.items(), key=lambda item: item[1])
    if best_overlap == 0:
        raise ValueError(
            f"the rig cameras {sorted(camera_id.name for camera_id in camera_ids)} fill no "
            f"dataset's front-camera slots; see FRONT_CAMERAS.",
        )
    return best_dataset


def rig_camera_metadatas(
    sensor_rig_file: str,
    camera_ids: Collection[CameraID],
) -> dict[CameraID, BaseCameraMetadata]:
    """The rig's full camera calibrations, for the entries that carry one; bridge rigs carry none."""
    modalities = yaml.safe_load(Path(sensor_rig_file).read_text())["modalities"]
    metadatas: dict[CameraID, BaseCameraMetadata] = {}
    for key, modality in modalities.items():
        if not key.startswith("camera.") or "camera_model" not in modality:
            continue
        metadata = camera_metadata_from_dict(modality)
        if metadata.camera_id in camera_ids:
            metadatas[metadata.camera_id] = metadata
    return metadatas


def build_ego_metadata(config: EgoVehicleConfig) -> EgoStateSE3Metadata:
    return EgoStateSE3Metadata(
        vehicle_name=config.vehicle_name,
        width=config.width_m,
        length=config.length_m,
        height=config.height_m,
        wheel_base=config.wheel_base_m,
        center_to_imu_se3=PoseSE3(
            x=config.rear_axle_to_center_longitudinal_m,
            y=0.0,
            z=0.0,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _ftheta_poly(
    values: Sequence[float],
    name: str,
    logical_id: str,
) -> npt.NDArray[np.float64]:
    poly = np.zeros(6, dtype=np.float64)
    coefficients = np.asarray(list(values), dtype=np.float64)
    if coefficients.size > 6:
        if np.any(coefficients[6:] != 0.0):
            raise ValueError(
                f"camera {logical_id} announces {coefficients.size} {name} coefficients; "
                f"123D f-theta intrinsics carry 6.",
            )
        coefficients = coefficients[:6]
    poly[: coefficients.size] = coefficients
    return poly


def camera_metadata_from_announcement(
    announced: sensorsim_pb2.AvailableCamerasReturn.AvailableCamera,
    camera_id: CameraID,
    decoded_height: PositiveInt,
    decoded_width: PositiveInt,
) -> FThetaCameraMetadata | None:
    """
    The calibration a session announces for one camera, scaled from its native
    resolution to the decoded frame; None when the announcement is not f-theta.
    """
    if announced.intrinsics.WhichOneof("camera_param") != "ftheta_param":
        return None
    native_width = int(announced.intrinsics.resolution_w)
    native_height = int(announced.intrinsics.resolution_h)
    if native_width == 0 or native_height == 0:
        raise ValueError(
            f"camera {announced.logical_id} announces f-theta intrinsics without a native resolution, "
            f"so they cannot be scaled to the {decoded_width}x{decoded_height} decoded frame.",
        )
    scale_x = decoded_width / native_width
    scale_y = decoded_height / native_height
    if not np.isclose(scale_x, scale_y, atol=1e-6):
        LOG.warning(
            f"camera {announced.logical_id} decodes at {decoded_width}x{decoded_height} from a "
            f"{native_width}x{native_height} native calibration; the radial scale {scale_y:.6f} "
            f"leaves an anisotropic x-residual.",
        )

    param = announced.intrinsics.ftheta_param
    fw_poly = _ftheta_poly(param.angle_to_pixeldist_poly, "angle-to-pixel", announced.logical_id) * scale_y
    bw_poly = _ftheta_poly(param.pixeldist_to_angle_poly, "pixel-to-angle", announced.logical_id)
    bw_poly = bw_poly / np.power(scale_y, np.arange(6, dtype=np.float64))
    # The proto field is named rig_to_camera but carries the camera pose in the rig frame.
    pose = announced.rig_to_camera
    camera_to_imu_se3 = (
        PoseSE3(
            x=pose.vec.x,
            y=pose.vec.y,
            z=pose.vec.z,
            qw=pose.quat.w,
            qx=pose.quat.x,
            qy=pose.quat.y,
            qz=pose.quat.z,
        )
        if announced.HasField("rig_to_camera")
        else PoseSE3.identity()
    )
    return FThetaCameraMetadata(
        camera_name=announced.logical_id,
        camera_id=camera_id,
        intrinsics=FThetaIntrinsics(
            cx=param.principal_point_x * scale_x,
            cy=param.principal_point_y * scale_y,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
        ),
        width=decoded_width,
        height=decoded_height,
        camera_to_imu_se3=camera_to_imu_se3,
    )


def build_camera_metadatas(
    images: dict[CameraID, npt.NDArray[np.uint8]],
    camera_id_by_alpasim_id: dict[str, CameraID],
) -> dict[CameraID, PinholeCameraMetadata]:
    """Describes the rendered frames with placeholder calibration; a policy reading intrinsics or mount poses would silently get identity."""
    name_by_camera_id = {camera_id: name for name, camera_id in camera_id_by_alpasim_id.items()}
    return {
        camera_id: PinholeCameraMetadata(
            camera_name=name_by_camera_id[camera_id],
            camera_id=camera_id,
            intrinsics=None,
            distortion=None,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            camera_to_imu_se3=PoseSE3.identity(),
            is_undistorted=True,
        )
        for camera_id, image in images.items()
    }
