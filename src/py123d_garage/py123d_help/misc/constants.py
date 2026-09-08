"""The cameras and lidars filling a policy's input slots, per dataset."""

from __future__ import annotations

from py123d.datatypes import CameraID, LidarID

FRONT_CAMERAS: dict[str, list[CameraID]] = {
    "nuplan": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0],
    "carla": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0],
    "physical-ai-av": [CameraID.FTCAM_L0, CameraID.FTCAM_F0, CameraID.FTCAM_R0],
    "kesai": [CameraID.FTCAM_L0, CameraID.FTCAM_F0, CameraID.FTCAM_R0],
}

# HACK: Each dataset has a slightly different convention at conversion time.
# This should be fixed in the dataset conversion, but for now we just hardcode the differences here.
INPUT_LIDARS: dict[str, list[LidarID]] = {
    "nuplan": [LidarID.LIDAR_MERGED],
    "carla": [LidarID.LIDAR_TOP],
    "physical-ai-av": [LidarID.LIDAR_MERGED],
    "kesai": [LidarID.LIDAR_TOP],
}
