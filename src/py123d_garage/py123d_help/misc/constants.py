# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The cameras and lidars filling a policy's input slots, per dataset."""

from __future__ import annotations

from py123d.datatypes import CameraID, LidarID

# Left, front and right camera per dataset, in that order.
FRONT_CAMERAS: dict[str, list[CameraID]] = {
    "nuplan": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0],
    "carla": [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0],
    "physical-ai-av": [CameraID.FTCAM_L0, CameraID.FTCAM_F0, CameraID.FTCAM_R0],
    "nurec": [CameraID.FTCAM_L0, CameraID.FTCAM_F0, CameraID.FTCAM_R0],
}

FRONT_CENTER_CAMERA: dict[str, list[CameraID]] = {
    "nuplan": [CameraID.PCAM_F0],
    "physical-ai-av": [CameraID.FTCAM_F0],
    "nurec": [CameraID.FTCAM_F0],
}

# HACK: The datasets name their lidars differently after conversion; this should be fixed in the conversion.
INPUT_LIDARS: dict[str, list[LidarID]] = {
    "nuplan": [LidarID.LIDAR_MERGED],
    "carla": [LidarID.LIDAR_TOP],
    "physical-ai-av": [LidarID.LIDAR_MERGED],
    "nurec": [LidarID.LIDAR_MERGED],
}
