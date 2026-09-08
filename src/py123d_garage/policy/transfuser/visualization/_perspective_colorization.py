"""Colorization of the perspective semantic-segmentation and depth modalities."""

from __future__ import annotations

from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes.sensors.camera_segmentation_label import (
    DEFAULT_CAMERA_SEGMENTATION_RGB,
)


def semantic_to_rgb(
    class_map: jt.Int64[npt.NDArray[np.int64], "height width"],
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Colors the default-taxonomy semantic classes with the Cityscapes palette.

    Args:
        class_map: DefaultCameraSegmentationLabel value per pixel

    Returns:
        RGB image
    """
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label, rgb in DEFAULT_CAMERA_SEGMENTATION_RGB.items():
        palette[label.value] = rgb
    return palette[np.clip(class_map, 0, 255).astype(np.int64)]


def depth_to_rgb(
    depth_m: jt.Float32[npt.NDArray[np.float32], "height width"],
    max_depth_m: float,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Colorizes metric depth, near warm and far cold, against a fixed far plane.

    Args:
        depth_m: metric depth in meters
        max_depth_m: depth mapped to the far end of the colormap; fixed so a
            color means the same distance across frames

    Returns:
        RGB image
    """
    normalized = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    colored = cv2.applyColorMap(
        (255 * (1.0 - normalized)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cast(
        "jt.UInt8[npt.NDArray[np.uint8], 'height width 3']",
        cv2.cvtColor(colored, cv2.COLOR_BGR2RGB),
    )
