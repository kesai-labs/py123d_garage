# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from NVlabs/alpasim (Apache License 2.0).

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from py123d.datatypes import FThetaCameraMetadata

_RADIAL_INDICES = (0, 1, 4, 5, 6, 7)
_TANGENTIAL_INDICES = (2, 3)


def _distortion_vector(
    radial: Sequence[float],
    tangential: Sequence[float],
) -> npt.NDArray[np.float64]:
    dist = np.zeros(14, dtype=np.float64)
    for index, value in zip(_RADIAL_INDICES, radial, strict=True):
        dist[index] = value
    for index, value in zip(_TANGENTIAL_INDICES, tangential, strict=True):
        dist[index] = value
    return dist


def _undistorted_pixels(
    pixels: npt.NDArray[np.float64],
    focal: tuple[float, float],
    principal: tuple[float, float],
    dist: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    fx, fy = focal
    cx, cy = principal
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    normalized = cast(
        "npt.NDArray[np.float64]",
        cv2.undistortPoints(pixels.reshape((-1, 1, 2)), cameraMatrix=camera_matrix, distCoeffs=dist),
    ).reshape((-1, 2))
    return normalized * np.array([fx, fy]) + np.array([cx, cy])


def _overscan_scale(
    resolution_wh: tuple[int, int],
    focal: tuple[float, float],
    principal: tuple[float, float],
    dist: npt.NDArray[np.float64],
    safety_margin_px: float,
    max_scale: float,
) -> float:
    width, height = resolution_wh
    samples_x = np.linspace(0.0, width - 1.0, num=9, dtype=np.float64)
    samples_y = np.linspace(0.0, height - 1.0, num=9, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(samples_x, samples_y)
    undistorted = _undistorted_pixels(np.stack((grid_x, grid_y), axis=-1), focal, principal, dist)

    margin_x = max(0.0, -float(np.min(undistorted[:, 0])), float(np.max(undistorted[:, 0])) - (width - 1))
    margin_y = max(0.0, -float(np.min(undistorted[:, 1])), float(np.max(undistorted[:, 1])) - (height - 1))
    margin_x += safety_margin_px
    margin_y += safety_margin_px
    if margin_x <= 0.0 and margin_y <= 0.0:
        return 1.0
    scale = max(1.0 + 2.0 * margin_x / width, 1.0 + 2.0 * margin_y / height)
    return max(1.0, min(scale, max_scale))


def _ftheta_maps(
    metadata: FThetaCameraMetadata,
    focal: tuple[float, float],
    principal: tuple[float, float],
    canvas_wh: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    intrinsics = metadata.intrinsics
    assert intrinsics is not None, f"camera {metadata.camera_name} carries no f-theta intrinsics."
    fx, fy = focal
    cx, cy = principal
    canvas_w, canvas_h = canvas_wh
    grid_x, grid_y = np.meshgrid(
        np.arange(canvas_w, dtype=np.float64),
        np.arange(canvas_h, dtype=np.float64),
    )
    rays = np.stack(((grid_x - cx) / fx, (grid_y - cy) / fy, np.ones_like(grid_x)), axis=-1)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-9)

    xy = rays[..., :2]
    z = rays[..., 2]
    xy_norm = np.linalg.norm(xy, axis=-1)
    theta = np.where(z > 0.0, np.arctan2(xy_norm, z), 0.0)
    radii = np.polynomial.polynomial.polyval(theta, np.asarray(intrinsics.fw_poly, dtype=np.float64))
    scales = np.divide(radii, xy_norm, out=np.zeros_like(radii), where=xy_norm > 1e-9)
    map_x = intrinsics.cx + xy[..., 0] * scales
    map_y = intrinsics.cy + xy[..., 1] * scales

    valid = (
        (z > 0.0)
        & (map_x >= -0.5)
        & (map_x <= metadata.width - 0.5)
        & (map_y >= -0.5)
        & (map_y <= metadata.height - 0.5)
    )
    map_x = np.where(valid, map_x, -1.0).astype(np.float32)
    map_y = np.where(valid, map_y, -1.0).astype(np.float32)
    return map_x, map_y


def _distortion_maps(
    focal: tuple[float, float],
    principal: tuple[float, float],
    dist: npt.NDArray[np.float64],
    canvas_wh: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    canvas_w, canvas_h = canvas_wh
    grid_x, grid_y = np.meshgrid(
        np.arange(canvas_w, dtype=np.float64),
        np.arange(canvas_h, dtype=np.float64),
    )
    undistorted = _undistorted_pixels(np.stack((grid_x, grid_y), axis=-1), focal, principal, dist)
    map_x = undistorted[:, 0].reshape((canvas_h, canvas_w)).astype(np.float32)
    map_y = undistorted[:, 1].reshape((canvas_h, canvas_w)).astype(np.float32)
    valid = (map_x >= -0.5) & (map_x <= canvas_w - 0.5) & (map_y >= -0.5) & (map_y <= canvas_h - 0.5)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y


class FThetaRectifier:
    def __init__(
        self,
        metadata: FThetaCameraMetadata,
        focal_length_px: float,
        principal_point_px: Sequence[float],
        radial: Sequence[float],
        tangential: Sequence[float],
        max_overscan_scale: float,
        safety_margin_px: float,
    ) -> None:
        """
        Builds the remap tables for one camera.

        Args:
            metadata: the recorded f-theta camera the images come from
            focal_length_px: focal length of the pinhole camera to rectify onto
            principal_point_px: principal point (x, y) of that pinhole camera
            radial: radial distortion coefficients to re-apply
            tangential: tangential distortion coefficients to re-apply
            max_overscan_scale: largest canvas growth allowed to keep the distorted border
            safety_margin_px: slack kept around that border
        """
        width, height = metadata.width, metadata.height
        focal = (focal_length_px, focal_length_px)
        principal_x, principal_y = principal_point_px
        dist = _distortion_vector(radial, tangential)
        has_distortion = bool(np.any(dist))

        scale = 1.0
        if has_distortion:
            scale = _overscan_scale(
                (width, height),
                focal,
                (principal_x, principal_y),
                dist,
                safety_margin_px,
                max_overscan_scale,
            )
        if scale > 1.0:
            canvas_w, canvas_h = math.ceil(width * scale), math.ceil(height * scale)
        else:
            canvas_w, canvas_h = width, height
        margin_x = round((canvas_w - width) / 2.0)
        margin_y = round((canvas_h - height) / 2.0)
        canvas_principal = (principal_x + margin_x, principal_y + margin_y)

        self._map_x, self._map_y = _ftheta_maps(metadata, focal, canvas_principal, (canvas_w, canvas_h))
        self._distort_maps: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None
        if has_distortion:
            self._distort_maps = _distortion_maps(focal, canvas_principal, dist, (canvas_w, canvas_h))
        self._crop_yx = (margin_y, margin_x) if (canvas_w, canvas_h) != (width, height) else None
        self._resolution_hw = (height, width)

    def rectify(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rectified = cast(
            "npt.NDArray[np.uint8]",
            cv2.remap(
                image,
                self._map_x,
                self._map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ),
        )
        if self._distort_maps is not None:
            rectified = cast(
                "npt.NDArray[np.uint8]",
                cv2.remap(
                    rectified,
                    self._distort_maps[0],
                    self._distort_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ),
            )
        if self._crop_yx is not None:
            y0, x0 = self._crop_yx
            h, w = self._resolution_hw
            rectified = rectified[y0 : y0 + h, x0 : x0 + w]
        return rectified
