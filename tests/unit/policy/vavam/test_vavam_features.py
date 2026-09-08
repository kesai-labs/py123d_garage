"""VaVAM context frames: timestamp matching, young-rollout padding, dropout rejection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from py123d.api import SceneAPI
from py123d.datatypes import CameraID, Timestamp
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata
from py123d.geometry.pose import PoseSE3

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.vavam_config import VavamConfig
from py123d_garage.policy.vavam.vavam_policy import VavamPolicy

# Pinhole metadata passes through _rectified unchanged; only f-theta cameras rectify.
_CAMERA_METADATA = PinholeCameraMetadata(
    camera_name="pcam_f0",
    camera_id=CameraID.PCAM_F0,
    intrinsics=None,
    distortion=None,
    width=32,
    height=32,
    camera_to_imu_se3=PoseSE3.identity(),
)


class _CameraScene:
    """SceneAPI stand-in serving timestamped frames of one camera."""

    def __init__(
        self,
        anchor_us: int,
        frames: dict[int, npt.NDArray[np.uint8]],
    ) -> None:
        self._anchor_us = anchor_us
        self._frames = frames

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    @property
    def scene_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(dataset="nuplan")

    def _camera(self, timestamp_us: int) -> SimpleNamespace:
        return SimpleNamespace(
            image=self._frames[timestamp_us],
            timestamp=Timestamp.from_us(timestamp_us),
            metadata=_CAMERA_METADATA,
        )

    def get_camera_at_iteration(
        self,
        iteration: int,
        camera_id: CameraID,
    ) -> SimpleNamespace | None:
        return self._camera(self._anchor_us)

    def get_camera_at_timestamp(
        self,
        timestamp: int,
        camera_id: CameraID,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
    ) -> SimpleNamespace | None:
        candidates_us = (
            [frame_us for frame_us in self._frames if frame_us <= timestamp]
            if criteria == "backward"
            else list(self._frames)
        )
        if not candidates_us:
            return None
        return self._camera(min(candidates_us, key=lambda frame_us: abs(frame_us - timestamp)))


SceneAPI.register(_CameraScene)


def _as_scene(stub: object) -> SceneAPI:
    """The stub is a virtual SceneAPI subclass (register); cast for the type checker."""
    return cast(SceneAPI, stub)


_ANCHOR_US = 10_000_000
_INTERVAL_US = 500_000


def _policy() -> VavamPolicy:
    return VavamPolicy(PolicyConfig(vavam_config=VavamConfig(context_length=3)))


def _image(value: int) -> npt.NDArray[np.uint8]:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def _normalized(value: int) -> float:
    return value / 255.0 * 2.0 - 1.0


def test_context_frames_come_oldest_first() -> None:
    scene = _CameraScene(
        _ANCHOR_US,
        {
            _ANCHOR_US - 2 * _INTERVAL_US: _image(10),
            _ANCHOR_US - _INTERVAL_US: _image(20),
            _ANCHOR_US: _image(30),
        },
    )
    frames = _policy().build_features(_as_scene(scene), {}).frames
    assert frames.shape[:2] == (3, 3)
    for index, value in enumerate((10, 20, 30)):
        assert torch.allclose(frames[index], torch.full_like(frames[index], _normalized(value)), atol=1e-5)


def test_off_grid_frame_within_tolerance_is_served() -> None:
    scene = _CameraScene(
        _ANCHOR_US,
        {
            _ANCHOR_US - 2 * _INTERVAL_US: _image(10),
            _ANCHOR_US - _INTERVAL_US + 100_000: _image(20),
            _ANCHOR_US: _image(30),
        },
    )
    frames = _policy().build_features(_as_scene(scene), {}).frames
    assert torch.allclose(frames[1], torch.full_like(frames[1], _normalized(20)), atol=1e-5)


def test_young_rollout_repeats_the_oldest_served_frame() -> None:
    scene = _CameraScene(
        _ANCHOR_US,
        {
            _ANCHOR_US - _INTERVAL_US: _image(20),
            _ANCHOR_US: _image(30),
        },
    )
    frames = _policy().build_features(_as_scene(scene), {}).frames
    assert torch.equal(frames[0], frames[1])
    assert torch.allclose(frames[1], torch.full_like(frames[1], _normalized(20)), atol=1e-5)


def test_dropout_inside_coverage_is_rejected() -> None:
    """A missing frame with older frames recorded is a dropout, not an episode start."""
    scene = _CameraScene(
        _ANCHOR_US,
        {
            _ANCHOR_US - 2 * _INTERVAL_US: _image(10),
            _ANCHOR_US: _image(30),
        },
    )
    with pytest.raises(ValueError, match="already recording"):
        _policy().build_features(_as_scene(scene), {})
