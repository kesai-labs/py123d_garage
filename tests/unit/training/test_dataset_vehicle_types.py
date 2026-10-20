# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The hood swap's vehicle type per sample: typed by camera mount, hoodless by dataset, and loud on anything else."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from py123d.api import SceneAPI
from py123d.datatypes import CameraID

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.common.sensor import hood_swap
from py123d_garage.config.schema.training.photometric_augmentation_config import HoodSwapConfig
from py123d_garage.training.dataset import TrainingDataset, _vehicle_types

_MOUNT_CAMERAS = (CameraID.FTCAM_F0, CameraID.FTCAM_L0, CameraID.FTCAM_R0)


class _Scene:
    """A scene of one dataset and log whose camera mounts are the given metadatas."""

    def __init__(self, dataset: str, log_name: str, metadatas: dict[CameraID, Any]) -> None:
        self.scene_metadata = SimpleNamespace(dataset=dataset)
        self.log_name = log_name
        self._metadatas = metadatas

    def get_camera_metadatas(self) -> dict[CameraID, Any]:
        return self._metadatas


SceneAPI.register(_Scene)


def _scene(dataset: str, log_name: str, mount_m: list[float] | None) -> Any:
    """
    A scene of a dataset whose front, left and right cameras sit at the given mount.

    Args:
        dataset: the dataset name.
        log_name: the log name.
        mount_m: nine translations in metres; None = a log without those cameras.

    Returns:
        the scene.
    """
    metadatas = (
        {
            camera_id: SimpleNamespace(camera_to_imu_se3=SimpleNamespace(x=x, y=y, z=z))
            for camera_id, (x, y, z) in zip(_MOUNT_CAMERAS, zip(*[iter(mount_m)] * 3, strict=True), strict=True)
        }
        if mount_m is not None
        else {}
    )
    return _Scene(dataset, log_name, metadatas)


def _vehicle_type(scene: Any, config: HoodSwapConfig | None = None) -> int:
    """The data loader's vehicle type of the scene under the config."""
    _vehicle_types.clear()
    loader = SimpleNamespace(_hood_swap_config=config)
    return TrainingDataset._vehicle_type(
        cast(TrainingDataset, loader),
        cast(SceneAPI, scene),
        scene.scene_metadata.dataset,
        scene.log_name,
    )


def test_typed_log_takes_its_nearest_vehicle_type() -> None:
    mount = list(hood_swap._pai_mount_centroids()[7])
    assert _vehicle_type(_scene("physical-ai-av", "log-of-type-7", mount), HoodSwapConfig()) == 7


def test_hoodless_dataset_and_hood_swap_off_need_no_mount() -> None:
    assert _vehicle_type(_scene("nuplan", "a-nuplan-log", None), HoodSwapConfig()) == -1
    assert _vehicle_type(_scene("physical-ai-av", "any-log", None), None) == -1


def test_typed_log_without_mount_cameras_fails() -> None:
    with pytest.raises(ValueError, match="lacks the mount cameras"):
        _vehicle_type(_scene("nurec", "a-clip-without-side-cameras", None), HoodSwapConfig())


def test_typed_log_of_an_unknown_vehicle_fails() -> None:
    with pytest.raises(ValueError, match="no known type"):
        _vehicle_type(_scene("physical-ai-av", "a-new-vehicle", [0.0] * 9), HoodSwapConfig())


def test_dataset_neither_typed_nor_hoodless_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="neither in typed_datasets nor in hoodless_datasets"):
        TrainingDataset(
            [cast(SceneAPI, _scene("kesai", "a-kesai-log", None))],
            MagicMock(spec=AbstractPolicy),
            hood_swap_config=HoodSwapConfig(),
            check_scene_coverage=False,
        )
