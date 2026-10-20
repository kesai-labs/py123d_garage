# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The sensor rig export: one sensor_rig_<dataset>.yaml per dataset, holding the first source's rig."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from py123d.api import SceneAPI

from py123d_garage.training.train import _export_sensor_rigs


class _Modality:
    """A modality metadata whose dict form is the given fields."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def to_dict(self) -> dict[str, Any]:
        return dict(self._fields)


class _Scene:
    """A scene of one dataset and log whose modalities are the given metadatas."""

    def __init__(self, dataset: str, log_name: str, modalities: dict[str, _Modality]) -> None:
        self.scene_metadata = SimpleNamespace(dataset=dataset)
        self._log_name = log_name
        self._modalities = modalities

    def get_log_metadata(self) -> Any:
        return SimpleNamespace(log_name=self._log_name)

    def get_all_modality_metadatas(self) -> dict[str, _Modality]:
        return self._modalities


SceneAPI.register(_Scene)


def _scene(dataset: str, log_name: str, focal_px: float) -> Any:
    return _Scene(dataset, log_name, {"camera.pcam_f0": _Modality(camera_name="front", focal_px=focal_px)})


def test_one_rig_per_dataset_named_by_dataset(tmp_path: Path) -> None:
    _export_sensor_rigs(
        [_scene("nuplan", "log_a", 1.0), _scene("nuplan", "log_b", 1.0), _scene("nurec", "log_c", 2.0)],
        str(tmp_path),
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["sensor_rig_nuplan.yaml", "sensor_rig_nurec.yaml"]
    assert yaml.safe_load((tmp_path / "sensor_rig_nurec.yaml").read_text()) == {
        "dataset": "nurec",
        "log_name": "log_c",
        "modalities": {"camera.pcam_f0": {"camera_name": "front", "focal_px": 2.0}},
    }


def test_disagreeing_sources_keep_the_first_rig_and_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="py123d_garage.training.train"):
        _export_sensor_rigs([_scene("nuplan", "log_a", 1.0), _scene("nuplan", "log_b", 3.0)], str(tmp_path))

    rig = yaml.safe_load((tmp_path / "sensor_rig_nuplan.yaml").read_text())
    assert rig["log_name"] == "log_a"
    assert rig["modalities"]["camera.pcam_f0"]["focal_px"] == 1.0
    assert "log_a" in caplog.text
    assert "log_b" in caplog.text
