# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""A preset's dict-typed fields replace the schema's entries: a narrowed dict stays narrow and enum names convert."""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig, OmegaConf
from py123d.datatypes import CameraID, DefaultBoxDetectionLabel

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.common.config_help.config_handling import typed_config
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BoxDetectionConfig, CameraConfig, TransfuserConfig
from py123d_garage.config.schema.training.training_config import TrainingConfig

_CLASSES = "policy_config.transfuser_config.box_detection_config.detection_classes"
_CAMERAS = "policy_config.transfuser_config.camera_config.input_cameras"


def _replayed(transfuser_config: TransfuserConfig) -> DictConfig:
    """The training config of the architecture, as a yaml plan with enum names, replayed onto the schema."""
    plan = OmegaConf.to_container(
        OmegaConf.structured(TrainingConfig(policy_config=PolicyConfig(transfuser_config=transfuser_config))),
        resolve=False,
        enum_to_str=True,
    )
    return typed_config(cast(DictConfig, OmegaConf.create(plan)), TrainingConfig)


def test_a_narrowed_int_keyed_dict_replaces_the_default() -> None:
    narrowed = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        box_detection_config=BoxDetectionConfig(detection_classes={0: [DefaultBoxDetectionLabel.VEHICLE]}),
    )
    assert len(BoxDetectionConfig().detection_classes) > 1
    classes = OmegaConf.to_container(OmegaConf.select(_replayed(narrowed), _CLASSES))
    assert classes == {0: [DefaultBoxDetectionLabel.VEHICLE]}


def test_a_str_keyed_dict_converts_its_enum_names() -> None:
    cameras = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(input_cameras={"nuplan": [CameraID.PCAM_F0]}),
    )
    assert OmegaConf.to_container(OmegaConf.select(_replayed(cameras), _CAMERAS)) == {"nuplan": [CameraID.PCAM_F0]}
