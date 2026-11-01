# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import inspect

import pytest
from omegaconf.errors import ConfigAttributeError

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.common.config_help import build_from_string, load_config
from py123d_garage.config.presets.training import latent_transfuser as latent_training_presets
from py123d_garage.config.schema.training.training_config import TrainingConfig

_POSTTRAIN_PRESET = "py123d_garage.config.presets.training.latent_transfuser:posttrain"

_TRAINING_MODULES = (latent_training_presets,)


@pytest.mark.parametrize(
    ("preset_module", "config_class"),
    [
        (latent_training_presets, TrainingConfig),
    ],
    ids=[
        "training-latent",
    ],
)
def test_every_preset_constructs(preset_module, config_class) -> None:
    factories = _preset_factories(preset_module)
    assert factories
    for factory in factories:
        assert isinstance(factory(), config_class), factory.__name__


def test_no_preset_declares_no_policy_block() -> None:
    config = load_config(TrainingConfig, "train", args=[])
    assert config.policy_config.transfuser_config is None


def test_preset_sets_policy_and_dataset_together() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_POSTTRAIN_PRESET}",
            "offline_data_sources.0.data_root=/data/123D",
        ],
    )
    transfuser = config.policy_config.transfuser_config
    assert transfuser is not None
    assert transfuser.trajectory_horizon_us == 6_400_000
    assert transfuser.trajectory_offsets_us == uniform_offsets_us(100_000, 6_400_000)
    source = config.offline_data_sources[0]
    assert source.data_root == "/data/123D"
    assert source.garage_scene_filter.split_names == ["physical-ai-av_train"]
    assert source.garage_scene_filter.future_duration_s == 6.4
    assert source.garage_scene_filter.min_remaining_route_m == 80.0
    assert source.garage_scene_filter.history_duration_s == 1.0


def _preset_factories(module) -> list:
    return [
        factory
        for name, factory in inspect.getmembers(module, inspect.isfunction)
        if factory.__module__ == module.__name__ and not name.startswith("_")
    ]


def test_every_training_preset_filter_serves_its_policy() -> None:
    for module in _TRAINING_MODULES:
        for factory in _preset_factories(module):
            config = factory()
            policy = build_from_string(config.policy_config, AbstractPolicy)
            for source in config.offline_data_sources:
                policy.verify_contract(offline_data_source_config=source, scene_filter=source.garage_scene_filter)


def test_every_preset_filter_fn_resolves() -> None:
    for module in _TRAINING_MODULES:
        for factory in _preset_factories(module):
            config = factory()
            for source in config.offline_data_sources:
                resolved = source.garage_scene_filter.to_py123d_scene_filter()
                assert all(callable(fn) for fn in resolved.custom_anchor_filter_fns or [])


def test_training_presets_keep_their_anchor_stride() -> None:
    physical_ai_av_train = latent_training_presets.pretrain().offline_data_sources[0]
    assert physical_ai_av_train.garage_scene_filter.timestamp_threshold_s == 2.0


def test_cli_override_beats_the_preset() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_POSTTRAIN_PRESET}",
            "offline_data_sources.0.data_root=/data/123D",
            "policy_config.transfuser_config.trajectory_offsets_us=[1000000,2000000,3000000]",
        ],
    )
    transfuser = config.policy_config.transfuser_config
    assert transfuser is not None
    assert transfuser.trajectory_horizon_us == 3_000_000
    assert transfuser.trajectory_offsets_us == [1_000_000, 2_000_000, 3_000_000]


def test_unknown_preset_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_config(TrainingConfig, "train", args=["training=nope:missing_preset"])


def test_preset_of_the_wrong_config_class_raises() -> None:
    with pytest.raises(ConfigAttributeError):
        load_config(
            TrainingConfig,
            "train",
            args=["training=py123d_garage.config.presets.policy.latent_transfuser:latent_transfuser"],
        )
