from __future__ import annotations

import inspect

import pytest
from omegaconf.errors import ConfigAttributeError

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.common.config_help import build_from_string, load_config
from py123d_garage.config.presets.python.training import latent_transfuser as latent_training_presets
from py123d_garage.config.presets.python.training import transfuser as transfuser_training_presets
from py123d_garage.config.schema.training.training_config import TrainingConfig

_CARLA_TRAINING_PRESET = "py123d_garage.config.presets.python.training.transfuser:tf_carla_train"

_TRAINING_MODULES = (
    transfuser_training_presets,
    latent_training_presets,
)


@pytest.mark.parametrize(
    ("preset_module", "config_class"),
    [
        (transfuser_training_presets, TrainingConfig),
        (latent_training_presets, TrainingConfig),
    ],
    ids=[
        "training-transfuser",
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
    assert config.policy_config.vavam_config is None


def test_preset_sets_policy_and_dataset_together() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_CARLA_TRAINING_PRESET}",
            "offline_data_sources.carla.data_root=/data/123D",
        ],
    )
    transfuser = config.policy_config.transfuser_config
    assert transfuser is not None
    assert transfuser.trajectory_horizon_us == 2_000_000
    assert transfuser.trajectory_interval_us == 250_000
    (source,) = config.offline_data_sources.values()
    assert source.garage_scene_filter.split_names == ["normal_view"]
    assert source.garage_scene_filter.future_duration_s == 2.0
    assert source.garage_scene_filter.min_remaining_route_m == 45.0
    assert source.garage_scene_filter.history_duration_s == 0.2


def _preset_factories(module) -> list:
    return [
        factory
        for name, factory in inspect.getmembers(module, inspect.isfunction)
        if factory.__module__ == module.__name__
    ]


def test_every_training_preset_filter_serves_its_policy() -> None:
    for module in _TRAINING_MODULES:
        for factory in _preset_factories(module):
            config = factory()
            policy = build_from_string(config.policy_config, AbstractPolicy)
            for source in config.offline_data_sources.values():
                policy.verify_contract(offline_data_source_config=source, scene_filter=source.garage_scene_filter)


def test_every_preset_filter_fn_resolves() -> None:
    for module in _TRAINING_MODULES:
        for factory in _preset_factories(module):
            config = factory()
            for source in config.offline_data_sources.values():
                resolved = source.garage_scene_filter.to_py123d_scene_filter()
                assert all(callable(fn) for fn in resolved.custom_anchor_filter_fns or [])


def test_training_presets_keep_their_anchor_stride() -> None:
    nuplan_train = transfuser_training_presets.tf_nuplan_trainval().offline_data_sources["nuplan_train"]
    assert nuplan_train.garage_scene_filter.timestamp_threshold_s == 1.0
    physical_ai_av_train = latent_training_presets.ltf_physical_ai_av_trainval().offline_data_sources[
        "physical_ai_av_train"
    ]
    assert physical_ai_av_train.garage_scene_filter.timestamp_threshold_s == 2.0


def test_cli_override_beats_the_preset() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_CARLA_TRAINING_PRESET}",
            "offline_data_sources.carla.data_root=/data/123D",
            "policy_config.transfuser_config.trajectory_horizon_us=3000000",
        ],
    )
    transfuser = config.policy_config.transfuser_config
    assert transfuser is not None
    assert transfuser.trajectory_horizon_us == 3_000_000
    assert transfuser.trajectory_interval_us == 250_000


def test_unknown_preset_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_config(TrainingConfig, "train", args=["training=nope:missing_preset"])


def test_preset_of_the_wrong_config_class_raises() -> None:
    with pytest.raises(ConfigAttributeError):
        load_config(
            TrainingConfig,
            "train",
            args=["training=py123d_garage.config.presets.python.policy.transfuser:tf_carla"],
        )
