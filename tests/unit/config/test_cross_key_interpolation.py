from __future__ import annotations

import pytest

from py123d_garage.common.config_help import load_config
from py123d_garage.config.schema.training.training_config import TrainingConfig


def test_cross_key_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="cross-key interpolation"):
        load_config(
            TrainingConfig,
            "train",
            args=["initial_weights_file=${compile_mode}"],
        )


def test_nested_cross_key_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="cross-key interpolation"):
        load_config(
            TrainingConfig,
            "train",
            args=["wandb_config.name=${wandb_config.project}"],
        )


def test_env_resolver_is_allowed() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=["training=py123d_garage.config.presets.python.training.transfuser:tf_carla_train"],
    )
    assert config.offline_data_sources["carla"].data_root == "/data/lead/123D"
