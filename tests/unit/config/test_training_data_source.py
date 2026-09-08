from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf.errors import InterpolationResolutionError

from py123d_garage.common.config_help import load_config, run_dir
from py123d_garage.config.schema.training.training_config import TrainingConfig

_CARLA_TRAINING_PRESET = "py123d_garage.config.presets.python.training.transfuser:tf_carla_train"


def test_preset_resolves_paths_from_the_environment() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[f"training={_CARLA_TRAINING_PRESET}"],
    )
    source = config.offline_data_sources["carla"]
    assert source.data_root == "/data/lead/123D"
    assert source.cache_root == "/data/lead/py123d_garage_cache/transfuser_carla"


def test_explicit_cache_root_wins() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_CARLA_TRAINING_PRESET}",
            "offline_data_sources.carla.data_root=/data/123D",
            "offline_data_sources.carla.cache_root=/stores/carla",
        ],
    )
    assert config.offline_data_sources["carla"].cache_root == "/stores/carla"


def test_missing_environment_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PY123D_GARAGE_DATA_ROOT")
    with pytest.raises(InterpolationResolutionError):
        load_config(
            TrainingConfig,
            "train",
            args=[f"training={_CARLA_TRAINING_PRESET}"],
        )


def test_run_dir_defaults_under_outputs() -> None:
    load_config(TrainingConfig, "train", args=[f"training={_CARLA_TRAINING_PRESET}"])
    assert run_dir().parts[0] == "outputs"


def test_run_dir_override_wins() -> None:
    load_config(TrainingConfig, "train", args=[f"training={_CARLA_TRAINING_PRESET}", "hydra.run.dir=/tmp/run"])
    assert run_dir() == Path("/tmp/run")
