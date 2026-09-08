from __future__ import annotations

from pathlib import Path

import pytest

from py123d_garage.common.config_help import compose_config, finalize_cache
from py123d_garage.config.presets.python.training.transfuser import tf_nuplan_trainval
from py123d_garage.config.schema.cache.cache_config import CacheConfig

_TRAINING_PRESET = "py123d_garage.config.presets.python.training.transfuser:tf_nuplan_trainval"


@pytest.fixture(autouse=True)
def data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")


def _load(*overrides: str) -> CacheConfig:
    return finalize_cache(compose_config("build_cache", CacheConfig, list(overrides)), CacheConfig)


def test_cache_config_densifies_the_training_plan() -> None:
    config = _load(f"training@plan={_TRAINING_PRESET}")
    training = tf_nuplan_trainval()
    assert config.policy_config == training.policy_config
    assert [Path(source.cache_root).name for source in config.offline_data_sources.values()] == [
        "transfuser_nuplan_train",
        "transfuser_nuplan_val",
    ]
    for source in config.offline_data_sources.values():
        assert source.garage_scene_filter.timestamp_threshold_s is None
    for source in training.offline_data_sources.values():
        assert source.garage_scene_filter.timestamp_threshold_s is not None


def test_override_beats_the_training_plan() -> None:
    config = _load(f"training@plan={_TRAINING_PRESET}", "dataloader_config.num_workers=4")
    assert config.dataloader_config.num_workers == 4


def test_missing_plan_raises() -> None:
    with pytest.raises(ValueError, match="training plan"):
        _load()


def test_sparse_cache_keeps_the_training_plan_stride() -> None:
    config = _load(f"training@plan={_TRAINING_PRESET}", "build_dense_cache=false")
    training = tf_nuplan_trainval()
    for name, source in config.offline_data_sources.items():
        assert (
            source.garage_scene_filter.timestamp_threshold_s
            == training.offline_data_sources[name].garage_scene_filter.timestamp_threshold_s
        )
