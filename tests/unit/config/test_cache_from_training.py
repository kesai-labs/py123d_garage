# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from pathlib import Path

import pytest

from py123d_garage.common.config_help import compose_config, finalize_cache
from py123d_garage.config.presets.training.latent_transfuser import pretrain as latent_transfuser_pretrain
from py123d_garage.config.schema.cache.cache_config import CacheConfig

_STRIDED_PRESET = "py123d_garage.config.presets.training.latent_transfuser:pretrain"


@pytest.fixture(autouse=True)
def data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")


def _load(*overrides: str) -> CacheConfig:
    return finalize_cache(compose_config("build_cache", CacheConfig, list(overrides)), CacheConfig)


def test_cache_config_keeps_the_training_plan_anchors_by_default() -> None:
    config = _load(f"training@plan={_STRIDED_PRESET}")
    training = latent_transfuser_pretrain()
    assert [source.garage_scene_filter for source in config.offline_data_sources] == [
        source.garage_scene_filter for source in training.offline_data_sources
    ]


def test_dense_cache_config_densifies_the_training_plan() -> None:
    config = _load(f"training@plan={_STRIDED_PRESET}", "build_dense_cache=true")
    training = latent_transfuser_pretrain()
    assert config.policy_config == training.policy_config
    assert [Path(source.cache_root).name for source in config.offline_data_sources] == [
        Path(source.cache_root).name for source in training.offline_data_sources
    ]
    for source in config.offline_data_sources:
        assert source.garage_scene_filter.timestamp_threshold_s is None
    assert any(source.garage_scene_filter.timestamp_threshold_s is not None for source in training.offline_data_sources)


def test_override_beats_the_training_plan() -> None:
    config = _load(f"training@plan={_STRIDED_PRESET}", "dataloader_config.num_workers=4")
    assert config.dataloader_config.num_workers == 4


def test_missing_plan_raises() -> None:
    with pytest.raises(ValueError, match="training plan"):
        _load()


def test_sparse_cache_keeps_the_training_plan_stride() -> None:
    config = _load(f"training@plan={_STRIDED_PRESET}", "build_dense_cache=false")
    training = latent_transfuser_pretrain()
    for source, planned in zip(config.offline_data_sources, training.offline_data_sources, strict=True):
        assert source.garage_scene_filter.timestamp_threshold_s == planned.garage_scene_filter.timestamp_threshold_s
