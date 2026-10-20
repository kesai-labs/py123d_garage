# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import replace

import pytest

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.common.config_help import build_from_string
from py123d_garage.config.presets.offline_data_sources.navsim import navtest
from py123d_garage.config.presets.policy.latent_transfuser import latent_transfuser
from py123d_garage.config.presets.training.latent_transfuser import midtrain, posttrain, pretrain
from py123d_garage.config.schema.evaluation.navsim_config import NavsimBenchmarkConfig
from py123d_garage.config.schema.evaluation.open_loop_config import OpenLoopBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, TransfuserConfig


@pytest.fixture(autouse=True)
def data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")


def _latent_architecture() -> TransfuserConfig:
    """The mixed architecture with navsim's 4 s plan, which the navtest scenes' 5 s of future cover."""
    return replace(latent_transfuser(), trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000))


def _lidar_architecture() -> TransfuserConfig:
    """The latent architecture with the lidar branch on, over 0.2 s of lidar history at the 10 Hz nuplan serves."""
    architecture = _latent_architecture()
    architecture.backbone_config = BackboneConfig(latent=False)
    architecture.lidar_config.lidar_interval_us = 100_000
    architecture.lidar_config.lidar_horizon_us = 200_000
    return architecture


def _policy_configs() -> list[TransfuserConfig]:
    return [_lidar_architecture(), _latent_architecture()]


def test_source_serves_its_policy() -> None:
    for policy_config in _policy_configs():
        benchmark_config = NavsimBenchmarkConfig()
        source = navtest(policy_config, benchmark_config)
        policy = build_from_string(PolicyConfig(transfuser_config=policy_config), AbstractPolicy)
        policy.verify_contract(
            benchmark_config=benchmark_config,
            offline_data_source_config=source,
            scene_filter=source.garage_scene_filter,
        )
        future_duration_s = source.garage_scene_filter.future_duration_s
        assert future_duration_s is not None
        assert round(future_duration_s * 1e6) >= NavsimBenchmarkConfig.navsim_scoring_future_duration_us
        resolved = source.garage_scene_filter.to_py123d_scene_filter()
        assert all(callable(fn) for fn in resolved.custom_anchor_filter_fns or [])
        assert source.data_root == "/data/nuplan/123D"


def test_navsim_source_asks_for_the_scoring_window_only() -> None:
    """Every navtest frame with the 5 s scoring window is scored, whatever the plan horizon and the route length."""
    for stage in (pretrain, midtrain, posttrain):
        policy_config = stage().policy_config.transfuser_config
        assert policy_config is not None
        benchmark_config = NavsimBenchmarkConfig()
        source = navtest(policy_config, benchmark_config)
        policy = build_from_string(PolicyConfig(transfuser_config=policy_config), AbstractPolicy)
        policy.verify_contract(
            benchmark_config=benchmark_config,
            offline_data_source_config=source,
            scene_filter=source.garage_scene_filter.to_py123d_scene_filter(),
        )
        future_duration_s = source.garage_scene_filter.future_duration_s
        assert future_duration_s is not None
        assert round(future_duration_s * 1e6) == NavsimBenchmarkConfig.navsim_scoring_future_duration_us
        assert policy_config.trajectory_horizon_us > NavsimBenchmarkConfig.navsim_scoring_future_duration_us
        assert source.garage_scene_filter.min_remaining_route_m is None
        assert source.expected_num_scenes is None


def test_open_loop_source_keeps_the_plan_horizon_and_the_route() -> None:
    """Open loop scores against the logged future, so its navtest slice needs the plan horizon and the route."""
    policy_config = posttrain().policy_config.transfuser_config
    assert policy_config is not None
    benchmark_config = OpenLoopBenchmarkConfig()
    source = navtest(policy_config, benchmark_config)
    policy = build_from_string(PolicyConfig(transfuser_config=policy_config), AbstractPolicy)
    policy.verify_contract(
        benchmark_config=benchmark_config,
        offline_data_source_config=source,
        scene_filter=source.garage_scene_filter.to_py123d_scene_filter(),
    )
    future_duration_s = source.garage_scene_filter.future_duration_s
    assert future_duration_s is not None
    assert round(future_duration_s * 1e6) == policy_config.trajectory_horizon_us
    assert source.garage_scene_filter.min_remaining_route_m == max(policy_config.required_target_point_distances_m)


def test_lidar_follows_the_architecture() -> None:
    full, latent = _policy_configs()
    full_modalities = navtest(full, NavsimBenchmarkConfig()).garage_scene_filter.required_scene_modalities
    latent_modalities = navtest(latent, NavsimBenchmarkConfig()).garage_scene_filter.required_scene_modalities
    assert full_modalities is not None
    assert "lidar.lidar_merged@history+initial" in full_modalities
    assert latent_modalities is not None
    assert not any("lidar" in modality for modality in latent_modalities)


def test_unset_data_root_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PY123D_GARAGE_DATA_ROOT")
    with pytest.raises(ValueError, match="PY123D_GARAGE_DATA_ROOT"):
        navtest(_policy_configs()[0], NavsimBenchmarkConfig())
