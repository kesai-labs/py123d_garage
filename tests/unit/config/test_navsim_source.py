from __future__ import annotations

import pytest

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.common.config_help import build_from_string
from py123d_garage.config.presets.python.offline_data_sources.navsim import navtest
from py123d_garage.config.presets.python.policy.latent_transfuser import ltf_nuplan as latent_architecture_preset
from py123d_garage.config.presets.python.policy.transfuser import tf_nuplan as transfuser_architecture_preset
from py123d_garage.config.schema.evaluation.navsim_config import NavsimBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig


@pytest.fixture(autouse=True)
def data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")


def _policy_configs() -> list[TransfuserConfig]:
    return [transfuser_architecture_preset(), latent_architecture_preset()]


def test_source_serves_its_policy() -> None:
    for policy_config in _policy_configs():
        source = navtest(policy_config)
        policy = build_from_string(PolicyConfig(transfuser_config=policy_config), AbstractPolicy)
        policy.verify_contract(offline_data_source_config=source, scene_filter=source.garage_scene_filter)
        future_duration_s = source.garage_scene_filter.future_duration_s
        assert future_duration_s is not None
        assert round(future_duration_s * 1e6) >= NavsimBenchmarkConfig.navsim_scoring_future_duration_us
        resolved = source.garage_scene_filter.to_py123d_scene_filter()
        assert all(callable(fn) for fn in resolved.custom_anchor_filter_fns or [])
        assert source.data_root == "/data/nuplan/123D"


def test_lidar_follows_the_architecture() -> None:
    full, latent = _policy_configs()
    full_modalities = navtest(full).garage_scene_filter.required_scene_modalities
    latent_modalities = navtest(latent).garage_scene_filter.required_scene_modalities
    assert full_modalities is not None
    assert "lidar.lidar_merged@history+initial" in full_modalities
    assert latent_modalities is not None
    assert not any("lidar" in modality for modality in latent_modalities)


def test_unset_data_root_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PY123D_GARAGE_DATA_ROOT")
    with pytest.raises(ValueError, match="PY123D_GARAGE_DATA_ROOT"):
        navtest(_policy_configs()[0])
