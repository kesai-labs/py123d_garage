# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf.errors import InterpolationResolutionError

from py123d_garage.common.config_help import load_config, run_dir
from py123d_garage.config.schema.training.training_config import TrainingConfig

_PRETRAIN_PRESET = "py123d_garage.config.presets.training.latent_transfuser:pretrain"


def test_preset_resolves_paths_from_the_environment() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[f"training={_PRETRAIN_PRESET}"],
    )
    source = config.offline_data_sources[0]
    assert source.data_root == "/data/physical-ai-av/123D"
    assert source.cache_root == (
        "/data/physical-ai-av/py123d_garage_cache/"
        "latent_transfuser_physical_ai_av_train_uniform_sparse_2s_temporal_front_camera"
    )


def test_explicit_cache_root_wins() -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            f"training={_PRETRAIN_PRESET}",
            "offline_data_sources.0.data_root=/data/123D",
            "offline_data_sources.0.cache_root=/stores/physical_ai_av",
        ],
    )
    assert config.offline_data_sources[0].cache_root == "/stores/physical_ai_av"


def test_missing_environment_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PY123D_GARAGE_DATA_ROOT")
    with pytest.raises(InterpolationResolutionError):
        load_config(
            TrainingConfig,
            "train",
            args=[f"training={_PRETRAIN_PRESET}"],
        )


def test_run_dir_defaults_under_outputs() -> None:
    load_config(TrainingConfig, "train", args=[f"training={_PRETRAIN_PRESET}"])
    assert run_dir().parts[0] == "outputs"


def test_run_dir_override_wins() -> None:
    load_config(TrainingConfig, "train", args=[f"training={_PRETRAIN_PRESET}", "hydra.run.dir=/tmp/run"])
    assert run_dir() == Path("/tmp/run")
