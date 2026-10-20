# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import pytest

from py123d_garage.common.config_help import load_config
from py123d_garage.config.schema.training.training_config import TrainingConfig


def test_cross_key_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="cross-key interpolation"):
        load_config(
            TrainingConfig,
            "train",
            args=["cache_mirror_root=${compile_mode}"],
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
        args=["training=py123d_garage.config.presets.training.latent_transfuser:pretrain"],
    )
    assert config.offline_data_sources[0].data_root == "/data/physical-ai-av/123D"
