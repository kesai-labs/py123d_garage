# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError, ValidationError

from py123d_garage.common.config_help import (
    CONFIG_PATH,
    finalize,
    load_config,
    register_schema,
    run_dir,
)
from py123d_garage.config.presets.training.latent_transfuser import pretrain as latent_transfuser_pretrain
from py123d_garage.config.schema.training.training_config import TrainingConfig

_PRESET = "py123d_garage.config.presets.training.latent_transfuser:pretrain"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A user config dir: a training plan selecting an optimizer group."""
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "optimizer_config").mkdir()
    (tmp_path / "training" / "optimizer_config" / "fast.yaml").write_text("learning_rate: 0.01\n")
    (tmp_path / "training" / "tweak.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - optimizer_config: fast\n"
        "  - _self_\n"
        "wandb_config:\n"
        "  group: from_yaml\n"
        "lightning_trainer_config:\n"
        "  max_epochs: 3\n",
    )
    return tmp_path


def _user_dir(config_dir: Path) -> str:
    return f"hydra.searchpath=[file://{config_dir}]"


def test_yaml_plan_composes_onto_the_schema(config_dir: Path) -> None:
    config = load_config(TrainingConfig, "train", args=[_user_dir(config_dir), "training=tweak"])
    assert config.wandb_config.group == "from_yaml"
    assert config.optimizer_config.learning_rate == 0.01
    assert config.lightning_trainer_config.max_epochs == 3


def test_override_beats_the_yaml_and_reaches_unset_keys(config_dir: Path) -> None:
    config = load_config(
        TrainingConfig,
        "train",
        args=[
            _user_dir(config_dir),
            "training=tweak",
            "lightning_trainer_config.max_epochs=5",
            "dataloader_config.batch_size=7",
        ],
    )
    assert config.lightning_trainer_config.max_epochs == 5
    assert config.dataloader_config.batch_size == 7


def test_yaml_plan_builds_on_a_python_preset(config_dir: Path) -> None:
    (config_dir / "training" / "on_preset.yaml").write_text(
        f"# @package _global_\ndefaults:\n  - /preset: {_PRESET}\n  - _self_\nlightning_trainer_config:\n  max_epochs: 3\n",
    )
    config = load_config(TrainingConfig, "train", args=[_user_dir(config_dir), "training=on_preset"])
    preset = load_config(TrainingConfig, "train", args=[f"training={_PRESET}"])
    assert config.policy_config == preset.policy_config
    assert config.offline_data_sources == preset.offline_data_sources
    assert config.lightning_trainer_config.max_epochs == 3


def test_unknown_key_and_wrong_type_fail(config_dir: Path) -> None:
    (config_dir / "training" / "unknown.yaml").write_text("# @package _global_\nnope: 1\n")
    with pytest.raises(ConfigAttributeError, match="nope"):
        load_config(TrainingConfig, "train", args=[_user_dir(config_dir), "training=unknown"])
    with pytest.raises(ConfigCompositionException, match="nope"):
        load_config(TrainingConfig, "train", args=["nope=1"])
    with pytest.raises(ValidationError):
        load_config(TrainingConfig, "train", args=["optimizer_config.learning_rate=abc"])


def test_cross_key_interpolation_in_yaml_is_rejected(config_dir: Path) -> None:
    (config_dir / "training" / "interpolated.yaml").write_text(
        "# @package _global_\nwandb_config:\n  name: ${wandb_config.project}\n",
    )
    with pytest.raises(ValueError, match="cross-key"):
        load_config(TrainingConfig, "train", args=[_user_dir(config_dir), "training=interpolated"])


def test_hydra_target_key_names_the_policy(config_dir: Path) -> None:
    (config_dir / "training" / "targeted.yaml").write_text(
        "# @package _global_\npolicy_config:\n  _target_: py123d_garage.policy.vavam.vavam_policy.VavamPolicy\n",
    )
    config = load_config(TrainingConfig, "train", args=[_user_dir(config_dir), "training=targeted"])
    assert config.policy_config.target == "py123d_garage.policy.vavam.vavam_policy:VavamPolicy"


def test_env_overrides_sit_below_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_CONFIG", "optimizer_config.learning_rate=0.5 num_seeds=7")
    config = load_config(TrainingConfig, "train", args=["num_seeds=9"])
    assert config.optimizer_config.learning_rate == 0.5
    assert config.num_seeds == 9


def test_hydra_main_composes_the_plan_and_owns_the_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    @hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
    def task(cfg: DictConfig) -> None:
        seen["config"] = finalize(cfg, TrainingConfig)
        seen["run_dir"] = run_dir()

    register_schema("train", TrainingConfig)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", f"training={_PRESET}", f"hydra.run.dir={tmp_path}", "num_seeds=3"],
    )
    task()
    assert seen["config"].num_seeds == 3
    assert [source.garage_scene_filter.split_names for source in seen["config"].offline_data_sources] == [
        source.garage_scene_filter.split_names for source in latent_transfuser_pretrain().offline_data_sources
    ]
    assert seen["run_dir"] == tmp_path
    assert (tmp_path / "hydra" / "overrides.yaml").is_file()
    assert Path.cwd() != tmp_path
