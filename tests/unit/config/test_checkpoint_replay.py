# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from py123d_garage.common.config_help import load_evaluation_config
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.config.schema.evaluation.carla_config import CarlaBenchmarkConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserVisualizationConfig

_SAVED_CONFIG = """
policy_config:
  target: py123d_garage.policy.transfuser.transfuser_policy:TransfuserPolicy
  transfuser_config:
    trajectory_offsets_us: [500000]
    required_target_point_distances_m: [20.0]
    planning_config:
      transfuser_token_dim: 128
"""


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> str:
    (tmp_path / "config.yaml").write_text(_SAVED_CONFIG)
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    return str(checkpoint)


def _carla_args(checkpoint_file: str) -> list[str]:
    return [
        f"policy_config.evaluation_checkpoint_files=[{checkpoint_file}]",
        "policy_config.evaluation_sensor_rig_file=rig.yaml",
        "routes_file=routes.xml",
    ]


def test_replays_the_saved_policy_config(checkpoint_file: str) -> None:
    config = load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=_carla_args(checkpoint_file))
    assert config.policy_config.target == "py123d_garage.policy.transfuser.transfuser_policy:TransfuserPolicy"
    assert config.policy_config.transfuser_config is not None
    assert config.policy_config.transfuser_config.planning_config.transfuser_token_dim == 128
    assert config.policy_config.evaluation_checkpoint_files == [checkpoint_file]


def test_override_beats_the_saved_config(checkpoint_file: str) -> None:
    config = load_evaluation_config(
        CarlaBenchmarkConfig,
        "evaluate_carla",
        args=[
            *_carla_args(checkpoint_file),
            "+policy_config.transfuser_config.planning_config.transfuser_token_dim=64",
        ],
    )
    assert config.policy_config.transfuser_config is not None
    assert config.policy_config.transfuser_config.planning_config.transfuser_token_dim == 64


def test_works_for_every_benchmark_class(checkpoint_file: str) -> None:
    config = load_evaluation_config(
        AlpasimBenchmarkConfig,
        "evaluate_alpasim",
        args=[
            f"policy_config.evaluation_checkpoint_files=[{checkpoint_file}]",
            "policy_config.evaluation_sensor_rig_file=rig.yaml",
        ],
    )
    assert config.policy_config.transfuser_config is not None
    assert config.policy_config.transfuser_config.planning_config.transfuser_token_dim == 128


def test_missing_checkpoint_override_raises() -> None:
    with pytest.raises(ValueError, match="evaluation_checkpoint_files"):
        load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=[])


def test_replays_an_ensemble_of_one_recipe(checkpoint_file: str, tmp_path: Path) -> None:
    second = tmp_path / "second.pth"
    second.touch()
    config = load_evaluation_config(
        CarlaBenchmarkConfig,
        "evaluate_carla",
        args=[
            f"policy_config.evaluation_checkpoint_files=[{checkpoint_file},{second}]",
            "policy_config.evaluation_sensor_rig_file=rig.yaml",
            "routes_file=routes.xml",
        ],
    )
    assert config.policy_config.evaluation_checkpoint_files == [checkpoint_file, str(second)]
    assert config.policy_config.transfuser_config is not None


def test_mixed_recipes_are_rejected(checkpoint_file: str, tmp_path: Path) -> None:
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "config.yaml").write_text(
        _SAVED_CONFIG.replace("transfuser_token_dim: 128", "transfuser_token_dim: 64"),
    )
    other = other_dir / "model.pth"
    other.touch()
    with pytest.raises(ValueError, match="different policy_config"):
        load_evaluation_config(
            CarlaBenchmarkConfig,
            "evaluate_carla",
            args=[
                f"policy_config.evaluation_checkpoint_files=[{checkpoint_file},{other}]",
                "policy_config.evaluation_sensor_rig_file=rig.yaml",
                "routes_file=routes.xml",
            ],
        )


def test_missing_saved_config_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match=r"no config\.yaml"):
        load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=_carla_args(str(checkpoint)))


def test_training_plan_is_rejected(checkpoint_file: str) -> None:
    with pytest.raises(ConfigCompositionException, match="training"):
        load_evaluation_config(
            CarlaBenchmarkConfig,
            "evaluate_carla",
            args=[*_carla_args(checkpoint_file), "training=some.module:factory"],
        )


def test_drawing_styles_come_from_the_code(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        _SAVED_CONFIG + "    visualization_config:\n      prediction_trajectory_style:\n        pose_radius_pixel: 9\n",
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    config = load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=_carla_args(str(checkpoint)))
    assert config.policy_config.transfuser_config is not None
    assert config.policy_config.transfuser_config.planning_config.transfuser_token_dim == 128
    assert config.policy_config.transfuser_config.visualization_config == TransfuserVisualizationConfig()


def test_replays_an_empty_saved_block(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "policy_config:\n  target: py123d_garage.policy.transfuser.transfuser_policy:TransfuserPolicy\n  transfuser_config: {}\n",
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    config = load_evaluation_config(
        CarlaBenchmarkConfig,
        "evaluate_carla",
        args=[
            *_carla_args(str(checkpoint)),
            "+policy_config.transfuser_config.trajectory_offsets_us=[500000]",
            "+policy_config.transfuser_config.required_target_point_distances_m=[20.0]",
        ],
    )
    assert config.policy_config.transfuser_config is not None
