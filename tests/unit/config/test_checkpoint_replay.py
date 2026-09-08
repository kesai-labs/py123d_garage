from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from py123d_garage.common.config_help import load_evaluation_config
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.config.schema.evaluation.carla_config import CarlaBenchmarkConfig

_SAVED_CONFIG = """
policy_config:
  target: py123d_garage.policy.vavam.vavam_policy:VavamPolicy
  vavam_config:
    context_length: 4
"""


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> str:
    (tmp_path / "config.yaml").write_text(_SAVED_CONFIG)
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    return str(checkpoint)


def _carla_args(checkpoint_file: str) -> list[str]:
    return [
        f"policy_config.evaluation_checkpoint_file={checkpoint_file}",
        "policy_config.evaluation_sensor_rig_file=rig.yaml",
        "routes_file=routes.xml",
    ]


def test_replays_the_saved_policy_config(checkpoint_file: str) -> None:
    config = load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=_carla_args(checkpoint_file))
    assert config.policy_config.target == "py123d_garage.policy.vavam.vavam_policy:VavamPolicy"
    assert config.policy_config.vavam_config is not None
    assert config.policy_config.vavam_config.context_length == 4
    assert config.policy_config.transfuser_config is None
    assert config.policy_config.evaluation_checkpoint_file == checkpoint_file


def test_override_beats_the_saved_config(checkpoint_file: str) -> None:
    config = load_evaluation_config(
        CarlaBenchmarkConfig,
        "evaluate_carla",
        args=[*_carla_args(checkpoint_file), "+policy_config.vavam_config.context_length=1"],
    )
    assert config.policy_config.vavam_config is not None
    assert config.policy_config.vavam_config.context_length == 1


def test_works_for_every_benchmark_class(checkpoint_file: str) -> None:
    config = load_evaluation_config(
        AlpasimBenchmarkConfig,
        "evaluate_alpasim",
        args=[
            f"policy_config.evaluation_checkpoint_file={checkpoint_file}",
            "policy_config.evaluation_sensor_rig_file=rig.yaml",
        ],
    )
    assert config.policy_config.vavam_config is not None
    assert config.policy_config.vavam_config.context_length == 4


def test_missing_checkpoint_override_raises() -> None:
    with pytest.raises(ValueError, match="evaluation_checkpoint_file"):
        load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=[])


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


def test_replays_an_empty_saved_block(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "policy_config:\n  target: py123d_garage.policy.vavam.vavam_policy:VavamPolicy\n  vavam_config: {}\n",
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    config = load_evaluation_config(CarlaBenchmarkConfig, "evaluate_carla", args=_carla_args(str(checkpoint)))
    assert config.policy_config.vavam_config is not None
    assert config.policy_config.transfuser_config is None
