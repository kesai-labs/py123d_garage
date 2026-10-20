# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig
from py123d_garage.evaluation.help import expected_num_scenes, merge_results_if_last_shard, save_shard_results


def _source(count: int | None) -> OfflineEvaluationDataSourceConfig:
    return OfflineEvaluationDataSourceConfig(
        data_root="/data/nuplan/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=None,
        expected_num_scenes=count,
    )


def _merge(tmp_path: Path, rows: list[dict[str, object]], expected: int | None) -> pd.DataFrame:
    save_shard_results(rows, str(tmp_path), 0)
    merge_results_if_last_shard(str(tmp_path), 1, 0, expected)
    return pd.read_csv(tmp_path / "results.csv")


def test_expected_num_scenes_sums_the_declared_counts() -> None:
    assert expected_num_scenes([_source(2), _source(3)]) == 5
    assert expected_num_scenes([_source(2), _source(None)]) is None
    assert expected_num_scenes([]) is None


def test_merge_passes_the_declared_count(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"scene_uuid": "a", "pdm_score": 1.0}, {"scene_uuid": "b", "pdm_score": 0.5}]
    results = _merge(tmp_path, rows, 2)
    average = results[results["scene_uuid"] == "average"].iloc[0]
    assert average["pdm_score"] == pytest.approx(0.75)


def test_merge_fails_on_a_wrong_scene_count_after_writing(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"scene_uuid": "a", "pdm_score": 1.0}, {"scene_uuid": "b", "pdm_score": 0.5}]
    with pytest.raises(RuntimeError, match="expected_num_scenes=3"):
        _merge(tmp_path, rows, 3)
    assert (tmp_path / "results.csv").is_file()


def test_merge_fails_on_an_unscorable_scene_after_writing(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"scene_uuid": "a", "pdm_score": 1.0},
        {"scene_uuid": "b", "scoring_error": "ValueError: no route"},
    ]
    with pytest.raises(RuntimeError, match="1/2 scenes are unscorable"):
        _merge(tmp_path, rows, None)
    results = pd.read_csv(tmp_path / "results.csv")
    average = results[results["scene_uuid"] == "average"].iloc[0]
    assert average["pdm_score"] == pytest.approx(1.0)
