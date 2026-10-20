# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The shard machinery offline evaluation protocols share: scene building, inference, and the results table."""

from __future__ import annotations

from py123d_garage.evaluation.help.scene_inference import (
    OfflineEvaluationDataset,
    build_source_scenes,
    run_shard_inference,
)
from py123d_garage.evaluation.help.sharding_help import (
    expected_num_scenes,
    merge_results_if_last_shard,
    save_shard_results,
)

__all__ = [
    "OfflineEvaluationDataset",
    "build_source_scenes",
    "expected_num_scenes",
    "merge_results_if_last_shard",
    "run_shard_inference",
    "save_shard_results",
]
