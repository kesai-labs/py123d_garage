# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Scores a policy's open-loop displacement errors on offline data sources."""

from __future__ import annotations

import faulthandler
import logging
import os
from dataclasses import replace
from typing import cast

import hydra
import numpy as np
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.geometry.geometry_index import PoseSE2Index

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize_evaluation,
    hydra_overrides,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import setup_logging
from py123d_garage.config.presets.offline_data_sources.navsim import navtest
from py123d_garage.config.schema.evaluation.open_loop_config import OpenLoopBenchmarkConfig
from py123d_garage.config.schema.evaluation.parallel_offline_evaluation_config import ParallelizationConfig
from py123d_garage.datatypes.numerics import PositiveInt
from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.evaluation.help import (
    build_source_scenes,
    expected_num_scenes,
    merge_results_if_last_shard,
    run_shard_inference,
    save_shard_results,
)
from py123d_garage.evaluation.help.policy_building import build_evaluation_policy
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor
from py123d_garage.py123d_help.scene_readers.ego_state import sample_ego_se2

LOG = logging.getLogger(__name__)


register_schema("evaluate_open_loop", OpenLoopBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_open_loop", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()

    # Print the C-level stack trace when a worker dies on a fatal signal.
    faulthandler.enable()

    benchmark_config: OpenLoopBenchmarkConfig = finalize_evaluation(cfg, OpenLoopBenchmarkConfig, hydra_overrides())
    output_dir = run_dir()

    py123d_garage_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )

    # Falls back to navtest so the entry point runs without any config.
    if not benchmark_config.benchmark_offline_data_sources:
        benchmark_config = replace(
            benchmark_config,
            benchmark_offline_data_sources=[navtest(py123d_garage_policy.policy_config, benchmark_config)],
        )

    parallelization: ParallelizationConfig = benchmark_config.parallelization_config
    if not 0 <= parallelization.shard_index < parallelization.num_shards:
        raise ValueError(
            f"shard_index {parallelization.shard_index} outside [0, {parallelization.num_shards})",
        )

    # One writer only: every shard shares the output dir.
    if parallelization.shard_index == 0:
        save_config(benchmark_config)
    LOG.info(
        f"Path where all results are stored: {output_dir!s}",
    )

    py123d_garage_policy.verify_contract(benchmark_config=benchmark_config)

    max_workers = parallelization.max_workers
    if max_workers is None:
        # os.cpu_count sees the whole node; the affinity set is the allocation.
        max_workers = len(os.sched_getaffinity(0))
    executor = VerboseProcessPoolExecutor(max_workers=max_workers)

    # An un-initialized copy, with no weights on the device, builds the features in the loader workers.
    py123d_garage_feature_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )

    py123d_garage_policy = build_evaluation_policy(benchmark_config.policy_config)
    py123d_garage_policy.to(parallelization.device)

    results: list[dict[str, object]] = []
    for source in benchmark_config.benchmark_offline_data_sources:
        scenes = build_source_scenes(
            source,
            parallelization.shard_index,
            parallelization.num_shards,
            executor,
            py123d_garage_policy,
            benchmark_config,
        )
        LOG.info("Running Inference")
        trajectories = run_shard_inference(
            scenes,
            py123d_garage_policy,
            py123d_garage_feature_policy,
            source.cache_root,
            parallelization.inference_batch_size,
            max_workers,
            output_dir / "visualizations" if benchmark_config.save_visualizations else None,
        )
        LOG.info("Running Scoring")
        results.extend(_score_scenes(scenes, trajectories, py123d_garage_policy.policy_config))

    save_shard_results(results, str(output_dir), parallelization.shard_index)
    merge_results_if_last_shard(
        str(output_dir),
        parallelization.num_shards,
        parallelization.shard_index,
        expected_num_scenes(benchmark_config.benchmark_offline_data_sources),
    )


def _score_scenes(
    scenes: list[SceneAPI],
    trajectories: list[TrajectorySE2],
    policy_config: AbstractPolicyConfig,
) -> list[dict[str, object]]:
    """
    Computes the average and final displacement error of every scene.

    Args:
        scenes: the shard's scenes, in inference order.
        trajectories: the predicted trajectory of each scene, ego-relative.
        policy_config: provides the trajectory grid the errors are scored on.

    Returns:
        one metric dict per scene.
    """
    results: list[dict[str, object]] = []
    for scene, trajectory in zip(scenes, trajectories, strict=True):
        result: dict[str, object] = {"scene_uuid": scene.scene_uuid}
        try:
            result.update(
                _compute_displacement_errors(
                    scene,
                    trajectory,
                    policy_config.trajectory_offsets_us,
                ),
            )
        except Exception as error:
            result["scoring_error"] = f"{type(error).__name__}: {error}"
            LOG.warning(
                f"scene {scene.scene_uuid} of log {scene.log_name} is unscorable: {error}",
            )
        results.append(result)
    return results


def _compute_displacement_errors(
    scene: SceneAPI,
    trajectory: TrajectorySE2,
    offsets_us: list[PositiveInt],
) -> dict[str, float]:
    """
    Compares one predicted trajectory against the logged ego poses on the scoring grid.

    Args:
        scene: the scene the trajectory was predicted for, anchored at its current frame.
        trajectory: the predicted trajectory, relative to the anchor ego pose.
        offsets_us: times past the anchor the scoring grid holds a pose at.

    Returns:
        the average and the final displacement error, in meters.
    """
    ground_truth = sample_ego_se2(
        scene_api=scene,
        offsets_us=offsets_us,
        relative_to_anchor=True,
    )
    ego_state_se3 = scene.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for displacement scoring!"
    # The first predicted pose is one policy step past the anchor; prepending the anchor origin covers finer grids.
    anchored_prediction = TrajectorySE2(
        pose_se2_array=np.vstack(
            [np.zeros((1, len(PoseSE2Index)), dtype=np.float64), trajectory.pose_se2_array],
        ),
        timestamps_us=np.concatenate(
            [np.array([ego_state_se3.timestamp.time_us], dtype=np.int64), trajectory.timestamps_us],
        ),
    )
    predicted_se2_array = anchored_prediction.interpolate(ground_truth.timestamps_us)
    position_columns = [PoseSE2Index.X, PoseSE2Index.Y]
    displacements_m = np.linalg.norm(
        predicted_se2_array[:, position_columns] - ground_truth.pose_se2_array[:, position_columns],
        axis=1,
    )
    return {
        "average_displacement_error_m": float(displacements_m.mean()),
        "final_displacement_error_m": float(displacements_m[-1]),
    }


if __name__ == "__main__":
    main()
