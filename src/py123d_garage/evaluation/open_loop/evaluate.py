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
from py123d_garage.config.presets.python.offline_data_sources.navsim import navtest
from py123d_garage.config.schema.evaluation.open_loop_config import OpenLoopBenchmarkConfig
from py123d_garage.config.schema.evaluation.parallel_offline_evaluation_config import ParallelizationConfig
from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.evaluation.help import (
    build_source_scenes,
    merge_results_if_last_shard,
    run_shard_inference,
    save_shard_results,
)
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor
from py123d_garage.py123d_help.scene_readers.ego_state import sample_ego_se2

LOG = logging.getLogger(__name__)


register_schema("evaluate_open_loop", OpenLoopBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_open_loop", version_base=None)
def main(cfg: DictConfig) -> None:
    # Setup logging
    setup_logging()

    # Print the C-level stack trace when a worker dies on a fatal signal.
    faulthandler.enable()

    # Compose the benchmark config from the Hydra config and command line overrides.
    benchmark_config: OpenLoopBenchmarkConfig = finalize_evaluation(cfg, OpenLoopBenchmarkConfig, hydra_overrides())
    output_dir = run_dir()

    # Build the policy from the config
    py123d_garage_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )

    # Evaluate default on the navtest source if no sources are provided,
    # so the entry point can be run without any config.
    if not benchmark_config.benchmark_offline_data_sources:
        benchmark_config = replace(
            benchmark_config,
            benchmark_offline_data_sources={"navtest": navtest(py123d_garage_policy.policy_config)},
        )

    # Sanity check sharding config
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

    # Verify the policy's config is compatible with the benchmark's sources.
    py123d_garage_policy.verify_contract(benchmark_config=benchmark_config)
    scored_horizon_us = py123d_garage_policy.policy_config.trajectory_horizon_us
    for scored_source in benchmark_config.benchmark_offline_data_sources.values():
        declared_s = scored_source.garage_scene_filter.future_duration_s
        if declared_s is None or round(declared_s * 1e6) < scored_horizon_us:
            raise ValueError(
                f"the source at '{scored_source.data_root}' authors future_duration_s={declared_s} "
                f"but the errors cover {scored_horizon_us} µs past each anchor.",
            )

    max_workers = parallelization.max_workers
    if max_workers is None:
        # os.cpu_count sees the whole node; the affinity set is the allocation.
        max_workers = len(os.sched_getaffinity(0))
    executor = VerboseProcessPoolExecutor(max_workers=max_workers)

    # Un-initialized copy for the loader workers.
    # This policy object does not have Torch tensors on the device, and is not used for inference.
    py123d_garage_feature_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )

    # Initialize the inference policy's weight and move it to the device, so the workers can use it for inference.
    py123d_garage_policy.initialize(
        benchmark_config.policy_config.evaluation_checkpoint_file,
    )
    py123d_garage_policy.to(parallelization.device)

    # Run the inferences and score the results, one shard at a time.
    # Each worker loads its own scenes and runs inference on them.
    results: list[dict[str, object]] = []
    for source in benchmark_config.benchmark_offline_data_sources.values():
        scenes = build_source_scenes(
            source,
            parallelization.shard_index,
            parallelization.num_shards,
            executor,
            py123d_garage_policy,
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

    # Save the shard's results and merge them if this is the last shard to finish.
    save_shard_results(results, str(output_dir), parallelization.shard_index)
    merge_results_if_last_shard(
        str(output_dir),
        parallelization.num_shards,
        parallelization.shard_index,
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
                    policy_config.trajectory_num_steps,
                    policy_config.trajectory_interval_us,
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
    num_steps: int,
    interval_us: int,
) -> dict[str, float]:
    """
    Compares one predicted trajectory against the logged ego poses on the scoring grid.

    Args:
        scene: the scene the trajectory was predicted for, anchored at its current frame.
        trajectory: the predicted trajectory, relative to the anchor ego pose.
        num_steps: how many poses the scoring grid holds.
        interval_us: spacing of the scoring grid.

    Returns:
        the average and the final displacement error, in meters.
    """
    ground_truth = sample_ego_se2(
        scene_api=scene,
        num_steps=num_steps,
        interval_us=interval_us,
        relative_to_anchor=True,
    )
    ego_state_se3 = scene.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for displacement scoring!"
    # The prediction's first pose sits one policy step past the anchor, so a finer
    # scoring grid would query before it; in its own frame the ego is at the origin
    # at the anchor, and prepending that pose makes the whole grid interpolable.
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
