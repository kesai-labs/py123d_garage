from __future__ import annotations

import faulthandler
import logging
import os
from dataclasses import dataclass, replace
from typing import cast

import hydra
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.common.execution import executor_map_chunked_list

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
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
from py123d_garage.config.schema.evaluation.navsim_config import NavsimBenchmarkConfig
from py123d_garage.config.schema.evaluation.parallel_offline_evaluation_config import ParallelizationConfig
from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.evaluation.help import (
    build_source_scenes,
    merge_results_if_last_shard,
    run_shard_inference,
    save_shard_results,
)
from py123d_garage.evaluation.navsim.help.pdm_metric import PDMMetric
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor

LOG = logging.getLogger(__name__)


register_schema("evaluate_navsim", NavsimBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_navsim", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for evaluating an agent on one shard of the split."""
    setup_logging()
    # Print the C-level stack trace when a worker dies on a fatal signal.
    faulthandler.enable()
    benchmark_config: NavsimBenchmarkConfig = finalize_evaluation(cfg, NavsimBenchmarkConfig, hydra_overrides())
    output_dir = run_dir()
    py123d_garage_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )
    if not benchmark_config.benchmark_offline_data_sources:
        benchmark_config = replace(
            benchmark_config,
            benchmark_offline_data_sources={"navtest": navtest(py123d_garage_policy.policy_config)},
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
    for scored_source in benchmark_config.benchmark_offline_data_sources.values():
        declared_s = scored_source.garage_scene_filter.future_duration_s
        if declared_s is None or round(declared_s * 1e6) < benchmark_config.navsim_scoring_future_duration_us:
            raise ValueError(
                f"the source at '{scored_source.data_root}' authors future_duration_s={declared_s} "
                f"but navsim scores {benchmark_config.navsim_scoring_future_duration_us} µs past each anchor.",
            )

    max_workers = parallelization.max_workers
    if max_workers is None:
        # os.cpu_count sees the whole node; the affinity set is the allocation.
        max_workers = len(os.sched_getaffinity(0))
    executor = VerboseProcessPoolExecutor(max_workers=max_workers)

    # Un-initialized copy for the loader workers: no weights, no device state.
    py123d_garage_feature_policy = cast(
        AnyPolicy,
        build_from_string(benchmark_config.policy_config, AbstractPolicy),
    )
    py123d_garage_policy.initialize(
        benchmark_config.policy_config.evaluation_checkpoint_file,
    )
    py123d_garage_policy.to(parallelization.device)

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
        results.extend(
            executor_map_chunked_list(
                executor,
                _score_scenes,
                [
                    _SceneTrajectoryPair(scene, trajectory)
                    for scene, trajectory in zip(scenes, trajectories, strict=True)
                ],
                name="Scoring",
            ),
        )

    save_shard_results(results, str(output_dir), parallelization.shard_index)
    merge_results_if_last_shard(
        str(output_dir),
        parallelization.num_shards,
        parallelization.shard_index,
    )


@dataclass(frozen=True)
class _SceneTrajectoryPair:
    scene: SceneAPI
    trajectory: TrajectorySE2


def _score_scenes(
    scene_trajectory_pairs: list[_SceneTrajectoryPair],
) -> list[dict[str, object]]:
    """
    Runs the PDM metric for a chunk of scene-trajectory pairs. Built per worker.

    A scene the metric cannot score keeps its row with the error and no metric
    values: one unscorable scene must not abort a shard of thousands, and the
    empty values drop out of the average instead of biasing it.
    """
    metric = PDMMetric()
    results: list[dict[str, object]] = []
    for pair in scene_trajectory_pairs:
        scene, trajectory = pair.scene, pair.trajectory
        result: dict[str, object] = {"scene_uuid": scene.scene_uuid}
        try:
            result.update(metric.compute_metric(scene, agent_trajectory=trajectory))
        except Exception as error:
            result["scoring_error"] = f"{type(error).__name__}: {error}"
            LOG.warning(
                f"scene {scene.scene_uuid} of log {scene.log_name} is unscorable: {error}",
            )
        results.append(result)
    return results


if __name__ == "__main__":
    main()
