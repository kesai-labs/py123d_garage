# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Fill a training run's session cache with the session tensors of every sample, so its first epoch reads them.

A training forks start_session_cache_fillers right after its dataset build. Run as a module, it fills the session cache
of the run whose dir holds its own run dir ahead of that run, e.g. the next stage's while a stage trains:

    python -m py123d_garage.training.fill_session_cache <the run's overrides> hydra.run.dir=<run dir>/session_cache_fill
"""

from __future__ import annotations

import ctypes
import gc
import logging
import multiprocessing
import os
from collections.abc import Sequence
from time import perf_counter
from typing import cast

import cv2
import hydra
import lightning as L
import numba
import torch
from omegaconf import DictConfig

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.common.config_help import CONFIG_PATH, build_from_string, finalize, register_schema, run_dir
from py123d_garage.common.logging_setup import setup_logging
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.training.dataset import TrainingDataset

LOG = logging.getLogger(__name__)

# glibc's mallopt parameter for the arena cap.
_M_ARENA_MAX = -8


def single_threaded(helper_cpus: set[int] | None) -> None:
    """
    Keep a loader-side process to one thread per library and one glibc malloc arena; the parallelism is across processes.

    Args:
        helper_cpus: the CPUs to pin the process to; None keeps its CPU set
    """
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    numba.set_num_threads(1)  # pyright: ignore[reportUnknownMemberType]
    # Extra arenas for the worker's helper threads fragment and stay resident in every one of the ~150 workers.
    ctypes.CDLL("libc.so.6").mallopt(_M_ARENA_MAX, 1)
    if helper_cpus is not None:
        os.sched_setaffinity(0, helper_cpus)


def fill_session_cache_chunk(
    datasets: Sequence[TrainingDataset],
    chunk_index: int,
    num_chunks: int,
    helper_cpus: set[int] | None,
) -> None:
    """
    Fill the session cache with one chunk of every dataset, at the lowest priority.

    Args:
        datasets: every source's dataset
        chunk_index: this process's chunk of each dataset
        num_chunks: chunks per dataset across the node
        helper_cpus: the CPUs to pin the process to; None keeps its CPU set
    """
    os.nice(19)
    single_threaded(helper_cpus)
    start = perf_counter()
    built_samples = 0
    # Contiguous chunks keep a log's samples in one process, which then opens each log once.
    for dataset in datasets:
        built_samples += dataset.fill_session_cache(
            len(dataset) * chunk_index // num_chunks,
            len(dataset) * (chunk_index + 1) // num_chunks,
        )
    if chunk_index == 0:
        LOG.info(
            f"Session cache: the first filler built {built_samples} samples in {(perf_counter() - start) / 60:.1f} min",
        )


def start_session_cache_fillers(
    datasets: Sequence[TrainingDataset],
    num_processes: int,
    helper_cpus: set[int] | None,
    local_rank: int,
    ranks_per_node: int,
) -> None:
    """
    Fork processes that fill the session cache with this rank's share of the node's samples ahead of the loader.

    The processes run at the lowest priority, so they yield to the loader.

    Args:
        datasets: every source's dataset
        num_processes: filler processes on this rank
        helper_cpus: the CPUs to pin the processes to; None keeps their CPU set
        local_rank: this process's rank on its node
        ranks_per_node: the ranks that share the node
    """
    context = multiprocessing.get_context("fork")
    for process_index in range(num_processes):
        context.Process(
            target=fill_session_cache_chunk,
            args=(datasets, local_rank * num_processes + process_index, ranks_per_node * num_processes, helper_cpus),
            daemon=True,
        ).start()
    LOG.info(f"Session cache: {num_processes} filler processes started on this rank")


# The training registers the same schema; the fill composes it as its own entry point too.
register_schema("train", TrainingConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="fill_session_cache", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Fill the session cache of the run whose dir holds this one's, with session_cache_fill_processes processes.

    Args:
        cfg: the training config of that run
    """
    # The training module imports this one, so it is imported here.
    from py123d_garage.training.train import build_source_datasets, launcher_ranks

    setup_logging()
    training_config: TrainingConfig = finalize(cfg, TrainingConfig)
    if not (training_config.session_cache and training_config.cache_mirror_root):
        LOG.info("No session cache without session_cache and cache_mirror_root: nothing to fill")
        return
    target_run_dir = run_dir().parent
    L.seed_everything(0)
    policy = cast(AnyPolicy, build_from_string(training_config.policy_config, AbstractPolicy))
    policy.initialize()
    datasets = build_source_datasets(training_config, policy, target_run_dir, None, None, launcher_ranks(1, 1))
    # The forked fillers would otherwise copy the pages every collection touches.
    gc.collect()
    gc.freeze()
    num_processes = training_config.session_cache_fill_processes
    start = perf_counter()
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=fill_session_cache_chunk, args=(datasets, index, num_processes, None))
        for index in range(num_processes)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    assert not failed, f"{len(failed)} of {num_processes} fillers failed with exit codes {failed}"
    LOG.info(f"Filled the session cache of {target_run_dir} in {(perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
