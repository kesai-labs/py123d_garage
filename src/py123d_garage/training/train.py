# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Training entry point that builds the policies, datasets and trainer, then fits one policy per seed."""

from __future__ import annotations

import bisect
import faulthandler
import gc
import hashlib
import logging
import math
import multiprocessing
import os
import pickle
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, cast

import hydra
import jaxtyping as jt
import lightning as L
import numpy as np
import numpy.typing as npt
import torch
import torch._inductor.config
import wandb
import yaml
from diskcache import FanoutCache
from lightning.fabric.utilities.seed import pl_worker_init_function
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loops.progress import _ProcessedTracker  # pyright: ignore[reportPrivateUsage]
from lightning.pytorch.overrides.distributed import _sync_module_states  # pyright: ignore[reportPrivateUsage]
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,  # pyright: ignore[reportPrivateUsage]
)
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from torch.nn.attention import SDPBackend
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from typing_extensions import override

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.contract_verifications import verify_offline_data_source_scenes_declared_intervals
from py123d_garage.cache import CacheStoreReader
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import log_stage, setup_logging
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor, build_scene_builder
from py123d_garage.training.callbacks import (
    BatchesProgressBar,
    EpochLoggingCallback,
    FreezeHeapCallback,
    PinMainProcessCallback,
    RecycleWorkersCallback,
    ShutdownCompileWorkersCallback,
    StepTraceCallback,
    StepWatchdogCallback,
    ThroughputLoggingCallback,
    VisualizationCallback,
)
from py123d_garage.training.checkpointing import merged_resume_checkpoint
from py123d_garage.training.compile_ahead import (
    CompileAhead,
    FirstStepCompileCheckCallback,
    example_batch,
)
from py123d_garage.training.dataset import TrainingDataset, TrainingSample
from py123d_garage.training.fill_session_cache import single_threaded, start_session_cache_fillers
from py123d_garage.training.kernel_verification import kernel_patches_of
from py123d_garage.training.lightning_module import PolicyLightningModule
from py123d_garage.training.pinned_batches import PinnedBatchSlots, ReleaseBatchSlotsCallback

LOG = logging.getLogger(__name__)

# Lookup threads per loader worker for the batch prefetch; 16 closed 2/3 of the cold-read gap at 16 workers.
_PREFETCH_THREADS = 16

# Splits an evaluation suite scores: training on them contaminates the scores.
HELD_OUT_SPLITS = frozenset({"nurec_val", "nuplan_test"})
# NuRec reconstructs physical-ai-av clips under the same uuid, so those splits hold the nurec suite's clips.
NUREC_SOURCE_SPLIT_PREFIX = "physical-ai-av"
# The anchor filter that drops the clips NuRec reconstructs.
NUREC_CLIP_FILTER = "py123d_garage.py123d_help.scene_filters:is_not_nurec_clip"


class _UnwrappedDDPStrategy(DDPStrategy):
    """DDP's process group, sampler and initial state sync without the module wrapper: the recorded step all-reduces itself."""

    @override
    def configure_ddp(self) -> None:
        """Sync the initial module states across ranks without wrapping the module."""
        assert self.model is not None
        _sync_module_states(self.model)


def _build_loggers(training_config: TrainingConfig) -> list[Logger]:
    """
    Build the trainer loggers.

    Args:
        training_config: the training config

    Returns:
        the W&B logger, or nothing when it is disabled
    """
    wandb_config = training_config.wandb_config
    if not wandb_config.enabled:
        return []
    return [
        WandbLogger(
            project=wandb_config.project,
            name=wandb_config.name,
            save_dir=str(run_dir()),
            log_model=wandb_config.log_model,
            entity=wandb_config.entity,
            group=wandb_config.group,
            tags=list(wandb_config.tags or []),
            mode=wandb_config.mode,
        ),
    ]


def _worker_init(worker_id: int, helper_cpus: set[int] | None = None) -> None:
    """
    Make a DataLoader worker single-threaded and seed it.

    Args:
        worker_id: the worker's index
        helper_cpus: the CPUs to pin the worker to; None keeps its CPU set
    """
    single_threaded(helper_cpus)
    # A custom worker_init_fn stops Lightning from installing this itself.
    pl_worker_init_function(worker_id)


def _parse_cpu_list(text: str) -> set[int]:
    """
    Parse a sysfs CPU list such as ``0-3,72``.

    Args:
        text: the list's text

    Returns:
        the listed CPUs
    """
    cpus: set[int] = set()
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


@dataclass(frozen=True)
class Ranks:
    """This process's place among the job's ranks."""

    global_rank: int
    world_size: int
    local_rank: int
    ranks_per_node: int


def launcher_ranks(devices_per_node: int, num_nodes: int) -> Ranks:
    """
    Read this process's ranks from the environment of the launcher that started it.

    srun and torchrun each export the ranks under their own names. Lightning's own launcher exports the local and node
    rank to the processes it starts and nothing to the one it was started from, which is rank 0.

    Args:
        devices_per_node: the ranks per node Lightning's launcher starts
        num_nodes: the nodes of the job

    Returns:
        the ranks
    """
    environment = os.environ
    if "SLURM_PROCID" in environment:
        return Ranks(
            int(environment["SLURM_PROCID"]),
            int(environment.get("SLURM_NTASKS", "1")),
            int(environment.get("SLURM_LOCALID", "0")),
            int(environment.get("SLURM_NTASKS_PER_NODE", "1")),
        )
    if "RANK" in environment:
        return Ranks(
            int(environment["RANK"]),
            int(environment["WORLD_SIZE"]),
            int(environment["LOCAL_RANK"]),
            int(environment["LOCAL_WORLD_SIZE"]),
        )
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    node_rank = int(environment.get("NODE_RANK", "0"))
    return Ranks(node_rank * devices_per_node + local_rank, devices_per_node * num_nodes, local_rank, devices_per_node)


def _cpu_partition(ranks: Ranks) -> tuple[tuple[set[int], set[int]], set[int]] | None:
    """
    Split this rank's CPUs into two physical cores for the training process and the rest for its helpers.

    The training thread and the loader's pin-memory thread both stay busy, so each gets a core of its own. The main
    cores are the highest-numbered ones, away from CPU 0 where the kernel's interrupt work lands. A launcher that leaves
    every rank the whole node, unlike srun with CPU binding, gets the node's cores cut into one slice per local rank.

    Args:
        ranks: this process's place among the job's ranks

    Returns:
        The training thread's and the pin-memory thread's core CPUs and the helpers' CPUs, or None below four physical
        cores or without sysfs topology.
    """
    cpus = os.sched_getaffinity(0)
    cores: dict[int, set[int]] = {}
    try:
        for cpu in cpus:
            siblings_file = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            siblings = _parse_cpu_list(siblings_file.read_text()) & cpus
            cores[min(siblings)] = siblings
    except OSError:
        return None
    core_ids = sorted(cores)
    if ranks.ranks_per_node > 1 and cpus == _all_node_cpus():
        first = len(core_ids) * ranks.local_rank // ranks.ranks_per_node
        last = len(core_ids) * (ranks.local_rank + 1) // ranks.ranks_per_node
        core_ids = core_ids[first:last]
        cpus = {cpu for core_id in core_ids for cpu in cores[core_id]}
    if len(core_ids) < 4:
        return None
    training_core, pin_core = (cores[core_id] for core_id in core_ids[-2:])
    return (training_core, pin_core), cpus - training_core - pin_core


def _scene_share_dir(output_dir: Path) -> Path | None:
    """
    Find the run-dir directory the ranks of this job pass filtered scenes through.

    Named by the job, so a resumed run never reads a previous attempt's files.

    Args:
        output_dir: the run's output directory

    Returns:
        the directory, or None outside Slurm
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    return output_dir / f"scene_filter_{job_id}" if job_id else None


def _all_node_cpus() -> set[int]:
    """
    List every CPU of the node, regardless of this process's CPU set.

    Returns:
        the node's CPUs
    """
    return set(range(os.cpu_count() or 1))


def _scene_filter_executor(
    share_dir: Path | None,
    scene_filter_workers: int,
    ranks: Ranks,
) -> VerboseProcessPoolExecutor | None:
    """
    Create the process pool every source's scene filter runs on, over the whole node on rank 0.

    All sources share one pool, since each worker's import of the training module takes about a minute.

    Args:
        share_dir: the directory scenes are shared through; None filters on every process
        scene_filter_workers: worker processes per rank
        ranks: this process's place among the job's ranks

    Returns:
        the pool, or None on the ranks that wait for rank 0
    """
    if share_dir is None:
        return VerboseProcessPoolExecutor(max_workers=scene_filter_workers)
    if ranks.global_rank != 0:
        return None
    node_cpus = _all_node_cpus()
    return VerboseProcessPoolExecutor(
        max_workers=min(scene_filter_workers * ranks.ranks_per_node, len(node_cpus)),
        cpus=node_cpus,
    )


def _filter_scenes_once(
    share_dir: Path | None,
    source_index: int,
    wait_timeout_s: float,
    filter_scenes: Callable[[], Sequence[SceneAPI]],
    ranks: Ranks,
) -> Sequence[SceneAPI]:
    """
    Filter a source's scenes on rank 0 and let the other ranks load its result from the run dir.

    Args:
        share_dir: where the file goes; None filters in this process.
        source_index: the source's position, which names the shared file.
        wait_timeout_s: how long the other ranks wait for rank 0's file.
        filter_scenes: runs the filter.
        ranks: this process's place among the job's ranks.

    Returns:
        The source's selected scenes.
    """
    if share_dir is None:
        return filter_scenes()
    scenes_file = share_dir / f"scenes_{source_index}.pkl"
    if ranks.global_rank == 0:
        scenes = filter_scenes()
        share_dir.mkdir(parents=True, exist_ok=True)
        partial_file = scenes_file.with_suffix(".partial")
        with partial_file.open("wb") as file:
            pickle.dump(scenes, file, protocol=pickle.HIGHEST_PROTOCOL)
        partial_file.replace(scenes_file)
        LOG.info(
            f"Shared the {len(scenes)} scenes of source {source_index + 1} as {scenes_file} ({scenes_file.stat().st_size / 1e6:.0f} MB)",
        )
        return scenes
    _wait_for_file(scenes_file, wait_timeout_s, f"filter source {source_index + 1}")
    with scenes_file.open("rb") as file:
        scenes = cast(Sequence[SceneAPI], pickle.load(file))
    LOG.info(f"Loaded the {len(scenes)} scenes of source {source_index + 1} from {scenes_file}")
    return scenes


def _wait_for_file(path: Path, wait_timeout_s: float, task: str) -> None:
    """
    Wait until rank 0 has written a file.

    Args:
        path: the file to wait for
        wait_timeout_s: how long to wait before failing
        task: rank 0's task that writes the file, for the log
    """
    with log_stage(f"Waiting for rank 0 to {task}"):
        start = perf_counter()
        while not path.exists():
            if perf_counter() - start > wait_timeout_s:
                raise TimeoutError(f"{path} did not appear within {wait_timeout_s:.0f} s")
            time.sleep(2.0)


def _share_scene_uuids(
    share_dir: Path | None,
    source_index: int,
    scene_uuids: jt.UInt8[npt.NDArray[np.uint8], "scenes 16"] | None,
    wait_timeout_s: float,
    ranks: Ranks,
) -> jt.UInt8[npt.NDArray[np.uint8], "scenes 16"] | None:
    """
    Share rank 0's scene uuids of a source, read by its coverage check, with the ranks that skip the check.

    Args:
        share_dir: where the file goes; None shares nothing.
        source_index: the source's position, which names the shared file.
        scene_uuids: rank 0's uuids; ignored on the other ranks.
        wait_timeout_s: how long the other ranks wait for rank 0's file.
        ranks: this process's place among the job's ranks.

    Returns:
        the uuids; None when rank 0 has none.
    """
    if share_dir is None:
        return scene_uuids
    uuids_file = share_dir / f"scene_uuids_{source_index}.npy"
    if ranks.global_rank == 0:
        share_dir.mkdir(parents=True, exist_ok=True)
        partial_file = uuids_file.with_suffix(".partial")
        with partial_file.open("wb") as file:
            # An empty array stands for none, so the other ranks never wait in vain.
            np.save(file, scene_uuids if scene_uuids is not None else np.empty((0, 16), dtype=np.uint8))
        partial_file.replace(uuids_file)
        return scene_uuids
    _wait_for_file(uuids_file, wait_timeout_s, f"share the scene uuids of source {source_index + 1}")
    shared = cast("npt.NDArray[np.uint8]", np.load(uuids_file))
    return shared if len(shared) else None


def _filter_scenes_with_pool(
    scene_builder: ArrowSceneBuilder,
    scene_filter: SceneFilter,
    executor: VerboseProcessPoolExecutor | None,
) -> Sequence[SceneAPI]:
    """
    Filter a source's scenes on the shared process pool.

    Args:
        scene_builder: the source's scene builder
        scene_filter: the source's scene filter
        executor: the scene filter pool

    Returns:
        the selected scenes
    """
    assert executor is not None, "only the ranks with a scene filter pool filter"
    return scene_builder.get_scenes(filter=scene_filter, executor=executor)


def _filtered_scenes_memo(mirror_root: str, data_root: Path | str, scene_filter: SceneFilter) -> Path | None:
    """
    Name the node-local file that keeps a source's filtered scenes for the later runs of the job.

    The name hashes the pickled data root and filter, which name functions by module and qualified name rather than
    by address, so a run with any other filter filters again.

    Args:
        mirror_root: the job's node-local mirror root
        data_root: the source's data root
        scene_filter: the source's scene filter

    Returns:
        the memo file, or None for a filter that does not pickle, such as one with a lambda
    """
    try:
        pickled = pickle.dumps((str(data_root), scene_filter), protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, AttributeError, TypeError):
        return None
    return Path(mirror_root) / "filtered_scenes" / f"{hashlib.sha1(pickled).hexdigest()[:16]}.pkl"


def _memoized_scenes(memo_file: Path, filter_scenes: Callable[[], Sequence[SceneAPI]]) -> Sequence[SceneAPI]:
    """
    Load a source's scenes that an earlier run of the job filtered with the same filter, or filter and keep them.

    Args:
        memo_file: the memo, from _filtered_scenes_memo
        filter_scenes: runs the filter

    Returns:
        the selected scenes
    """
    if memo_file.exists():
        with memo_file.open("rb") as file:
            scenes = cast(Sequence[SceneAPI], pickle.load(file))
        LOG.info(f"Loaded the {len(scenes)} scenes an earlier run of the job filtered from {memo_file}")
        return scenes
    scenes = filter_scenes()
    memo_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = memo_file.with_suffix(f".{os.getpid()}.partial")
    with partial_file.open("wb") as file:
        pickle.dump(scenes, file, protocol=pickle.HIGHEST_PROTOCOL)
    partial_file.replace(memo_file)
    return scenes


def _open_session_cache(training_config: TrainingConfig, output_dir: Path) -> FanoutCache | None:
    """
    Open the run's node-local memo of the policy's session tensors under the mirror root.

    Each run gets its own directory, named after its run dir: the session tensors depend on the stage's settings, so
    the stages of one job share the mirror of the permanent cache but never each other's session tensors.

    Args:
        training_config: the training config
        output_dir: the run's dir

    Returns:
        the session cache, or None when it is off or there is no mirror root
    """
    if not (training_config.session_cache and training_config.cache_mirror_root):
        return None
    output_dir = output_dir.resolve()
    run_id = hashlib.sha1(str(output_dir).encode()).hexdigest()[:8]
    directory = Path(training_config.cache_mirror_root) / "session_cache" / f"{output_dir.name}_{run_id}"
    LOG.info(f"Session cache under {directory}")
    # Values are small and stay inline in the shards' SQLite files; nothing is ever evicted.
    return FanoutCache(
        str(directory),
        shards=16,
        timeout=1,
        size_limit=2**40,
        cull_limit=0,
        # diskcache takes its settings as keyword arguments that its stubs leave out.
        sqlite_cache_size=training_config.session_cache_sqlite_cache_pages,  # pyright: ignore[reportCallIssue]
    )


def build_source_datasets(
    training_config: TrainingConfig,
    policy: AnyPolicy,
    output_dir: Path,
    scene_share_dir: Path | None,
    mirror_copier: MirrorCopier | None,
    ranks: Ranks,
    on_first_dataset: Callable[[TrainingDataset], None] | None = None,
) -> list[TrainingDataset]:
    """
    Build one dataset per data source; each source's logs go to the copier as soon as its dataset exists.

    Args:
        training_config: the training config
        policy: the policy whose builders define a sample
        output_dir: the run's dir, which names its session cache
        scene_share_dir: the directory scenes are shared through; None filters on every rank
        mirror_copier: copies the stores' logs to the mirror; None copies nothing
        ranks: this process's place among the job's ranks
        on_first_dataset: called with the first source's dataset as soon as it exists; None calls nothing

    Returns:
        one dataset per source, in config order
    """
    assert training_config.offline_data_sources, "offline_data_sources must not be empty"
    session_cache = _open_session_cache(training_config, output_dir)
    datasets: list[TrainingDataset] = []
    augmentation = training_config.camera_augmentation
    scene_filter_executor = _scene_filter_executor(scene_share_dir, training_config.startup_num_workers, ranks)
    for source_index, source in enumerate(training_config.offline_data_sources):
        scene_builder: ArrowSceneBuilder = build_scene_builder(
            source.data_root,
        )
        with log_stage(
            f"Source {source_index + 1} of {len(training_config.offline_data_sources)}: "
            f"filtering scenes of splits {source.garage_scene_filter.split_names} "
            f"under {source.data_root} (this can take minutes on large datasets)",
        ):
            scene_filter = source.garage_scene_filter.to_py123d_scene_filter()
            policy.verify_contract(offline_data_source_config=source, scene_filter=scene_filter)
            # NOTE@ln2697: change to a thread pool if this causes problems.
            filter_scenes = partial(_filter_scenes_with_pool, scene_builder, scene_filter, scene_filter_executor)
            memo_file = (
                _filtered_scenes_memo(training_config.cache_mirror_root, source.data_root, scene_filter)
                if training_config.cache_mirror_root is not None
                else None
            )
            if memo_file is not None:
                # The stages of one job filter each distinct filter once.
                filter_scenes = partial(_memoized_scenes, memo_file, filter_scenes)
            scenes: Sequence[SceneAPI] = _filter_scenes_once(
                scene_share_dir,
                source_index,
                training_config.scene_filter_wait_timeout_s,
                filter_scenes,
                ranks,
            )
        assert len(scenes), f"source {source.data_root} selected no scenes"
        LOG.info(
            f"Source {source.data_root}: {len(scenes)} scenes passed the filter",
        )
        verify_offline_data_source_scenes_declared_intervals(source, scenes[0])

        permanent_cache: CacheStoreReader | None = None
        if training_config.read_from_cache_store:
            with log_stage(f"Opening the cache store at {source.cache_root}"):
                permanent_cache = CacheStoreReader(
                    source.cache_root,
                    policy.cache_signature(scenes[0].scene_metadata.dataset),
                    mirror_root=training_config.cache_mirror_root,
                )
        else:
            LOG.info(
                "read_from_cache_store is off: every tensor is built live, which is several times slower per sample",
            )
        # Rank 0 checks the store once; a failed check ends the whole job.
        check_scene_coverage = scene_share_dir is None or ranks.global_rank == 0
        dataset = TrainingDataset(
            scenes,
            policy,
            permanent_cache=permanent_cache,
            session_cache=session_cache,
            startup_num_workers=training_config.startup_num_workers,
            check_scene_coverage=check_scene_coverage,
            verify_store_coverage=training_config.check_cache_coverage,
            source_index=source_index,
            hood_swap_config=augmentation.hood_swap_config if augmentation.hood_swap else None,
            has_box_annotations=source.has_box_annotations,
        )
        if permanent_cache is not None:
            # The check reads every scene's uuid; with them a sample builds its scene only when a cache misses.
            dataset.scene_uuids = _share_scene_uuids(
                scene_share_dir,
                source_index,
                dataset.scene_uuids,
                training_config.scene_filter_wait_timeout_s,
                ranks,
            )
        datasets.append(dataset)
        if mirror_copier is not None and permanent_cache is not None:
            mirror_copier.submit(f"source {source_index + 1}", permanent_cache, datasets[-1].log_names)
        if on_first_dataset is not None and source_index == 0:
            on_first_dataset(dataset)
    LOG.info(
        f"Total: {sum(len(dataset) for dataset in datasets)} scenes across {len(datasets)} sources",
    )
    return datasets


class _PrefetchingConcatDataset(ConcatDataset[TrainingSample]):
    """The sources' samples back to back; a worker's batched fetch first asks the kernel to read every sample's records."""

    # Created on a worker's first batch, so no thread pool is ever forked.
    _executor: ThreadPoolExecutor | None = None

    def __getitems__(self, indices: list[int]) -> list[TrainingSample]:
        """
        Fetch a batch's samples after starting the disk reads of all of them.

        Args:
            indices: the batch's sample indices

        Returns:
            the samples, in order
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(_PREFETCH_THREADS)
        offsets_by_source: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            source_index = bisect.bisect_right(self.cumulative_sizes, index)
            offsets_by_source[source_index].append(
                index - (self.cumulative_sizes[source_index - 1] if source_index else 0),
            )
        for source_index, offsets in offsets_by_source.items():
            cast(TrainingDataset, self.datasets[source_index]).prefetch(offsets, self._executor)
        return [self[index] for index in indices]


class _MixtureSampler(Sampler[int]):
    """
    Draws every source's samples as often as its weight says, a whole weight being exact passes.

    The epochs of a super-epoch share one order, and each rank yields its own stripe of it.
    """

    def __init__(
        self,
        source_lengths: Sequence[int],
        source_weights: Sequence[float],
        seed: int,
        epochs_per_super_epoch: int,
    ) -> None:
        """
        Plan the epoch of a concatenation of sources.

        Args:
            source_lengths: sample count of each source, in the order the sources are concatenated.
            source_weights: how often each source is drawn per epoch, relative to its own length.
            seed: base seed of the shuffling, one per super-epoch.
            epochs_per_super_epoch: consecutive epochs that share one shuffling.
        """
        super().__init__()
        self._sources: list[tuple[int, float]] = list(zip(source_lengths, source_weights, strict=True))
        self._seed = seed
        self._epochs_per_super_epoch = epochs_per_super_epoch
        # Set after construction; Lightning iterates the loader before any epoch hook, so the epoch is read from it.
        self.trainer: L.Trainer | None = None

    def __len__(self) -> int:
        """This rank's number of samples per epoch."""
        assert self.trainer is not None, "the trainer must be set before the first epoch"
        return math.ceil(sum(round(length * weight) for length, weight in self._sources) / self.trainer.world_size)

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield this rank's stripe of the current super-epoch's order."""
        assert self.trainer is not None, "the trainer must be set before the first epoch"
        progress = self.trainer.fit_loop.epoch_progress.current
        assert isinstance(progress, _ProcessedTracker)
        # The count Lightning hands its own samplers: the epoch about to train, also right after a resume.
        super_epoch = progress.processed // self._epochs_per_super_epoch
        # Every rank materializes the sampler itself, so the seed must not depend on the rank.
        generator = torch.Generator().manual_seed(self._seed + super_epoch)
        draws: list[torch.Tensor] = []
        source_offset = 0
        for length, weight in self._sources:
            source_indices = torch.arange(source_offset, source_offset + length)
            num_passes, remainder = divmod(round(length * weight), length)
            draws.append(source_indices.repeat(num_passes))
            draws.append(source_indices[torch.randperm(length, generator=generator)[:remainder]])
            source_offset += length
        epoch_indices = torch.cat(draws)
        shuffled = epoch_indices[torch.randperm(len(epoch_indices), generator=generator)]
        # Padded to equal stripes like torch's DistributedSampler, so no rank runs an extra step.
        world_size = self.trainer.world_size
        padding = -len(shuffled) % world_size
        stripe = torch.cat([shuffled, shuffled[:padding]])[self.trainer.global_rank :: world_size]
        return iter(cast("list[int]", stripe.tolist()))  # pyright: ignore[reportUnknownMemberType]


class MirrorCopier:
    """
    Copies this rank's share of the selected logs to the mirror, in a process of its own.

    A process, not a thread, since forked loader workers would inherit a copy lock a thread holds.
    """

    def __init__(self, helper_cpus: set[int] | None, copy_threads: int, ranks: Ranks) -> None:
        """
        Start the copier process.

        Args:
            helper_cpus: the CPUs the copier is pinned to; None leaves it on the process's set.
            copy_threads: parallel NFS reads.
            ranks: this process's place among the job's ranks, which pick its share of the logs.
        """
        self._ranks = ranks
        context = multiprocessing.get_context("fork")
        self._queue: multiprocessing.Queue[tuple[str, CacheStoreReader, list[str]] | None] = context.Queue()
        self._process = context.Process(
            target=_copy_queued_logs,
            args=(self._queue, helper_cpus, copy_threads),
            daemon=True,
        )
        self._process.start()

    def submit(self, label: str, cache_reader: CacheStoreReader, log_names: Sequence[str]) -> None:
        """
        Queue this rank's stride of a source's logs.

        Args:
            label: names the source in the log
            cache_reader: the source's store reader, which copies a log on first access
            log_names: every selected log of the source
        """
        share = list(log_names[self._ranks.global_rank :: self._ranks.world_size])
        LOG.info(f"Mirror copier: queued {len(share)} of {len(log_names)} logs of {label}")
        self._queue.put((label, cache_reader, share))

    def close(self) -> None:
        """Signal that no more sources come; the process finishes the queued copies and exits."""
        self._queue.put(None)


def _copy_queued_logs(
    queue: multiprocessing.Queue[tuple[str, CacheStoreReader, list[str]] | None],
    helper_cpus: set[int] | None,
    copy_threads: int,
) -> None:
    """
    Copy every queued source's logs to the mirror until the queue ends.

    Args:
        queue: the sources to copy, ended by None
        helper_cpus: the CPUs to pin the process to; None keeps its CPU set
        copy_threads: parallel NFS reads
    """
    if helper_cpus is not None:
        os.sched_setaffinity(0, helper_cpus)
    LOG.info(f"Mirror copier: started with {copy_threads} threads")
    start = perf_counter()
    copied_logs = 0
    copied_bytes = 0
    with ThreadPoolExecutor(copy_threads) as executor:
        for label, cache_reader, log_names in iter(queue.get, None):
            source_start = perf_counter()
            source_bytes = 0

            def copy_log(log_name: str, reader: CacheStoreReader = cache_reader) -> int:
                """
                Copy one log to the mirror through the reader.

                Args:
                    log_name: the log to copy
                    reader: the source's store reader

                Returns:
                    the log's size in bytes
                """
                return sum(path.stat().st_size for path in reader.log_dir(log_name).iterdir())

            for log_bytes in executor.map(copy_log, log_names):
                copied_logs += 1
                copied_bytes += log_bytes
                source_bytes += log_bytes
                if copied_logs % 1000 == 0:
                    elapsed = perf_counter() - start
                    LOG.info(
                        f"Mirror copier: {copied_logs} logs, {copied_bytes / 1e9:.1f} GB "
                        f"in {elapsed / 60:.1f} min, {copied_bytes / elapsed / 1e6:.0f} MB/s on this rank",
                    )
            source_elapsed = perf_counter() - source_start
            LOG.info(
                f"Mirror copier: {label} done, {len(log_names)} logs, {source_bytes / 1e9:.1f} GB "
                f"in {source_elapsed / 60:.1f} min, {source_bytes / max(source_elapsed, 1e-9) / 1e6:.0f} MB/s on this rank",
            )
    elapsed = perf_counter() - start
    LOG.info(
        f"Mirror copier: finished, {copied_logs} logs, {copied_bytes / 1e9:.1f} GB "
        f"in {elapsed / 60:.1f} min, {copied_bytes / max(elapsed, 1e-9) / 1e6:.0f} MB/s on this rank",
    )


def _fast_fail_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> None:
    """
    Print an uncaught exception and end the process at once.

    Args:
        exc_type: the exception's type
        exc_value: the exception
        exc_tb: the exception's traceback
    """
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def _export_sensor_rigs(
    scenes: list[SceneAPI],
    output_dir: str,
) -> None:
    """
    Write every dataset's modality metadata next to the training outputs, as sensor_rig_<dataset>.yaml.

    The rig is assumed identical across a dataset, so one scene stands for every source of it; when the sources of a
    dataset disagree, the first one's rig is kept and the disagreement logged.

    Args:
        scenes: one scene per training source
        output_dir: the run's output directory
    """
    rigs: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        dataset = scene.scene_metadata.dataset
        rig: dict[str, Any] = {
            "dataset": dataset,
            "log_name": scene.get_log_metadata().log_name,
            "modalities": {key: metadata.to_dict() for key, metadata in scene.get_all_modality_metadatas().items()},
        }
        if dataset not in rigs:
            rigs[dataset] = rig
        elif rigs[dataset]["modalities"] != rig["modalities"]:
            LOG.warning(
                f"The {dataset} sources disagree on the sensor rig; sensor_rig_{dataset}.yaml keeps the rig of log "
                f"{rigs[dataset]['log_name']}, not that of log {rig['log_name']}",
            )
    for dataset, rig in rigs.items():
        with (Path(output_dir) / f"sensor_rig_{dataset}.yaml").open("w") as file:
            yaml.safe_dump(rig, file, sort_keys=False)


def _reject_held_out_splits(
    offline_data_sources: list[OfflineTrainingDataSourceConfig],
) -> None:
    """
    Fail when a training source draws from a split an evaluation suite scores.

    Args:
        offline_data_sources: the sources the run would train on.

    Raises:
        ValueError: if a source's filter names a held-out split, or draws from physical-ai-av without
            dropping the clips NuRec reconstructs.
    """
    unfiltered = sorted(
        {
            split
            for source in offline_data_sources
            for split in source.garage_scene_filter.split_names or []
            if split.startswith(NUREC_SOURCE_SPLIT_PREFIX)
            and NUREC_CLIP_FILTER not in (source.garage_scene_filter.custom_anchor_filter_fns or [])
        },
    )
    if unfiltered:
        raise ValueError(
            f"Training sources draw from {unfiltered} without {NUREC_CLIP_FILTER}; "
            f"those splits hold the clips the AlpaSim nurec suite drives.",
        )
    held_out = sorted(
        {
            split
            for source in offline_data_sources
            for split in source.garage_scene_filter.split_names or []
            if split in HELD_OUT_SPLITS
        },
    )
    if held_out:
        raise ValueError(
            f"Training sources draw from the held-out splits {held_out}; "
            f"nurec_val is AlpaSim's val suite and nuplan_test is navtest.",
        )


def _shutdown_distributed(trainer: L.Trainer, lightning_module: PolicyLightningModule, timeout_s: float) -> None:
    """
    Leave the process group in lockstep, before the ranks can race each other out.

    Otherwise a rank outliving rank 0's store server at exit crashes the whole srun step.

    Args:
        trainer: the trainer whose ranks have finished fitting.
        lightning_module: the module, whose recorded step is released first.
        timeout_s: how long the teardown may take before the rank exits on its own.
    """
    lightning_module.release_recorded_step()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    def teardown() -> None:
        """Wait for every rank, then destroy the process group."""
        trainer.strategy.barrier("shutdown")
        torch.distributed.destroy_process_group()

    teardown_thread = threading.Thread(target=teardown, name="shutdown", daemon=True)
    teardown_thread.start()
    teardown_thread.join(timeout_s)
    if teardown_thread.is_alive():
        LOG.warning(f"The process group teardown hung for {timeout_s:.0f} s after training; exiting")
        _exit_without_teardown()


register_schema("train", TrainingConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    # Print the C-level stack trace when a rank dies on a fatal signal.
    faulthandler.enable()
    sys.excepthook = _fast_fail_excepthook
    training_config: TrainingConfig = finalize(cfg, TrainingConfig)
    _reject_held_out_splits(training_config.offline_data_sources)
    output_dir = run_dir()
    save_config(training_config)
    trainer_config = training_config.lightning_trainer_config
    # The trainer's own resolution of the devices and the step's precision context; it also sets cuDNN's flags.
    connector = _AcceleratorConnector(
        devices=trainer_config.devices,
        num_nodes=trainer_config.num_nodes,
        accelerator=trainer_config.accelerator,
        precision=cast(Any, trainer_config.precision),
        benchmark=trainer_config.benchmark,
    )
    parallel_devices = connector._parallel_devices  # pyright: ignore[reportPrivateUsage]
    ranks = launcher_ranks(len(parallel_devices), trainer_config.num_nodes)
    partition = _cpu_partition(ranks) if training_config.dedicated_main_core else None
    main_cores, helper_cpus = partition if partition is not None else (None, None)

    if training_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if training_config.cudnn_attention:
        cudnn_first = [SDPBackend.CUDNN_ATTENTION.value]
        default_order: list[int] = torch._C._get_sdp_priority_order()  # pyright: ignore[reportPrivateUsage]
        torch._C._set_sdp_priority_order(cudnn_first + [b for b in default_order if b not in cudnn_first])  # pyright: ignore[reportPrivateUsage]
    # DDP's graph splitting would turn 2 compiled graphs per seed into ~30 for little all-reduce overlap.
    torch._dynamo.config.optimize_ddp = False  # pyright: ignore[reportPrivateUsage]
    # A frozen backbone's eval-mode batch norms fold into the convolution before each of them.
    torch._inductor.config.efficient_conv_bn_eval_fx_passes = True  # pyright: ignore[reportPrivateUsage]
    for name, value in training_config.inductor_config.items():
        setattr(torch._inductor.config, name, value)  # pyright: ignore[reportPrivateUsage]

    LOG.info(
        f"{training_config.num_seeds} seeds, results in {output_dir!s}",
    )
    seed_batch_size = training_config.dataloader_config.batch_size
    loaded_batch_factor = training_config.effective_loaded_batch_factor
    loaded_batch_size = round(seed_batch_size * loaded_batch_factor)
    assert loaded_batch_factor >= 1, f"loaded_batch_factor {loaded_batch_factor} < 1"
    LOG.info(
        f"{trainer_config.max_epochs} epochs on {trainer_config.devices} devices x "
        f"{trainer_config.num_nodes} nodes, {training_config.num_seeds} seeds each drawing "
        f"{seed_batch_size} of the {loaded_batch_size} loaded samples and "
        f"{training_config.dataloader_config.num_workers} workers per device, "
        f"compile={training_config.compile_mode}",
    )

    policies: list[AnyPolicy] = []
    with log_stage(f"Building policy {training_config.policy_config.target} x {training_config.num_seeds}"):
        for seed in range(training_config.num_seeds):
            L.seed_everything(seed)
            policy = cast(
                AnyPolicy,
                build_from_string(training_config.policy_config, AbstractPolicy),
            )
            policy.initialize()
            policy.prepare_training()
            policies.append(policy)
    L.seed_everything(0, workers=True)
    policy = policies[0]
    LOG.info(
        f"Policy has {sum(p.numel() for p in policy.parameters()):,} trainable parameters",
    )
    lightning_module = PolicyLightningModule(
        policies=policies,
        training_config=training_config,
    )

    compile_aheads: list[CompileAhead] = []
    on_first_dataset: Callable[[TrainingDataset], None] | None = None
    compile_ahead_on = training_config.compile_ahead and training_config.compile_mode is not None
    kernel_check_on = training_config.kernel_check and bool(kernel_patches_of(training_config.policy_config))
    if compile_ahead_on or kernel_check_on:

        def check_kernels_and_compile_ahead(dataset: TrainingDataset) -> None:
            """
            Check the kernel patches on a batch of the first source, then start compiling on it.

            Args:
                dataset: the first source's dataset
            """
            batch = example_batch(dataset, loaded_batch_size)
            device = connector.strategy.root_device
            if kernel_check_on:
                with log_stage("Checking the kernel patches against the plain torch policy"):
                    lightning_module.to(device)
                    lightning_module.verify_kernel_patches(
                        batch.to(device),
                        connector.precision_plugin.train_step_context,
                        output_dir / "kernel_report.json" if ranks.global_rank == 0 else None,
                    )
            if compile_ahead_on:
                compile_aheads.append(
                    CompileAhead(
                        lightning_module,
                        batch,
                        device,
                        connector.precision_plugin.train_step_context,
                    ),
                )

        on_first_dataset = check_kernels_and_compile_ahead

    with log_stage(
        f"Building the datasets of {len(training_config.offline_data_sources)} data sources",
    ):
        mirror_copier = (
            MirrorCopier(helper_cpus, training_config.cache_mirror_copy_threads, ranks)
            if training_config.cache_mirror_root is not None
            else None
        )
        source_datasets: list[TrainingDataset] = build_source_datasets(
            training_config,
            policy,
            output_dir,
            _scene_share_dir(output_dir),
            mirror_copier,
            ranks,
            on_first_dataset,
        )
        if mirror_copier is not None:
            mirror_copier.close()
    if training_config.freeze_heap:
        # The forked fillers and loader workers inherit these millions of objects; frozen, their collections skip them
        # instead of each worker walking all of them at once every few minutes (2+ s, copying the shared pages).
        gc.collect()
        gc.freeze()
    if (
        training_config.session_cache
        and training_config.cache_mirror_root
        and training_config.session_cache_fill_processes
    ):
        start_session_cache_fillers(
            source_datasets,
            training_config.session_cache_fill_processes,
            helper_cpus,
            ranks.local_rank,
            ranks.ranks_per_node,
        )
    train_data: ConcatDataset[Any] = (
        _PrefetchingConcatDataset(source_datasets)
        if training_config.cache_read_prefetch
        else ConcatDataset(source_datasets)
    )
    LOG.info(f"{len(train_data)} train samples")
    epochs_per_super_epoch = int(loaded_batch_factor)
    sampler = _MixtureSampler(
        [len(dataset) for dataset in source_datasets],
        [source.source_weight for source in training_config.offline_data_sources],
        0,
        epochs_per_super_epoch,
    )
    # batch_size and num_workers are per device.
    dataloader_kwargs = asdict(training_config.dataloader_config)
    dataloader_kwargs["batch_size"] = loaded_batch_size
    if dataloader_kwargs["num_workers"] == 0:  # the DataLoader refuses the combinations
        dataloader_kwargs["persistent_workers"] = False
        dataloader_kwargs["prefetch_factor"] = None
        dataloader_kwargs["in_order"] = True
    collate_fn: Callable[[list[TrainingSample]], Any] = TrainingSample.collate
    batch_slots: PinnedBatchSlots | None = None
    if training_config.pinned_batch_slots and dataloader_kwargs["num_workers"] > 0 and torch.cuda.is_available():
        batch_slots = PinnedBatchSlots.for_batch_of_one(
            example_batch(source_datasets[0], 1),
            loaded_batch_size,
            dataloader_kwargs["num_workers"],
            # A worker never waits on a slot whose copy the training process has not started yet.
            (training_config.dataloader_config.prefetch_factor or 2) + 2,
            TrainingSample.collate,
        )
        collate_fn = batch_slots.collate
        # The slots are page-locked already; the pin-memory thread would only copy them again.
        dataloader_kwargs["pin_memory"] = False
        lightning_module.batch_slots = batch_slots
    train_dataloader = DataLoader(
        train_data,
        worker_init_fn=partial(_worker_init, helper_cpus=helper_cpus),
        collate_fn=collate_fn,
        sampler=sampler,
        **dataloader_kwargs,
    )

    callbacks: list[L.Callback] = [
        *policy.get_training_callbacks(),
        # Lightning's default RichProgressBar writes nothing into a redirected log file.
        BatchesProgressBar(
            loaded_batch_size,
            training_config.progress_bar_refresh_rate,
        ),
        VisualizationCallback(training_config.log_images_every_n_steps),
        ThroughputLoggingCallback(
            loaded_batch_size,
            seed_batch_size * training_config.num_seeds,
            training_config.lightning_trainer_config.log_every_n_steps,
        ),
        EpochLoggingCallback(),
    ]
    if training_config.trace_from_step is not None:
        callbacks.append(StepTraceCallback(training_config.trace_from_step, output_dir))
    if training_config.compile_mode is not None:
        callbacks.append(ShutdownCompileWorkersCallback())
    if compile_aheads:
        callbacks.append(FirstStepCompileCheckCallback())
    if main_cores is not None:
        callbacks.append(PinMainProcessCallback(*main_cores))
    if batch_slots is not None:
        callbacks.append(ReleaseBatchSlotsCallback(batch_slots))

    training_loggers: list[Logger] = _build_loggers(training_config)
    trainer_kwargs = asdict(training_config.lightning_trainer_config)
    single_device = trainer_kwargs["devices"] == 1 and trainer_kwargs["num_nodes"] == 1
    if trainer_kwargs["strategy"] == "auto" and not single_device:
        # Skips DDP's per-forward broadcast of the BatchNorm running statistics; one device needs no DDP.
        trainer_kwargs["strategy"] = (
            _UnwrappedDDPStrategy()
            if training_config.num_seeds > 1
            else DDPStrategy(
                broadcast_buffers=False,
                gradient_as_bucket_view=training_config.gradient_as_bucket_view,
                bucket_cap_mb=training_config.gradient_bucket_cap_mb,
            )
        )
    # Lightning's own distributed sampler reshuffles the training order every epoch; the sampler splits the ranks itself.
    trainer_kwargs["use_distributed_sampler"] = False
    if training_config.num_seeds > 1:
        # Lightning refuses automatic clipping under manual optimization; the module clips from these values.
        trainer_kwargs["gradient_clip_val"] = None
        trainer_kwargs["gradient_clip_algorithm"] = None
        callbacks.append(StepWatchdogCallback(training_config.step_watchdog_timeout_s, ranks.global_rank))
    if dataloader_kwargs["persistent_workers"] and training_config.persistent_workers_recycle_epochs:
        callbacks.append(RecycleWorkersCallback(train_dataloader, training_config.persistent_workers_recycle_epochs))
    if training_config.freeze_heap:
        callbacks.append(FreezeHeapCallback())
    trainer = L.Trainer(
        **trainer_kwargs,
        # Keeps Lightning's own files inside the run's output dir.
        default_root_dir=str(output_dir),
        # True would add Lightning's own default ModelCheckpoint.
        enable_checkpointing=False,
        logger=training_loggers,
        callbacks=callbacks,
    )
    sampler.trainer = trainer
    # The SIGTERM flag broadcast syncs the device every batch; srun signals every rank, so log steps suffice.
    epoch_loop = trainer.fit_loop.epoch_loop
    broadcast_sigterm = epoch_loop._broadcast_sigterm_tensor  # pyright: ignore[reportPrivateUsage]
    log_every_n_steps = max(1, trainer_kwargs["log_every_n_steps"])

    def broadcast_sigterm_on_log_steps() -> None:
        """Broadcast Lightning's SIGTERM flag only on log steps."""
        if trainer.global_step % log_every_n_steps == 0:
            broadcast_sigterm()

    epoch_loop._broadcast_sigterm_tensor = broadcast_sigterm_on_log_steps  # pyright: ignore[reportPrivateUsage]

    for training_logger in training_loggers:
        training_logger.log_hyperparams(asdict(training_config))
    if trainer.is_global_zero:
        _export_sensor_rigs([dataset.scenes[0] for dataset in source_datasets], str(output_dir))

    resume = (
        merged_resume_checkpoint(output_dir, list(range(training_config.num_seeds)), ranks.global_rank)
        if training_config.resume_from_last_checkpoint
        else None
    )

    resume_note = f", resuming from {resume}" if resume else ""
    compile_note = (
        f" (compiling {training_config.compile_mode}: the first step can take minutes)"
        if training_config.compile_mode is not None
        else ""
    )
    # Before the loader workers fork. The fillers may fork while it compiles: a filler stuck on a lock it inherited
    # only leaves its share of the session cache to the loader.
    for compile_ahead in compile_aheads:
        compile_ahead.join(training_config.step_watchdog_timeout_s)
    LOG.info(f"Starting training{resume_note}{compile_note}")
    # sys.excepthook does not run for SystemExit or fatal signals, so a crashing rank would otherwise exit silently.
    try:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            ckpt_path=resume,
        )
    except BaseException:
        sys.stderr.write(
            f"=== rank {ranks.global_rank} exiting ===\n",
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    LOG.info(
        f"Training finished after {trainer.current_epoch} epochs / {trainer.global_step} global steps",
    )
    _shutdown_distributed(trainer, lightning_module, training_config.shutdown_timeout_s)
    _exit_without_teardown()


def _exit_without_teardown() -> None:
    """
    Exit 0 without the interpreter's teardown, once the run is complete on disk.

    A crash while unloading native libraries (e.g. SIGBUS from an extension whose file was replaced on NFS) would
    otherwise fail a finished job and every job waiting on it.
    """
    # Uploads what the W&B run still holds; a no-op without a run.
    wandb.finish()
    for child in multiprocessing.active_children():
        child.terminate()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
