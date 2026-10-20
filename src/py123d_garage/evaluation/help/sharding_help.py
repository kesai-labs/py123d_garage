# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The per-scene results table every offline benchmark writes: one file per shard, merged by the last one."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig

LOG = logging.getLogger(__name__)


def expected_num_scenes(sources: Sequence[OfflineEvaluationDataSourceConfig]) -> int | None:
    """
    Sums the scene counts the sources declare, for the merge to check.

    Args:
        sources: the benchmark's sources.

    Returns:
        the total, or None when any source declares no count.
    """
    counts = [source.expected_num_scenes for source in sources]
    if not counts or any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


def save_shard_results(
    results: list[dict[str, object]],
    output_dir: str,
    shard_index: int,
) -> None:
    """
    Atomically writes this shard's per-scene results table.

    Args:
        results: per-scene metric dicts.
        output_dir: the run's output dir, shared by every shard.
        shard_index: this process's slice.
    """
    shards_dir = Path(output_dir) / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    # The merge counts results_*.csv, so the file appears only when complete.
    temporary_file = shards_dir / f"results_{shard_index:05d}.csv.tmp"
    pd.DataFrame(results).to_csv(temporary_file, index=False)
    shard_file = shards_dir / f"results_{shard_index:05d}.csv"
    temporary_file.replace(shard_file)
    LOG.info(f"Saved shard results to {shard_file}")


def _read_shard_results(shard_file: Path) -> pd.DataFrame:
    """
    Reads one shard's table.

    Args:
        shard_file: the shard's results file.

    Returns:
        the shard's rows; empty when its slice scored no scene, which writes a table without columns.
    """
    try:
        return pd.read_csv(shard_file)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_results_if_last_shard(
    output_dir: str,
    num_shards: int,
    shard_index: int,
    expected_num_scenes: int | None = None,
) -> None:
    """
    Merges every shard's table and appends an average row, once all shards have written.

    Every shard calls this after its own atomic write, so the last one sees all files; a duplicate merge is harmless.

    Args:
        output_dir: the run's output dir, shared by every shard.
        num_shards: how many shard files the merge waits for.
        shard_index: this process's slice, naming its temporary file.
        expected_num_scenes: how many scenes the sources declare; None skips the count check.

    Raises:
        ValueError: if no shard scored a single scene.
        RuntimeError: once results.csv is written, if a scene is unscorable or the scene count is not the declared one.
    """
    shards_dir = Path(output_dir) / "shards"
    shard_files = sorted(shards_dir.glob("results_*.csv"))
    if len(shard_files) < num_shards:
        LOG.info(
            f"{len(shard_files)}/{num_shards} shards finished; the last one merges.",
        )
        return

    shard_frames = [frame for frame in map(_read_shard_results, shard_files) if not frame.empty]
    if not shard_frames:
        raise ValueError(
            f"no scene passed the filter in any of the {num_shards} shards of "
            f"{output_dir}; widen the scene filter or check that the "
            f"logs carry every modality it requires.",
        )
    df = pd.concat(shard_frames, ignore_index=True)
    LOG.info(
        f"{len(df)} scenes passed the filter across all shards",
    )
    num_unscorable = int(df["scoring_error"].notna().sum()) if "scoring_error" in df else 0
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    average_row: dict[str, object] = {"scene_uuid": "average"}
    average_row.update({col: df[col].mean() for col in numeric_cols})
    df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

    results_file = Path(output_dir) / "results.csv"
    temporary_file = results_file.with_name(
        f"results.csv.{shard_index:05d}.tmp",
    )
    df.to_csv(temporary_file, index=False)
    temporary_file.replace(results_file)
    LOG.info(f"Merged {len(shard_files)} shard results into {results_file}")
    num_scenes = len(df) - 1
    if num_unscorable:
        raise RuntimeError(
            f"{num_unscorable}/{num_scenes} scenes are unscorable and carry no metrics; the average row of "
            f"{results_file} covers the remaining {num_scenes - num_unscorable}, see its scoring_error column.",
        )
    if expected_num_scenes is not None and num_scenes != expected_num_scenes:
        raise RuntimeError(
            f"{num_scenes} scenes were scored but the sources declare expected_num_scenes={expected_num_scenes}: "
            f"logs are missing or the scene filter dropped frames.",
        )
