"""The per-scene results table every offline benchmark writes: one file per shard, merged by the last one."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)


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
) -> None:
    """
    Merges every shard's table and appends an average row, once all shards have written.

    Every shard calls this after its own atomic write, so at least the last one
    to finish sees all num_shards files; a concurrent duplicate merge writes
    identical content.

    Args:
        output_dir: the run's output dir, shared by every shard.
        num_shards: how many shard files the merge waits for.
        shard_index: this process's slice, naming its temporary file.

    Raises:
        ValueError: if no shard scored a single scene.
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
    if num_unscorable:
        LOG.warning(
            f"{num_unscorable}/{len(df)} scenes are unscorable and carry no metrics; "
            f"the average is over the remaining {len(df) - num_unscorable}. "
            f"See the scoring_error column of results.csv.",
        )
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
