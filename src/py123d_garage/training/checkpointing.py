from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

LOG = logging.getLogger(__name__)

# How long a non-zero rank waits for rank 0's merged resume checkpoint.
_RESUME_MERGE_TIMEOUT_S = 1800


def checkpoint_files(output_dir: Path, epoch: int) -> tuple[Path, Path, Path]:
    """The model, optimizer and trainer-state file paths of one epoch."""
    return (
        output_dir / f"model_{epoch:04d}.pth",
        output_dir / f"optimizer_{epoch:04d}.pth",
        output_dir / f"trainer_state_{epoch:04d}.pth",
    )


def split_checkpoint(
    lightning_monolith_file: Path,
    first_raw_key: str,
    output_dir: Path,
    epoch: int,
) -> None:
    """Splits a dumped Lightning checkpoint into the three epoch files."""
    checkpoint: dict[str, Any] = torch.load(
        lightning_monolith_file,
        map_location="cpu",
        weights_only=False,
    )
    state_dict: dict[str, Tensor] = checkpoint.pop("state_dict")

    # The monolith prefixes every weight key with the module path down to the
    # policy (e.g. ``policy._orig_mod.`` when compiled); stripped for the model
    # file, recorded for the resume merge.
    first_full_key = next(iter(state_dict))
    assert first_full_key.endswith(first_raw_key)
    prefix = first_full_key[: len(first_full_key) - len(first_raw_key)]
    assert all(key.startswith(prefix) for key in state_dict)

    model_file, optimizer_file, state_file = checkpoint_files(output_dir, epoch)
    torch.save(
        {key.removeprefix(prefix): value for key, value in state_dict.items()},
        model_file,
    )
    torch.save(checkpoint.pop("optimizer_states"), optimizer_file)
    checkpoint["state_dict_prefix"] = prefix
    torch.save(checkpoint, state_file)
    lightning_monolith_file.unlink()


def prune_checkpoint(
    output_dir: Path,
    epoch: int,
    keep_epochs: list[int],
) -> None:
    """Removes an epoch's checkpoint files, keeping listed epochs' model file."""
    removable = list(output_dir.glob(".resume_merged_*.pth"))
    if epoch >= 0:
        model_file, optimizer_file, state_file = checkpoint_files(
            output_dir,
            epoch,
        )
        removable += [optimizer_file, state_file]
        if epoch not in keep_epochs:
            removable.append(model_file)
    for checkpoint_file in removable:
        if checkpoint_file.is_file():
            checkpoint_file.unlink()


def merged_resume_checkpoint(output_dir: Path) -> Path | None:
    """
    Merges the newest trio back into a monolith; None with nothing to resume.

    Rank zero writes the merged file atomically; every other rank waits for it
    to appear. Runs before the process group exists, so the coordination is
    filesystem-based.

    Args:
        output_dir: The run's output directory holding the checkpoint files.

    Returns:
        Path of the merged checkpoint, or None when there is nothing to resume.

    Raises:
        TimeoutError: If rank zero's merged file never appears.
    """
    state_files = sorted(output_dir.glob("trainer_state_*.pth"))
    if not state_files:
        return None
    epoch = int(state_files[-1].stem.rsplit("_", 1)[-1])
    model_file, optimizer_file, state_file = checkpoint_files(output_dir, epoch)
    if not (model_file.is_file() and optimizer_file.is_file()):
        LOG.warning(
            f"Incomplete checkpoint trio for epoch {epoch}, starting fresh.",
        )
        return None

    merged_file = output_dir / f".resume_merged_{epoch:04d}.pth"
    rank = int(
        os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")),
    )
    if rank != 0:
        deadline = time.monotonic() + _RESUME_MERGE_TIMEOUT_S
        while not merged_file.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Rank 0 never produced {merged_file}.")
            time.sleep(1.0)
        return merged_file
    if merged_file.is_file():
        # A completed merge from an earlier attempt; never half-written thanks
        # to the atomic rename below.
        return merged_file

    checkpoint: dict[str, Any] = torch.load(
        state_file,
        map_location="cpu",
        weights_only=False,
    )
    prefix: str = checkpoint.pop("state_dict_prefix")
    weights: dict[str, Tensor] = torch.load(
        model_file,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["state_dict"] = {prefix + key: value for key, value in weights.items()}
    checkpoint["optimizer_states"] = torch.load(
        optimizer_file,
        map_location="cpu",
        weights_only=False,
    )
    partial_file = merged_file.with_name(f"{merged_file.name}.tmp")
    torch.save(checkpoint, partial_file)
    partial_file.replace(merged_file)
    return merged_file
