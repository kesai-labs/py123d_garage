# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Splits training checkpoints into per-seed files, prunes old ones and merges them back for resuming."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

LOG = logging.getLogger(__name__)

# How long a non-zero rank waits for rank 0's merged resume checkpoint.
_RESUME_MERGE_TIMEOUT_S = 1800


def checkpoint_files(output_dir: Path, epoch: int, seeds: Sequence[int]) -> tuple[list[Path], Path, Path]:
    """
    Name the checkpoint files of one epoch.

    Args:
        output_dir: the run's output directory
        epoch: the epoch the files belong to
        seeds: the run's seeds, one model file each

    Returns:
        the per-seed model files, the optimizer file and the trainer-state file
    """
    return (
        [output_dir / f"model_{seed}_{epoch:04d}.pth" for seed in seeds],
        output_dir / f"optimizer_{epoch:04d}.pth",
        output_dir / f"trainer_state_{epoch:04d}.pth",
    )


def _save_whole(obj: object, file: Path) -> None:
    """
    Save through a temporary file in the same directory, so a crash mid-write leaves no truncated file under the name.

    Args:
        obj: what to save
        file: the final file
    """
    partial_file = file.with_name(f"{file.name}.tmp")
    torch.save(obj, partial_file)
    partial_file.replace(file)


def split_checkpoint(
    checkpoint: dict[str, Any],
    first_raw_keys: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    epoch: int,
) -> None:
    """
    Write a Lightning checkpoint as one model file per seed plus the optimizer and trainer-state files.

    Args:
        checkpoint: the trainer's checkpoint with its tensors on the CPU; its entries are moved out
        first_raw_keys: the first state-dict key of each seed's policy, used to find its key prefix
        seeds: the run's seeds
        output_dir: the run's output directory
        epoch: the epoch the checkpoint belongs to
    """
    state_dict: dict[str, Tensor] = checkpoint.pop("state_dict")
    model_files, optimizer_file, state_file = checkpoint_files(output_dir, epoch, seeds)

    # Weight keys carry the module path to the policy (e.g. ``policies.0._orig_mod.``), kept for the resume merge.
    prefixes: list[str] = []
    for seed_index, (first_raw_key, model_file) in enumerate(zip(first_raw_keys, model_files, strict=True)):
        policy_prefix = f"policies.{seed_index}."
        first_full_key = next(
            key for key in state_dict if key.startswith(policy_prefix) and key.endswith(first_raw_key)
        )
        prefix = first_full_key[: len(first_full_key) - len(first_raw_key)]
        prefixes.append(prefix)
        _save_whole(
            {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)},
            model_file,
        )
    assert all(any(key.startswith(prefix) for prefix in prefixes) for key in state_dict)

    _save_whole(checkpoint.pop("optimizer_states"), optimizer_file)
    checkpoint["state_dict_prefixes"] = prefixes
    checkpoint["seeds"] = list(seeds)
    _save_whole(checkpoint, state_file)


def prune_checkpoints(
    output_dir: Path,
    before_epoch: int,
    keep_epochs: Sequence[int],
    keep_training_state: bool = False,
) -> None:
    """
    Remove every earlier epoch's checkpoint files and any resume merge, keeping the listed epochs' model files.

    Args:
        output_dir: the run's output directory
        before_epoch: files of this epoch and later are kept
        keep_epochs: earlier epochs whose model files are kept
        keep_training_state: also keep the listed epochs' optimizer and trainer-state files
    """
    removable = list(output_dir.glob(".resume_merged_*.pth"))
    for pattern, keepable in (
        ("optimizer_*.pth", keep_training_state),
        ("trainer_state_*.pth", keep_training_state),
        ("model_*_*.pth", True),
    ):
        for checkpoint_file in output_dir.glob(pattern):
            epoch_text = checkpoint_file.stem.rsplit("_", 1)[-1]
            if not epoch_text.isdigit():
                continue
            epoch = int(epoch_text)
            if epoch < before_epoch and not (keepable and epoch in keep_epochs):
                removable.append(checkpoint_file)
    for checkpoint_file in removable:
        checkpoint_file.unlink()


def merged_resume_checkpoint(output_dir: Path, seeds: Sequence[int], global_rank: int) -> Path | None:
    """
    Merge the newest epoch's files back into one Lightning checkpoint for resuming.

    Rank zero writes the merged file and the other ranks wait for it on disk, since no process group exists yet.

    Args:
        output_dir: The run's output directory holding the checkpoint files.
        seeds: The run's seeds, which must be the ones the checkpoint holds.
        global_rank: This process's rank across the job.

    Returns:
        Path of the merged checkpoint, or None when there is nothing to resume.

    Raises:
        TimeoutError: If rank zero's merged file never appears.
    """
    state_files = sorted(output_dir.glob("trainer_state_*.pth"))
    if not state_files:
        return None
    epoch = int(state_files[-1].stem.rsplit("_", 1)[-1])
    checkpoint: dict[str, Any] = torch.load(
        state_files[-1],
        map_location="cpu",
        weights_only=False,
    )
    saved_seeds: list[int] = checkpoint.pop("seeds")
    assert saved_seeds == list(seeds), f"The checkpoint holds seeds {saved_seeds}, the run seeds {list(seeds)}"
    model_files, optimizer_file, _ = checkpoint_files(output_dir, epoch, seeds)
    if not (all(model_file.is_file() for model_file in model_files) and optimizer_file.is_file()):
        LOG.warning(
            f"Incomplete checkpoint files for epoch {epoch}, starting fresh.",
        )
        return None

    merged_file = output_dir / f".resume_merged_{epoch:04d}.pth"
    if global_rank != 0:
        deadline = time.monotonic() + _RESUME_MERGE_TIMEOUT_S
        while not merged_file.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Rank 0 never produced {merged_file}.")
            time.sleep(1.0)
        return merged_file
    if merged_file.is_file():
        # Left by an earlier attempt; the atomic rename below means it is complete.
        return merged_file

    prefixes: list[str] = checkpoint.pop("state_dict_prefixes")
    state_dict: dict[str, Tensor] = {}
    for prefix, model_file in zip(prefixes, model_files, strict=True):
        weights: dict[str, Tensor] = torch.load(
            model_file,
            map_location="cpu",
            weights_only=True,
        )
        state_dict.update({prefix + key: value for key, value in weights.items()})
    checkpoint["state_dict"] = state_dict
    checkpoint["optimizer_states"] = torch.load(
        optimizer_file,
        map_location="cpu",
        weights_only=False,
    )
    _save_whole(checkpoint, merged_file)
    return merged_file
