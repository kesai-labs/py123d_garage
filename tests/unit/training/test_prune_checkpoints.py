# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from pathlib import Path

from py123d_garage.training.checkpointing import prune_checkpoints


def _write_epoch(output_dir: Path, epoch: int) -> None:
    for name in (f"model_0_{epoch:04d}.pth", f"optimizer_{epoch:04d}.pth", f"trainer_state_{epoch:04d}.pth"):
        (output_dir / name).touch()


def _names(output_dir: Path) -> set[str]:
    return {path.name for path in output_dir.iterdir()}


def test_kept_epochs_keep_only_their_model_files(tmp_path: Path) -> None:
    for epoch in range(3):
        _write_epoch(tmp_path, epoch)
    (tmp_path / ".resume_merged_0001.pth").touch()
    prune_checkpoints(tmp_path, 2, [0])
    assert _names(tmp_path) == {
        "model_0_0000.pth",
        "model_0_0002.pth",
        "optimizer_0002.pth",
        "trainer_state_0002.pth",
    }


def test_keep_training_state_keeps_the_kept_epochs_whole(tmp_path: Path) -> None:
    for epoch in range(3):
        _write_epoch(tmp_path, epoch)
    (tmp_path / ".resume_merged_0001.pth").touch()
    prune_checkpoints(tmp_path, 2, [0, 1], keep_training_state=True)
    assert _names(tmp_path) == {
        f"{prefix}_{epoch:04d}.pth" for epoch in range(3) for prefix in ("model_0", "optimizer", "trainer_state")
    }
