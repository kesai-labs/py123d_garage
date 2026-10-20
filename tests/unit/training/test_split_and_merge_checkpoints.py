# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""A split checkpoint gives every seed its own prefix-free weights, and the merge restores the whole checkpoint."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from py123d_garage.training import checkpointing
from py123d_garage.training.checkpointing import merged_resume_checkpoint, split_checkpoint

_SEEDS = [0, 1]
_FIRST_RAW_KEYS = ["layer.weight", "layer.weight"]


def _checkpoint() -> dict[str, Any]:
    state_dict = {}
    for seed in _SEEDS:
        state_dict[f"policies.{seed}._orig_mod.layer.weight"] = torch.full((2, 2), float(seed))
        state_dict[f"policies.{seed}._orig_mod.layer.bias"] = torch.full((2,), 10.0 + seed)
    return {"state_dict": state_dict, "optimizer_states": [{"state": {"step": 7}}], "epoch": 3, "global_step": 42}


def test_split_writes_one_prefix_free_file_per_seed(tmp_path: Path) -> None:
    split_checkpoint(_checkpoint(), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 3)
    for seed in _SEEDS:
        weights = torch.load(tmp_path / f"model_{seed}_0003.pth", weights_only=True)
        assert set(weights) == {"layer.weight", "layer.bias"}
        assert torch.equal(weights["layer.weight"], torch.full((2, 2), float(seed)))
        assert torch.equal(weights["layer.bias"], torch.full((2,), 10.0 + seed))
    assert torch.load(tmp_path / "optimizer_0003.pth", weights_only=False) == [{"state": {"step": 7}}]
    state = torch.load(tmp_path / "trainer_state_0003.pth", weights_only=False)
    assert state["state_dict_prefixes"] == ["policies.0._orig_mod.", "policies.1._orig_mod."]
    assert state["seeds"] == _SEEDS
    assert state["epoch"] == 3
    assert "state_dict" not in state
    assert "optimizer_states" not in state


def test_merge_restores_the_split_checkpoint(tmp_path: Path) -> None:
    original = _checkpoint()
    split_checkpoint(copy.deepcopy(original), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 3)
    merged_file = merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=0)
    assert merged_file is not None
    assert merged_file == tmp_path / ".resume_merged_0003.pth"
    merged = torch.load(merged_file, weights_only=False)
    assert merged["state_dict"].keys() == original["state_dict"].keys()
    assert all(torch.equal(merged["state_dict"][key], value) for key, value in original["state_dict"].items())
    assert merged["optimizer_states"] == original["optimizer_states"]
    assert merged["epoch"] == 3
    assert merged["global_step"] == 42
    assert "seeds" not in merged
    assert "state_dict_prefixes" not in merged


def test_merge_takes_the_newest_epoch_and_refuses_other_seeds(tmp_path: Path) -> None:
    split_checkpoint(_checkpoint(), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 2)
    split_checkpoint(_checkpoint(), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 3)
    assert merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=0) == tmp_path / ".resume_merged_0003.pth"
    with pytest.raises(AssertionError, match="seeds"):
        merged_resume_checkpoint(tmp_path, [0], global_rank=0)


def test_merge_returns_none_without_a_complete_epoch(tmp_path: Path) -> None:
    assert merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=0) is None
    split_checkpoint(_checkpoint(), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 3)
    (tmp_path / "model_1_0003.pth").unlink()
    assert merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=0) is None


def test_other_ranks_wait_for_rank_zeros_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_checkpoint(_checkpoint(), _FIRST_RAW_KEYS, _SEEDS, tmp_path, 3)
    monkeypatch.setattr(checkpointing, "_RESUME_MERGE_TIMEOUT_S", 0)
    with pytest.raises(TimeoutError):
        merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=1)
    merged_file = merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=0)
    assert merged_resume_checkpoint(tmp_path, _SEEDS, global_rank=1) == merged_file
