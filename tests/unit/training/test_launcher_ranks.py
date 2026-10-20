# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import os
from pathlib import Path

import pytest

from py123d_garage.training.train import Ranks, _cpu_partition, launcher_ranks

_LAUNCHER_VARIABLES = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NTASKS_PER_NODE",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
)


@pytest.fixture(autouse=True)
def _no_launcher_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LAUNCHER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_srun_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
    assert launcher_ranks(1, 1) == Ranks(5, 8, 1, 4)


def test_torchrun_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert launcher_ranks(1, 1) == Ranks(5, 8, 1, 4)


def test_lightning_launcher_parent_is_rank_zero() -> None:
    assert launcher_ranks(4, 2) == Ranks(0, 8, 0, 4)


def test_lightning_launcher_child_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert launcher_ranks(4, 2) == Ranks(5, 8, 1, 4)


def _fake_topology(monkeypatch: pytest.MonkeyPatch, affinity: set[int]) -> None:
    def thread_siblings(path: Path, *args: object, **kwargs: object) -> str:
        cpu = int(path.parts[-3].removeprefix("cpu"))
        return f"{cpu % 8},{cpu % 8 + 8}"

    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: affinity)
    monkeypatch.setattr(Path, "read_text", thread_siblings)


def test_cpu_partition_slices_a_shared_node_per_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_topology(monkeypatch, set(range(16)))
    assert _cpu_partition(Ranks(0, 2, 0, 2)) == (({2, 10}, {3, 11}), {0, 1, 8, 9})
    assert _cpu_partition(Ranks(1, 2, 1, 2)) == (({6, 14}, {7, 15}), {4, 5, 12, 13})


def test_cpu_partition_keeps_a_bound_cpu_set_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_topology(monkeypatch, set(range(8)))
    assert _cpu_partition(Ranks(1, 2, 1, 2)) == (({6}, {7}), {0, 1, 2, 3, 4, 5})


def test_cpu_partition_needs_four_cores_per_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_topology(monkeypatch, set(range(16)))
    assert _cpu_partition(Ranks(0, 4, 0, 4)) is None
