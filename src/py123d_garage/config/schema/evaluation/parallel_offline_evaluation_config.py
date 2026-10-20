# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelizationConfig:
    """Sharding and per-shard resources every offline benchmark runs under."""

    # Number of independent shard processes; the sources' logs are sliced [shard_index::num_shards].
    num_shards: int = 1
    # Index of this process's shard, e.g. the SLURM array task id.
    shard_index: int = 0
    # Inference device, e.g. "cuda".
    device: str = "cpu"
    # Scenes per forward pass.
    inference_batch_size: int = 32
    # Worker processes for scene building and metric scoring; None uses every CPU this process may use.
    max_workers: int | None = None
