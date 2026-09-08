from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelizationConfig:
    """Sharding and per-shard resources every offline benchmark runs under."""

    # Independent shard processes; the sources' logs are sliced [shard_index::num_shards].
    num_shards: int = 1
    # This process's slice, e.g. the SLURM array task id.
    shard_index: int = 0
    # Inference device, e.g. "cuda"; restrict per-shard GPU visibility at launch time.
    device: str = "cpu"
    # Scenes per forward pass.
    inference_batch_size: int = 32
    # Worker processes for scene building and metric scoring; null means every
    # CPU this process may use (the SLURM allocation, not the node).
    max_workers: int | None = None
