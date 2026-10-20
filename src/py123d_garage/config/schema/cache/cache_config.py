# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig


@dataclass
class CacheDataLoaderConfig:
    """Torch DataLoader settings that parallelize the cache build inside one shard."""

    # Samples per write batch.
    batch_size: int = 8
    # Number of worker processes.
    num_workers: int = 32
    # Pins host memory; off because no GPU consumes the batches.
    pin_memory: bool = False
    # Shuffles the samples; off because cache order does not matter.
    shuffle: bool = False


@dataclass
class CacheConfig:
    """Config of the py123d_garage.cache.build_cache entry point."""

    # -- Config objects --

    # The policy whose builders fill the stores.
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    # Offline data sources to cache; each gets one cache store.
    offline_data_sources: list[OfflineTrainingDataSourceConfig] = field(default_factory=list)
    # DataLoader settings of the cache build.
    dataloader_config: CacheDataLoaderConfig = field(
        default_factory=CacheDataLoaderConfig,
    )

    # -- Atomic settings --

    # Global RNG seed.
    seed: int = 0
    # Overwrite stored tensors; required after a config change that affects cached content.
    force_cache_rebuild: bool = False
    # 0-based index of the shard this process builds.
    shard_index: int = 0
    # Count of shards to split the cache build into; 1 means no sharding.
    shard_count: int = 1
    # Id that separates this build from other builds of the same config.
    cache_id: str = ""
    # Caches every frame for any later anchor stride; false caches only the anchors the training plan samples.
    build_dense_cache: bool = False
