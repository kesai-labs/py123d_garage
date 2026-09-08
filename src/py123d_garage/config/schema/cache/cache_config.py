from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig


@dataclass
class CacheDataLoaderConfig:
    """
    We parallelize the cache build inside a shard with a torch DataLoader, so we can use its batching and multiprocessing.

    This configures the DataLoader.
    """

    # Samples per write batch.
    batch_size: int = 8
    # Number of worker processes.
    num_workers: int = 32
    # No GPU consumer.
    pin_memory: bool = False
    # We build cache and doesn't need to shuffle the data.
    shuffle: bool = False


@dataclass
class CacheConfig:
    """
    Config for py123d_garage.cache.build_cache entry point.
    """

    # -- Config objects --

    # The policy whose builders fill the stores.
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    # Each of these offline data sources get one cache store.
    offline_data_sources: dict[str, OfflineTrainingDataSourceConfig] = field(default_factory=dict)
    # DataLoader kwargs.
    dataloader_config: CacheDataLoaderConfig = field(
        default_factory=CacheDataLoaderConfig,
    )

    # -- Atomic settings --

    # Global RNG seed.
    seed: int = 0
    # Overwrite stored tensors; required after a config change that affects cached content.
    force_cache_rebuild: bool = False
    # Parallelization cache shard index; 0-based.
    shard_index: int = 0
    # Count of shards to split the cache build into; 1 means no sharding.
    shard_count: int = 1
    # Each cache build gets an id to differentiate it from other builds of the same config.
    cache_id: str = ""
    # True builds every frame so a cache store serves any anchor stride a later training run picks.
    # False caches only the anchors the training plan itself selects, e.g. for CI.
    build_dense_cache: bool = True
