from __future__ import annotations

import faulthandler
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2
import hydra
import lightning as L
import numba
import torch
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from typing_extensions import override

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.contract_verifications import verify_offline_data_source_scenes_declared_intervals
from py123d_garage.cache._lmdb_backend import LmdbCacheWriter, read_cached_tensor_addresses
from py123d_garage.cache.cache_store import (
    check_builder_cache_signature,
    delete_manifest,
    read_manifest,
    write_manifest,
)
from py123d_garage.cache.codec import TensorCodec, encode_tensor
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize_cache,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import log_stage, setup_logging
from py123d_garage.config.schema.cache.cache_config import CacheConfig
from py123d_garage.py123d_help.scene_builders import VerboseThreadPoolExecutor, build_scene_builder, find_log_names

LOG = logging.getLogger(__name__)

# One record per finished shard; the manifest is written once all are there.
_SHARD_RECORD_DIR = "shards"


class _SceneToBlobsDataset(TorchDataset[tuple[str, str, dict[str, bytes]]]):
    def __init__(
        self,
        scenes: Sequence[SceneAPI],
        policy: AnyPolicy,
        tensor_codecs: dict[str, TensorCodec],
    ):
        self._scenes = scenes
        self._policy = policy
        self._tensor_codecs = tensor_codecs

    def __len__(self) -> int:
        return len(self._scenes)

    @override
    def __getitem__(self, idx: int) -> tuple[str, str, dict[str, bytes]]:
        """Returns one scene's log name, scene uuid, and its encoded blobs by tensor name."""
        scene: SceneAPI = self._scenes[idx]
        sample_tensors: dict[str, torch.Tensor] = (
            self._policy.build_features(
                scene,
                {},
            ).as_dict()
            | self._policy.build_labels(scene, {}).as_dict()
        )
        missing_tensor_names: set[str] = self._tensor_codecs.keys() - sample_tensors.keys()
        if missing_tensor_names:
            raise ValueError(
                f"sample {scene.scene_uuid} of log {scene.log_name} built no value for "
                f"{sorted(missing_tensor_names)}; builders must emit the same tensors for every scene.",
            )
        encoded_blobs: dict[str, bytes] = {
            tensor_name: encode_tensor(sample_tensors[tensor_name], codec)
            for tensor_name, codec in self._tensor_codecs.items()
        }
        return scene.log_name, scene.scene_uuid, encoded_blobs


def _worker_init(_worker_id: int) -> None:
    """One thread per worker: there are already as many workers as cores."""
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    numba.set_num_threads(1)  # pyright: ignore[reportUnknownMemberType]


def _group_scenes_by_log_name(
    scenes: Sequence[SceneAPI],
) -> dict[str, list[SceneAPI]]:
    scenes_by_log_name: dict[str, list[SceneAPI]] = defaultdict(list)
    for scene in scenes:
        scenes_by_log_name[scene.log_name].append(scene)
    return scenes_by_log_name


def _shard_record_file(
    cache_root: str,
    cache_id: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    return Path(cache_root) / _SHARD_RECORD_DIR / f"{cache_id}_{shard_index:04d}_of_{shard_count:04d}.yaml"


def _record_finished_shard(
    cache_root: str,
    cache_id: str,
    shard_index: int,
    shard_count: int,
    manifest: dict[str, Any] | None,
) -> bool:
    """
    Records this shard's finished build; True once every shard of the run has one.
    Completion counts only this run's cache_id, ignoring older runs' leftovers.
    """
    import yaml

    record_file = _shard_record_file(
        cache_root,
        cache_id,
        shard_index,
        shard_count,
    )
    record_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = record_file.with_suffix(".tmp")
    temporary_file.write_text(yaml.safe_dump(manifest, sort_keys=False))
    temporary_file.replace(record_file)
    return all(_shard_record_file(cache_root, cache_id, index, shard_count).is_file() for index in range(shard_count))


def _manifest_from_shard_records(
    cache_root: str,
    cache_id: str,
    shard_count: int,
) -> dict[str, Any] | None:
    """The manifest a non-empty shard of the run recorded, or None when all were empty."""
    import yaml

    for index in range(shard_count):
        manifest = yaml.safe_load(
            _shard_record_file(
                cache_root,
                cache_id,
                index,
                shard_count,
            ).read_text(),
        )
        if manifest is not None:
            return manifest
    return None


def _finish_shard(
    cache_config: CacheConfig,
    cache_root: str,
    manifest: dict[str, Any] | None,
) -> None:
    """
    Records this shard and seals the store when it is the run's last. An empty
    shard records None; the seal takes a non-empty shard's manifest.
    """
    if not _record_finished_shard(
        cache_root,
        cache_config.cache_id,
        cache_config.shard_index,
        cache_config.shard_count,
        manifest,
    ):
        LOG.info(
            f"Shard {cache_config.shard_index} of {cache_config.shard_count} built; "
            f"the store is sealed once every shard is",
        )
        return
    if manifest is None:
        manifest = _manifest_from_shard_records(
            cache_root,
            cache_config.cache_id,
            cache_config.shard_count,
        )
    if manifest is None:
        raise RuntimeError(
            f"every shard of {cache_config.shard_count} was empty: no scenes pass the filter anywhere; nothing to seal",
        )
    write_manifest(cache_root, manifest)
    LOG.info(
        f"Every shard of {cache_config.shard_count} is built; wrote the manifest",
    )


def _check_no_unsealed_leftovers(
    cache_root: str,
    shard_log_names: list[str],
) -> None:
    """
    Refuse to resume onto unsealed blobs: their config is unrecorded. Probes only this
    shard's logs, since sibling shards write elsewhere before the store is sealed.
    """
    if read_manifest(cache_root) is not None:
        return
    leftover_log_names = [name for name in shard_log_names if (Path(cache_root) / name).is_dir()]
    if leftover_log_names:
        raise RuntimeError(
            f"{cache_root} has no manifest but already holds {len(leftover_log_names)} of this "
            f"shard's logs (first: {leftover_log_names[0]}): a build crashed or never sealed, "
            f"and those blobs' config is unknown. Rebuild with force_cache_rebuild=true, or "
            f"delete the store.",
        )


def _is_scene_fully_stored(
    scene: SceneAPI,
    required_tensor_names: list[str],
    stored_tensor_keys: set[str],
) -> bool:
    return all(f"{scene.scene_uuid}/{tensor_name}" in stored_tensor_keys for tensor_name in required_tensor_names)


def _select_scenes_missing_from_store(
    scenes: Sequence[SceneAPI],
    cache_root: str,
    required_tensor_names: list[str],
    force_rebuild: bool,
) -> list[SceneAPI]:
    """
    Selects the scenes not yet fully stored, ordered log by log so the write
    loop keeps a single LMDB writer open.
    """
    scenes_by_log_name: dict[str, list[SceneAPI]] = _group_scenes_by_log_name(
        scenes,
    )

    selected_scenes: list[SceneAPI] = []
    for log_name in sorted(scenes_by_log_name):
        stored_tensor_keys: set[str] = set() if force_rebuild else read_cached_tensor_addresses(cache_root, log_name)
        selected_scenes.extend(
            scene
            for scene in scenes_by_log_name[log_name]
            if force_rebuild
            or not _is_scene_fully_stored(
                scene,
                required_tensor_names,
                stored_tensor_keys,
            )
        )
    return selected_scenes


register_schema("build_cache", CacheConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="build_cache", version_base=None)
def main(cfg: DictConfig) -> None:
    """Builds or extends one sealed cache store per data source; resuming a crashed (unsealed) build needs force_cache_rebuild=true."""
    setup_logging()
    # Print the C-level stack trace when a worker dies on a fatal signal.
    faulthandler.enable()
    cache_config: CacheConfig = finalize_cache(cfg, CacheConfig)
    assert cache_config.offline_data_sources, "offline_data_sources must not be empty"
    assert cache_config.shard_count == 1 or cache_config.cache_id, (
        "sharded builds need a run-unique cache_id shared by every shard"
    )
    assert 0 <= cache_config.shard_index < cache_config.shard_count, (
        f"shard_index {cache_config.shard_index} outside [0, {cache_config.shard_count})"
    )
    save_config(cache_config)

    L.seed_everything(cache_config.seed, workers=True)
    LOG.info(
        f"Seed {cache_config.seed}, run outputs in {run_dir()}",
    )
    LOG.info(
        f"Shard {cache_config.shard_index} of {cache_config.shard_count}, "
        f"{len(cache_config.offline_data_sources)} data sources, "
        f"{cache_config.dataloader_config.num_workers} workers"
        f"{', forced rebuild' if cache_config.force_cache_rebuild else ''}",
    )

    # The builders do not need the neural network, so the policy is never initialize()d here.
    with log_stage(f"Building policy {cache_config.policy_config.target}"):
        policy = cast(
            AnyPolicy,
            build_from_string(cache_config.policy_config, AbstractPolicy),
        )

    for source_index, source in enumerate(cache_config.offline_data_sources.values()):
        LOG.info(
            f"Source {source_index + 1} of {len(cache_config.offline_data_sources)}: {source.data_root} -> {source.cache_root}",
        )
        _build_store(cache_config, policy, source)
    LOG.info("Every data source of this shard is built")


def _build_store(
    cache_config: CacheConfig,
    policy: AnyPolicy,
    source: OfflineTrainingDataSourceConfig,
) -> None:
    """Builds or extends one source's store; this job writes only its shard of it."""
    cache_root: str = source.cache_root

    if cache_config.force_cache_rebuild:
        delete_manifest(cache_root)

    # Shard before reading anything: only this job's logs get opened and filtered.
    scene_filter: SceneFilter = source.garage_scene_filter.to_py123d_scene_filter()
    policy.verify_contract(offline_data_source_config=source, scene_filter=scene_filter)
    with log_stage(f"Listing logs of splits {scene_filter.split_names}"):
        log_names: list[str] = scene_filter.log_names or find_log_names(
            source.data_root,
            scene_filter.split_names,
        )
    shard_log_names = sorted(log_names)[cache_config.shard_index :: cache_config.shard_count]
    if not shard_log_names:
        LOG.warning(
            f"shard {cache_config.shard_index} of {cache_config.shard_count} covers none of "
            f"the {len(log_names)} logs; recording an empty shard",
        )
        _finish_shard(cache_config, cache_root, manifest=None)
        return
    scene_filter.log_names = shard_log_names
    if not cache_config.force_cache_rebuild:
        _check_no_unsealed_leftovers(cache_root, shard_log_names)

    scene_builder: ArrowSceneBuilder = build_scene_builder(
        source.data_root,
    )
    with log_stage(
        f"Filtering scenes of this shard's {len(shard_log_names)} of {len(log_names)} logs",
    ):
        scenes = scene_builder.get_scenes(
            filter=scene_filter,
            executor=VerboseThreadPoolExecutor(),
        )
    if not scenes:
        LOG.warning(
            f"shard {cache_config.shard_index} of {cache_config.shard_count} covers "
            f"{len(shard_log_names)} logs, none of whose scenes pass the filter; "
            f"recording an empty shard",
        )
        _finish_shard(cache_config, cache_root, manifest=None)
        return
    LOG.info(
        f"Shard {cache_config.shard_index} of {cache_config.shard_count}: "
        f"{len(scenes)} scenes from {len(shard_log_names)} of {len(log_names)} logs "
        f"into {cache_root}",
    )
    verify_offline_data_source_scenes_declared_intervals(source, scenes[0])

    cache_signature: dict[str, dict[str, str]] = policy.cache_signature(
        scenes[0].scene_metadata.dataset,
    )
    if not cache_config.force_cache_rebuild:
        check_builder_cache_signature(
            read_manifest(cache_root),
            cache_signature,
        )

    tensor_codecs = policy.cache_codecs()
    tensor_codec_specs: dict[str, dict[str, Any]] = {name: codec.spec for name, codec in tensor_codecs.items()}
    LOG.info(f"Stored tensors and their codecs: {tensor_codec_specs}")

    with log_stage(f"Scanning {cache_root} for scenes it already holds"):
        scenes_to_build: list[SceneAPI] = _select_scenes_missing_from_store(
            scenes,
            cache_root,
            list(tensor_codecs),
            cache_config.force_cache_rebuild,
        )
    if scenes_to_build:
        LOG.info(
            f"Building {len(scenes_to_build)} scenes; {len(scenes) - len(scenes_to_build)} are already stored",
        )
        encoding_dataset = _SceneToBlobsDataset(
            scenes_to_build,
            policy,
            tensor_codecs,
        )

        # collate_fn keeps the batch a plain list of records; the default would try to stack bytes.
        encoding_loader = DataLoader(
            encoding_dataset,
            collate_fn=list,
            worker_init_fn=_worker_init,
            **asdict(cache_config.dataloader_config),
        )

        LOG.info(
            f"Caching {len(scenes_to_build)} scenes with "
            f"{cache_config.dataloader_config.num_workers} workers; the first batch "
            f"lands once a worker has built {cache_config.dataloader_config.batch_size} scenes",
        )

        # LMDB permits one writer; scenes arrive log-contiguous, so one open writer suffices.
        writer: LmdbCacheWriter | None = None
        writer_log_name: str | None = None
        written_scenes = written_bytes = 0
        try:
            for encoded_batch in tqdm(encoding_loader, desc="Caching Dataset"):
                for log_name, scene_uuid, encoded_blobs in encoded_batch:
                    if log_name != writer_log_name:
                        if writer is not None:
                            writer.close()
                        writer = LmdbCacheWriter(
                            cache_root,
                            log_name,
                        )
                        writer_log_name = log_name
                    assert writer is not None
                    for tensor_name, encoded_blob in encoded_blobs.items():
                        written_bytes += writer.write(
                            f"{scene_uuid}/{tensor_name}",
                            encoded_blob,
                        )
                    written_scenes += 1
        finally:
            if writer is not None:
                writer.close()
        LOG.info(
            f"Cached {written_scenes} samples ({written_bytes / 1e9:.2f} GB) into {cache_root}",
        )
    else:
        LOG.info(
            f"Nothing to build: all {len(scenes)} scenes of the shard are already stored",
        )

    _finish_shard(
        cache_config,
        cache_root,
        manifest={
            "tensors": tensor_codec_specs,
            "cache_signature": cache_signature,
        },
    )


if __name__ == "__main__":
    main()
