# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Training dataset that builds a policy's samples from scenes, reading cached tensors where it can."""

from __future__ import annotations

import logging
import math
import pickle
import uuid
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from diskcache import FanoutCache
from py123d.api import SceneAPI
from py123d.api.scene.arrow.lazy_scene_sequence import LazySceneSequence
from py123d.datatypes import CameraID
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import override

from py123d_garage.api.abstract_offline_data_source_config import HAS_BOX_ANNOTATIONS
from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    NavigationConditioning,
)
from py123d_garage.cache import CacheStoreReader
from py123d_garage.common.logging_setup import log_stage
from py123d_garage.common.sensor.hood_swap import vehicle_type_from_mount
from py123d_garage.config.schema.training.photometric_augmentation_config import HoodSwapConfig

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSample:
    """One sample from the policy's builders; collated, one training batch."""

    # The scene of each row.
    scene_apis: list[SceneAPI]
    # The policy's input features.
    features: AbstractFeatures
    # The policy's training targets.
    labels: AbstractLabels
    # The target points the policy is conditioned on.
    navigation: NavigationConditioning
    # The offline data source each row came from, as its index in the training config's sources.
    source_indices: torch.Tensor
    # The physical-ai-av vehicle type of each row's recording vehicle for the hood swap, -1 for a hoodless one.
    vehicle_types: torch.Tensor

    def to(self, *args: Any, **kwargs: Any) -> TrainingSample:
        """
        Move every tensor with Tensor.to; also serves Lightning's device transfer.

        Args:
            *args: positional arguments passed to Tensor.to
            **kwargs: keyword arguments passed to Tensor.to

        Returns:
            a new sample with the moved tensors
        """
        return replace(
            self,
            features=self.features.to(*args, **kwargs),
            labels=self.labels.to(*args, **kwargs),
            navigation=self.navigation.to(*args, **kwargs),
            source_indices=self.source_indices.to(*args, **kwargs),
            vehicle_types=self.vehicle_types.to(*args, **kwargs),
        )

    def pin_memory(self) -> TrainingSample:
        """
        Pin every tensor; the DataLoader's pin_memory hook.

        Returns:
            a new sample with the pinned tensors
        """
        return replace(
            self,
            features=self.features.pin_memory(),
            labels=self.labels.pin_memory(),
            navigation=self.navigation.pin_memory(),
            source_indices=self.source_indices.pin_memory(),
            vehicle_types=self.vehicle_types.pin_memory(),
        )

    @classmethod
    def collate(cls, samples: list[TrainingSample]) -> TrainingSample:
        """
        Stack the samples of one batch; bundles collate field-wise, the scene lists concatenate.

        Args:
            samples: the samples of one batch, as returned by TrainingDataset.__getitem__.

        Returns:
            the collated batch.
        """
        return cls(
            scene_apis=[scene_api for sample in samples for scene_api in sample.scene_apis],
            features=type(samples[0].features).collate(
                [sample.features for sample in samples],
            ),
            labels=type(samples[0].labels).collate(
                [sample.labels for sample in samples],
            ),
            navigation=NavigationConditioning.collate(
                [sample.navigation for sample in samples],
            ),
            source_indices=torch.cat([sample.source_indices for sample in samples]),
            vehicle_types=torch.cat([sample.vehicle_types for sample in samples]),
        )


# The cameras whose mounts tell the physical-ai-av vehicle types apart.
_VEHICLE_TYPE_CAMERAS = (CameraID.FTCAM_F0, CameraID.FTCAM_L0, CameraID.FTCAM_R0)
# Each log's vehicle type by (dataset, log name), memoized per DataLoader worker.
_vehicle_types: dict[tuple[str, str], int] = {}


# Every dataset's scenes by source index, so a scene handle pickled by a DataLoader worker resolves in the main process.
_scene_sequences: dict[int, Sequence[SceneAPI]] = {}


class _SceneHandle:
    """
    The scene at a position of a source's scene sequence, built on first attribute access.

    Building a scene reads its log's sync table, usually off NFS, which a sample served from the caches never needs.
    """

    def __init__(
        self,
        source_index: int,
        position: int,
        scenes: Sequence[SceneAPI] | None = None,
        scene: SceneAPI | None = None,
    ) -> None:
        """
        Point the handle at one scene without building it.

        Args:
            source_index: the scene's dataset index, used to find its sequence after unpickling
            position: the scene's index in its sequence
            scenes: the scene sequence; None looks it up by source_index
            scene: the already built scene, if any
        """
        self._source_index = source_index
        self._position = position
        self._scenes = scenes
        self._scene = scene

    @property
    def scene(self) -> SceneAPI:
        """The scene, built on the first call."""
        if self._scene is None:
            scenes = self._scenes if self._scenes is not None else _scene_sequences[self._source_index]
            self._scene = scenes[self._position]
        return self._scene

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the scene, building it on first use.

        Args:
            name: the attribute name

        Returns:
            the scene's attribute
        """
        if name.startswith("__") or name in ("_source_index", "_position", "_scenes", "_scene"):
            raise AttributeError(name)
        return getattr(self.scene, name)

    @override
    def __reduce__(self) -> tuple[type[_SceneHandle], tuple[int, int, None, SceneAPI | None]]:
        """Pickle the handle as its position, plus the scene only if already built."""
        return (_SceneHandle, (self._source_index, self._position, None, self._scene))


SceneAPI.register(_SceneHandle)


def pack_scene_uuids(scene_uuids: Sequence[str]) -> jt.UInt8[npt.NDArray[np.uint8], "scenes 16"]:
    """
    Pack scene uuids as one row of 16 bytes each, so forked workers have no per-string objects to copy on access.

    Args:
        scene_uuids: normalized uuid strings, one per scene.

    Returns:
        the uuids as 16 bytes per scene.
    """
    packed = b"".join(uuid.UUID(scene_uuid).bytes for scene_uuid in scene_uuids)
    return np.frombuffer(packed, dtype=np.uint8).reshape(-1, 16).copy()


def _dense_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten a tensor with stride 1, which view(torch.uint8) needs, copying only where its strides differ.

    A one-element tensor counts as contiguous whatever its stride, so contiguous() alone is not enough.

    Args:
        tensor: the tensor to flatten

    Returns:
        the flat tensor with stride 1
    """
    flat = tensor.detach().cpu().reshape(-1)
    return flat if flat.stride(0) == 1 else flat.clone(memory_format=torch.contiguous_format)


def encode_session_tensors(tensors: dict[str, torch.Tensor]) -> bytes:
    """
    Pack tensors into one blob, a pickled header of names, dtypes and shapes followed by their raw bytes.

    Args:
        tensors: CPU tensors by name.

    Returns:
        the blob decode_session_tensors reads.
    """
    flat = {name: _dense_flat(tensor) for name, tensor in tensors.items()}
    header = pickle.dumps(
        [(name, str(tensor.dtype).removeprefix("torch."), tuple(tensors[name].shape)) for name, tensor in flat.items()],
    )
    return b"".join(
        [
            len(header).to_bytes(8, "little"),
            header,
            *(
                tensor.view(torch.uint8).numpy().tobytes()  # pyright: ignore[reportUnknownMemberType]
                for tensor in flat.values()
            ),
        ],
    )


def decode_session_tensors(blob: bytes | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Unpack a blob from encode_session_tensors into views of one copy, passing an older run's pickled dict through.

    Args:
        blob: the session cache's value.

    Returns:
        the tensors by name.
    """
    if isinstance(blob, dict):
        return blob
    header_size = int.from_bytes(blob[:8], "little")
    header = cast("list[tuple[str, str, tuple[int, ...]]]", pickle.loads(blob[8 : 8 + header_size]))
    buffer = bytearray(blob)
    offset = 8 + header_size
    tensors: dict[str, torch.Tensor] = {}
    for name, dtype_name, shape in header:
        dtype = cast(torch.dtype, getattr(torch, dtype_name))
        num_bytes = math.prod(shape) * dtype.itemsize
        if num_bytes == 0:
            tensors[name] = torch.empty(shape, dtype=dtype)
        else:
            raw = torch.frombuffer(buffer, dtype=torch.uint8, count=num_bytes, offset=offset)
            tensors[name] = raw.view(dtype).reshape(shape)
        offset += num_bytes
    return tensors


# Scenes per uuid-reading task of the coverage check's worker pool.
_UUID_SPAN = 8192

# The lazy scenes of the coverage check's worker pool, set in each worker by its initializer.
_pool_scenes: Sequence[SceneAPI] | None = None


def _set_pool_scenes(scenes: Sequence[SceneAPI] | None) -> None:
    """
    Hand the coverage check's worker its scenes.

    Args:
        scenes: the lazy scene sequence, or None for a materialized one
    """
    global _pool_scenes
    _pool_scenes = scenes


def _scene_uuids_between(start: int, stop: int) -> list[str]:
    """
    Read the uuids of a range of the worker's scenes.

    Args:
        start: the first scene position
        stop: the position after the last

    Returns:
        the scene uuids, in order
    """
    assert _pool_scenes is not None
    return [_pool_scenes[position].scene_uuid for position in range(start, stop)]


class TrainingDataset(TorchDataset[TrainingSample]):
    """Training samples built by a policy from scenes, with cached tensors read from the store and session cache."""

    def __init__(
        self,
        scenes: Sequence[SceneAPI],
        policy: AnyPolicy,
        permanent_cache: CacheStoreReader | None = None,
        session_cache: FanoutCache | None = None,
        startup_num_workers: int = 16,
        check_scene_coverage: bool = True,
        verify_store_coverage: bool = True,
        source_index: int = 0,
        hood_swap_config: HoodSwapConfig | None = None,
        scene_uuids: jt.UInt8[npt.NDArray[np.uint8], "scenes 16"] | None = None,
        has_box_annotations: bool = True,
    ):
        """
        Set up the dataset over the selected scenes, checking the cache store covers them.

        Args:
            scenes: the selected scenes, one sample each.
            policy: the policy whose builders turn a scene into a sample.
            permanent_cache: the store built by build_cache with the policy's store tensors; None builds all live.
            session_cache: this job's node-local memo of the policy's session tensors; None builds them every time.
            startup_num_workers: worker processes for the coverage check.
            check_scene_coverage: verify the store holds every selected scene; False only indexes the logs.
            verify_store_coverage: with check_scene_coverage, read the store's keys; False only groups the scenes and
                reads their uuids.
            source_index: the dataset's index in the training config's sources, carried by every sample.
            hood_swap_config: the hood swap's dataset lists when it is on; None makes every sample hoodless (-1).
            scene_uuids: every scene's packed uuid, from a rank that checked the same scenes; None derives them.
            has_box_annotations: whether the source annotates other agents' boxes; False skips box supervision.

        Raises:
            ValueError: when the hood swap is on and the dataset is neither typed nor hoodless.
        """
        self.scenes = scenes
        self._policy = policy
        self._source_index = source_index
        _scene_sequences[source_index] = scenes
        self._hood_swap_config = hood_swap_config
        # With every uuid known, a sample builds its scene only when a cache misses.
        self.scene_uuids = scene_uuids
        self._has_box_annotations = has_box_annotations
        assert scene_uuids is None or len(scene_uuids) == len(scenes), "one uuid per scene"
        self._dataset_name = ""
        if hood_swap_config is not None and scenes:
            dataset = scenes[0].scene_metadata.dataset
            self._dataset_name = dataset
            if dataset not in (*hood_swap_config.typed_datasets, *hood_swap_config.hoodless_datasets):
                raise ValueError(
                    f"hood_swap: dataset {dataset} is neither in typed_datasets nor in hoodless_datasets; add it to one",
                )
        self._target_point_distances_m = policy.policy_config.required_target_point_distances_m
        self._permanent_cache = permanent_cache
        self._session_cache = session_cache
        self._session_tensor_names = policy.session_cache_names() if session_cache is not None else frozenset[str]()
        self.log_names: list[str] = []
        self._startup_num_workers = startup_num_workers
        # The lazy sequence's log name per scene position; scene.log_name would parse every modality schema of its log dir.
        self._scene_log_names: tuple[list[str], Sequence[int]] | None = None
        declared_tensor_names = set(policy.cache_codecs())
        if self._session_tensor_names:
            LOG.info(f"Memoizing {sorted(self._session_tensor_names)} per scene in the session cache")
        if permanent_cache is None:
            LOG.info(
                f"Cache store off: building every tensor live, including {sorted(declared_tensor_names)}",
            )
        else:
            LOG.info(
                f"Reading {sorted(permanent_cache.tensor_names)} from the cache store",
            )
            undeclared_tensor_names = declared_tensor_names - permanent_cache.tensor_names
            if undeclared_tensor_names:
                LOG.warning(
                    f"The store is missing declared tensors {sorted(undeclared_tensor_names)}; they are built live",
                )
            if check_scene_coverage:
                self._check_scene_coverage(permanent_cache, verify_store_coverage)
            else:
                self._index_logs()

    def _index_logs(self) -> None:
        """Index the scenes' log names as the coverage check does, without the check."""
        if isinstance(self.scenes, LazySceneSequence):
            log_names, name_indices, _ = cast(
                "tuple[list[str], Sequence[int], Sequence[int]]",
                self.scenes.anchor_columns(),  # pyright: ignore[reportUnknownMemberType]
            )
            self._scene_log_names = (log_names, name_indices)
            self.log_names = list(dict.fromkeys(log_names[index] for index in name_indices))
        else:
            self.log_names = list(dict.fromkeys(scene.log_name for scene in self.scenes))

    def _check_scene_coverage(self, permanent_cache: CacheStoreReader, verify_store: bool = True) -> None:
        """
        Fail fast when the filter selects scenes the cache store does not hold.

        Args:
            permanent_cache: the store serving this dataset.
            verify_store: read the store's keys; False only groups the scenes and reads their uuids.

        Raises:
            KeyError: if a selected scene is missing from the store.
        """
        # Both stages are GIL-bound, so processes rather than threads.
        lazy_scenes = self.scenes if isinstance(self.scenes, LazySceneSequence) else None
        with ProcessPoolExecutor(
            max_workers=self._startup_num_workers,
            initializer=_set_pool_scenes,
            initargs=(lazy_scenes,),
        ) as executor:
            scene_uuids_by_log: dict[str, list[str]] = defaultdict(list)
            with log_stage(f"Grouping the {len(self.scenes)} selected scenes by log"):
                if lazy_scenes is not None:
                    log_names, name_indices, _ = cast(
                        "tuple[list[str], Sequence[int], Sequence[int]]",
                        lazy_scenes.anchor_columns(),  # pyright: ignore[reportUnknownMemberType]
                    )
                    self._scene_log_names = (log_names, name_indices)
                    starts = range(0, len(self.scenes), _UUID_SPAN)
                    scene_uuids: list[str] = []
                    for span_uuids in executor.map(
                        _scene_uuids_between,
                        starts,
                        [min(start + _UUID_SPAN, len(self.scenes)) for start in starts],
                    ):
                        scene_uuids.extend(span_uuids)
                        if len(scene_uuids) % (25 * _UUID_SPAN) == 0:
                            LOG.info(f"Grouped {len(scene_uuids)}/{len(self.scenes)} scenes")
                    for scene_uuid, name_index in zip(scene_uuids, name_indices, strict=True):
                        scene_uuids_by_log[log_names[name_index]].append(scene_uuid)
                    self.scene_uuids = pack_scene_uuids(scene_uuids)
                else:
                    for scene in self.scenes:
                        scene_uuids_by_log[scene.log_name].append(scene.scene_uuid)
            self.log_names = list(scene_uuids_by_log)
            if verify_store:
                permanent_cache.check_covers(scene_uuids_by_log, executor)
            else:
                LOG.info("check_cache_coverage is off: the store is trusted to hold every selected scene")

    def _vehicle_type(self, scene: SceneAPI, dataset: str, log_name: str) -> int:
        """
        The physical-ai-av vehicle type of the scene's recording vehicle, from its camera mounts.

        Args:
            scene: the scene, read once per log for its camera mounts.
            dataset: the scene's dataset.
            log_name: the scene's log.

        Returns:
            the vehicle type; -1 when the hood swap is off or the dataset is hoodless.

        Raises:
            ValueError: when a typed log lacks a mount camera or matches no known type.
        """
        config = self._hood_swap_config
        if config is None or dataset in config.hoodless_datasets:
            return -1
        key = (dataset, log_name)
        if key not in _vehicle_types:
            metadatas = scene.get_camera_metadatas()
            missing = [camera_id.name for camera_id in _VEHICLE_TYPE_CAMERAS if camera_id not in metadatas]
            if missing:
                raise ValueError(f"hood_swap: log {log_name} of {dataset} lacks the mount cameras {missing}")
            mounts = [metadatas[camera_id].camera_to_imu_se3 for camera_id in _VEHICLE_TYPE_CAMERAS]
            try:
                _vehicle_types[key] = vehicle_type_from_mount(
                    [value for pose in mounts for value in (pose.x, pose.y, pose.z)],
                )
            except ValueError as error:
                raise ValueError(f"hood_swap: log {log_name} of {dataset}: {error}") from error
        return _vehicle_types[key]

    def __len__(self) -> int:
        """The number of selected scenes."""
        return len(self.scenes)

    def prefetch(self, indices: Sequence[int], executor: Executor) -> None:
        """
        Start reading samples' stored records from disk ahead of __getitem__.

        Without a store, scene uuids or per-scene log names there is nothing to look up, so it does nothing.

        Args:
            indices: the samples' indices
            executor: the threads doing the lookups
        """
        if self._permanent_cache is None or self.scene_uuids is None or self._scene_log_names is None:
            return
        log_names, name_indices = self._scene_log_names
        scene_uuids = self.scene_uuids
        self._permanent_cache.prefetch(
            [(log_names[name_indices[idx]], str(uuid.UUID(bytes=scene_uuids[idx].tobytes()))) for idx in indices],
            executor,
        )

    @override
    def __getitem__(self, idx: int) -> TrainingSample:
        """Build one sample from the caches, reading the scene only for tensors they miss."""
        if self.scene_uuids is None:
            scene: SceneAPI = self.scenes[idx]
            scene_uuid = scene.scene_uuid
        else:
            scene = cast(SceneAPI, _SceneHandle(self._source_index, idx, self.scenes))
            scene_uuid = str(uuid.UUID(bytes=self.scene_uuids[idx].tobytes()))
        if self._scene_log_names is None:
            log_name = scene.log_name if self._permanent_cache is not None or self._hood_swap_config else ""
        else:
            log_names, name_indices = self._scene_log_names
            log_name = log_names[name_indices[idx]]
        permanent_tensors: dict[str, torch.Tensor] = {}
        if self._permanent_cache is not None:
            permanent_tensors = self._permanent_cache.read(log_name, scene_uuid)
            permanent_tensors[HAS_BOX_ANNOTATIONS] = torch.tensor(self._has_box_annotations)
        # One entry per scene holding every session tensor: one lookup on a hit, one write on a miss.
        session_tensors: dict[str, torch.Tensor] | None = None
        if self._session_cache is not None:
            blob = cast("bytes | dict[str, torch.Tensor] | None", self._session_cache.get(scene_uuid))
            session_tensors = None if blob is None else decode_session_tensors(blob)
        cached_tensors = permanent_tensors | (session_tensors or {})
        sample = TrainingSample(
            scene_apis=[scene],
            features=self._policy.build_features(scene, cached_tensors),
            labels=self._policy.build_labels(scene, cached_tensors),
            navigation=NavigationConditioning(
                target_points=cached_tensors["target_points"],
                target_point_headings=cached_tensors["target_point_headings"],
            )
            if "target_points" in cached_tensors
            else NavigationConditioning.from_scene(scene, self._target_point_distances_m),
            source_indices=torch.tensor([self._source_index]),
            vehicle_types=torch.tensor([self._vehicle_type(scene, self._dataset_name, log_name)]),
        )
        if self._session_cache is not None and session_tensors is None:
            built_tensors = sample.features.as_dict() | sample.labels.as_dict() | sample.navigation.as_dict()
            # A locked shard skips this write; the next epoch builds the tensors again.
            self._session_cache.set(
                scene_uuid,
                encode_session_tensors(
                    {
                        name: tensor
                        for name in self._session_tensor_names
                        if (tensor := built_tensors.get(name)) is not None
                    },
                ),
                retry=False,
            )
        return sample

    def fill_session_cache(self, first_index: int, end_index: int) -> int:
        """
        Build the samples in a range whose session tensors the session cache lacks, which stores them.

        Args:
            first_index: the first sample index
            end_index: the index after the last

        Returns:
            the number of samples built
        """
        assert self._session_cache is not None, "no session cache to fill"
        built_samples = 0
        for idx in range(first_index, end_index):
            if self.scene_uuids is None:
                scene_uuid = self.scenes[idx].scene_uuid
            else:
                scene_uuid = str(uuid.UUID(bytes=self.scene_uuids[idx].tobytes()))
            if scene_uuid not in self._session_cache:
                self[idx]
                built_samples += 1
        return built_samples
