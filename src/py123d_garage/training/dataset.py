from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from py123d.api import SceneAPI
from py123d.api.scene.arrow.lazy_scene_sequence import LazySceneSequence
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    NavigationConditioning,
)
from py123d_garage.cache import CacheStoreReader
from py123d_garage.common.logging_setup import log_stage

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSample:
    """One sample from the policy's builders; collated, one training batch."""

    scene_apis: list[SceneAPI]
    features: AbstractFeatures
    labels: AbstractLabels
    navigation: NavigationConditioning

    def to(self, *args: Any, **kwargs: Any) -> TrainingSample:
        """A new sample with every tensor moved via Tensor.to; also serves Lightning's device transfer."""
        return replace(
            self,
            features=self.features.to(*args, **kwargs),
            labels=self.labels.to(*args, **kwargs),
            navigation=self.navigation.to(*args, **kwargs),
        )

    def pin_memory(self) -> TrainingSample:
        """A new sample with every tensor pinned; the DataLoader's pin_memory hook."""
        return replace(
            self,
            features=self.features.pin_memory(),
            labels=self.labels.pin_memory(),
            navigation=self.navigation.pin_memory(),
        )

    @classmethod
    def collate(cls, samples: list[TrainingSample]) -> TrainingSample:
        """
        Stacks the samples of one batch; bundles collate field-wise, the scene lists concatenate.

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
        )


# -- Start of internals of TrainingDataset._check_scene_coverage's worker pool. --
_UUID_SPAN = 8192

_pool_scenes: Sequence[SceneAPI] | None = None


def _set_pool_scenes(scenes: Sequence[SceneAPI] | None) -> None:
    global _pool_scenes
    _pool_scenes = scenes


def _scene_uuids_between(start: int, stop: int) -> list[str]:
    assert _pool_scenes is not None
    return [_pool_scenes[position].scene_uuid for position in range(start, stop)]


# -- End of internals of TrainingDataset._check_scene_coverage's worker pool. --


class TrainingDataset(TorchDataset[TrainingSample]):
    def __init__(
        self,
        scenes: Sequence[SceneAPI],
        policy: AnyPolicy,
        cache_reader: CacheStoreReader | None = None,
        startup_num_workers: int = 16,
    ):
        """The policy's builders define what a sample is; cache_reader serves its cached tensors."""
        self.scenes = scenes
        self._policy = policy
        self._target_point_distances_m = policy.policy_config.required_target_point_distances_m
        self._cache_reader = cache_reader
        self._startup_num_workers = startup_num_workers
        declared_tensor_names = set(policy.cache_codecs())
        if cache_reader is None:
            LOG.info(
                f"Cache store off: building every tensor live, including {sorted(declared_tensor_names)}",
            )
        else:
            LOG.info(
                f"Reading {sorted(cache_reader.tensor_names)} from the cache store",
            )
            undeclared_tensor_names = declared_tensor_names - cache_reader.tensor_names
            if undeclared_tensor_names:
                LOG.warning(
                    f"The store is missing declared tensors {sorted(undeclared_tensor_names)}; they are built live",
                )
            self._check_scene_coverage(cache_reader)

    def _check_scene_coverage(self, cache_reader: CacheStoreReader) -> None:
        """
        Fails fast when the filter selects scenes the cache store does not hold.

        Args:
            cache_reader: the store serving this dataset.

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
                    # scene.log_name would parse every modality schema of its log dir.
                    log_names, name_indices, _ = cast(
                        "tuple[list[str], Sequence[int], Sequence[int]]",
                        lazy_scenes.anchor_columns(),  # pyright: ignore[reportUnknownMemberType]
                    )
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
                else:
                    for scene in self.scenes:
                        scene_uuids_by_log[scene.log_name].append(scene.scene_uuid)
            cache_reader.check_covers(scene_uuids_by_log, executor)

    def __len__(self) -> int:
        return len(self.scenes)

    @override
    def __getitem__(self, idx: int) -> TrainingSample:
        """One sample: the scene's features, labels, and target points."""
        scene: SceneAPI = self.scenes[idx]
        cached: dict[str, torch.Tensor] = {}
        if self._cache_reader is not None:
            cached = self._cache_reader.read(
                scene.log_name,
                scene.scene_uuid,
            )
        return TrainingSample(
            scene_apis=[scene],
            features=self._policy.build_features(scene, cached),
            labels=self._policy.build_labels(scene, cached),
            navigation=NavigationConditioning.from_scene(
                scene,
                self._target_point_distances_m,
            ),
        )
