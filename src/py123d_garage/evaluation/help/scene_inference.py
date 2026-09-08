"""Turning one offline data source into this shard's scenes, then into predicted trajectories."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import Executor
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import override

from py123d_garage.api.abstract_offline_data_source_config import AbstractOfflineDataSourceConfig
from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.api.contract_verifications import verify_offline_data_source_scenes_declared_intervals
from py123d_garage.cache import CacheStoreReader
from py123d_garage.datatypes.tensor import TensorBundle
from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.py123d_help.scene_builders import build_scene_builder, find_log_names

LOG = logging.getLogger(__name__)

BundleT = TypeVar("BundleT", bound=TensorBundle)


def build_source_scenes(
    source: AbstractOfflineDataSourceConfig,
    shard_index: int,
    num_shards: int,
    executor: Executor,
    py123d_garage_policy: AnyPolicy,
) -> list[SceneAPI]:
    """
    Builds this shard's slice of one source's benchmark scenes.

    Args:
        source: the dataset to read.
        shard_index: this process's slice of the source's logs.
        num_shards: how many slices the source's logs are cut into.
        executor: worker pool for scene building.
        py123d_garage_policy: the policy the scenes will feed; refuses an unservable filter.

    Returns:
        the shard's scenes of this source.
    """
    authored_filter = source.garage_scene_filter
    log_names = authored_filter.log_names or find_log_names(
        source.data_root,
        authored_filter.split_names,
    )
    shard_log_names = sorted(log_names)[shard_index::num_shards]
    scene_filter: SceneFilter = replace(
        authored_filter.to_py123d_scene_filter(),
        log_names=shard_log_names,
    )
    py123d_garage_policy.verify_contract(offline_data_source_config=source, scene_filter=scene_filter)
    scene_builder: ArrowSceneBuilder = build_scene_builder(
        source.data_root,
        lazy=False,
    )
    scenes: list[SceneAPI] = list(
        scene_builder.get_scenes(
            filter=scene_filter,
            executor=executor,
        ),
    )
    LOG.info(
        f"Source {source.data_root}, shard "
        f"{shard_index}/{num_shards} "
        f"({len(shard_log_names)}/{len(log_names)} logs): "
        f"{len(scenes)} scenes passed the filter",
    )
    if scenes:
        verify_offline_data_source_scenes_declared_intervals(source, scenes[0])
    return scenes


def run_shard_inference(
    scenes: list[SceneAPI],
    py123d_garage_policy: AnyPolicy,
    py123d_garage_feature_policy: AnyPolicy,
    cache_root: str | None,
    inference_batch_size: int,
    max_workers: int,
    visualization_dir: Path | None = None,
) -> list[TrajectorySE2]:
    """
    Predicts one trajectory per scene, building the features in loader workers.

    Args:
        scenes: the shard's scenes of one source.
        py123d_garage_policy: the initialized policy, already on its device.
        py123d_garage_feature_policy: an un-initialized copy the loader workers build features with.
        cache_root: feature cache store of the source; None = build every feature from the scene.
        inference_batch_size: scenes per forward pass.
        max_workers: loader worker processes.
        visualization_dir: writes every scene's ground-truth and prediction views side by side here as one
            JPEG, building the labels the ground-truth view draws; None = no views, no labels.

    Returns:
        the predicted trajectory of every scene, in the order the scenes were given.
    """
    cache_reader = (
        CacheStoreReader(
            cache_root,
            py123d_garage_policy.cache_signature(scenes[0].scene_metadata.dataset),
        )
        if cache_root is not None
        else None
    )
    device = next(py123d_garage_policy.parameters()).device
    device_type: str = device.type
    py123d_garage_policy.eval()
    data_loader: DataLoader[SceneSample] = DataLoader(
        OfflineEvaluationDataset(
            scenes,
            py123d_garage_feature_policy,
            cache_reader,
            build_labels=visualization_dir is not None,
        ),
        batch_size=inference_batch_size,
        num_workers=max_workers,
        collate_fn=_collate_scene_samples,
        pin_memory=device_type != "cpu",
    )
    trajectories: list[TrajectorySE2] = []
    for batch_index, (features, labels, navigation) in enumerate(
        tqdm(data_loader, desc="Inference"),
    ):
        # The loader preserves dataset order, so batch i covers this slice.
        batch_scenes = scenes[batch_index * inference_batch_size : (batch_index + 1) * inference_batch_size]
        assert len(navigation.target_points) == len(batch_scenes), (
            f"batch {batch_index} holds {len(navigation.target_points)} samples for {len(batch_scenes)} scenes"
        )
        with torch.no_grad():
            predictions = py123d_garage_policy.forward(features.to(device), navigation.to(device))
        trajectories.extend(
            py123d_garage_policy.trajectories_from_predictions(predictions, batch_scenes),
        )
        if visualization_dir is not None:
            _save_scene_views(
                py123d_garage_policy,
                features,
                labels,
                navigation,
                predictions.to("cpu"),
                batch_scenes,
                visualization_dir,
            )
    return trajectories


def _save_scene_views(
    py123d_garage_policy: AnyPolicy,
    features: AbstractFeatures,
    labels: AbstractLabels | None,
    navigation: NavigationConditioning,
    predictions: AbstractPredictions,
    scenes: list[SceneAPI],
    visualization_dir: Path,
) -> None:
    """
    Writes one JPEG per scene: the policy's ground-truth view left, its prediction view right.

    Other views the policy renders are left out.

    Args:
        py123d_garage_policy: the policy that rendered the batch.
        features: the collated feature bundle of the batch.
        labels: the collated label bundle of the batch; None leaves out the ground-truth view.
        navigation: the collated navigation conditioning of the batch.
        predictions: the collated prediction bundle of the batch.
        scenes: the batch's scenes, in batch order.
        visualization_dir: the output dir of the JPEGs.
    """
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(scenes):
        views = py123d_garage_policy.visualize_batch(
            _sample_at(features, index),
            _sample_at(labels, index) if labels is not None else None,
            _sample_at(navigation, index),
            _sample_at(predictions, index),
            scene,
        )
        panels = [
            cast("npt.NDArray[np.uint8]", views[name].numpy())  # pyright: ignore[reportUnknownMemberType]
            for name in ("labels", "predictions")
            if name in views
        ]
        if not panels:
            continue
        cv2.imwrite(
            str(visualization_dir / f"{scene.log_name}_{scene.scene_uuid}.jpg"),
            cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR),
        )


def _sample_at(bundle: BundleT, index: int) -> BundleT:
    """One sample of a collated bundle, kept batched with a leading size of one."""
    return bundle.apply(lambda tensor: tensor[index : index + 1])


SceneSample: TypeAlias = tuple[AbstractFeatures, AbstractLabels | None, NavigationConditioning]


class OfflineEvaluationDataset(Dataset[SceneSample]):
    """One scene's features, optional labels, and navigation conditioning per index; runs in loader workers."""

    def __init__(
        self,
        scenes: list[SceneAPI],
        py123d_garage_policy: AnyPolicy,
        cache_reader: CacheStoreReader | None = None,
        build_labels: bool = False,
    ) -> None:
        self._scenes = scenes
        self._policy = py123d_garage_policy
        self._cache_reader = cache_reader
        self._build_labels = build_labels
        self._target_point_distances_m: list[float] = (
            py123d_garage_policy.policy_config.required_target_point_distances_m
        )

    def __len__(self) -> int:
        return len(self._scenes)

    @override
    def __getitem__(self, index: int) -> SceneSample:
        scene = self._scenes[index]
        cached: dict[str, Tensor] = {}
        if self._cache_reader is not None:
            cached = self._cache_reader.read(
                scene.log_name,
                scene.scene_uuid,
            )
        features: AbstractFeatures = self._policy.build_features(scene, cached)
        labels: AbstractLabels | None = self._policy.build_labels(scene, cached) if self._build_labels else None
        navigation = NavigationConditioning.from_scene(
            scene,
            self._target_point_distances_m,
        )
        return features, labels, navigation


def _collate_scene_samples(samples: list[SceneSample]) -> SceneSample:
    """
    Stacks loader samples into one collated feature bundle, label bundle, and navigation batch.

    Args:
        samples: the loader's (features, labels, navigation) triples of one batch.

    Returns:
        the collated bundles; labels stay None when the samples carry none.
    """
    features = type(samples[0][0]).collate([sample_features for sample_features, _, _ in samples])
    sample_labels = [labels for _, labels, _ in samples if labels is not None]
    labels = type(sample_labels[0]).collate(sample_labels) if sample_labels else None
    navigation = NavigationConditioning.collate([sample_navigation for _, _, sample_navigation in samples])
    return features, labels, navigation
