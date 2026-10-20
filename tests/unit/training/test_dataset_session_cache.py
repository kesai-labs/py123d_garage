# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The session cache memoizes the policy's declared tensors per scene: built once, then served without a scene read."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from diskcache import FanoutCache
from py123d.api import SceneAPI
from torch import Tensor
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, CacheTensorSpec
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig, uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.training.dataset import (
    TrainingDataset,
    TrainingSample,
    _SceneHandle,  # pyright: ignore[reportPrivateUsage]
    decode_session_tensors,
    encode_session_tensors,
    pack_scene_uuids,
)


@dataclass(frozen=True)
class _Features(AbstractFeatures):
    velocity: Tensor


@dataclass(frozen=True)
class _Predictions(AbstractPredictions):
    trajectory: Tensor

    @property
    @override
    def ego_trajectory_xy(self) -> Tensor:
        return self.trajectory[..., :2]

    @property
    @override
    def ego_trajectory_se2(self) -> Tensor:
        return self.trajectory


class _Scene:
    def __init__(self, scene_uuid: str, velocity: float) -> None:
        self.scene_uuid = scene_uuid
        self.velocity = velocity
        self.reads = 0


SceneAPI.register(_Scene)


class _CountingPolicy(AbstractPolicy[_Features, AbstractLabels, _Predictions]):
    """Reads the velocity from the scene and counts every read."""

    @property
    @override
    def policy_config(self) -> AbstractPolicyConfig:
        return TransfuserConfig(
            trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
            required_target_point_distances_m=[45.0],
        )

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        pass

    @override
    def forward(self, features: _Features, navigation: NavigationConditioning) -> _Predictions:
        return _Predictions(trajectory=torch.zeros(1, 1, 3))

    @override
    def build_features(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> _Features:
        if "velocity" in cached:
            return _Features(velocity=cached["velocity"])
        scene = cast(_Scene, scene_api)
        scene.reads += 1
        return _Features(velocity=torch.tensor([scene.velocity]))

    @override
    def build_labels(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> AbstractLabels:
        return AbstractLabels()

    @override
    def _permanent_cache_specs(self) -> dict[str, CacheTensorSpec]:
        return {}

    @override
    def session_cache_names(self) -> frozenset[str]:
        return frozenset({"velocity", "target_points", "target_point_headings"})

    @override
    def compute_loss(self, labels: AbstractLabels, predictions: _Predictions) -> dict[str, Tensor]:
        return {}

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        return {}


@pytest.fixture
def scenes(monkeypatch: pytest.MonkeyPatch) -> list[_Scene]:
    """Two scenes whose target points come from the scene's velocity, counted like a scene read."""

    def from_scene(scene_api: SceneAPI, target_point_distances_m: list[float]) -> NavigationConditioning:
        scene = cast(_Scene, scene_api)
        scene.reads += 1
        return NavigationConditioning(
            target_points=torch.tensor([[scene.velocity, 0.0]]),
            target_point_headings=torch.zeros(1),
        )

    monkeypatch.setattr(NavigationConditioning, "from_scene", staticmethod(from_scene))
    return [_Scene("a", 1.0), _Scene("b", 2.0)]


def _velocity(sample: TrainingSample) -> Tensor:
    return cast(_Features, sample.features).velocity


def _dataset(scenes: list[_Scene], session_cache: FanoutCache | None) -> TrainingDataset:
    return TrainingDataset(
        cast("list[SceneAPI]", scenes),
        _CountingPolicy(),
        permanent_cache=None,
        session_cache=session_cache,
        check_scene_coverage=False,
    )


def test_second_access_reads_nothing_from_the_scene(scenes: list[_Scene], tmp_path: Path) -> None:
    dataset = _dataset(scenes, FanoutCache(str(tmp_path / "session_cache"), shards=2))

    first = dataset[0]
    assert scenes[0].reads == 2  # velocity and target points

    second = dataset[0]
    assert scenes[0].reads == 2
    torch.testing.assert_close(_velocity(second), _velocity(first))
    torch.testing.assert_close(second.navigation.target_points, first.navigation.target_points)
    assert scenes[1].reads == 0


def test_scenes_are_memoized_separately(scenes: list[_Scene], tmp_path: Path) -> None:
    dataset = _dataset(scenes, FanoutCache(str(tmp_path / "session_cache"), shards=2))

    torch.testing.assert_close(_velocity(dataset[0]), torch.tensor([1.0]))
    torch.testing.assert_close(_velocity(dataset[1]), torch.tensor([2.0]))
    torch.testing.assert_close(dataset[1].navigation.target_points, torch.tensor([[2.0, 0.0]]))
    assert (scenes[0].reads, scenes[1].reads) == (2, 2)


def test_without_session_cache_every_access_reads_the_scene(scenes: list[_Scene]) -> None:
    dataset = _dataset(scenes, None)

    dataset[0]
    dataset[0]
    assert scenes[0].reads == 4


def test_fill_builds_only_the_missing_samples(scenes: list[_Scene], tmp_path: Path) -> None:
    dataset = _dataset(scenes, FanoutCache(str(tmp_path / "session_cache"), shards=2))
    dataset[1]

    assert dataset.fill_session_cache(0, 2) == 1
    assert dataset.fill_session_cache(0, 2) == 0
    torch.testing.assert_close(_velocity(dataset[0]), torch.tensor([1.0]))
    assert (scenes[0].reads, scenes[1].reads) == (2, 2)


_UUIDS = ["00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"]


def _uuid_dataset(scenes: list[_Scene], tmp_path: Path) -> TrainingDataset:
    return TrainingDataset(
        cast(
            "list[SceneAPI]",
            [_Scene(scene_uuid, scene.velocity) for scene_uuid, scene in zip(_UUIDS, scenes, strict=True)],
        ),
        _CountingPolicy(),
        session_cache=FanoutCache(str(tmp_path / "session_cache"), shards=2),
        check_scene_coverage=False,
        source_index=7,
        scene_uuids=pack_scene_uuids(_UUIDS),
    )


def _is_built(sample: TrainingSample) -> bool:
    """Whether the sample's lazy scene handle has built its scene."""
    return cast(_SceneHandle, sample.scene_apis[0])._scene is not None  # pyright: ignore[reportPrivateUsage]


def test_known_uuids_build_the_scene_only_on_a_cache_miss(scenes: list[_Scene], tmp_path: Path) -> None:
    dataset = _uuid_dataset(scenes, tmp_path)

    first = dataset[0]
    assert _is_built(first)
    second = dataset[0]
    assert not _is_built(second)
    torch.testing.assert_close(_velocity(second), _velocity(first))
    assert len(second.scene_apis) == 1


def test_an_unbuilt_scene_pickles_as_its_position(scenes: list[_Scene], tmp_path: Path) -> None:
    dataset = _uuid_dataset(scenes, tmp_path)
    dataset[1]
    handle = cast(_SceneHandle, dataset[1].scene_apis[0])
    assert handle._scene is None  # pyright: ignore[reportPrivateUsage]

    restored = cast(_SceneHandle, pickle.loads(pickle.dumps(handle)))
    assert restored._scene is None  # pyright: ignore[reportPrivateUsage]
    assert restored.scene_uuid == _UUIDS[1]
    assert restored._scene is not None  # pyright: ignore[reportPrivateUsage]


def test_session_blob_round_trips_every_dtype_and_shape() -> None:
    tensors = {
        "camera": torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8),
        "trajectory": torch.randn(6, 3).to(torch.bfloat16),
        "velocity": torch.tensor(3.5),
        "mask": torch.tensor([True, False, True]),
        "empty": torch.zeros(0, 2, dtype=torch.int16),
        "strided": torch.arange(12.0).reshape(3, 4).T,
        # One target point's heading, a column of its (x, y, heading) row: one element with stride 3.
        "single_strided": torch.zeros(1, 3)[:, 2],
    }
    decoded = decode_session_tensors(encode_session_tensors(tensors))
    assert decoded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert decoded[name].dtype == tensor.dtype, name
        assert torch.equal(decoded[name], tensor), name


def test_session_value_of_an_older_run_reads_as_is() -> None:
    tensors = {"velocity": torch.tensor([1.0])}
    assert decode_session_tensors(tensors) is tensors
