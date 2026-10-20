# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from py123d.api import SceneAPI
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from py123d_garage.api.abstract_policy_tensors import AbstractFeatures, AbstractLabels, NavigationConditioning
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.pinned_batches import PinnedBatchSlots, SlotBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="page-locking needs CUDA")

_BATCH_SIZE = 4


@dataclass(frozen=True)
class _Features(AbstractFeatures):
    image: Tensor
    speed: Tensor
    unused: Tensor | None = None


@dataclass(frozen=True)
class _Labels(AbstractLabels):
    semantic: Tensor
    mask: Tensor


@dataclass(frozen=True)
class _Scene:
    index: int


SceneAPI.register(_Scene)


def _sample(index: int) -> TrainingSample:
    return TrainingSample(
        scene_apis=[cast(SceneAPI, _Scene(index))],
        features=_Features(
            image=torch.full((3, 5, 7), index % 256, dtype=torch.uint8),
            speed=torch.tensor([index * 0.5]),
        ),
        labels=_Labels(
            semantic=torch.arange(6, dtype=torch.int64) + index,
            mask=torch.tensor([index % 2 == 0]),
        ),
        navigation=NavigationConditioning(
            target_points=torch.full((3, 2), float(index)),
            target_point_headings=torch.zeros(3),
        ),
        source_indices=torch.tensor([index % 3]),
        vehicle_types=torch.tensor([-1]),
    )


class _Samples(Dataset[TrainingSample]):
    def __len__(self) -> int:
        return 40

    @override
    def __getitem__(self, index: int) -> TrainingSample:
        return _sample(index)


def _assert_equal(moved: TrainingSample, expected: TrainingSample) -> None:
    assert moved.scene_apis == expected.scene_apis
    for (name, actual), (_, reference) in zip(
        [*moved.features.named_tensors(), *moved.labels.named_tensors(), *moved.navigation.named_tensors()],
        [*expected.features.named_tensors(), *expected.labels.named_tensors(), *expected.navigation.named_tensors()],
        strict=True,
    ):
        assert actual.dtype == reference.dtype, name
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(moved.source_indices.cpu(), expected.source_indices)
    assert cast(_Features, moved.features).unused is None


@pytest.mark.parametrize("recycle", [False, True])
def test_slot_batches_match_the_collated_batches(recycle: bool) -> None:
    slots = PinnedBatchSlots.for_batch_of_one(
        TrainingSample.collate([_sample(0)]),
        _BATCH_SIZE,
        2,
        4,
        TrainingSample.collate,
    )
    loader = DataLoader(
        _Samples(),
        batch_size=_BATCH_SIZE,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=not recycle,
        collate_fn=slots.collate,
    )
    stream = torch.cuda.Stream()
    seen = 0
    for _ in range(3):
        slots.release_all()
        for batch_index, batch in enumerate(loader):
            assert isinstance(batch, SlotBatch)
            materialized = slots.materialize(batch)
            assert cast(_Features, materialized.features).image.is_pinned()
            with torch.cuda.stream(stream):
                moved = materialized.to(torch.device("cuda"), non_blocking=True)
                slots.release(batch, stream)
            stream.synchronize()
            first = batch_index * _BATCH_SIZE
            _assert_equal(
                moved,
                TrainingSample.collate([_sample(index) for index in range(first, first + _BATCH_SIZE)]),
            )
            seen += 1
    assert seen == 3 * 40 // _BATCH_SIZE


def test_a_batch_larger_than_its_slot_travels_as_usual() -> None:
    slots = PinnedBatchSlots(1, 4, 64, TrainingSample.collate)
    loader = DataLoader(_Samples(), batch_size=_BATCH_SIZE, num_workers=1, collate_fn=slots.collate)
    assert isinstance(next(iter(loader)), TrainingSample)


def test_a_worker_skips_slots_left_busy_by_dropped_batches() -> None:
    slots = PinnedBatchSlots.for_batch_of_one(
        TrainingSample.collate([_sample(0)]),
        _BATCH_SIZE,
        1,
        4,
        TrainingSample.collate,
    )
    # As if two batches the loader dropped at the end of a cut-short epoch had claimed slots after release_all.
    slots._busy[0, 1] = 1  # pyright: ignore[reportPrivateUsage]
    slots._busy[0, 2] = 1  # pyright: ignore[reportPrivateUsage]
    # A worker stuck on a busy slot fails the test after the timeout instead of hanging it.
    loader = DataLoader(
        _Samples(),
        batch_size=_BATCH_SIZE,
        num_workers=1,
        prefetch_factor=2,
        timeout=60,
        collate_fn=slots.collate,
    )
    stream = torch.cuda.Stream()
    for batch_index, batch in enumerate(loader):
        assert isinstance(batch, SlotBatch)
        assert batch.slot in (0, 3)
        with torch.cuda.stream(stream):
            moved = slots.materialize(batch).to(torch.device("cuda"), non_blocking=True)
            slots.release(batch, stream)
        stream.synchronize()
        first = batch_index * _BATCH_SIZE
        _assert_equal(moved, TrainingSample.collate([_sample(index) for index in range(first, first + _BATCH_SIZE)]))


def test_batches_still_arrive_when_page_locking_fails() -> None:
    slots = PinnedBatchSlots.for_batch_of_one(
        TrainingSample.collate([_sample(0)]),
        _BATCH_SIZE,
        1,
        4,
        TrainingSample.collate,
    )
    # Registering the buffer first makes the slots' own registration fail, like a page-locking limit would.
    cudart = cast(Any, torch.cuda.cudart())
    buffer = slots._buffer  # pyright: ignore[reportPrivateUsage]
    torch.cuda.check_error(cudart.cudaHostRegister(buffer.data_ptr(), buffer.nbytes, 0))
    try:
        loader = DataLoader(_Samples(), batch_size=_BATCH_SIZE, num_workers=1, collate_fn=slots.collate)
        for batch_index, batch in enumerate(loader):
            assert isinstance(batch, SlotBatch)
            moved = slots.materialize(batch).to(torch.device("cuda"))
            slots.release(batch, torch.cuda.current_stream())
            first = batch_index * _BATCH_SIZE
            _assert_equal(
                moved,
                TrainingSample.collate([_sample(index) for index in range(first, first + _BATCH_SIZE)]),
            )
    finally:
        cudart.cudaHostUnregister(buffer.data_ptr())
