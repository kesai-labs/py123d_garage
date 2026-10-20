# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Loader batches in per-worker slots of one shared, page-locked buffer.

By default each DataLoader batch travels in fresh shared memory, one segment per tensor, which the training process
maps, copies into pinned memory and frees again, all on its single pin-memory thread. Here a worker collates its batch
into a slot it owns and sends only the slot's layout; the training process copies straight from the slot to the GPU and
hands the slot back once that copy is done.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import os
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import lightning as L
import torch
import torch.utils._pytree as pytree
import torch.version
from torch import Tensor
from torch.utils.data import get_worker_info
from typing_extensions import override

from py123d_garage.training.dataset import TrainingSample

LOG = logging.getLogger(__name__)

# Byte alignment of every tensor in a slot.
_ALIGNMENT = 256
# Seconds a worker waits for its next slot before it warns.
_SLOT_WAIT_WARN_S = 30.0
# cudaHostRegisterPortable: the pinning holds for every CUDA context of the process.
_HOST_REGISTER_PORTABLE = 1


def _clear_cuda_error() -> None:
    """Reset the CUDA runtime's last error, which torch would otherwise report at its next CUDA call."""
    # torch's runtime bindings lack cudaGetLastError; the runtime library torch loaded has it.
    ctypes.CDLL(f"libcudart.so.{(torch.version.cuda or '12').split('.')[0]}").cudaGetLastError()


def _unregister(cudart: Any, data_ptr: int, registering_pid: int) -> None:
    """
    End the page-locking of a buffer, in the process that registered it only.

    Args:
        cudart: torch's CUDA runtime bindings
        data_ptr: the buffer's address
        registering_pid: the process that registered the buffer; a forked worker never touches CUDA
    """
    if os.getpid() == registering_pid:
        cudart.cudaHostUnregister(data_ptr)


@dataclass(frozen=True)
class SlotBatch:
    """A collated batch that lives in a slot: what a worker sends instead of the tensors."""

    # The worker that owns the slot.
    worker: int
    # The slot's index among the worker's slots.
    slot: int
    # Byte offset, dtype and shape of every tensor, in pytree leaf order.
    layout: list[tuple[int, torch.dtype, torch.Size]]
    # The structure that turns the tensors back into (features, labels, navigation, source_indices, vehicle_types).
    spec: pytree.TreeSpec
    # The scene of each row.
    scene_apis: list[Any]


def _tensor_fields(batch: TrainingSample) -> tuple[Any, ...]:
    """The batch's tensor-carrying fields, in the order SlotBatch.spec rebuilds them."""
    return batch.features, batch.labels, batch.navigation, batch.source_indices, batch.vehicle_types


def _layout(tensors: Sequence[Tensor]) -> tuple[list[tuple[int, torch.dtype, torch.Size]], int]:
    """
    Place tensors back to back at aligned offsets.

    Args:
        tensors: the tensors to place

    Returns:
        every tensor's offset, dtype and shape, and the bytes they span
    """
    layout: list[tuple[int, torch.dtype, torch.Size]] = []
    offset = 0
    for tensor in tensors:
        layout.append((offset, tensor.dtype, tensor.shape))
        offset += -(-tensor.nbytes // _ALIGNMENT) * _ALIGNMENT
    return layout, offset


class PinnedBatchSlots:
    """
    A shared buffer of slots_per_worker batch slots for each loader worker, page-locked once in the training process.

    Create it before the DataLoader forks its workers and pass collate as the DataLoader's collate_fn. A worker writes
    its batches into its own slots in turn and waits while the next one is still busy; the training process turns a
    SlotBatch into tensor views with materialize and frees the slot with release once the copy has finished.
    """

    def __init__(
        self,
        num_workers: int,
        slots_per_worker: int,
        slot_bytes: int,
        collate_fn: Callable[[list[TrainingSample]], TrainingSample],
    ) -> None:
        """
        Allocate the shared buffer and the slots' busy flags.

        Args:
            num_workers: the DataLoader's workers
            slots_per_worker: slots of each worker; at least prefetch_factor + 2 keeps a worker from waiting
            slot_bytes: bytes of one slot
            collate_fn: the collate function whose batches go into the slots
        """
        self._num_workers = num_workers
        self._slots_per_worker = slots_per_worker
        self._slot_bytes = -(-slot_bytes // _ALIGNMENT) * _ALIGNMENT
        self._collate_fn = collate_fn
        self._buffer = torch.empty(num_workers * slots_per_worker * self._slot_bytes, dtype=torch.uint8).share_memory_()
        # 1 while a slot holds a batch the training process has not copied yet.
        self._busy = torch.zeros(num_workers, slots_per_worker, dtype=torch.int32).share_memory_()
        # Per process: the worker's next slot.
        self._next_slot = 0
        self._registered = False
        self._warned_too_large = False
        # Training process: the copied slots with the event their copy recorded, oldest first.
        self._in_flight: collections.deque[tuple[int, int, torch.cuda.Event]] = collections.deque()
        LOG.info(
            f"Pinned batch slots: {num_workers} workers x {slots_per_worker} slots x {self._slot_bytes / 2**20:.0f} MB "
            f"= {self._buffer.nbytes / 2**30:.1f} GB",
        )

    @classmethod
    def for_batch_of_one(
        cls,
        batch_of_one: TrainingSample,
        batch_size: int,
        num_workers: int,
        slots_per_worker: int,
        collate_fn: Callable[[list[TrainingSample]], TrainingSample],
    ) -> PinnedBatchSlots:
        """
        Size the slots from a collated batch of one sample, with headroom for samples larger than this one.

        Args:
            batch_of_one: one sample, collated
            batch_size: samples per batch
            num_workers: the DataLoader's workers
            slots_per_worker: slots of each worker
            collate_fn: the collate function whose batches go into the slots

        Returns:
            the slots
        """
        leaves, _ = pytree.tree_flatten(_tensor_fields(batch_of_one))
        batch_bytes = sum(-(-leaf.nbytes * batch_size // _ALIGNMENT) * _ALIGNMENT for leaf in leaves)
        return cls(num_workers, slots_per_worker, batch_bytes + batch_bytes // 8 + 2**20, collate_fn)

    def _slot_view(self, worker: int, slot: int) -> Tensor:
        """The bytes of one slot."""
        start = (worker * self._slots_per_worker + slot) * self._slot_bytes
        return self._buffer[start : start + self._slot_bytes]

    def _claim_slot(self, worker: int) -> int:
        """
        Mark the worker's first free slot from its next one on as busy, waiting while all of them are.

        Any free slot rather than strictly the next one: a batch the loader drops at an epoch cut short by
        limit_train_batches can claim a slot after release_all, which then stays busy until the next epoch.

        Args:
            worker: the worker's index

        Returns:
            the slot's index
        """
        wait_start = time.monotonic()
        warned = False
        while True:
            for step in range(self._slots_per_worker):
                slot = (self._next_slot + step) % self._slots_per_worker
                if int(self._busy[worker, slot]) == 0:
                    self._busy[worker, slot] = 1
                    self._next_slot = (slot + 1) % self._slots_per_worker
                    return slot
            time.sleep(0.0005)
            if not warned and time.monotonic() - wait_start > _SLOT_WAIT_WARN_S:
                LOG.warning(f"Worker {worker} has waited {_SLOT_WAIT_WARN_S:.0f} s for a free slot")
                warned = True

    def collate(self, samples: list[TrainingSample]) -> TrainingSample | SlotBatch:
        """
        Collate a batch and, in a loader worker, move it into the worker's next slot.

        Args:
            samples: the batch's samples

        Returns:
            the slot's SlotBatch, or the collated batch outside a worker or when it does not fit a slot
        """
        batch = self._collate_fn(samples)
        worker_info = get_worker_info()
        if worker_info is None:
            return batch
        leaves, spec = pytree.tree_flatten(_tensor_fields(batch))
        layout, batch_bytes = _layout(leaves)
        if batch_bytes > self._slot_bytes:
            if not self._warned_too_large:
                LOG.warning(f"A {batch_bytes / 2**20:.0f} MB batch exceeds the {self._slot_bytes / 2**20:.0f} MB slot")
                self._warned_too_large = True
            return batch
        worker = worker_info.id
        slot = self._claim_slot(worker)
        slot_bytes = self._slot_view(worker, slot)
        for tensor, (offset, dtype, shape) in zip(leaves, layout, strict=True):
            slot_bytes[offset : offset + tensor.nbytes].view(dtype).view(shape).copy_(tensor)
        return SlotBatch(worker, slot, layout, spec, batch.scene_apis)

    def materialize(self, batch: SlotBatch) -> TrainingSample:
        """
        Rebuild a SlotBatch as tensor views of its page-locked slot, in the training process.

        Args:
            batch: the slot's batch

        Returns:
            the batch, its tensors viewing the slot
        """
        if not self._registered:
            # Page-locks the buffer once, so copies from any slot run asynchronously.
            self._registered = True
            start = time.monotonic()
            # torch ships no stubs for its cudart bindings.
            cudart = cast(Any, torch.cuda.cudart())
            try:
                torch.cuda.check_error(
                    cudart.cudaHostRegister(self._buffer.data_ptr(), self._buffer.nbytes, _HOST_REGISTER_PORTABLE),
                )
            except RuntimeError as error:
                _clear_cuda_error()
                # E.g. a container's page-locking limit: the slots still spare the per-batch shared memory, but their
                # copies to the GPU then block the host.
                LOG.warning(f"Could not page-lock the {self._buffer.nbytes / 2**30:.1f} GB of batch slots: {error}")
            else:
                # The registration must end before the buffer is unmapped, or a later mapping there cannot be locked.
                weakref.finalize(self, _unregister, cudart, self._buffer.data_ptr(), os.getpid())
                LOG.info(f"Page-locked the batch slots in {time.monotonic() - start:.1f} s")
        slot_bytes = self._slot_view(batch.worker, batch.slot)
        leaves = [
            slot_bytes[offset : offset + dtype.itemsize * shape.numel()].view(dtype).view(shape)
            for offset, dtype, shape in batch.layout
        ]
        features, labels, navigation, source_indices, vehicle_types = pytree.tree_unflatten(leaves, batch.spec)
        return TrainingSample(
            scene_apis=batch.scene_apis,
            features=features,
            labels=labels,
            navigation=navigation,
            source_indices=source_indices,
            vehicle_types=vehicle_types,
        )

    def release(self, batch: SlotBatch, stream: torch.cuda.Stream) -> None:
        """
        Hand the slot back once the copies enqueued on stream so far are done; frees the slots of earlier copies.

        Args:
            batch: the batch whose copy was just enqueued
            stream: the stream the copy runs on
        """
        event = torch.cuda.Event()
        event.record(stream)
        # Every older copy finished long ago, since the stream runs them in order; waiting on them keeps a slow poll
        # from ever holding a worker back.
        while self._in_flight:
            worker, slot, older_event = self._in_flight.popleft()
            older_event.synchronize()
            self._busy[worker, slot] = 0
        self._in_flight.append((batch.worker, batch.slot, event))

    def release_all(self) -> None:
        """Free every slot: the copies so far are waited for, and batches a recycled worker pool dropped are void."""
        while self._in_flight:
            _, _, event = self._in_flight.popleft()
            event.synchronize()
        self._busy.zero_()


class ReleaseBatchSlotsCallback(L.Callback):
    """Frees every batch slot when an epoch starts, before its loader iterator resumes or forks a new worker pool."""

    def __init__(self, batch_slots: PinnedBatchSlots) -> None:
        """
        Set up the release.

        Args:
            batch_slots: the slots to free
        """
        self._batch_slots = batch_slots

    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Free every slot."""
        self._batch_slots.release_all()
