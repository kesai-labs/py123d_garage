# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Compiles the policies on an early real batch in a thread while the rest of the training startup runs."""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from time import perf_counter
from typing import Any, TextIO, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch._dynamo.utils import counters
from torch._inductor.async_compile import shutdown_compile_workers
from typing_extensions import override

from py123d_garage.training.dataset import TrainingDataset, TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule

LOG = logging.getLogger(__name__)

# Held around every write to sys.stdout and sys.stderr while compiling ahead, and by the main thread across a fork.
_STREAM_GATE = threading.Lock()
# Whether a compile thread runs, so a fork must not inherit a stream or logging lock it holds.
_gate_forks = False


class _GatedStream:
    """Writes to a stream under the stream gate, so a fork never inherits the stream's buffer lock mid-write."""

    def __init__(self, stream: TextIO) -> None:
        """
        Wrap a stream.

        Args:
            stream: the stream to write to
        """
        self.stream = stream

    def write(self, text: str) -> int:
        """
        Write under the gate.

        Args:
            text: the text to write

        Returns:
            the number of characters written
        """
        with _STREAM_GATE:
            return self.stream.write(text)

    def flush(self) -> None:
        """Flush under the gate."""
        with _STREAM_GATE:
            self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        """
        Pass every other attribute through.

        Args:
            name: the attribute

        Returns:
            the stream's attribute
        """
        return getattr(self.stream, name)


def _logging_handlers() -> list[logging.Handler]:
    """
    List every live logging handler.

    Returns:
        the handlers
    """
    handler_refs = cast(list[Callable[[], logging.Handler | None]], logging._handlerList)  # pyright: ignore[reportAttributeAccessIssue]
    return [handler for handler_ref in handler_refs if (handler := handler_ref()) is not None]


def _before_fork() -> None:
    """Wait until no thread is inside a logging handler or a stream write, and keep them out until the fork is done."""
    if not _gate_forks:
        return
    # Handlers first: a handler made during the compile may write to a gated stream while holding its own lock.
    for handler in _logging_handlers():
        handler.acquire()
    _STREAM_GATE.acquire()


def _after_fork_in_parent() -> None:
    """Let the threads write again."""
    if not _gate_forks:
        return
    _STREAM_GATE.release()
    for handler in _logging_handlers():
        handler.release()


def _after_fork_in_child() -> None:
    """Release the gate in the child; logging's own fork hook renews the handler locks there."""
    if _gate_forks:
        _STREAM_GATE.release()


os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_in_parent, after_in_child=_after_fork_in_child)


def example_batch(dataset: TrainingDataset, batch_size: int, start: int = 0) -> TrainingSample:
    """
    Load one batch of a dataset's consecutive samples as the DataLoader would, leaving the global RNG streams untouched.

    Args:
        dataset: the dataset to read
        batch_size: the loaded batch size
        start: the index of the batch's first sample; the indices wrap around the dataset

    Returns:
        the collated batch, on the CPU
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        batch = TrainingSample.collate([dataset[(start + index) % len(dataset)] for index in range(batch_size)])
    random.setstate(python_state)
    np.random.set_state(numpy_state)  # noqa: NPY002
    return batch


class CompileAhead:
    """
    Runs every seed's compiled step once in a thread, so the training's first step finds the graphs compiled.

    The thread touches the device only: the dataset build and the forks that follow run in subprocesses and the main
    thread, which never use the device or the global torch generator before training starts.
    """

    def __init__(
        self,
        lightning_module: PolicyLightningModule,
        batch: TrainingSample,
        device: torch.device,
        train_step_context: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """
        Start the compile thread.

        Args:
            lightning_module: the module whose policies to compile
            batch: a loaded batch on the CPU with the shapes of every training batch
            device: the device the training runs on
            train_step_context: the trainer's context around a training step, e.g. its autocast
        """
        self._lightning_module = lightning_module
        self._batch: TrainingSample | None = batch
        self._device = device
        self._train_step_context = train_step_context
        self._start = perf_counter()
        self._compile_start = self._start
        self._end: float | None = None
        self._error: BaseException | None = None
        global _gate_forks
        _gate_forks = True
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = cast(TextIO, _GatedStream(sys.stdout))
        sys.stderr = cast(TextIO, _GatedStream(sys.stderr))
        self._thread = threading.Thread(target=self._run, name="compile_ahead", daemon=True)
        self._thread.start()
        LOG.info(f"Compile ahead: started on {device}")

    def _run(self) -> None:
        """Compile, recording the end time or the error."""
        try:
            self._compile_start = perf_counter()
            torch.cuda.set_device(self._device)
            self._lightning_module.to(self._device)
            assert self._batch is not None
            self._lightning_module.warm_up_compile(self._batch.to(self._device), self._train_step_context)
            # The fillers and loader workers fork after this; none of them should inherit the worker pool's pipes.
            shutdown_compile_workers()
            torch.cuda.empty_cache()
        except BaseException as error:
            self._error = error
        finally:
            self._batch = None
            self._end = perf_counter()

    def join(self, timeout_s: float) -> None:
        """
        Wait for the compile and log how much of it overlapped the startup; a failed compile leaves it to the first step.

        Args:
            timeout_s: how long to wait before ending the run

        Raises:
            TimeoutError: when the compile hangs; the first step would block on its compile lock.
        """
        join_start = perf_counter()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(f"Compile ahead: still compiling after {timeout_s:.0f} s of waiting")
        global _gate_forks
        sys.stdout, sys.stderr = self._streams
        _gate_forks = False
        assert self._end is not None
        waited_s = perf_counter() - join_start
        compile_s = self._end - self._compile_start
        overlap_s = max(0.0, min(self._end, join_start) - self._compile_start)
        if self._error is not None:
            LOG.warning(
                f"Compile ahead: failed after {compile_s / 60:.1f} min, the first step compiles instead: {self._error!r}",
            )
        else:
            LOG.info(
                f"Compile ahead: compiled in {compile_s / 60:.1f} min, {overlap_s / 60:.1f} min of it overlapping the "
                f"startup; training waited {waited_s / 60:.1f} min for it; Inductor cache hits / misses "
                f"{counters['inductor']['fxgraph_cache_hit']} / {counters['inductor']['fxgraph_cache_miss']}",
            )


class FirstStepCompileCheckCallback(L.Callback):
    """Logs how many frames the first training step still compiled after the compile ahead; any means a guard missed."""

    def __init__(self) -> None:
        """Set up the check."""
        self._frames_before: int | None = None
        self._done = False

    @override
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Count the frames compiled so far, before the first step."""
        if self._frames_before is None:
            self._frames_before = counters["frames"]["ok"]

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Log the frames the first step compiled."""
        if self._done or self._frames_before is None:
            return
        self._done = True
        recompiled_frames = counters["frames"]["ok"] - self._frames_before
        if recompiled_frames:
            LOG.warning(f"Compile ahead: the first step still compiled {recompiled_frames} frames")
        else:
            LOG.info("Compile ahead: the first step compiled nothing")
