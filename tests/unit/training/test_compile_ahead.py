# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The compile ahead leaves the training state as it found it and leaves a failed compile to the first step."""

from __future__ import annotations

import logging
import os
import random
import signal
import sys
import threading
import time
from contextlib import nullcontext
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from py123d.api import SceneAPI
from typing_extensions import override

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import DataLoaderConfig, TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training import compile_ahead
from py123d_garage.training.compile_ahead import CompileAhead, example_batch
from py123d_garage.training.dataset import TrainingDataset, TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compile ahead runs on the GPU")


class _RandomDataset(TrainingDataset):
    """Draws from every global generator per sample, as an augmenting dataset would."""

    def __init__(self) -> None:
        pass

    @override
    def __len__(self) -> int:
        return 3

    @override
    def __getitem__(self, index: int) -> TrainingSample:
        random.random()
        np.random.rand()  # noqa: NPY002
        torch.rand(1)
        return cast(TrainingSample, index)


def _collated(samples: list[object]) -> TrainingSample:
    batch = MagicMock(spec=TrainingSample)
    batch.samples = samples
    return batch


def test_example_batch_keeps_the_rng_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TrainingSample, "collate", classmethod(lambda cls, samples: _collated(samples)))
    random.seed(1)
    np.random.seed(1)  # noqa: NPY002
    torch.manual_seed(1)
    expected = (random.random(), np.random.rand(), torch.rand(1))  # noqa: NPY002
    random.seed(1)
    np.random.seed(1)  # noqa: NPY002
    torch.manual_seed(1)

    batch = example_batch(_RandomDataset(), 5)

    assert cast(Any, batch).samples == [0, 1, 2, 0, 1]
    assert (random.random(), np.random.rand(), torch.rand(1)) == expected  # noqa: NPY002


@_needs_gpu
def test_warm_up_compile_leaves_no_trace(config) -> None:  # noqa: F811
    """Two seeds' train-mode forward and backward on one loaded batch; buffers, weights, gradients and RNG stay."""
    policies: list[TransfuserPolicy] = []
    for seed in range(2):
        torch.manual_seed(seed)
        policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
        policy.initialize()
        policies.append(policy.cuda().train())
    module = PolicyLightningModule(
        policies,
        TrainingConfig(
            offline_data_sources=[],
            num_seeds=2,
            dataloader_config=DataLoaderConfig(batch_size=2),
            compile_mode=None,
        ),
    )
    batch = TrainingSample(
        scene_apis=[MagicMock(spec=SceneAPI) for _ in range(4)],
        features=_features(config, 4),
        labels=_labels(config, batch_size=4),
        navigation=_navigation(config, 4),
        source_indices=torch.zeros(4, dtype=torch.long),
        vehicle_types=torch.full((4,), -1),
    ).to("cuda")
    weights = [parameter.detach().clone() for parameter in module.parameters()]
    buffers = [buffer.clone() for buffer in module.buffers()]
    cuda_rng = torch.cuda.get_rng_state()
    cpu_rng = torch.random.get_rng_state()

    module.warm_up_compile(batch, nullcontext)

    for parameter, weight in zip(module.parameters(), weights, strict=True):
        torch.testing.assert_close(parameter, weight, rtol=0, atol=0)
        assert parameter.grad is None
    for buffer, saved in zip(module.buffers(), buffers, strict=True):
        torch.testing.assert_close(buffer, saved, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)


class _RecordingModule(PolicyLightningModule):
    """Stands in for the Lightning module: records when it is compiled, or fails."""

    def __init__(self, fail: bool = False) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.compiled = threading.Event()
        self._fail = fail

    @override
    def to(self, device: torch.device) -> _RecordingModule:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self

    @override
    def warm_up_compile(self, batch: object, train_step_context: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._fail:
            raise RuntimeError("out of memory")
        self.compiled.set()


def _compile_ahead(module: _RecordingModule) -> CompileAhead:
    batch = MagicMock(spec=TrainingSample)
    batch.to.return_value = batch
    return CompileAhead(module, batch, torch.device("cuda", 0), nullcontext)


@_needs_gpu
def test_a_failed_compile_leaves_the_compile_to_the_first_step(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _compile_ahead(_RecordingModule(fail=True)).join(60.0)
    assert "the first step compiles instead" in caplog.text


def test_a_fork_while_another_thread_writes_never_inherits_a_held_stream_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Children forked while a thread floods sys.stderr and a logging handler all exit; ungated, most hang."""
    monkeypatch.setattr(compile_ahead, "_gate_forks", True)
    monkeypatch.setattr(sys, "stderr", compile_ahead._GatedStream(sys.stderr))  # pyright: ignore[reportPrivateUsage]
    handler = logging.StreamHandler(sys.stderr)
    writer_logger = logging.getLogger("test_compile_ahead_writer")
    writer_logger.addHandler(handler)
    writer_logger.propagate = False
    stop = threading.Event()

    def write() -> None:
        while not stop.is_set():
            sys.stderr.write("x" * 20000 + "\n")
            writer_logger.warning("y" * 2000)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    hung = 0
    try:
        for _ in range(20):
            pid = os.fork()
            if pid == 0:
                sys.stderr.write("child\n")
                writer_logger.warning("child")
                os._exit(0)
            deadline = time.monotonic() + 5.0
            while not os.waitpid(pid, os.WNOHANG)[0]:
                if time.monotonic() > deadline:
                    hung += 1
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    break
                time.sleep(0.01)
    finally:
        stop.set()
        writer.join()
        writer_logger.removeHandler(handler)
    assert hung == 0
