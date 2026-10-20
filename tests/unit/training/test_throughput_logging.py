# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import lightning as L
from typing_extensions import override

from py123d_garage.training.callbacks import ThroughputLoggingCallback
from py123d_garage.training.dataset import TrainingSample


class _RecordingModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.logged: list[dict[str, float]] = []

    @override
    def log_dict(self, dictionary: Any, **_: Any) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.logged.append(dictionary)


def test_window_logs_per_gpu_rate_and_data_wait() -> None:
    """Two steps per window, the second batch arriving 50 ms late: the wait shows up as a share of the window."""
    callback = ThroughputLoggingCallback(
        loaded_batch_size_per_device=6,
        seed_samples_per_batch_per_device=6,
        every_n_steps=2,
    )
    trainer = MagicMock(spec=L.Trainer)
    trainer.global_rank, trainer.world_size, trainer.global_step = 0, 4, 0
    module = _RecordingModule()
    batch = MagicMock(spec=TrainingSample)
    callback.on_train_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_start(trainer, module, batch, 0)
    callback.on_train_batch_end(trainer, module, None, batch, 0)
    for batch_idx, data_wait_s in ((1, 0.0), (2, 0.05)):
        time.sleep(data_wait_s)
        callback.on_train_batch_start(trainer, module, batch, batch_idx)
        time.sleep(0.01)
        callback.on_train_batch_end(trainer, module, None, batch, batch_idx)

    (metrics,) = module.logged
    assert metrics["train/seed_samples_per_gpu_second"] * 4 == metrics["train/seed_samples_per_second"]
    assert 0.5 < metrics["train/data_wait_fraction"] < 1.0
    assert {"system/cpu_busy", "system/cpu_iowait"} <= metrics.keys()
