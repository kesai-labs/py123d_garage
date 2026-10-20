# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Resuming from an epoch-end checkpoint continues the super-epoch order with the next epoch, none repeated or skipped."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from py123d_garage.training.train import _MixtureSampler  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.slow

_NUM_SAMPLES = 8
_EPOCHS_PER_SUPER_EPOCH = 2


class _Indices(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return _NUM_SAMPLES

    @override
    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)


class _Recorder(L.LightningModule):
    """Records the sample order of every epoch and saves a checkpoint at each epoch's end."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.output_dir = output_dir
        self.orders: dict[int, list[int]] = {}

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.orders.setdefault(self.current_epoch, []).extend(batch.tolist())
        return self.weight.sum() * 0.0

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)

    @override
    def on_train_epoch_end(self) -> None:
        self.trainer.save_checkpoint(self.output_dir / f"epoch_{self.current_epoch}.ckpt")


def _fit(output_dir: Path, max_epochs: int, checkpoint: Path | None) -> dict[int, list[int]]:
    sampler = _MixtureSampler([_NUM_SAMPLES], [1.0], 0, _EPOCHS_PER_SUPER_EPOCH)
    module = _Recorder(output_dir)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        default_root_dir=str(output_dir),
    )
    sampler.trainer = trainer
    trainer.fit(module, DataLoader(_Indices(), batch_size=4, sampler=sampler), ckpt_path=checkpoint)
    return module.orders


def test_resume_continues_with_the_next_epoch_of_the_super_epoch(tmp_path: Path) -> None:
    orders = _fit(tmp_path, max_epochs=2, checkpoint=None)
    resumed = _fit(tmp_path, max_epochs=4, checkpoint=tmp_path / "epoch_1.ckpt")

    assert sorted(orders) == [0, 1]
    assert sorted(resumed) == [2, 3]
    assert orders[0] == orders[1]
    assert resumed[2] == resumed[3]
    assert resumed[2] != orders[0]
    for order in [*orders.values(), *resumed.values()]:
        assert sorted(order) == list(range(_NUM_SAMPLES))
