"""Lightning callbacks for training: per-epoch logging and W&B visualization."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.training.dataset import TrainingSample

LOG = logging.getLogger(__name__)


class _SamplesPerSecondPostfix:
    """
    Live samples/s readout for a progress bar's postfix.

    An object rather than a string because tqdm prefixes ", " to any string
    postfix; a non-string object is rendered verbatim.
    """

    def __init__(self, bar: Tqdm, samples_per_batch: int) -> None:
        self._bar = bar
        self._samples_per_batch = samples_per_batch

    @override
    def __str__(self) -> str:
        """
        Renders the current rate in samples per second.

        Returns:
            the readout, empty until the bar has a rate.
        """
        # tqdm keeps its smoothed rate in per-iteration units, and leaves it
        # None between smoothing updates, where tqdm falls back to the average.
        bar_state = cast(dict[str, Any], self._bar.format_dict)
        rate = bar_state["rate"]
        if rate is None and bar_state["elapsed"]:
            rate = bar_state["n"] / bar_state["elapsed"]
        if not rate:
            return ""
        return f" {rate * self._samples_per_batch:.1f} samples/s"


class BatchesProgressBar(TQDMProgressBar):
    """
    Progress bar counting batches, with a samples/s postfix.

    Batch counts keep the n/total readout short; the postfix scales the rate by
    the global batch size, so runs at different batch sizes report comparable
    samples/s.
    """

    def __init__(self, batch_size_per_device: int, refresh_rate: int) -> None:
        super().__init__(refresh_rate=refresh_rate)
        self._batch_size_per_device = batch_size_per_device
        self._rate_postfix: _SamplesPerSecondPostfix | None = None

    @override
    def init_train_tqdm(self) -> Tqdm:
        bar = super().init_train_tqdm()
        # The leading space separates the rate number from its unit.
        bar.unit = " batches"
        self._rate_postfix = _SamplesPerSecondPostfix(
            bar,
            self._batch_size_per_device * self.trainer.world_size,
        )
        bar.postfix = self._rate_postfix
        return bar

    @override
    def get_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        # The truncated logger version; the output dir and W&B page identify the run.
        metrics.pop("v_num", None)
        return metrics

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        # The superclass pushes get_metrics through set_postfix here, which would
        # replace the rate object with a metrics string.
        # tqdm ships no stubs, so the bar's members reach us untyped.
        batches_done = batch_idx + 1
        progress_bar = cast(Tqdm, self.train_progress_bar)
        if self._should_update(batches_done, cast(int, progress_bar.total)):
            progress_bar.n = batches_done
            progress_bar.refresh()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self._train_progress_bar is not None:  # pyright: ignore[reportUnknownMemberType]
            cast(Tqdm, self.train_progress_bar).postfix = self._rate_postfix


class ThroughputLoggingCallback(L.Callback):
    """
    Batch and sample throughput, to the log stream and to the loggers.

    Rates are measured over the trainer's logging interval rather than over
    single steps, so one stalled batch does not define the number. Every rank
    steps in lockstep under DDP, so rank zero's batch rate is the run's.
    """

    def __init__(self, batch_size_per_device: int, every_n_steps: int) -> None:
        self._batch_size_per_device = batch_size_per_device
        self._every_n_steps = every_n_steps
        self._window_start: float | None = None
        self._window_batches = 0

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        # The first batch carries the epoch's startup, so the window opens after it.
        if self._window_start is None:
            self._window_start = perf_counter()
            return

        self._window_batches += 1
        if self._window_batches < self._every_n_steps:
            return

        elapsed = perf_counter() - self._window_start
        batches_per_second = self._window_batches / elapsed
        samples_per_second = batches_per_second * self._batch_size_per_device * trainer.world_size
        pl_module.log_dict(  # pyright: ignore[reportUnknownMemberType]
            {
                "train/batches_per_second": batches_per_second,
                "train/samples_per_second": samples_per_second,
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        LOG.info(
            f"step {trainer.global_step} | {batches_per_second:.2f} batches/s | {samples_per_second:.1f} samples/s",
        )
        self._window_start = perf_counter()
        self._window_batches = 0

    @override
    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._window_start = None
        self._window_batches = 0


class EpochLoggingCallback(L.Callback):
    """One log line per epoch, plus the epoch time as a metric."""

    def __init__(self) -> None:
        self._epoch_start: float | None = None

    @override
    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch_start = perf_counter()
        LOG.info(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | {trainer.num_training_batches} batches",
        )

    @override
    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        elapsed = perf_counter() - self._epoch_start if self._epoch_start is not None else float("nan")
        loss = trainer.callback_metrics.get("train/loss_epoch")
        loss_note = f"{float(loss):.4f}" if loss is not None else "n/a"
        samples_per_second = (
            trainer.num_training_batches * trainer.train_dataloader.batch_size * trainer.world_size / elapsed
            if elapsed > 0 and trainer.train_dataloader is not None
            else float("nan")
        )
        LOG.info(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} done "
            f"| train/loss_epoch={loss_note} | {elapsed:.1f}s "
            f"| {samples_per_second:.1f} samples/s "
            f"| global_step={trainer.global_step}",
        )
        pl_module.log_dict(  # pyright: ignore[reportUnknownMemberType]
            {"time_epoch": elapsed, "step": pl_module.current_epoch},
            rank_zero_only=True,
        )


class VisualizationCallback(L.Callback):
    """Logs policy-rendered images of the current batch to W&B every N steps."""

    def __init__(self, log_images_every_n_steps: int) -> None:
        """
        Initializes the visualization callback.

        Args:
            log_images_every_n_steps: rendering cadence in optimizer steps.
        """
        self._log_images_every_n_steps = max(1, log_images_every_n_steps)

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if (trainer.global_step + 1) % self._log_images_every_n_steps != 0:
            return
        wandb_loggers = [logger for logger in trainer.loggers if isinstance(logger, WandbLogger)]
        if not wandb_loggers or not trainer.is_global_zero:
            return

        # The batch already sits on the training device.
        policy: AnyPolicy = cast(AnyPolicy, pl_module.policy)
        with torch.no_grad():
            predictions = policy.forward(batch.features, batch.navigation)

        images = policy.visualize_batch(
            batch.features.to("cpu"),
            batch.labels.to("cpu"),
            batch.navigation.to("cpu"),
            predictions.to("cpu"),
            batch.scene_apis[0],
        )
        for name, image in images.items():
            for wandb_logger in wandb_loggers:
                wandb_logger.log_image(
                    key=f"train_plot/{name}",
                    images=[image.numpy()],  # pyright: ignore[reportUnknownMemberType]
                    step=trainer.global_step,
                )
