# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

"""Lightning callbacks for training: per-epoch logging and W&B visualization."""

from __future__ import annotations

import gc
import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch._inductor.async_compile import shutdown_compile_workers
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule

LOG = logging.getLogger(__name__)

# Whole training steps one trace holds.
_TRACE_STEPS = 5


class _SamplesPerSecondPostfix:
    """
    Live samples/s readout for a progress bar's postfix.

    An object rather than a string, because tqdm prefixes ", " to any string postfix.
    """

    def __init__(self, bar: Tqdm, samples_per_batch: int) -> None:
        """
        Set up the readout.

        Args:
            bar: the progress bar whose rate is shown
            samples_per_batch: global samples per batch the rate is scaled by
        """
        self._bar = bar
        self._samples_per_batch = samples_per_batch

    @override
    def __str__(self) -> str:
        """
        Render the current rate in samples per second.

        Returns:
            the readout, empty until the bar has a rate.
        """
        # tqdm leaves its smoothed rate None between smoothing updates, where it falls back to the average.
        bar_state = cast(dict[str, Any], self._bar.format_dict)
        rate = bar_state["rate"]
        if rate is None and bar_state["elapsed"]:
            rate = bar_state["n"] / bar_state["elapsed"]
        if not rate:
            return ""
        return f" {rate * self._samples_per_batch:.1f} samples/s"


class BatchesProgressBar(TQDMProgressBar):
    """Progress bar counting batches, with a samples/s postfix comparable across batch sizes."""

    def __init__(self, batch_size_per_device: int, refresh_rate: int) -> None:
        """
        Set up the bar.

        Args:
            batch_size_per_device: samples per batch on each device
            refresh_rate: batches between bar refreshes
        """
        super().__init__(refresh_rate=refresh_rate)
        self._batch_size_per_device = batch_size_per_device
        self._rate_postfix: _SamplesPerSecondPostfix | None = None

    @override
    def init_train_tqdm(self) -> Tqdm:
        """Create the training bar counting batches, with the samples/s postfix."""
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
        """Drop the logger version from the bar's metrics."""
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
        """Advance the bar without replacing the rate postfix."""
        # Skips the superclass, whose set_postfix would replace the rate object with a metrics string.
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
        """Restore the rate postfix the superclass replaced at epoch end."""
        super().on_train_epoch_end(trainer, pl_module)
        if self._train_progress_bar is not None:  # pyright: ignore[reportUnknownMemberType]
            cast(Tqdm, self.train_progress_bar).postfix = self._rate_postfix


class _GpuActivity:
    """
    Average SM, tensor-core and DRAM activity of one GPU between two calls, from NVML's GPM counters.

    Inactive when pynvml is missing or the GPU has no GPM support (before Hopper).
    """

    _METRIC_NAMES = ("gpu_sm_active", "gpu_tensor_active", "gpu_dram_bw")

    def __init__(self, device_index: int) -> None:
        """
        Start sampling one GPU's counters, or stay inactive without GPM support.

        Args:
            device_index: the CUDA device to watch
        """
        self._nvml: Any = None
        try:
            nvml: Any = importlib.import_module("pynvml")
            nvml.nvmlInit()
            properties = cast(Any, torch.cuda.get_device_properties)(device_index)
            self._handle = nvml.nvmlDeviceGetHandleByUUID(f"GPU-{properties.uuid}")
            if not nvml.nvmlGpmQueryDeviceSupport(self._handle).isSupportedDevice:
                return
            self._samples = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
            nvml.nvmlGpmSampleGet(self._handle, self._samples[0])
            self._metric_ids = (
                nvml.NVML_GPM_METRIC_SM_UTIL,
                nvml.NVML_GPM_METRIC_ANY_TENSOR_UTIL,
                nvml.NVML_GPM_METRIC_DRAM_BW_UTIL,
            )
            self._nvml = nvml
        except Exception as error:  # pynvml missing, no driver access, or an NVML error
            LOG.info(f"No GPU activity logging: {error!r}")

    def since_last_call(self) -> dict[str, float]:
        """
        Read the GPU activity since the previous call, or since construction.

        Returns:
            the activity percentages by metric name; empty when inactive
        """
        if self._nvml is None:
            return {}
        nvml = self._nvml
        nvml.nvmlGpmSampleGet(self._handle, self._samples[1])
        request = nvml.c_nvmlGpmMetricsGet_t()
        request.version = nvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(self._metric_ids)
        request.sample1, request.sample2 = self._samples
        for slot, metric_id in enumerate(self._metric_ids):
            request.metrics[slot].metricId = metric_id
        nvml.nvmlGpmMetricsGet(request)
        self._samples.reverse()
        return {
            name: float(request.metrics[slot].value)
            for slot, name in enumerate(self._METRIC_NAMES)
            if request.metrics[slot].nvmlReturn == nvml.NVML_SUCCESS
        }


class _CpuActivity:
    """The node's busy and I/O-wait share of CPU time between two calls, from /proc/stat; inactive off Linux."""

    def __init__(self) -> None:
        """Take the first /proc/stat reading."""
        self._previous = self._read()

    @staticmethod
    def _read() -> tuple[int, int, int] | None:
        """
        Read the node's cumulative CPU time counters.

        Returns:
            the busy, I/O-wait and total jiffies, or None when /proc/stat is unreadable
        """
        try:
            fields = [int(field) for field in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        except OSError:
            return None
        user, nice, system, idle, iowait, irq, softirq, steal = fields[:8]
        busy = user + nice + system + irq + softirq + steal
        return busy, iowait, busy + idle + iowait

    def since_last_call(self) -> dict[str, float]:
        """
        Read the CPU activity since the previous call, or since construction.

        Returns:
            the busy and I/O-wait percentages; empty when /proc/stat is unreadable
        """
        current = self._read()
        previous, self._previous = self._previous, current
        if current is None or previous is None or current[2] == previous[2]:
            return {}
        total = current[2] - previous[2]
        return {
            "cpu_busy": 100 * (current[0] - previous[0]) / total,
            "cpu_iowait": 100 * (current[1] - previous[1]) / total,
        }


class ThroughputLoggingCallback(L.Callback):
    """
    Logs batch and sample throughput, the data-wait share and rank zero's GPU and CPU activity.

    Rates are measured over a window of steps, so one stalled batch does not define the number.
    """

    def __init__(
        self,
        loaded_batch_size_per_device: int,
        seed_samples_per_batch_per_device: int,
        every_n_steps: int,
    ) -> None:
        """
        Set up the throughput window.

        Args:
            loaded_batch_size_per_device: samples each device loads per batch
            seed_samples_per_batch_per_device: samples each device trains on per batch, summed over seeds
            every_n_steps: batches per measurement window
        """
        self._loaded_batch_size_per_device = loaded_batch_size_per_device
        self._seed_samples_per_batch_per_device = seed_samples_per_batch_per_device
        self._every_n_steps = every_n_steps
        self._window_start: float | None = None
        self._window_batches = 0
        self._last_batch_end: float | None = None
        self._window_data_wait = 0.0
        self._gpu_activity: _GpuActivity | None = None
        self._cpu_activity: _CpuActivity | None = None

    @override
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the activity readers on rank zero."""
        if trainer.global_rank == 0:
            if torch.cuda.is_available():
                self._gpu_activity = _GpuActivity(torch.cuda.current_device())
            self._cpu_activity = _CpuActivity()

    @override
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Add the time since the last batch ended to the data wait."""
        # Fetching, transferring and augmenting the batch all happen before this hook.
        if self._window_start is not None and self._last_batch_end is not None:
            self._window_data_wait += perf_counter() - self._last_batch_end

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Log the throughput once a window of every_n_steps batches is complete."""
        # The first batch carries the epoch's startup, so the window opens after it.
        if self._window_start is None:
            self._open_window()
            return

        self._window_batches += 1
        if self._window_batches < self._every_n_steps:
            self._last_batch_end = perf_counter()
            return

        elapsed = perf_counter() - self._window_start
        batches_per_second = self._window_batches / elapsed
        samples_per_second = batches_per_second * self._loaded_batch_size_per_device * trainer.world_size
        seed_samples_per_second = batches_per_second * self._seed_samples_per_batch_per_device * trainer.world_size
        data_wait_fraction = self._window_data_wait / elapsed
        system_metrics = {
            **(self._gpu_activity.since_last_call() if self._gpu_activity is not None else {}),
            **(self._cpu_activity.since_last_call() if self._cpu_activity is not None else {}),
        }
        if torch.cuda.is_available():
            system_metrics["gpu_memory_reserved_peak_gb"] = torch.cuda.max_memory_reserved() / 2**30
        pl_module.log_dict(  # pyright: ignore[reportUnknownMemberType]
            {
                "train/batches_per_second": batches_per_second,
                "train/samples_per_second": samples_per_second,
                "train/seed_samples_per_second": seed_samples_per_second,
                "train/seed_samples_per_gpu_second": seed_samples_per_second / trainer.world_size,
                "train/data_wait_fraction": data_wait_fraction,
                **{f"system/{name}": value for name, value in system_metrics.items()},
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        system_readout = "".join(f" | {name} {value:.0f}" for name, value in system_metrics.items())
        LOG.info(
            f"step {trainer.global_step} | {batches_per_second:.2f} batches/s | {samples_per_second:.1f} samples/s "
            f"loaded | {seed_samples_per_second:.1f} seed samples/s | waiting for data "
            f"{100 * data_wait_fraction:.0f} %{system_readout}",
        )
        self._open_window()

    def _open_window(self) -> None:
        """Start a new measurement window."""
        self._window_start = perf_counter()
        self._window_batches = 0
        self._window_data_wait = 0.0
        self._last_batch_end = self._window_start
        if self._gpu_activity is not None:
            self._gpu_activity.since_last_call()
        if self._cpu_activity is not None:
            self._cpu_activity.since_last_call()

    @override
    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Reset the window so each epoch's startup is excluded."""
        self._window_start = None
        self._window_batches = 0
        self._last_batch_end = None


class StepTraceCallback(L.Callback):
    """Writes a per-rank Chrome trace of a few whole training steps, with hook markers splitting the CPU timeline."""

    def __init__(self, first_step: int, output_dir: Path) -> None:
        """
        Set up the profiler.

        Args:
            first_step: training steps the profiler waits before its one warmup step
            output_dir: directory the traces are written to
        """
        self._output_dir = output_dir
        self._rank = 0
        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                wait=first_step,
                warmup=1,
                active=_TRACE_STEPS,
                repeat=1,
            ),
            on_trace_ready=self._export,
        )

    def _export(self, profiler: torch.profiler.profile) -> None:
        """
        Write this rank's finished trace.

        Args:
            profiler: the profiler holding the trace
        """
        trace_file = self._output_dir / f"trace_rank{self._rank}.json"
        profiler.export_chrome_trace(str(trace_file))
        LOG.info(f"Wrote {trace_file}")

    @staticmethod
    def _mark(hook: str) -> None:
        """
        Place a named marker in the trace.

        Args:
            hook: the hook name
        """
        with torch.profiler.record_function(f"hook/{hook}"):
            pass

    @override
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start profiling."""
        self._rank = trainer.global_rank
        self._profiler.start()

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop profiling."""
        self._profiler.stop()

    @override
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Mark the hook in the trace."""
        self._mark("on_train_batch_start")

    @override
    def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: torch.Tensor) -> None:
        """Mark the hook in the trace."""
        self._mark("on_before_backward")

    @override
    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Mark the hook in the trace."""
        self._mark("on_after_backward")

    @override
    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: Optimizer,
    ) -> None:
        """Mark the hook in the trace."""
        self._mark("on_before_optimizer_step")

    @override
    def on_before_zero_grad(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer) -> None:
        """Mark the hook in the trace."""
        self._mark("on_before_zero_grad")

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Mark the hook in the trace and advance the profiler schedule."""
        self._mark("on_train_batch_end")
        self._profiler.step()


class FreezeHeapCallback(L.Callback):
    """
    Takes every object alive after the first batch and after each epoch out of the garbage collector's walks.

    The compiled graphs (first batch) and whatever the first epoch builds up (1.4 M objects in the midtrain) would
    otherwise make each rank's full collections take 0.3-1.2 s every minute or two while the other ranks wait.
    """

    def __init__(self) -> None:
        """Set up the freeze."""
        self._done = False

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Collect and freeze once the first batch is done."""
        if self._done:
            return
        gc.collect()
        gc.freeze()
        self._done = True

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Collect and freeze at every epoch end, before the next loader workers fork."""
        gc.collect()
        gc.freeze()


class ShutdownCompileWorkersCallback(L.Callback):
    """Shuts Inductor's idle compile worker pool down once the first batch is done; a later compile starts a new one."""

    def __init__(self) -> None:
        """Set up the shutdown."""
        self._done = False

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Shut the compile workers down once the first batch is done."""
        if self._done:
            return
        shutdown_compile_workers()
        self._done = True
        LOG.info("Shut down the compile workers")


class PinMainProcessCallback(L.Callback):
    """
    Pins the training thread and the loader's pin-memory thread to a physical core each, after the compile workers forked.

    Affinity is per thread, and the two must not share a core's hyperthreads: the training thread spins while it waits
    for the GPU, and the pin-memory thread, which receives, pins and frees every batch, then falls behind.
    """

    def __init__(self, training_cpus: set[int], pin_memory_cpus: set[int]) -> None:
        """
        Set up the pinning.

        Args:
            training_cpus: the CPUs of the training thread's core
            pin_memory_cpus: the CPUs of the pin-memory thread's core
        """
        self._training_cpus = training_cpus
        self._pin_memory_cpus = pin_memory_cpus
        self._pinned = False
        self._pinned_pin_memory_ids: set[int] = set()

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Pin the training thread once the first batch is done, and every new loader's pin-memory thread."""
        if not self._pinned:
            os.sched_setaffinity(0, self._training_cpus)
            self._pinned = True
            LOG.info(f"Pinned the training thread to CPUs {sorted(self._training_cpus)}")
        if batch_idx != 0:
            return
        # A recycled worker pool comes with a new pin-memory thread.
        for thread in threading.enumerate():
            native_id = thread.native_id
            if (
                "_pin_memory_loop" in thread.name
                and native_id is not None
                and native_id not in self._pinned_pin_memory_ids
            ):
                os.sched_setaffinity(native_id, self._pin_memory_cpus)
                self._pinned_pin_memory_ids.add(native_id)
                LOG.info(f"Pinned the pin-memory thread to CPUs {sorted(self._pin_memory_cpus)}")


class RecycleWorkersCallback(L.Callback):
    """Shuts a persistent loader worker pool down every N epochs, so the next epoch forks a fresh one."""

    def __init__(self, dataloader: DataLoader[Any], every_n_epochs: int) -> None:
        """
        Set up the recycling.

        Args:
            dataloader: the training loader with persistent workers
            every_n_epochs: epochs between worker restarts
        """
        self._dataloader = dataloader
        self._every_n_epochs = every_n_epochs

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Shut the workers down at every N-th epoch end."""
        if (trainer.current_epoch + 1) % self._every_n_epochs != 0:
            return
        iterator = getattr(self._dataloader, "_iterator", None)
        if iterator is None:
            return
        iterator._shutdown_workers()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
        self._dataloader._iterator = None  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
        LOG.info(f"Recycled the loader workers after epoch {trainer.current_epoch}")


class StepWatchdogCallback(L.Callback):
    """
    Ends the process when no training batch finishes within the limit, once the first one has.

    A rank dying inside a replayed CUDA graph otherwise leaves the others hanging in an all-reduce until the walltime.
    """

    def __init__(self, timeout_s: float, global_rank: int) -> None:
        """
        Set up the watchdog.

        Args:
            timeout_s: seconds allowed between two finished batches
            global_rank: this process's rank across the job, for the log
        """
        self._timeout_s = timeout_s
        self._global_rank = global_rank
        # None until the first batch finishes: compiling and recording the step take as long as they take.
        self._last_batch_end: float | None = None
        self._running = False

    def _watch(self) -> None:
        """Poll the time since the last batch and exit the process past the limit."""
        while self._running:
            time.sleep(min(30.0, self._timeout_s / 4))
            if (
                self._running
                and self._last_batch_end is not None
                and perf_counter() - self._last_batch_end > self._timeout_s
            ):
                sys.stderr.write(
                    f"=== rank {self._global_rank}: no training batch finished in "
                    f"{self._timeout_s:.0f} s, aborting ===\n",
                )
                sys.stderr.flush()
                os._exit(1)

    @override
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the watchdog thread."""
        self._running = True
        threading.Thread(target=self._watch, name="step-watchdog", daemon=True).start()

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Record when the batch finished."""
        self._last_batch_end = perf_counter()

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop the watchdog thread."""
        self._running = False


class EpochLoggingCallback(L.Callback):
    """One log line per epoch, plus the epoch time as a metric."""

    def __init__(self) -> None:
        """Set up the epoch timer."""
        self._epoch_start: float | None = None

    @override
    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Start the epoch timer and log the epoch's start."""
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
        """Log the epoch's loss, time and throughput."""
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
        """Render and log the batch's images every N steps."""
        del outputs, batch_idx
        if (trainer.global_step + 1) % self._log_images_every_n_steps != 0:
            return
        wandb_loggers = [logger for logger in trainer.loggers if isinstance(logger, WandbLogger)]
        if not wandb_loggers or not trainer.is_global_zero:
            return

        # The raw module, since a no_grad forward through the compiled one recompiles.
        policy: AnyPolicy = cast(AnyPolicy, getattr(pl_module.policy, "_orig_mod", pl_module.policy))
        rows = cast(PolicyLightningModule, pl_module).seed_rows(0)
        features = batch.features.apply(lambda tensor: tensor[rows])
        labels = batch.labels.apply(lambda tensor: tensor[rows])
        navigation = batch.navigation.apply(lambda tensor: tensor[rows])
        # Eval mode: this extra forward must neither update the batch-norm statistics nor draw dropout masks.
        policy.eval()
        try:
            with torch.no_grad(), trainer.precision_plugin.train_step_context():
                predictions = policy.forward(features, navigation)
        finally:
            policy.train()

        images = policy.visualize_batch(
            features.to("cpu"),
            labels.to("cpu"),
            navigation.to("cpu"),
            predictions.to("cpu"),
            batch.scene_apis[cast(int, rows.start)],
        )
        for name, image in images.items():
            for wandb_logger in wandb_loggers:
                wandb_logger.log_image(
                    key=f"train_plot/{name}",
                    images=[image.numpy()],  # pyright: ignore[reportUnknownMemberType]
                    step=trainer.global_step,
                )
