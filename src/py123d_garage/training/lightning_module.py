# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

"""Lightning module that trains one policy per seed, one after the other or as one recorded CUDA graph."""

from __future__ import annotations

import functools
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jaxtyping as jt
import lightning as L
import torch
import torch.distributed as dist
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torch.optim import Optimizer
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.common.config_help import run_dir
from py123d_garage.common.sensor.hood_swap import swap_hoods
from py123d_garage.common.sensor.photometric_augmentation import augment_camera_batch
from py123d_garage.config.schema.training.training_config import (
    OptimizerConfig,
    TrainingConfig,
)
from py123d_garage.training.checkpointing import (
    prune_checkpoints,
    split_checkpoint,
)
from py123d_garage.training.concurrent_seeds_training import (
    CapturedSeedsStep,
    SeedInputs,
    policy_forward_and_loss,
)
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.flat_adamw import FlatAdamW
from py123d_garage.training.kernel_verification import verify_kernel_patches
from py123d_garage.training.lr_scheduler import (
    CosineAnnealingWarmRestartsWithWarmup,
    cycle_end_epochs,
)
from py123d_garage.training.pinned_batches import PinnedBatchSlots, SlotBatch

if TYPE_CHECKING:
    # Exists only in NCCL builds of torch.
    from torch.distributed import ProcessGroupNCCL

LOG = logging.getLogger(__name__)

# Each cosine cycle is this many times longer than the previous one; the first lasts one epoch.
_CYCLE_LENGTH_MULT = 2
# Keeps the seeds of re-initializations at different epochs apart from each other and from the seeds' own.
_REINIT_SEED_STRIDE = 1000


# torch.compile modes without inductor's CUDA graphs, for the policies inside the hand-recorded step.
_CONCURRENT_COMPILE_MODES = {
    "reduce-overhead": "default",
    "max-autotune": "max-autotune-no-cudagraphs",
}


def _multi_rank_nccl_backend() -> ProcessGroupNCCL | None:
    """
    Find the default process group's NCCL backend.

    Returns:
        the backend, or None without a process group of more than one rank
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return None
    group = dist.distributed_c10d._get_default_group()  # pyright: ignore[reportPrivateUsage]
    return cast("ProcessGroupNCCL", group._get_backend(torch.device("cuda")))  # pyright: ignore[reportPrivateUsage]


class PolicyLightningModule(L.LightningModule):
    """Trains one policy per seed on its own rows of each loaded batch."""

    def __init__(self, policies: Sequence[AnyPolicy], training_config: TrainingConfig):
        """
        Initialise the module, loading initial weights and compiling the policies.

        Args:
            policies: one policy per seed; policy i was built under seed i
            training_config: the training config
        """
        super().__init__()
        self.training_config: TrainingConfig = training_config
        self._seeds: list[int] = list(range(training_config.num_seeds))
        assert len(policies) == len(self._seeds), f"{len(policies)} policies for seeds {self._seeds}"
        assert training_config.num_seeds <= training_config.effective_loaded_batch_factor, (
            f"{training_config.num_seeds} seeds need loaded_batch_factor >= {training_config.num_seeds}"
        )
        self._loss_weights: dict[str, float] = {}
        self._h2d_stream: torch.cuda.Stream | None = None
        self._h2d_event: torch.cuda.Event | None = None
        # The loader's page-locked batch slots, if its workers collate into them.
        self.batch_slots: PinnedBatchSlots | None = None
        self._concurrent = training_config.num_seeds > 1
        # The recorded step owns backward and all-reduce, so the module takes the optimizer step itself.
        self.automatic_optimization = not self._concurrent
        assert not self._concurrent or training_config.lightning_trainer_config.accumulate_grad_batches == 1, (
            "concurrent seeds training records one optimizer step per batch"
        )
        self._captured: CapturedSeedsStep | None = None
        self._flat_adamw: FlatAdamW | None = None
        self._gradient_pool: torch.cuda.MemPool | None = None
        self._loss_weight_tensors: dict[str, Tensor] = {}

        initial_weights_files = list(training_config.initial_weights_files)
        if initial_weights_files:
            assert len(initial_weights_files) in (1, len(policies)), (
                f"{len(initial_weights_files)} initial_weights_files for {len(policies)} seeds; give one or one per seed"
            )
            if len(initial_weights_files) == 1:
                initial_weights_files *= len(policies)
            state_dicts: dict[str, dict[str, Tensor]] = {}
            for policy, initial_weights_file in zip(policies, initial_weights_files, strict=True):
                if initial_weights_file not in state_dicts:
                    LOG.info(f"Loading initial policy weights from {initial_weights_file}")
                    state_dicts[initial_weights_file] = torch.load(
                        initial_weights_file,
                        map_location="cpu",
                        weights_only=True,
                    )
                state_dict = state_dicts[initial_weights_file]
                missing, unexpected = cast(
                    tuple[list[str], list[str]],
                    policy.load_state_dict(
                        state_dict,
                        strict=training_config.initial_weights_strict,
                    ),
                )
                if missing or unexpected:
                    LOG.info(
                        f"Initial weights: {len(missing)} missing keys, {len(unexpected)} unexpected keys",
                    )

        prepared_policies: list[AnyPolicy] = []
        for policy in policies:
            if training_config.channels_last:
                # cuDNN's bf16 convolutions expect NHWC; NCHW would transpose every convolution's input and output.
                policy = cast(
                    AnyPolicy,
                    policy.to(memory_format=torch.channels_last),  # pyright: ignore[reportCallIssue]
                )
            # mode=None is torch.compile's default mode, not "do not compile".
            if training_config.compile_mode is not None:
                # The recorded step captures the kernels itself, so inductor must not add its own CUDA graphs; a compile
                # ahead must not either, since inductor's CUDA graph trees belong to the thread that creates them.
                mode = training_config.compile_mode
                if self._concurrent or training_config.compile_ahead:
                    mode = _CONCURRENT_COMPILE_MODES.get(mode, mode)
                compiled_policy = cast(
                    AnyPolicy,
                    torch.compile(  # pyright: ignore[reportUnknownMemberType]
                        policy,
                        dynamic=False,
                        backend="inductor",
                        mode=mode,
                    ),
                )
                # A flow decoder's loss reruns the planner, so the loss is a large share of the step.
                compiled_policy.compute_loss = torch.compile(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
                    policy.compute_loss,
                    dynamic=False,
                    backend="inductor",
                    mode=mode,
                )
                policy = compiled_policy
            prepared_policies.append(policy)
        self.policies = nn.ModuleList(prepared_policies)

    def warm_up_compile(
        self,
        batch: TrainingSample,
        train_step_context: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """
        Run every seed's forward, loss and backward once as the first training step would, which compiles them.

        Leaves no trace in the training state: the CPU and device generators, every buffer (batch-norm statistics) and
        the gradients are restored, and the optimizer does not exist yet.

        Args:
            batch: a loaded batch on the policies' device
            train_step_context: the trainer's context around a training step, e.g. its autocast
        """
        device = batch.source_indices.device
        raw_policies = self._raw_policies()
        saved_buffers = [{name: buffer.clone() for name, buffer in policy.named_buffers()} for policy in raw_policies]
        loss_weights = self.policy.loss_weights(self.current_epoch)
        try:
            with torch.random.fork_rng(devices=[device]):  # pyright: ignore[reportUnknownMemberType]
                inputs = self._seed_inputs(self._augmented(batch))
                with train_step_context():
                    for policy_module, (features, labels, navigation) in zip(self.policies, inputs, strict=True):
                        total_loss, _, _ = policy_forward_and_loss(
                            cast(AnyPolicy, policy_module),
                            features,
                            labels,
                            navigation,
                            loss_weights,
                            self.training_config.channels_last,
                        )
                        total_loss.backward()  # pyright: ignore[reportUnknownMemberType]
                torch.cuda.synchronize(device)
        finally:
            for policy, buffers in zip(raw_policies, saved_buffers, strict=True):
                for name, buffer in policy.named_buffers():
                    buffer.copy_(buffers[name])
            for parameter in self.parameters():
                parameter.grad = None

    def verify_kernel_patches(
        self,
        batch: TrainingSample,
        train_step_context: Callable[[], AbstractContextManager[None]],
        report_file: Path | None,
    ) -> None:
        """
        Check policy 0's kernel patches against the plain torch policy on seed 0's rows of a loaded batch, and write
        the kernel report; see kernel_verification.verify_kernel_patches.

        Args:
            batch: a loaded batch on the policies' device
            train_step_context: the trainer's context around a training step, e.g. its autocast
            report_file: where the kernel report goes; None writes none
        """
        device = batch.source_indices.device
        with torch.random.fork_rng(devices=[device]):  # pyright: ignore[reportUnknownMemberType]
            inputs = self._seed_inputs(self._augmented(batch))[0]
        verify_kernel_patches(
            self.policy,
            self.training_config,
            inputs,
            self.policy.loss_weights(self.current_epoch),
            train_step_context,
            device,
            report_file,
        )

    def release_recorded_step(self) -> None:
        """Drop the recorded CUDA graph and its captured collectives; called before the process group goes."""
        self._captured = None

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients, through FlatAdamW's buffers when it is the optimizer."""
        if self._flat_adamw is None:
            super().zero_grad(set_to_none)
        else:
            self._flat_adamw.zero_grad()

    @property
    def policy(self) -> AnyPolicy:
        """The first policy without its torch.compile wrapper; datasets, loss weights and visualizations are its."""
        return self._raw_policies()[0]

    def _raw_policies(self) -> list[AnyPolicy]:
        """
        Unwrap the compiled policies.

        Returns:
            every seed's policy without its torch.compile wrapper
        """
        return [cast(AnyPolicy, getattr(policy, "_orig_mod", policy)) for policy in self.policies]

    def _graphed_optimizer(self) -> FlatAdamW | None:
        """
        Find the optimizer whose updates the step's graph records.

        Returns:
            the FlatAdamW recorded in the graph, or None when the updates run after the replay
        """
        if self._flat_adamw is None or not self.training_config.optimizer_config.flat_adamw_in_graph:
            return None
        return self._flat_adamw

    def _across_seeds(self, values_by_name: dict[str, list[Tensor]]) -> dict[str, Tensor]:
        """
        Summarize per-seed scalars as each seed's value, the non-NaN mean and the spread.

        Args:
            values_by_name: every seed's value of each scalar, in seed order

        Returns:
            the scalars under <name>_seed<seed>, <name> for the mean and <name>_std for the spread
        """
        scalars: dict[str, Tensor] = {}
        for name, values in values_by_name.items():
            for seed, value in zip(self._seeds, values, strict=True):
                scalars[f"{name}_seed{seed}"] = value
            stacked = torch.stack(values)
            scalars[name] = stacked.nanmean()
            if len(values) > 1:
                scalars[f"{name}_std"] = stacked.std()
        return scalars

    @override
    def training_step(
        self,
        batch: TrainingSample,
        batch_idx: int,
    ) -> jt.Float[Tensor, ""]:
        """
        Run every policy on its own rows of the loaded batch and return the summed loss.

        In the concurrent mode the seeds run as one replayed CUDA graph and the module takes the optimizer step itself.
        """
        log_step = self._is_scalar_log_step()
        inputs = self._seed_inputs(batch)
        seed_totals: list[Tensor] = []
        seed_losses: list[dict[str, Tensor]] = []
        seed_predictions: list[AbstractPredictions] = []
        seed_labels: list[AbstractLabels] = [labels for _, labels, _ in inputs]
        if self._concurrent:
            graphed_optimizer = self._graphed_optimizer()
            if self._captured is None:
                self._loss_weight_tensors = {
                    name: torch.tensor(weight, device=self.device) for name, weight in self._loss_weights.items()
                }
                self._captured = CapturedSeedsStep(
                    cast(list[AnyPolicy], list(self.policies)),
                    self._loss_weight_tensors,
                    self.training_config.channels_last,
                    flat_grads=None if self._flat_adamw is None else self._flat_adamw.flat_gradients,
                    seed_updates=None if graphed_optimizer is None else graphed_optimizer.graph_updates(),
                    copy_gradients=self.training_config.batched_gradient_copy,
                    skip_gradient_zeroing=self.training_config.skip_gradient_zeroing,
                )
                self._captured.capture(inputs)
            if graphed_optimizer is not None:
                graphed_optimizer.load_hyperparameters()
            for seed_index, (features, labels, navigation) in enumerate(inputs):
                self._captured.copy_inputs(seed_index, features, labels, navigation)
            self._captured.replay()
            seed_totals = self._captured.totals
            seed_losses = self._captured.losses
            seed_predictions = self._captured.predictions
            seed_labels = self._captured.labels
        else:
            for policy_module, (features, labels, navigation) in zip(self.policies, inputs, strict=True):
                total_loss, losses, predictions = policy_forward_and_loss(
                    cast(AnyPolicy, policy_module),
                    features,
                    labels,
                    navigation,
                    self._loss_weights,
                    self.training_config.channels_last,
                )
                seed_totals.append(total_loss)
                seed_losses.append(losses)
                seed_predictions.append(predictions)

        values_by_name: dict[str, list[Tensor]] = defaultdict(list)
        debug_scalars: dict[str, Tensor] = {}
        if log_step:
            for total_loss, losses in zip(seed_totals, seed_losses, strict=True):
                for name, value in losses.items():
                    values_by_name[f"train/unscaled_{name}"].append(value.float())
                    values_by_name[f"train/scaled_{name}"].append(self._loss_weights[name] * value.float().reshape(()))
                values_by_name["train/loss_step"].append(total_loss)
            with torch.no_grad():
                for policy_module, seed, labels, predictions in zip(
                    self.policies,
                    self._seeds,
                    seed_labels,
                    seed_predictions,
                    strict=True,
                ):
                    for name, tensor in predictions.named_tensors():
                        debug_scalars[f"outputs/{name}_min_seed{seed}"] = tensor.min().float()
                        debug_scalars[f"outputs/{name}_max_seed{seed}"] = tensor.max().float()
                    policy = cast(AnyPolicy, policy_module)
                    metrics: dict[str, Tensor] = policy.compute_metrics(labels, predictions)
                    for name, value in metrics.items():
                        values_by_name[f"metric/{name}"].append(value.float())
                debug_scalars |= self._data_source_scalars(batch.source_indices, seed_labels, seed_predictions)

        # Disjoint parameters: the summed loss gives every policy the gradient of its own loss.
        summed_loss = torch.stack(seed_totals).sum()
        self.log(  # pyright: ignore[reportUnknownMemberType]
            "train/loss_epoch",
            summed_loss / len(self.policies),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Every synced on_step scalar costs a barrier and an all_reduce, and Lightning writes them only on log steps.
        if log_step:
            scalars = self._across_seeds(values_by_name)
            metric_names = [name for name in scalars if name.startswith("metric/")]
            self.log_dict(  # pyright: ignore[reportUnknownMemberType]
                {name: value for name, value in scalars.items() if name not in metric_names},
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log_dict(  # pyright: ignore[reportUnknownMemberType]
                {**{name: scalars[name] for name in metric_names}, **debug_scalars},
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
        if self._concurrent:
            # Manual optimization: Lightning's wrapper counts the step and fires on_before_optimizer_step, which clips.
            cast(Optimizer, self.optimizers()).step()
            cast(CosineAnnealingWarmRestartsWithWarmup, self.lr_schedulers()).step()
        return summed_loss

    def _seed_inputs(self, batch: TrainingSample) -> list[SeedInputs]:
        """
        Split the already augmented loaded batch into every policy's inputs.

        Args:
            batch: the loaded batch

        Returns:
            each seed's features, labels and navigation
        """
        seed_batch_size = self.training_config.dataloader_config.batch_size
        disjoint_rows = len(batch.scene_apis) > seed_batch_size
        inputs: list[SeedInputs] = []
        for seed_index in range(len(self.policies)):
            features: AbstractFeatures = batch.features
            labels: AbstractLabels = batch.labels
            navigation: NavigationConditioning = batch.navigation
            if disjoint_rows:
                rows = self.seed_rows(seed_index)
                features = features.apply(lambda tensor, rows=rows: tensor[rows])
                labels = labels.apply(lambda tensor, rows=rows: tensor[rows])
                navigation = navigation.apply(lambda tensor, rows=rows: tensor[rows])
            inputs.append((features, labels, navigation))
        return inputs

    def _augmented(self, batch: TrainingSample) -> TrainingSample:
        """
        Augment the loaded batch's cameras once for all seeds.

        Args:
            batch: the loaded batch on the device

        Returns:
            the batch with augmented features
        """
        augment_camera = functools.partial(self._augment_camera, vehicle_types=batch.vehicle_types)
        return replace(batch, features=self.policy.augment_features(batch.features, augment_camera))

    def _augment_camera(
        self,
        rgb: jt.Shaped[Tensor, "batch 3 height width"],
        num_tiles: int,
        num_rows: int,
        front_tile: int,
        *,
        vehicle_types: jt.Int[Tensor, " batch"],
    ) -> jt.Shaped[Tensor, "batch 3 height width"]:
        """
        Augment a stitched camera batch with the configured hood swap, then the photometric effects.

        Args:
            rgb: batched channel-first stitched camera images in [0, 255], on device.
            num_tiles: cameras side by side in a row.
            num_rows: frames stacked top to bottom.
            front_tile: the tile of the front camera.
            vehicle_types: the recording vehicle's physical-ai-av type per sample, -1 for a hoodless one.

        Returns:
            the augmented batch, same dtype, device and layout.
        """
        augmentation = self.training_config.camera_augmentation
        # The step runs under mixed-precision autocast, which would turn the image filters' outputs into bf16.
        with torch.autocast(rgb.device.type, enabled=False):
            if augmentation.hood_swap:
                hood_swap = augmentation.hood_swap_config
                rgb = swap_hoods(
                    rgb,
                    vehicle_types,
                    hood_swap.probability,
                    hood_swap.alpasim_probability,
                    num_tiles,
                    num_rows,
                    front_tile,
                )
            if augmentation.photometric_augmentation:
                photometric = augmentation.photometric_augmentation_config
                rgb = augment_camera_batch(
                    rgb,
                    photometric.effect_probabilities,
                    photometric.max_effects_per_category,
                    photometric.max_effects_per_camera,
                    photometric.clean_fraction,
                    num_tiles,
                    num_rows,
                )
        return rgb

    def seed_rows(self, seed_index: int) -> slice:
        """
        Pick the rows of the loaded batch a seed trains on in the current epoch.

        Args:
            seed_index: the seed's index

        Returns:
            the seed's rows
        """
        seed_batch_size = self.training_config.dataloader_config.batch_size
        epochs_per_super_epoch = int(self.training_config.effective_loaded_batch_factor)
        chunk = (seed_index + self.current_epoch) % epochs_per_super_epoch
        return slice(chunk * seed_batch_size, (chunk + 1) * seed_batch_size)

    def _data_source_scalars(
        self,
        source_indices: Tensor,
        seed_labels: Sequence[AbstractLabels],
        seed_predictions: Sequence[AbstractPredictions],
    ) -> dict[str, Tensor]:
        """
        Score every loss and metric per offline data source, averaged over the seeds where it is not NaN.

        Args:
            source_indices: the data source index of each loaded row
            seed_labels: every seed's labels
            seed_predictions: every seed's predictions

        Returns:
            the scores as data_sources_<name>/<source index>
        """
        loaded_batch_size = len(source_indices)
        values: dict[str, list[Tensor]] = defaultdict(list)
        # The raw policies: the compiled compute_loss would recompile for every per-source row count.
        for seed_index, (policy, labels, predictions) in enumerate(
            zip(self._raw_policies(), seed_labels, seed_predictions, strict=True),
        ):
            seed_sources = source_indices
            if loaded_batch_size > self.training_config.dataloader_config.batch_size:
                seed_sources = source_indices[self.seed_rows(seed_index)]
            for source_index in cast(list[int], seed_sources.unique().tolist()):  # pyright: ignore[reportUnknownMemberType]
                rows = seed_sources == source_index
                source_labels = labels.apply(lambda tensor, rows=rows: tensor[rows])
                source_predictions = predictions.apply(lambda tensor, rows=rows: tensor[rows])
                scores = policy.compute_loss(source_labels, source_predictions) | policy.compute_metrics(
                    source_labels,
                    source_predictions,
                )
                for name, value in scores.items():
                    values[f"data_sources_{name}/{source_index}"].append(value.float().reshape(()))
        return {name: torch.stack(seed_values).nanmean() for name, seed_values in values.items()}

    def _is_scalar_log_step(self) -> bool:
        """
        Check whether Lightning writes on_step scalars at this step.

        Returns:
            True on a scalar log step
        """
        log_every_n_steps = max(
            1,
            self.training_config.lightning_trainer_config.log_every_n_steps,
        )
        return (self.global_step + 1) % log_every_n_steps == 0

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """
        Log the learning rate and gradient norms on log steps, and clip in the concurrent mode unless FlatAdamW does.
        """
        gradient_clip_val = self.training_config.lightning_trainer_config.gradient_clip_val
        if self._is_scalar_log_step():
            norms_by_name: dict[str, list[Tensor]] = defaultdict(list)
            for raw_policy in self._raw_policies():
                grads_by_module: dict[str, list[Tensor]] = defaultdict(list)
                for name, parameter in raw_policy.named_parameters():
                    if parameter.grad is not None:
                        grads_by_module[name.split(".", 1)[0]].append(parameter.grad)
                for module_name, grads in grads_by_module.items():
                    norms_by_name[f"grad_norm/{module_name}"].append(torch.nn.utils.get_total_norm(grads))
                total_norm = torch.nn.utils.get_total_norm(
                    [grad for grads in grads_by_module.values() for grad in grads],
                )
                norms_by_name["grad_norm/total"].append(total_norm)
                if gradient_clip_val:
                    norms_by_name["grad_norm/clip_scale"].append((gradient_clip_val / total_norm).clamp(max=1.0))
            self.log_dict(  # pyright: ignore[reportUnknownMemberType]
                {"trainer/lr": optimizer.param_groups[0]["lr"], **self._across_seeds(norms_by_name)},
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
        if self._concurrent and self._flat_adamw is None:
            trainer_config = self.training_config.lightning_trainer_config
            self.configure_gradient_clipping(
                optimizer,
                trainer_config.gradient_clip_val,
                trainer_config.gradient_clip_algorithm,
            )

    @override
    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip every policy's gradient norm on its own, since a joint clip would couple the policies."""
        del optimizer
        assert gradient_clip_algorithm is None or gradient_clip_algorithm == "norm", gradient_clip_algorithm
        if gradient_clip_val:
            for policy in self.policies:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip_val)

    @override
    def transfer_batch_to_device(
        self,
        batch: TrainingSample | SlotBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> TrainingSample:
        """Copy and augment the batch on a separate stream, overlapping the previous step's compute."""
        slot_batch = batch if isinstance(batch, SlotBatch) else None
        if slot_batch is not None:
            assert self.batch_slots is not None
            batch = self.batch_slots.materialize(slot_batch)
        assert isinstance(batch, TrainingSample)
        if device.type != "cuda" or not self.training_config.copy_batch_on_side_stream:
            copied = super().transfer_batch_to_device(batch, device, dataloader_idx)
            if slot_batch is not None and self.batch_slots is not None:
                self.batch_slots.release(slot_batch, torch.cuda.current_stream(device))
            return self._augmented(copied)
        stream = self._h2d_stream
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._h2d_stream = stream
            self._h2d_event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            # The batch is pinned, so its copy need not block the host.
            copied = batch.to(device, non_blocking=True)
            if slot_batch is not None and self.batch_slots is not None:
                self.batch_slots.release(slot_batch, stream)
            moved = self._augmented(copied)
        assert self._h2d_event is not None
        self._h2d_event.record(stream)
        return moved

    @override
    def on_train_batch_start(
        self,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Make the compute stream wait until the batch copy and augmentation have finished."""
        del batch_idx
        if self._h2d_event is not None:
            compute_stream = torch.cuda.current_stream()
            self._h2d_event.wait(compute_stream)
            for bundle in (batch.features, batch.labels, batch.navigation):
                for _, tensor in bundle.named_tensors():
                    tensor.record_stream(compute_stream)

    @override
    def on_train_epoch_start(self) -> None:
        """Refresh the loss weights and run every policy's epoch hook, clearing re-initialized parameters' moments."""
        self._loss_weights = self.policy.loss_weights(self.current_epoch)
        for name, tensor in self._loss_weight_tensors.items():
            tensor.fill_(self._loss_weights[name])
        for seed, raw_policy in zip(self._seeds, self._raw_policies(), strict=True):
            # A re-initialization draws from its own seed and epoch, the same on every rank.
            with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
                torch.manual_seed(seed + _REINIT_SEED_STRIDE * self.current_epoch)  # pyright: ignore[reportUnknownMemberType]
                reinitialized = raw_policy.on_train_epoch_start(self.current_epoch)
            if not reinitialized:
                continue
            if self._flat_adamw is not None:
                self._flat_adamw.clear_moments(reinitialized)
            else:
                optimizer = cast(Optimizer, cast(Any, self.optimizers()).optimizer)
                for parameter in reinitialized:
                    optimizer.state.pop(parameter, None)

    def _weight_decay_param_groups(self, raw_policy: AnyPolicy, weight_decay: float) -> list[dict[str, Any]]:
        """
        Split one policy's parameters into a decayed and an undecayed group.

        Only the weight matrices and kernels of leaf modules decay; embedding tables and free parameters do not.

        Args:
            raw_policy: the uncompiled policy
            weight_decay: the decayed group's weight decay

        Returns:
            the decayed group, then the undecayed group
        """
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        seen: set[int] = set()
        for module in raw_policy.modules():
            # MultiheadAttention owns its in_proj weights directly but has out_proj as a child.
            is_leaf = next(module.children(), None) is None or isinstance(module, torch.nn.MultiheadAttention)
            for parameter in module.parameters(recurse=False):
                if id(parameter) in seen or not parameter.requires_grad:
                    continue
                seen.add(id(parameter))
                decays = parameter.ndim >= 2 and is_leaf and not isinstance(module, torch.nn.Embedding)
                (decay if decays else no_decay).append(parameter)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Build one AdamW over every policy with a per-step cosine warm-restart schedule.

        FlatAdamW serves the concurrent mode when configured; either is element-wise, so the policies stay independent.
        """
        optimizer_config: OptimizerConfig = self.training_config.optimizer_config
        trainer_config = self.training_config.lightning_trainer_config
        policy_groups = [
            self._weight_decay_param_groups(raw_policy, optimizer_config.weight_decay)
            for raw_policy in self._raw_policies()
        ]
        LOG.info(
            f"Weight decay {optimizer_config.weight_decay} on "
            f"{sum(parameter.numel() for groups in policy_groups for parameter in groups[0]['params'])} parameters, "
            f"none on {sum(parameter.numel() for groups in policy_groups for parameter in groups[1]['params'])}, "
            f"over {len(policy_groups)} policies",
        )
        optimizer: Optimizer
        if optimizer_config.flat_adamw and self._concurrent:
            assert not optimizer_config.amsgrad, "flat_adamw has no amsgrad"
            assert trainer_config.gradient_clip_algorithm in (None, "norm"), trainer_config.gradient_clip_algorithm
            nccl_backend = _multi_rank_nccl_backend() if self.training_config.symmetric_gradient_memory else None
            if nccl_backend is not None:
                # torch.cuda.MemPool refuses expandable segments; they only matter for the caching allocator's own blocks.
                torch.cuda.memory._set_allocator_settings("expandable_segments:False")  # pyright: ignore[reportPrivateUsage]
                self._gradient_pool = torch.cuda.MemPool(nccl_backend.mem_allocator, symmetric=True)
            self._flat_adamw = FlatAdamW(
                policy_groups,
                lr=optimizer_config.learning_rate,
                betas=(0.9, optimizer_config.adam_beta2),
                max_grad_norm=trainer_config.gradient_clip_val,
                gradient_pool=self._gradient_pool,
            )
            if nccl_backend is not None and self._gradient_pool is not None:
                nccl_backend.register_mem_pool(self._gradient_pool)
                LOG.info("Gradient buffers in NCCL symmetric memory")
                torch.cuda.memory._set_allocator_settings(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))  # pyright: ignore[reportPrivateUsage]
            optimizer = self._flat_adamw
        else:
            if optimizer_config.flat_adamw:
                LOG.info("flat_adamw needs more than one seed; using torch's AdamW")
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [parameter for groups in policy_groups for parameter in groups[index]["params"]],
                        "weight_decay": weight_decay,
                    }
                    for index, weight_decay in ((0, optimizer_config.weight_decay), (1, 0.0))
                ],
                lr=optimizer_config.learning_rate,
                betas=(0.9, optimizer_config.adam_beta2),
                amsgrad=optimizer_config.amsgrad,
                fused=optimizer_config.fused and torch.cuda.is_available(),
                foreach=optimizer_config.foreach,
            )
        total_steps = int(self.trainer.estimated_stepping_batches)
        assert self.trainer.max_epochs is not None, "Trainer must be built with max_epochs set"
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            T_0=steps_per_epoch * self.training_config.effective_lr_first_cycle_epochs,
            T_mult=_CYCLE_LENGTH_MULT,
            warmup_fraction=optimizer_config.lr_warmup_fraction,
            eta_min=optimizer_config.lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _keep_epochs(self, max_epochs: int) -> list[int]:
        """
        List the epochs whose model checkpoints are kept.

        Args:
            max_epochs: the training's epoch count

        Returns:
            the configured epochs, or else every cosine cycle's last epoch
        """
        keep_epochs = self.training_config.epochs_to_keep_checkpoints_for
        if keep_epochs is not None:
            return keep_epochs
        return cycle_end_epochs(
            max_epochs,
            _CYCLE_LENGTH_MULT,
            self.training_config.effective_lr_first_cycle_epochs,
        )

    @override
    def on_train_epoch_end(self) -> None:
        """
        Log every policy's per-module parameter norms, then save and prune the checkpoint files on the kept epochs.
        """
        weight_norms_by_name: dict[str, list[Tensor]] = defaultdict(list)
        for raw_policy in self._raw_policies():
            parameters_by_module: dict[str, list[Tensor]] = defaultdict(list)
            for name, parameter in raw_policy.named_parameters():
                parameters_by_module[".".join(name.split(".")[:2])].append(parameter.detach())
            for module_name, parameters in parameters_by_module.items():
                weight_norms_by_name[f"weight_norm/{module_name}"].append(torch.nn.utils.get_total_norm(parameters))
        self.log_dict(  # pyright: ignore[reportUnknownMemberType]
            self._across_seeds(weight_norms_by_name),
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        if not self.training_config.save_model_checkpoint:
            return
        assert self.trainer.max_epochs is not None, "Trainer must be built with max_epochs set"
        epoch = self.current_epoch
        keep_all = self.training_config.keep_all_checkpoints
        keep_epochs = list(range(self.trainer.max_epochs)) if keep_all else self._keep_epochs(self.trainer.max_epochs)
        on_cadence = keep_all or (epoch + 1) % self.training_config.effective_checkpoint_every_n_epochs == 0
        if not (on_cadence or epoch in keep_epochs or epoch + 1 >= self.trainer.max_epochs):
            return
        output_dir = run_dir()
        # Every rank builds the checkpoint like Trainer.save_checkpoint does; rank 0 writes it straight as split files.
        checkpoint = cast(
            dict[str, Any],
            self.trainer._checkpoint_connector.dump_checkpoint(),  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        )
        if not self.trainer.is_global_zero:
            return
        split_checkpoint(
            move_data_to_device(checkpoint, "cpu"),
            [next(iter(raw_policy.state_dict())) for raw_policy in self._raw_policies()],
            self._seeds,
            output_dir,
            epoch,
        )
        LOG.info(f"Saved checkpoint files for epoch {epoch}.")
        prune_checkpoints(output_dir, epoch, keep_epochs, keep_training_state=keep_all)
