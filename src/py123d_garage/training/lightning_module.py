from __future__ import annotations

import logging
from collections import defaultdict
from typing import cast

import jaxtyping as jt
import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.api.abstract_policy_tensors import AbstractFeatures
from py123d_garage.common.config_help import run_dir
from py123d_garage.config.schema.training.training_config import (
    OptimizerConfig,
    TrainingConfig,
)
from py123d_garage.training.checkpointing import (
    prune_checkpoint,
    split_checkpoint,
)
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.lr_scheduler import (
    CosineAnnealingWarmRestartsWithWarmup,
)

LOG = logging.getLogger(__name__)


class PolicyLightningModule(L.LightningModule):
    def __init__(self, policy: AnyPolicy, training_config: TrainingConfig):
        """Initialise the lightning module wrapper around the policy to train."""
        super().__init__()
        self.training_config: TrainingConfig = training_config
        self.policy: AnyPolicy = policy
        self._loss_weights: dict[str, float] = {}
        self._h2d_stream: torch.cuda.Stream | None = None
        self._h2d_event: torch.cuda.Event | None = None

        if training_config.initial_weights_file:
            LOG.info(
                f"Loading initial policy weights from {training_config.initial_weights_file}",
            )
            state_dict = torch.load(
                training_config.initial_weights_file,
                map_location="cpu",
                weights_only=True,
            )
            missing, unexpected = cast(
                tuple[list[str], list[str]],
                self.policy.load_state_dict(
                    state_dict,
                    strict=training_config.initial_weights_strict,
                ),
            )
            if missing or unexpected:
                LOG.info(
                    f"Initial weights: {len(missing)} missing keys, {len(unexpected)} unexpected keys",
                )

        if training_config.channels_last:
            # cuDNN's bf16 convolutions expect NHWC; with the default NCHW layout
            # every convolution transposes its input and output.
            self.policy = cast(
                AnyPolicy,
                self.policy.to(memory_format=torch.channels_last),  # pyright: ignore[reportCallIssue]
            )

        # mode=None is torch.compile's default mode, not "do not compile", so
        # eager has to skip the call rather than pass the null through.
        if training_config.compile_mode is not None:
            self.policy = cast(
                AnyPolicy,
                torch.compile(  # pyright: ignore[reportUnknownMemberType]
                    self.policy,
                    dynamic=False,
                    backend="inductor",
                    mode=training_config.compile_mode,
                ),
            )

    @override
    def training_step(
        self,
        batch: TrainingSample,
        batch_idx: int,
    ) -> jt.Float[Tensor, ""]:
        """Runs the forward pass, weights the per-task losses, logs the scalars, and returns the total loss."""
        labels = batch.labels
        features = cast(
            AbstractFeatures,
            self.policy.augment_features(batch.features),
        )
        if self.training_config.channels_last:

            def to_channels_last(tensor: Tensor) -> Tensor:
                # Only 4D (N,C,H,W) tensors have a channels-last layout.
                if tensor.dim() != 4:
                    return tensor
                return tensor.to(memory_format=torch.channels_last)

            features = features.apply(to_channels_last)
            labels = labels.apply(to_channels_last)
        predictions = self.policy.forward(features, batch.navigation)
        losses: dict[str, Tensor] = self.policy.compute_loss(
            labels,
            predictions,
        )

        total_loss = torch.zeros((), device=self.device)
        scalars: dict[str, Tensor | float] = {}
        for name, value in losses.items():
            weighted = self._loss_weights[name] * value.float().reshape(())
            total_loss = total_loss + weighted
            scalars[f"train/unscaled_{name}"] = value.float()
            scalars[f"train/scaled_{name}"] = weighted
        self.log_dict(scalars, on_step=True, on_epoch=False, sync_dist=True)  # pyright: ignore[reportUnknownMemberType]
        self.log(  # pyright: ignore[reportUnknownMemberType]
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self._is_scalar_log_step():
            with torch.no_grad():
                debug_scalars: dict[str, Tensor | float] = {}
                for name, tensor in predictions.named_tensors():
                    debug_scalars[f"outputs/{name}_min"] = tensor.min().float()
                    debug_scalars[f"outputs/{name}_max"] = tensor.max().float()
                metrics: dict[str, Tensor] = self.policy.compute_metrics(
                    labels,
                    predictions,
                )
                for name, value in metrics.items():
                    debug_scalars[f"metric/{name}"] = value.float()
            self.log_dict(  # pyright: ignore[reportUnknownMemberType]
                debug_scalars,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
        return total_loss

    def _is_scalar_log_step(self) -> bool:
        """Whether this step is one Lightning writes on_step scalars for; gates the debug logging."""
        log_every_n_steps = max(
            1,
            self.training_config.lightning_trainer_config.log_every_n_steps,
        )
        return (self.global_step + 1) % log_every_n_steps == 0

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Logs the learning rate and per-module gradient norms on the scalar-log cadence."""
        if not self._is_scalar_log_step():
            return
        raw_policy = cast(
            AnyPolicy,
            getattr(self.policy, "_orig_mod", self.policy),
        )
        grads_by_module: dict[str, list[Tensor]] = defaultdict(list)
        for name, parameter in raw_policy.named_parameters():
            if parameter.grad is not None:
                grads_by_module[name.split(".", 1)[0]].append(parameter.grad)
        grad_norms = {
            f"grad_norm/{module_name}": torch.nn.utils.get_total_norm(grads)
            for module_name, grads in grads_by_module.items()
        }
        grad_norms["grad_norm/total"] = torch.nn.utils.get_total_norm(
            [grad for grads in grads_by_module.values() for grad in grads],
        )
        self.log_dict(  # pyright: ignore[reportUnknownMemberType]
            {"trainer/lr": optimizer.param_groups[0]["lr"], **grad_norms},
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    @override
    def transfer_batch_to_device(
        self,
        batch: TrainingSample,
        device: torch.device,
        dataloader_idx: int,
    ) -> TrainingSample:
        """Copies the batch to the GPU on a separate stream, overlapping the previous step's compute."""
        if device.type != "cuda" or not self.training_config.copy_batch_on_side_stream:
            return super().transfer_batch_to_device(
                batch,
                device,
                dataloader_idx,
            )
        stream = self._h2d_stream
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._h2d_stream = stream
            self._h2d_event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            moved = super().transfer_batch_to_device(
                batch,
                device,
                dataloader_idx,
            )
        assert self._h2d_event is not None
        self._h2d_event.record(stream)
        return moved

    @override
    def on_train_batch_start(
        self,
        batch: TrainingSample,
        batch_idx: int,
    ) -> None:
        """Makes the compute stream wait until the batch copy has finished."""
        del batch_idx
        if self._h2d_event is not None:
            compute_stream = torch.cuda.current_stream()
            self._h2d_event.wait(compute_stream)
            for bundle in (batch.features, batch.labels, batch.navigation):
                for _, tensor in bundle.named_tensors():
                    tensor.record_stream(compute_stream)

    @override
    def on_train_epoch_start(self) -> None:
        """Refresh the policy's loss weights for the current epoch."""
        self._loss_weights = self.policy.loss_weights(self.current_epoch)

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """AdamW with a per-step cosine annealing warm-restart schedule."""
        optimizer_config: OptimizerConfig = self.training_config.optimizer_config
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=optimizer_config.learning_rate,
            amsgrad=optimizer_config.amsgrad,
            weight_decay=optimizer_config.weight_decay,
            fused=optimizer_config.fused and torch.cuda.is_available(),
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        assert self.trainer.max_epochs is not None, "Trainer must be built with max_epochs set"
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=2,
            warmup_fraction=optimizer_config.lr_warmup_fraction,
            eta_min=optimizer_config.lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @override
    def on_train_epoch_end(self) -> None:
        """Save this epoch's checkpoint trio, then prune the previous epoch's."""
        if not self.training_config.save_model_checkpoint:
            return
        output_dir = run_dir()
        # Every rank must enter the dump; splitting and pruning are rank-0 only.
        lightning_monolith_file = output_dir / ".checkpoint_dump.pth"
        self.trainer.save_checkpoint(lightning_monolith_file)
        if not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch
        raw_policy = getattr(self.policy, "_orig_mod", self.policy)
        split_checkpoint(
            lightning_monolith_file,
            next(iter(raw_policy.state_dict())),
            output_dir,
            epoch,
        )
        LOG.info(f"Saved checkpoint files for epoch {epoch}.")
        prune_checkpoint(
            output_dir,
            epoch - 1,
            self.training_config.epochs_to_keep_checkpoints_for,
        )
