from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig


@dataclass
class WandbConfig:
    """W&B streaming, the trainer's only logger."""

    # Stream to W&B; false trains without any logger.
    enabled: bool = True
    # W&B project the runs land in.
    project: str = "py123d_garage"
    # null = your W&B account default.
    entity: str | None = None
    # online | offline | disabled.
    mode: str = "online"
    # null = W&B names the run.
    name: str | None = None
    # Groups related runs in the W&B UI.
    group: str | None = None
    # Freeform labels for filtering runs.
    tags: list[str] = field(default_factory=list)
    # Upload checkpoints as W&B artifacts.
    log_model: bool = False


@dataclass
class OptimizerConfig:
    """AdamW; the cosine warm-restart schedule is derived in the training loop."""

    # Peak learning rate; the cosine schedule decays from it.
    learning_rate: float = 3.0e-4
    # Decoupled AdamW weight decay.
    weight_decay: float = 0.01
    # Fraction of every cosine cycle spent ramping the learning rate up to
    # ``learning_rate`` before the cycle anneals; 0 restarts at the full rate.
    lr_warmup_fraction: float = 0.0
    # Learning rate every cosine cycle anneals down to.
    lr_min: float = 0.0
    # Normalize by the running max of the second moment instead of its current value.
    amsgrad: bool = True
    # Fused CUDA optimizer kernel; ignored on CPU. Incompatible with trainer-side
    # gradient clipping under mixed precision (the kernel unscales internally).
    fused: bool = False


@dataclass
class DataLoaderConfig:
    """Passed to the DataLoader as-is; batch_size and num_workers are per device."""

    # Samples per step.
    batch_size: int = 64
    # Worker processes building features in parallel.
    num_workers: int = 8
    # Use non-persistent worker process to avoid OOM when work with Py123D and LMDB.
    persistent_workers: bool = False
    # Page-locked host memory; required for async H2D copies.
    pin_memory: bool = True
    # Reshuffle every epoch.
    shuffle: bool = True
    # A compiled model wants static shapes.
    drop_last: bool = True


@dataclass
class LightningTrainerConfig:
    """Passed to lightning.Trainer as-is, plus the resolved strategy."""

    # Also sizes the cosine schedule's restart periods.
    max_epochs: int = 31
    # Stops before max_epochs when set; -1 = no step cap.
    max_steps: int = -1
    # float = fraction, int = num_batches.
    limit_train_batches: Any = 1.0
    # Batches summed into one optimizer step; the LR schedule counts optimizer steps.
    accumulate_grad_batches: int = 1
    # null = no gradient clipping.
    gradient_clip_val: float | None = None
    # norm | value; null = norm.
    gradient_clip_algorithm: str | None = None
    # gpu | cpu.
    accelerator: str = "gpu"
    # "auto" takes every GPU the node has, not the job's allocation; set it in single-GPU jobs.
    devices: Any = "auto"
    # World size = devices * num_nodes; match the SLURM allocation.
    num_nodes: int = 1
    # "auto" becomes DDP without the per-forward buffer broadcast; single-device on devices=1.
    strategy: str = "auto"
    # Autocast dtype for forward and loss.
    precision: str = "bf16-mixed"
    # cuDNN algorithm search; pays off once input shapes stop changing.
    benchmark: bool = True
    # Write cadence of on_step scalars; also gates the debug metrics, output
    # min/max, and gradient-norm computation.
    log_every_n_steps: int = 100


@dataclass
class TrainingConfig:
    """The train entry point; policy-agnostic, reads the cache store when configured."""

    # -- Config objects --

    # The policy to train: model, feature builders, losses.
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    # The datasets to train over; a mixture is more than one entry.
    offline_data_sources: dict[str, OfflineTrainingDataSourceConfig] = field(default_factory=dict)
    # AdamW hyperparameters.
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    # DataLoader kwargs.
    dataloader_config: DataLoaderConfig = field(
        default_factory=DataLoaderConfig,
    )
    # lightning.Trainer kwargs.
    lightning_trainer_config: LightningTrainerConfig = field(
        default_factory=LightningTrainerConfig,
    )
    # W&B streaming.
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    # -- Atomic settings --

    # Global RNG seed.
    seed: int = 0
    # Serve the policy's cached tensors from each source's prebuilt store; the rest build live.
    read_from_cache_store: bool = True
    # Worker processes for the startup scene filtering and cache-store coverage check. On 18 CPUs we oversubscribe to 36.
    startup_num_workers: int = 36

    # NHWC layout: cuDNN's bf16 conv kernels want channels-last.
    channels_last: bool = True
    # Upload batches on a side CUDA stream, overlapping the previous step; needs pin_memory.
    copy_batch_on_side_stream: bool = True
    # null (eager) | default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs.
    compile_mode: str | None = "default"
    # TF32 tensor cores for the ops autocast leaves in fp32.
    allow_tf32: bool = True

    # W&B image cadence: the current batch is re-rendered every this many steps.
    log_images_every_n_steps: int = 2000
    # Batches between progress-bar refreshes. 0 hides the bar.
    progress_bar_refresh_rate: int = 5

    # Per-epoch model_/optimizer_/trainer_state_{epoch:04d}.pth files;
    # evaluation loads the model file.
    save_model_checkpoint: bool = True
    # Epochs whose model_*.pth survive the per-epoch cleanup; empty = only the latest survives.
    epochs_to_keep_checkpoints_for: list[int] = field(default_factory=list)
    # Continue from the newest checkpoint trio in the run dir.
    resume_from_last_checkpoint: bool = False
    # Warm-start weights loaded into the policy before training; null = from scratch.
    initial_weights_file: str | None = None
    # false tolerates new heads without pretrained weights.
    initial_weights_strict: bool = True
