# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.photometric_augmentation_config import CameraAugmentationConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt


@dataclass
class WandbConfig:
    """W&B logging, the trainer's only logger."""

    # Log to W&B; false trains without any logger.
    enabled: bool = True
    # W&B project the runs go to.
    project: str = "py123d_garage"
    # W&B user or team; null uses the account default.
    entity: str | None = None
    # W&B mode: online, offline or disabled.
    mode: str = "online"
    # Run name; null lets W&B pick one.
    name: str | None = None
    # Group name that bundles related runs in the W&B UI.
    group: str | None = None
    # Free labels to filter runs by.
    tags: list[str] = field(default_factory=list)
    # Upload checkpoints to W&B as artifacts.
    log_model: bool = False


@dataclass
class OptimizerConfig:
    """AdamW settings; the cosine warm-restart schedule is built in the training loop."""

    # Peak learning rate that the cosine schedule starts from.
    learning_rate: float = 3.0e-4
    # AdamW weight decay.
    weight_decay: float = 0.01
    # AdamW beta2, how slowly the average of squared gradients forgets; lower values react faster.
    adam_beta2: float = 0.999
    # Part of every cosine cycle spent ramping up to learning_rate; 0 starts each cycle at the full rate.
    lr_warmup_fraction: float = 0.05
    # Learning rate at the end of every cosine cycle.
    lr_min: float = 0.0
    # Epochs in the first cosine cycle, each later cycle is twice as long; null means one pass per seed.
    lr_first_cycle_epochs: PositiveInt | None = None
    # Use the AMSGrad variant of AdamW.
    amsgrad: bool = False
    # With more than one seed, use our own fast clip and AdamW kernel per seed; one seed uses torch AdamW.
    flat_adamw: bool = True
    # Run that kernel right after each seed's backward; false runs all seeds' updates at the end of the step.
    flat_adamw_in_graph: bool = True
    # Use torch's fused CUDA AdamW kernel; ignored on CPU.
    fused: bool = True
    # Use torch's foreach AdamW update; torch does not allow it together with fused.
    foreach: bool = False


@dataclass
class DataLoaderConfig:
    """DataLoader arguments; batch_size and num_workers are per device."""

    # Samples per step for each seed on each device.
    batch_size: int = 64
    # Worker processes that build the features on each device.
    num_workers: int = 8
    # Keep the workers alive between epochs so the GPU does not wait for them to restart.
    persistent_workers: bool = True
    # Page-locked host memory, needed for async copies to the GPU.
    pin_memory: bool = True
    # Drop the last short batch so shapes stay fixed for the compiled model.
    drop_last: bool = True
    # Batches each worker loads ahead, ~0.1 GB of shared memory each; null keeps the DataLoader's 2.
    prefetch_factor: int | None = 2
    # Hand batches over in sampler order; false takes them as workers finish, so reruns are not byte-identical.
    in_order: bool = False


@dataclass
class LightningTrainerConfig:
    """Arguments passed to lightning.Trainer as-is."""

    # Number of epochs to train; also sets the length of the cosine schedule.
    max_epochs: int = 31
    # Stop after this many steps if that comes before max_epochs; -1 means no step limit.
    max_steps: int = -1
    # Batches per epoch: a float is a fraction of the data, an int is a count.
    limit_train_batches: Any = 1.0
    # Batches summed into one optimizer step.
    accumulate_grad_batches: int = 1
    # Gradient clipping threshold; null turns clipping off.
    gradient_clip_val: float | None = 1.0
    # Clip by norm or by value; null means norm.
    gradient_clip_algorithm: str | None = "norm"
    # Device type: gpu or cpu.
    accelerator: str = "gpu"
    # Devices per node; "auto" takes every GPU on the node, so set it in single-GPU jobs.
    devices: Any = "auto"
    # Number of nodes; must match the SLURM allocation.
    num_nodes: int = 1
    # Lightning strategy; "auto" picks DDP on many devices and a single device otherwise.
    strategy: str = "auto"
    # Training precision; bf16-mixed runs the forward and loss in bf16.
    precision: str = "bf16-mixed"
    # Let cuDNN search for the fastest kernels; good when input shapes stay fixed.
    benchmark: bool = True
    # Steps between scalar logs; also how often debug metrics and gradient norms are computed.
    log_every_n_steps: int = 1000
    # Lightning profiler, simple or advanced; null turns it off.
    profiler: str | None = None


@dataclass
class TrainingConfig:
    """Settings of the train entry point; works for any policy."""

    # The policy to train: model, feature builders and losses.
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    # Datasets to train on; more than one entry makes a mixture.
    offline_data_sources: list[OfflineTrainingDataSourceConfig] = field(default_factory=list)
    # AdamW and learning rate schedule settings.
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    # DataLoader settings.
    dataloader_config: DataLoaderConfig = field(
        default_factory=DataLoaderConfig,
    )
    # lightning.Trainer settings.
    lightning_trainer_config: LightningTrainerConfig = field(
        default_factory=LightningTrainerConfig,
    )
    # W&B logging settings.
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    # Camera image augmentation, used in training only.
    camera_augmentation: CameraAugmentationConfig = field(default_factory=CameraAugmentationConfig)

    # Policies trained side by side, one per seed; 3 epochs then make one pass over the data per seed.
    num_seeds: PositiveInt = 3
    # Loaded batch as a multiple of batch_size, split into one chunk per seed; null means num_seeds.
    loaded_batch_factor: float | None = None
    # Read the policy's tensors from each source's prebuilt cache store; the rest are built live.
    read_from_cache_store: bool = True
    # Check at startup that the cache stores hold every selected scene; a stage that follows a checked stage on the
    # same stores and a subset of its scenes can skip it.
    check_cache_coverage: bool = True
    # Worker processes for scene filtering and cache store checks at startup.
    startup_num_workers: int = 36
    # Local directory the cache store files are copied to on first read; null reads them in place.
    cache_mirror_root: str | None = None
    # Parallel copy threads per rank when filling the mirror; more threads copy faster.
    cache_mirror_copy_threads: PositiveInt = 16
    # Save per-scene tensors (ego status, labels, target points) under the mirror root so later epochs skip the logs.
    session_cache: bool = True
    # Low-priority processes per rank that fill the session cache after the dataset build; 0 leaves it to the loader.
    session_cache_fill_processes: NonNegativeInt = 9
    # SQLite page-cache pages per session-cache shard in every loader process; diskcache's 8192 cost 512 MB per worker.
    session_cache_sqlite_cache_pages: PositiveInt = 1024
    # Ask the kernel to read a whole batch's stored records in parallel before decoding them; helps when the stores outgrow the page cache.
    cache_read_prefetch: bool = True
    # Give the training process two CPU cores of its own, apart from the loader workers; needs at least 4 cores.
    dedicated_main_core: bool = True
    # Exempt the objects alive after the dataset build and after the first batch from garbage collection, so the
    # collections of the training process and of its forked loader workers stay short instead of stalling every rank.
    freeze_heap: bool = True

    # Store images channels-last (NHWC), which is faster for bf16 convolutions.
    channels_last: bool = True
    # Copy the next batch to the GPU while the current step runs; needs pin_memory.
    copy_batch_on_side_stream: bool = True
    # With loader workers on a GPU, workers collate into slots of one page-locked buffer instead of handing over fresh
    # shared memory per tensor, which the single pin-memory thread maps, copies and frees; replaces pin_memory.
    pinned_batch_slots: bool = True
    # Let DDP keep gradients in its all-reduce buckets to save a copy; no effect on one device.
    gradient_as_bucket_view: bool = True
    # Size of DDP's all-reduce buckets in MB; no effect on one device.
    gradient_bucket_cap_mb: int = 25
    # With more than one seed, copy all gradients into the flat buffer in one batched copy instead of one per parameter.
    batched_gradient_copy: bool = True
    # With batched_gradient_copy, zero only the gradients the backward leaves out instead of the whole flat buffer.
    skip_gradient_zeroing: bool = True
    # With more than one seed and rank, put the gradient buffers in NCCL symmetric memory, whose all-reduce kernels are cheaper.
    symmetric_gradient_memory: bool = False
    # The torch.compile mode, such as default or max-autotune; null runs the model eagerly.
    compile_mode: str | None = "reduce-overhead"
    # torch._inductor.config attributes set before compiling, e.g. +inductor_config.combo_kernels=true.
    inductor_config: dict[str, Any] = field(default_factory=dict)
    # Compile the policies in a thread on a batch of the first source while the other sources build, instead of in
    # the first step.
    compile_ahead: bool = False
    # Before training, run one forward and backward of policy 0 with its kernel patches and of a plain torch policy
    # with the same weights on the first batch, in the step's precision and once more in fp32, and compare the loss
    # and every gradient; nothing runs without patches.
    kernel_check: bool = True
    # Tolerance of the patched loss, relative to the plain policy's loss.
    kernel_check_loss_rtol: float = 1e-2
    # A patched gradient may lie this many times further from the fp32 gradient than the plain policy's gradient
    # does, plus this fraction of the norm of that parameter's fp32 gradient. A gradient smaller than the plain
    # policy's error, which bf16 cannot resolve, may lie up to 10 times further.
    kernel_check_grad_factor: float = 2.0
    kernel_check_grad_floor: float = 2e-2
    # End the run on a mismatch; false logs it and trains.
    kernel_check_refuse: bool = True
    # End the run when a requested patch's kernels do not run in the check: the GPU, the kernel build or its libraries
    # rule them out, the batch falls outside what they handle, or another patch replaces them; false logs it and trains
    # on the torch path.
    kernel_check_require_patches: bool = True
    # With more than one seed, seconds without a finished step before the rank aborts, so a hang ends the job.
    step_watchdog_timeout_s: float = 3600.0
    # Seconds the shutdown after training may take before the rank exits on its own.
    shutdown_timeout_s: float = 120.0
    # Seconds a rank waits for rank 0 to finish filtering a source's scenes.
    scene_filter_wait_timeout_s: float = 7200.0
    # Use TF32 tensor cores for the ops that stay in fp32.
    allow_tf32: bool = True
    # Prefer cuDNN's fused attention in scaled_dot_product_attention; its backward needs head dims <= 128.
    cudnn_attention: bool = False
    # Step at which every rank writes a Chrome trace of five steps into the run dir; null means no trace.
    trace_from_step: int | None = None

    # Steps between batch images logged to W&B.
    log_images_every_n_steps: int = 3000
    # Batches between progress bar updates; 0 hides the bar.
    progress_bar_refresh_rate: int = 20

    # Write model, optimizer and trainer state checkpoints during training.
    save_model_checkpoint: bool = True
    # Epochs between checkpoints, plus kept and last epochs; null means one pass per seed.
    checkpoint_every_n_epochs: PositiveInt | None = None
    # Restart persistent loader workers every this many epochs to cap their memory; 0 never restarts them.
    persistent_workers_recycle_epochs: int = 3
    # Epochs whose model files are kept; null keeps the end of each cosine cycle, empty keeps only the latest.
    epochs_to_keep_checkpoints_for: list[int] | None = None
    # Write and keep every epoch's checkpoints; overrides checkpoint_every_n_epochs and epochs_to_keep_checkpoints_for.
    keep_all_checkpoints: bool = False
    # Continue from the newest checkpoint in the run dir.
    resume_from_last_checkpoint: bool = False
    # Starting weights: empty trains from scratch, one file is used for all seeds, or give one file per seed.
    initial_weights_files: list[str] = field(default_factory=list)
    # Require every weight to load; false allows new heads without pretrained weights.
    initial_weights_strict: bool = True

    @property
    def effective_loaded_batch_factor(self) -> float:
        """loaded_batch_factor, num_seeds when unset."""
        return float(self.num_seeds) if self.loaded_batch_factor is None else self.loaded_batch_factor

    @property
    def effective_lr_first_cycle_epochs(self) -> int:
        """optimizer_config.lr_first_cycle_epochs, one full pass of every seed when unset."""
        if self.optimizer_config.lr_first_cycle_epochs is None:
            return int(self.effective_loaded_batch_factor)
        return self.optimizer_config.lr_first_cycle_epochs

    @property
    def effective_checkpoint_every_n_epochs(self) -> int:
        """checkpoint_every_n_epochs, one write per full pass of every seed when unset."""
        if self.checkpoint_every_n_epochs is None:
            return int(self.effective_loaded_batch_factor)
        return self.checkpoint_every_n_epochs
