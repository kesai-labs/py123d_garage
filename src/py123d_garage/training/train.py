from __future__ import annotations

import faulthandler
import logging
import os
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import cv2
import hydra
import lightning as L
import numba
import torch
import yaml
from lightning.fabric.utilities.seed import pl_worker_init_function
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.contract_verifications import verify_offline_data_source_scenes_declared_intervals
from py123d_garage.cache import CacheStoreReader
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import log_stage, setup_logging
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor, build_scene_builder
from py123d_garage.training.callbacks import (
    BatchesProgressBar,
    EpochLoggingCallback,
    ThroughputLoggingCallback,
    VisualizationCallback,
)
from py123d_garage.training.checkpointing import merged_resume_checkpoint
from py123d_garage.training.dataset import TrainingDataset, TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule

LOG = logging.getLogger(__name__)


def _build_loggers(training_config: TrainingConfig) -> list[Logger]:
    """Builds the trainer loggers; W&B is the only logger, empty when disabled."""
    wandb_config = training_config.wandb_config
    if not wandb_config.enabled:
        return []
    return [
        WandbLogger(
            project=wandb_config.project,
            name=wandb_config.name,
            save_dir=str(run_dir()),
            log_model=wandb_config.log_model,
            entity=wandb_config.entity,
            group=wandb_config.group,
            tags=list(wandb_config.tags or []),
            mode=wandb_config.mode,
        ),
    ]


def _worker_init(worker_id: int) -> None:
    """We parallelize across DataLoader workers."""
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    numba.set_num_threads(1)  # pyright: ignore[reportUnknownMemberType]
    # Our custom worker_init_fn stops Lightning from installing this itself.
    pl_worker_init_function(worker_id)


def _build_source_datasets(
    training_config: TrainingConfig,
    policy: AnyPolicy,
) -> list[TrainingDataset]:
    """Builds one dataset per data source over the policy's feature and target builders."""
    assert training_config.offline_data_sources, "offline_data_sources must not be empty"
    datasets: list[TrainingDataset] = []
    for source_index, source in enumerate(training_config.offline_data_sources.values()):
        scene_builder: ArrowSceneBuilder = build_scene_builder(
            source.data_root,
        )
        with log_stage(
            f"Source {source_index + 1} of {len(training_config.offline_data_sources)}: "
            f"filtering scenes of splits {source.garage_scene_filter.split_names} "
            f"under {source.data_root} (this can take minutes on large datasets)",
        ):
            scene_filter = source.garage_scene_filter.to_py123d_scene_filter()
            policy.verify_contract(offline_data_source_config=source, scene_filter=scene_filter)
            scenes: Sequence[SceneAPI] = scene_builder.get_scenes(
                filter=scene_filter,
                # NOTE@ln2697: change to a thread pool if this causes problems.
                executor=VerboseProcessPoolExecutor(
                    max_workers=training_config.startup_num_workers,
                ),
            )
        assert len(scenes), f"source {source.data_root} selected no scenes"
        LOG.info(
            f"Source {source.data_root}: {len(scenes)} scenes passed the filter",
        )
        verify_offline_data_source_scenes_declared_intervals(source, scenes[0])

        cache_reader: CacheStoreReader | None = None
        if training_config.read_from_cache_store:
            with log_stage(f"Opening the cache store at {source.cache_root}"):
                cache_reader = CacheStoreReader(
                    source.cache_root,
                    policy.cache_signature(scenes[0].scene_metadata.dataset),
                )
        else:
            LOG.info(
                "read_from_cache_store is off: every tensor is built live, which is several times slower per sample",
            )
        datasets.append(
            TrainingDataset(
                scenes,
                policy,
                cache_reader,
                startup_num_workers=training_config.startup_num_workers,
            ),
        )
    LOG.info(
        f"Total: {sum(len(dataset) for dataset in datasets)} scenes across {len(datasets)} sources",
    )
    return datasets


def _build_sampler(
    source_datasets: list[TrainingDataset],
    offline_data_sources: dict[str, OfflineTrainingDataSourceConfig],
) -> WeightedRandomSampler | None:
    """Weights every sample by its source; None when each weight is 1, i.e. plain concatenation."""
    if all(source.source_weight == 1.0 for source in offline_data_sources.values()):
        return None
    per_sample_weights = torch.cat(
        [
            torch.full(
                (len(dataset),),
                source.source_weight,
                dtype=torch.double,
            )
            for source, dataset in zip(
                offline_data_sources.values(),
                source_datasets,
                strict=True,
            )
        ],
    )
    # The epoch keeps the natural sample count; weights only reshape the mixture.
    return WeightedRandomSampler(
        per_sample_weights,  # pyright: ignore[reportArgumentType]
        num_samples=len(per_sample_weights),
        replacement=True,
    )


def _fast_fail_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> None:
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def _export_sensor_rig(
    scene: SceneAPI,
    output_dir: str,
    source_index: int,
) -> None:
    """Writes every modality's metadata next to the training outputs; the rig is assumed identical across the source."""
    rig: dict[str, Any] = {
        "log_name": scene.get_log_metadata().log_name,
        "modalities": {key: metadata.to_dict() for key, metadata in scene.get_all_modality_metadatas().items()},
    }
    with (Path(output_dir) / f"sensor_rig_{source_index}.yaml").open(
        "w",
    ) as file:
        yaml.safe_dump(rig, file, sort_keys=False)


register_schema("train", TrainingConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    # Print the C-level stack trace when a rank dies on a fatal signal.
    faulthandler.enable()
    sys.excepthook = _fast_fail_excepthook
    training_config: TrainingConfig = finalize(cfg, TrainingConfig)
    output_dir = run_dir()
    save_config(training_config)

    # Let fp32 matmuls and convolutions use the tensor cores via TF32
    if training_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    L.seed_everything(training_config.seed, workers=True)
    LOG.info(
        f"Seed {training_config.seed}, results in {output_dir!s}",
    )
    trainer_config = training_config.lightning_trainer_config
    LOG.info(
        f"{trainer_config.max_epochs} epochs on {trainer_config.devices} devices x "
        f"{trainer_config.num_nodes} nodes, batch {training_config.dataloader_config.batch_size} and "
        f"{training_config.dataloader_config.num_workers} workers per device, "
        f"compile={training_config.compile_mode}",
    )

    with log_stage(f"Building policy {training_config.policy_config.target}"):
        policy = cast(
            AnyPolicy,
            build_from_string(training_config.policy_config, AbstractPolicy),
        )
        policy.initialize()
    LOG.info(
        f"Policy has {sum(p.numel() for p in policy.parameters()):,} trainable parameters",
    )
    lightning_module = PolicyLightningModule(
        policy=policy,
        training_config=training_config,
    )

    with log_stage(
        f"Building the datasets of {len(training_config.offline_data_sources)} data sources",
    ):
        source_datasets: list[TrainingDataset] = _build_source_datasets(
            training_config,
            policy,
        )
    train_data: ConcatDataset[Any] = ConcatDataset(source_datasets)
    LOG.info(f"{len(train_data)} train samples")
    sampler = _build_sampler(source_datasets, training_config.offline_data_sources)
    # batch_size and num_workers are per device.
    dataloader_kwargs = asdict(training_config.dataloader_config)
    if sampler is not None:
        dataloader_kwargs["shuffle"] = False
    if dataloader_kwargs["num_workers"] == 0:  # the DataLoader refuses the combination
        dataloader_kwargs["persistent_workers"] = False
    train_dataloader = DataLoader(
        train_data,
        worker_init_fn=_worker_init,
        collate_fn=TrainingSample.collate,
        sampler=sampler,
        **dataloader_kwargs,
    )

    callbacks: list[L.Callback] = [
        *policy.get_training_callbacks(),
        # Lightning picks RichProgressBar whenever rich is installed, and its live
        # rendering writes nothing into a redirected log file.
        BatchesProgressBar(
            training_config.dataloader_config.batch_size,
            training_config.progress_bar_refresh_rate,
        ),
        VisualizationCallback(training_config.log_images_every_n_steps),
        ThroughputLoggingCallback(
            training_config.dataloader_config.batch_size,
            training_config.lightning_trainer_config.log_every_n_steps,
        ),
        EpochLoggingCallback(),
    ]

    training_loggers: list[Logger] = _build_loggers(training_config)
    trainer_kwargs = asdict(training_config.lightning_trainer_config)
    single_device = trainer_kwargs["devices"] == 1 and trainer_kwargs["num_nodes"] == 1
    if trainer_kwargs["strategy"] == "auto" and not single_device:
        # Skips DDP's per-forward broadcast of the BatchNorm running statistics; one device needs no DDP.
        trainer_kwargs["strategy"] = DDPStrategy(broadcast_buffers=False)
    trainer = L.Trainer(
        **trainer_kwargs,
        # Keeps Lightning's own files inside the run's output dir.
        default_root_dir=str(output_dir),
        # True would add Lightning's own default ModelCheckpoint.
        enable_checkpointing=False,
        logger=training_loggers,
        callbacks=callbacks,
    )

    for training_logger in training_loggers:
        training_logger.log_hyperparams(asdict(training_config))
    if trainer.is_global_zero:
        for source_index, dataset in enumerate(source_datasets):
            _export_sensor_rig(
                dataset.scenes[0],
                str(output_dir),
                source_index,
            )

    resume = merged_resume_checkpoint(output_dir) if training_config.resume_from_last_checkpoint else None

    resume_note = f", resuming from {resume}" if resume else ""
    compile_note = (
        f" (compiling {training_config.compile_mode}: the first step can take minutes)"
        if training_config.compile_mode is not None
        else ""
    )
    LOG.info(f"Starting training{resume_note}{compile_note}")
    # sys.excepthook does not run for SystemExit or fatal signals, so without this
    # a crashing rank would exit silently.
    try:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            ckpt_path=resume,
        )
    except BaseException:
        sys.stderr.write(
            f"=== rank {os.environ.get('SLURM_PROCID', '?')} exiting ===\n",
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    LOG.info(
        f"Training finished after {trainer.current_epoch} epochs / {trainer.global_step} global steps",
    )


if __name__ == "__main__":
    main()
