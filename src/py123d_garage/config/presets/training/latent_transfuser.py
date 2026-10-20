# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import replace
from typing import Final

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.presets.offline_data_sources.nuplan import (
    nuplan_train,
    nuplan_val,
)
from py123d_garage.config.presets.offline_data_sources.nurec import nurec_other, nurec_train
from py123d_garage.config.presets.offline_data_sources.physical_ai_av import (
    physical_ai_av_test,
    physical_ai_av_train,
    physical_ai_av_val,
)
from py123d_garage.config.presets.policy.latent_transfuser import latent_transfuser
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.config.schema.training.training_config import (
    LightningTrainerConfig,
    TrainingConfig,
)
from py123d_garage.py123d_help.misc import FRONT_CAMERAS

_MIXED_STORE_SUFFIX: Final[str] = "temporal_front_camera"


def _source_architecture() -> TransfuserConfig:
    """
    The architecture the sources are declared with.

    It requires all three front cameras, like the cache build, so every stage reads the anchors the stores hold.

    Returns:
        the TransFuser config of the sources.
    """
    architecture = latent_transfuser()
    return replace(architecture, camera_config=replace(architecture.camera_config, input_cameras=FRONT_CAMERAS))


def _nuplan_sources() -> list[OfflineTrainingDataSourceConfig]:
    """The nuPlan sources of the mixed recipe, which also make the nuPlan-only recipe."""
    architecture = _source_architecture()
    suffix = _MIXED_STORE_SUFFIX
    return [
        nuplan_train(
            architecture,
            cache_store=f"latent_transfuser_nuplan_train_sparse_200ms_{suffix}",
            timestamp_threshold_s=0.2,
        ),
        nuplan_val(
            architecture,
            cache_store=f"latent_transfuser_nuplan_val_sparse_200ms_{suffix}",
            timestamp_threshold_s=0.2,
        ),
    ]


def _mixed_sources() -> list[OfflineTrainingDataSourceConfig]:
    """The training sources of the mixed recipe."""
    architecture = _source_architecture()
    suffix = _MIXED_STORE_SUFFIX
    return [
        physical_ai_av_train(
            architecture,
            cache_store=f"latent_transfuser_physical_ai_av_train_uniform_sparse_2s_{suffix}",
            timestamp_threshold_s=2.0,
        ),
        physical_ai_av_val(
            architecture,
            cache_store=f"latent_transfuser_physical_ai_av_val_uniform_sparse_2s_{suffix}",
            timestamp_threshold_s=2.0,
        ),
        physical_ai_av_test(
            architecture,
            cache_store=f"latent_transfuser_physical_ai_av_test_uniform_sparse_2s_{suffix}",
            timestamp_threshold_s=2.0,
        ),
        nurec_train(architecture, cache_store=f"latent_transfuser_nurec_train_{suffix}", timestamp_threshold_s=None),
        nurec_other(architecture, cache_store=f"latent_transfuser_nurec_other_{suffix}", timestamp_threshold_s=None),
        *_nuplan_sources(),
    ]


def _stage(
    architecture: TransfuserConfig,
    max_epochs: int,
    offline_data_sources: list[OfflineTrainingDataSourceConfig],
) -> TrainingConfig:
    """
    Builds one stage of the recipe; three seeds train side by side, so 3 epochs are one pass over the data.

    Args:
        architecture: the TransFuser config of this stage.
        max_epochs: number of epochs to train.
        offline_data_sources: the training sources.

    Returns:
        the training config of the stage.
    """
    return TrainingConfig(
        offline_data_sources=offline_data_sources,
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(max_epochs=max_epochs),
        # Stages add or drop heads, so loading the previous stage's weights cannot be strict.
        initial_weights_strict=False,
    )


def _route_filtered(sources: list[OfflineTrainingDataSourceConfig]) -> list[OfflineTrainingDataSourceConfig]:
    """The sources, keeping only anchors with 80 m of route left, enough for the farthest target point."""
    return [
        replace(source, garage_scene_filter=replace(source.garage_scene_filter, min_remaining_route_m=80.0))
        for source in sources
    ]


def _pretrain(sources: list[OfflineTrainingDataSourceConfig]) -> TrainingConfig:
    """Perception pretrain: the backbone and the perception heads, without the planning decoder."""
    architecture = latent_transfuser()
    architecture = replace(
        architecture,
        planning_config=replace(architecture.planning_config, use_planning_decoder=False),
    )
    return replace(_stage(architecture, max_epochs=93, offline_data_sources=sources), initial_weights_strict=True)


def _midtrain(sources: list[OfflineTrainingDataSourceConfig]) -> TrainingConfig:
    """Unconditional midtrain: adds the planning decoder and trains the whole model, without target points."""
    architecture = replace(latent_transfuser(), required_target_point_distances_m=[])
    return _stage(architecture, max_epochs=45, offline_data_sources=_route_filtered(sources))


def _posttrain(sources: list[OfflineTrainingDataSourceConfig]) -> TrainingConfig:
    """Conditional posttrain: trains the heads on a frozen backbone, with target points that are mostly dropped."""
    architecture = latent_transfuser()
    architecture = replace(
        architecture,
        required_target_point_distances_m=[40.0, 60.0, 80.0],
        backbone_config=replace(architecture.backbone_config, freeze_backbone=True),
        planning_conditioning_config=replace(
            architecture.planning_conditioning_config,
            # Per-point chance that drops all three target points together in 95 % of the samples.
            target_point_dropout_probability=0.95 ** (1 / 3),
        ),
    )
    return _stage(architecture, max_epochs=45, offline_data_sources=_route_filtered(sources))


def pretrain() -> TrainingConfig:
    """Perception pretrain on the mixed sources."""
    return _pretrain(_mixed_sources())


def midtrain() -> TrainingConfig:
    """Unconditional midtrain on the mixed sources."""
    return _midtrain(_mixed_sources())


def posttrain() -> TrainingConfig:
    """Conditional posttrain on the mixed sources."""
    return _posttrain(_mixed_sources())


def nuplan_pretrain() -> TrainingConfig:
    """Perception pretrain on nuPlan only."""
    return _pretrain(_nuplan_sources())


def nuplan_midtrain() -> TrainingConfig:
    """Unconditional midtrain on nuPlan only."""
    return _midtrain(_nuplan_sources())


def nuplan_posttrain() -> TrainingConfig:
    """Conditional posttrain on nuPlan only."""
    return _posttrain(_nuplan_sources())
