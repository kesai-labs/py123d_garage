# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The release recipe on the smoke datasets, as training=smoke_presets:<stage>_<dataset> presets for the ci jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.config.presets.training import latent_transfuser
from py123d_garage.config.schema.training.training_config import TrainingConfig

_DATASETS: dict[str, tuple[str, ...]] = {
    "nuplan": ("nuplan",),
    "physical_ai_av": ("physical-ai-av",),
    "mixture": ("nuplan", "physical-ai-av"),
}


def _on_datasets(stage: Callable[[], TrainingConfig], datasets: tuple[str, ...]) -> TrainingConfig:
    """
    The stage's config with only the sources of the given datasets, their stores under a smoke name.

    Args:
        stage: the release recipe's stage factory.
        datasets: the 123D dataset directory names to keep.

    Returns:
        the training config of the stage on those datasets.
    """
    config = stage()
    sources: list[OfflineTrainingDataSourceConfig] = []
    for source in config.offline_data_sources:
        if any(f"/{dataset}/123D" in source.data_root for dataset in datasets):
            store_root, _, store_name = source.cache_root.rpartition("/")
            sources.append(replace(source, cache_root=f"{store_root}/smoke_{store_name}"))
    assert sources, f"the recipe has no source on {datasets}"
    return replace(config, offline_data_sources=sources)


def _preset(stage: Callable[[], TrainingConfig], datasets: tuple[str, ...]) -> Callable[[], TrainingConfig]:
    """
    A no-argument preset factory, as the hydra preset plugin calls them.

    Args:
        stage: the release recipe's stage factory.
        datasets: the 123D dataset directory names to keep.

    Returns:
        the factory.
    """
    return lambda: _on_datasets(stage, datasets)


pretrain_nuplan = _preset(latent_transfuser.pretrain, _DATASETS["nuplan"])
midtrain_nuplan = _preset(latent_transfuser.midtrain, _DATASETS["nuplan"])
posttrain_nuplan = _preset(latent_transfuser.posttrain, _DATASETS["nuplan"])
pretrain_physical_ai_av = _preset(latent_transfuser.pretrain, _DATASETS["physical_ai_av"])
midtrain_physical_ai_av = _preset(latent_transfuser.midtrain, _DATASETS["physical_ai_av"])
posttrain_physical_ai_av = _preset(latent_transfuser.posttrain, _DATASETS["physical_ai_av"])
pretrain_mixture = _preset(latent_transfuser.pretrain, _DATASETS["mixture"])
midtrain_mixture = _preset(latent_transfuser.midtrain, _DATASETS["mixture"])
posttrain_mixture = _preset(latent_transfuser.posttrain, _DATASETS["mixture"])
