# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The training's camera augmentation: under the mixed-precision autocast of the step, and once per loaded batch."""

from __future__ import annotations

import dataclasses
from typing import cast

import torch
from py123d.api import SceneAPI

from py123d_garage.api.abstract_policy_tensors import AbstractFeatures, AbstractLabels, NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.photometric_augmentation_config import (
    CameraAugmentationConfig,
    HoodSwapConfig,
    PhotometricAugmentationConfig,
)
from py123d_garage.config.schema.training.training_config import DataLoaderConfig, TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule
from tests.unit.policy.transfuser.test_transfuser_forward import config  # noqa: F401


def test_every_effect_and_the_hood_swap_run_under_bf16_autocast(config) -> None:  # noqa: F811
    every_effect = PhotometricAugmentationConfig(
        max_effects_per_category=99,
        max_effects_per_camera=99,
        clean_fraction=0.0,
        **{
            field.name: 1.0
            for field in dataclasses.fields(PhotometricAugmentationConfig)
            if field.name.endswith("_probability")
        },
    )
    augmentation = CameraAugmentationConfig(
        photometric_augmentation=True,
        photometric_augmentation_config=every_effect,
        hood_swap=True,
        hood_swap_config=HoodSwapConfig(probability=1.0),
    )
    module = PolicyLightningModule(
        [TransfuserPolicy(PolicyConfig(transfuser_config=config))],
        TrainingConfig(offline_data_sources=[], compile_mode=None, num_seeds=1, camera_augmentation=augmentation),
    )
    rgb = torch.rand(4, 3, 256, 1056) * 255
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module._augment_camera(rgb, 3, 1, 1, vehicle_types=torch.tensor([0, 0, 7, -1]))
    assert out.dtype == rgb.dtype
    assert out.shape == rgb.shape


@dataclasses.dataclass(frozen=True)
class _Rows(AbstractFeatures):
    tensor: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _RowLabels(AbstractLabels):
    tensor: torch.Tensor


class _Scene:
    pass


SceneAPI.register(_Scene)


def test_seeds_take_their_rows_of_one_augmented_loaded_batch(config, monkeypatch) -> None:  # noqa: F811
    num_seeds, seed_batch_size = 3, 2
    module = PolicyLightningModule(
        [TransfuserPolicy(PolicyConfig(transfuser_config=config)) for _ in range(num_seeds)],
        TrainingConfig(
            offline_data_sources=[],
            num_seeds=num_seeds,
            dataloader_config=DataLoaderConfig(batch_size=seed_batch_size),
        ),
    )
    calls: list[int] = []

    def augment_camera(rgb, num_tiles, num_rows, front_tile, *, vehicle_types):
        calls.append(len(rgb))
        return rgb + 1000

    monkeypatch.setattr(module, "_augment_camera", augment_camera)
    for policy in module.policies:
        monkeypatch.setattr(
            policy,
            "augment_features",
            lambda features, augment: _Rows(augment(features.tensor, 1, 1, 0)),
        )
    loaded = torch.arange(num_seeds * seed_batch_size, dtype=torch.float32)
    batch = TrainingSample(
        scene_apis=[cast(SceneAPI, _Scene()) for _ in loaded],
        features=_Rows(loaded),
        labels=_RowLabels(loaded),
        navigation=NavigationConditioning(
            target_points=loaded[:, None, None].expand(-1, 1, 2),
            target_point_headings=loaded[:, None],
        ),
        source_indices=torch.zeros(len(loaded), dtype=torch.long),
        vehicle_types=torch.zeros(len(loaded), dtype=torch.long),
    )

    inputs = module._seed_inputs(module._augmented(batch))

    assert calls == [len(loaded)]
    for seed_index, (features, labels, _) in enumerate(inputs):
        rows = module.seed_rows(seed_index)
        assert torch.equal(cast(_Rows, features).tensor, loaded[rows] + 1000)
        assert torch.equal(cast(_RowLabels, labels).tensor, loaded[rows])
