# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from py123d.api import SceneAPI
from typing_extensions import override

from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.callbacks import VisualizationCallback
from py123d_garage.training.dataset import TrainingSample
from py123d_garage.training.lightning_module import PolicyLightningModule
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401


class _RecordingPolicy(TransfuserPolicy):
    """Records the mode of every forward and skips the drawing, which needs a real scene."""

    def __init__(self, policy_config: PolicyConfig) -> None:
        super().__init__(policy_config)
        self.forward_modes: list[bool] = []

    @override
    def forward(self, features: TransfuserFeatures, navigation: NavigationConditioning) -> TransfuserPredictions:
        self.forward_modes.append(self.training)
        return super().forward(features, navigation)

    @override
    def visualize_batch(
        self,
        features: TransfuserFeatures,
        labels: TransfuserLabels | None,
        navigation: NavigationConditioning,
        predictions: TransfuserPredictions,
        scene_api: SceneAPI,
    ) -> dict[str, torch.Tensor]:
        assert predictions.trajectory is not None
        assert predictions.trajectory.shape[0] == 1, "the picture is drawn from seed 0's rows alone"
        return {"predictions": torch.zeros(2, 2, 3, dtype=torch.uint8)}


def test_visualization_forward_leaves_the_training_state_alone(config: TransfuserConfig) -> None:  # noqa: F811
    torch.manual_seed(0)
    policy = _RecordingPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    policy.train()
    running_statistics = {name: buffer.clone() for name, buffer in policy.named_buffers() if "running" in name}
    assert running_statistics, "the backbone's batch norms carry running statistics"
    batch = TrainingSample(
        scene_apis=[cast(SceneAPI, MagicMock(spec=SceneAPI)) for _ in range(2)],
        features=_features(config, batch_size=2),
        labels=_labels(config, batch_size=2),
        navigation=_navigation(config, batch_size=2),
        source_indices=torch.zeros(2, dtype=torch.long),
        vehicle_types=torch.full((2,), -1, dtype=torch.long),
    )
    trainer = MagicMock(spec=L.Trainer)
    trainer.global_step = 0
    trainer.is_global_zero = True
    trainer.loggers = [MagicMock(spec=WandbLogger)]
    trainer.precision_plugin.train_step_context = nullcontext
    module = MagicMock(spec=PolicyLightningModule)
    module.policy = policy
    module.seed_rows.return_value = slice(0, 1)

    VisualizationCallback(log_images_every_n_steps=1).on_train_batch_end(
        cast(L.Trainer, trainer),
        cast(L.LightningModule, module),
        None,
        batch,
        0,
    )

    assert policy.forward_modes == [False]
    assert policy.training
    for name, buffer in policy.named_buffers():
        if "running" in name:
            assert torch.equal(buffer, running_statistics[name]), name
    trainer.loggers[0].log_image.assert_called_once()
