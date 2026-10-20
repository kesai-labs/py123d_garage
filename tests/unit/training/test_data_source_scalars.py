# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import torch

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import DataLoaderConfig, TrainingConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from py123d_garage.training.lightning_module import PolicyLightningModule
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _labels, _navigation, config  # noqa: F401


def test_each_seed_scores_its_own_rows_per_source(config) -> None:  # noqa: F811
    """Two seeds on a loaded batch of four rows: seed 0 trains on source 0's rows, seed 1 on source 1's."""
    policies = []
    for seed in range(2):
        torch.manual_seed(seed)
        policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
        policy.initialize()
        policies.append(policy.eval())
    module = PolicyLightningModule(
        policies,
        TrainingConfig(
            offline_data_sources=[],
            compile_mode=None,
            num_seeds=2,
            dataloader_config=DataLoaderConfig(batch_size=2),
        ),
    )
    seed_labels = [_labels(config, batch_size=2) for _ in policies]
    with torch.no_grad():
        seed_predictions = [policy(_features(config, 2), _navigation(config, 2)) for policy in policies]
        scalars = module._data_source_scalars(torch.tensor([0, 0, 1, 1]), seed_labels, seed_predictions)
        expected = {
            source: policy.compute_loss(labels, predictions)["loss_trajectory"]
            for source, policy, labels, predictions in zip((0, 1), policies, seed_labels, seed_predictions, strict=True)
        }

    assert {"data_sources_loss_trajectory/0", "data_sources_loss_trajectory/1"} <= scalars.keys()
    assert "data_sources_waypoints_ade/1" in scalars
    for source, loss in expected.items():
        torch.testing.assert_close(scalars[f"data_sources_loss_trajectory/{source}"], loss.float())
