# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""EnsemblePolicy averages the members' trajectories, the yaw circularly, and hands everything else to its first member."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from py123d.api import SceneAPI
from torch import Tensor, nn
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, CacheTensorSpec
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig, uniform_offsets_us
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.api.ensemble_policy import EnsemblePolicy
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from tests.unit.policy.transfuser.test_transfuser_forward import _features, _navigation, config  # noqa: F401


@dataclass(frozen=True)
class _Predictions(AbstractPredictions):
    trajectory: Tensor

    @property
    @override
    def ego_trajectory_xy(self) -> Tensor:
        return self.trajectory[..., :2]

    @property
    @override
    def ego_trajectory_se2(self) -> Tensor:
        return self.trajectory


class _ConstantPolicy(AbstractPolicy[AbstractFeatures, AbstractLabels, _Predictions]):
    def __init__(self, trajectory: Tensor) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.trajectory = nn.Parameter(trajectory)

    @property
    @override
    def policy_config(self) -> AbstractPolicyConfig:
        return TransfuserConfig(
            trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
            required_target_point_distances_m=[45.0],
        )

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        pass

    @override
    def forward(self, features: AbstractFeatures, navigation: NavigationConditioning) -> _Predictions:
        return _Predictions(trajectory=self.trajectory)

    @override
    def build_features(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> AbstractFeatures:
        raise NotImplementedError

    @override
    def build_labels(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> AbstractLabels:
        raise NotImplementedError

    @override
    def _permanent_cache_specs(self) -> dict[str, CacheTensorSpec]:
        return {}

    @override
    def compute_loss(self, labels: AbstractLabels, predictions: _Predictions) -> dict[str, Tensor]:
        return {}

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        return {}


def _trajectory(x: float, y: float, yaw_deg: float) -> Tensor:
    return torch.tensor([[[x, y, math.radians(yaw_deg)]]])


def _forward(ensemble: EnsemblePolicy) -> Tensor:
    features = AbstractFeatures()
    navigation = NavigationConditioning(target_points=torch.zeros(1, 1, 2), target_point_headings=torch.zeros(1, 1))
    return ensemble.forward(features, navigation).ego_trajectory_se2


def test_positions_are_averaged() -> None:
    ensemble = EnsemblePolicy(
        [_ConstantPolicy(_trajectory(2.0, 0.0, 0.0)), _ConstantPolicy(_trajectory(4.0, 1.0, 0.0))],
    )
    trajectory = _forward(ensemble)
    torch.testing.assert_close(trajectory[..., :2], torch.tensor([[[3.0, 0.5]]]))


def test_yaw_is_averaged_on_the_circle() -> None:
    ensemble = EnsemblePolicy(
        [_ConstantPolicy(_trajectory(0.0, 0.0, 170.0)), _ConstantPolicy(_trajectory(0.0, 0.0, -170.0))],
    )
    yaw = _forward(ensemble)[..., 2]
    assert math.isclose(abs(float(yaw)), math.pi, abs_tol=1e-6)


def test_first_member_bundle_is_kept() -> None:
    first = _ConstantPolicy(_trajectory(1.0, 0.0, 0.0))
    ensemble = EnsemblePolicy([first, _ConstantPolicy(_trajectory(3.0, 0.0, 0.0))])
    predictions = ensemble.forward(
        AbstractFeatures(),
        NavigationConditioning(target_points=torch.zeros(1, 1, 2), target_point_headings=torch.zeros(1, 1)),
    )
    torch.testing.assert_close(predictions.first_member.ego_trajectory_se2, _trajectory(1.0, 0.0, 0.0))
    assert predictions.to(torch.float64).trajectory.dtype == torch.float64


def test_two_transfusers_average_their_plans(config: TransfuserConfig) -> None:  # noqa: F811
    members: list[TransfuserPolicy] = []
    for seed in range(2):
        torch.manual_seed(seed)
        member = TransfuserPolicy(PolicyConfig(transfuser_config=config))
        member.initialize()
        members.append(member.eval())
    features, navigation = _features(config), _navigation(config)
    ensemble = EnsemblePolicy(members).eval()

    with torch.no_grad():
        plans = [member(features, navigation).ego_trajectory_se2 for member in members]
        predictions = ensemble(features, navigation)

    assert not torch.allclose(plans[0], plans[1])
    torch.testing.assert_close(predictions.ego_trajectory_xy, torch.stack(plans)[..., :2].mean(dim=0))
    torch.testing.assert_close(predictions.first_member.ego_trajectory_se2, plans[0])
