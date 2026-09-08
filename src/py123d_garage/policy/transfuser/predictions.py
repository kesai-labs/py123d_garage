"""TransFuser's typed prediction bundle."""

from __future__ import annotations

from dataclasses import dataclass

import jaxtyping as jt
import torch
from py123d.geometry import Point2DIndex, PoseSE2Index
from typing_extensions import override

from py123d_garage.api.abstract_policy_tensors import AbstractPredictions
from py123d_garage.datatypes.trajectory import derive_yaw_from_positions
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
)


@dataclass(frozen=True)
class TransfuserPredictions(AbstractPredictions):
    """TransFuser's output tensors; fields of disabled heads are None."""

    trajectory: jt.Float[torch.Tensor, "*batch poses columns"] | None = None
    semantic: (
        jt.Float[
            torch.Tensor,
            "*batch semantic_classes semantic_height semantic_width",
        ]
        | None
    ) = None
    depth: jt.Float[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    bev_semantic: jt.Float[torch.Tensor, "*batch bev_classes bev_height bev_width"] | None = None
    boxes: CenterNetBoundingBoxPrediction | None = None

    @property
    @override
    def ego_trajectory_xy(self) -> jt.Float[torch.Tensor, "*batch poses 2"]:
        """Inherited, see superclass."""
        assert self.trajectory is not None, "The planning decoder is disabled, so there is no ego trajectory!"
        return self.trajectory[..., : len(Point2DIndex)]

    @property
    @override
    def ego_trajectory_se2(self) -> jt.Float[torch.Tensor, "*batch poses 3"]:
        """Inherited, see superclass."""
        assert self.trajectory is not None, "The planning decoder is disabled, so there is no ego trajectory!"
        if self.trajectory.shape[-1] == len(PoseSE2Index):
            return self.trajectory
        positions_xy = self.ego_trajectory_xy
        yaw = derive_yaw_from_positions(positions_xy)
        return torch.cat([positions_xy, yaw.unsqueeze(-1)], dim=-1)
