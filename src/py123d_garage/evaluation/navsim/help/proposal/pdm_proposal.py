# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from dataclasses import dataclass

from py123d.geometry import PolylineSE2
from shapely.geometry import LineString

from py123d_garage.evaluation.navsim.help.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)


@dataclass
class PDMProposal:
    """One proposal: a lateral path combined with a longitudinal IDM policy."""

    # Index of the proposal among all proposals.
    proposal_idx: int
    # Index of the path among the lateral proposals.
    lateral_idx: int
    # Index of the IDM policy among the longitudinal proposals.
    longitudinal_idx: int
    # Path the proposal follows.
    path: PolylineSE2

    @property
    def linestring(self) -> LineString:
        """
        The proposal's path as a linestring.

        Returns:
            The path linestring.
        """
        return self.path.linestring

    @property
    def length(self):
        """
        The length of the proposal's path.

        Returns:
            The path length in meters.
        """
        return self.path.length


class PDMProposalManager:
    """Holds every combination of lateral paths and longitudinal IDM policies."""

    def __init__(
        self,
        lateral_proposals: list[PolylineSE2],
        longitudinal_policies: BatchIDMPolicy,
    ):
        """
        Builds one proposal per path and IDM policy.

        Args:
            lateral_proposals: Paths to follow.
            longitudinal_policies: IDM policies to follow the paths with.
        """
        self._num_lateral_proposals: int = len(lateral_proposals)
        self._num_longitudinal_proposals: int = longitudinal_policies.num_policies
        self._longitudinal_policies: BatchIDMPolicy = longitudinal_policies

        self._proposals: list[PDMProposal] = []
        proposal_idx = 0

        for lateral_idx in range(self._num_lateral_proposals):
            for longitudinal_idx in range(self._num_longitudinal_proposals):
                self._proposals.append(
                    PDMProposal(
                        proposal_idx=proposal_idx,
                        lateral_idx=lateral_idx,
                        longitudinal_idx=longitudinal_idx,
                        path=lateral_proposals[lateral_idx],
                    ),
                )
                proposal_idx += 1

    def __len__(self) -> int:
        """
        Counts the proposals.

        Returns:
            The number of paths times the number of policies.
        """
        return len(self._proposals)

    def __getitem__(self, proposal_idx: int) -> PDMProposal:
        """
        Looks up a proposal by index.

        Args:
            proposal_idx: Index of the proposal.

        Returns:
            The proposal.
        """
        return self._proposals[proposal_idx]

    def update(self, speed_limit_mps: float | None) -> None:
        """
        Updates the target velocities of the IDM policies from the speed limit.

        Args:
            speed_limit_mps: Speed limit of the current lane in m/s, or None if unknown.
        """
        self._longitudinal_policies.update(speed_limit_mps)

    @property
    def num_lateral_proposals(self) -> int:
        """
        The number of lateral proposals.

        Returns:
            The number of paths.
        """
        return self._num_lateral_proposals

    @property
    def num_longitudinal_proposals(self) -> int:
        """
        The number of longitudinal proposals.

        Returns:
            The number of IDM policies.
        """
        return self._num_longitudinal_proposals

    @property
    def max_target_velocity(self) -> float:
        """
        The highest target velocity across the IDM policies.

        Returns:
            The target velocity in m/s.
        """
        return float(self._longitudinal_policies.max_target_velocity)

    @property
    def longitudinal_policies(self) -> BatchIDMPolicy:
        """
        The IDM policies shared by all proposals.

        Returns:
            The batch of IDM policies.
        """
        return self._longitudinal_policies
