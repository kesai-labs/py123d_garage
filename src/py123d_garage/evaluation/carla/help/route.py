"""The leaderboard's route as the 123D route polyline a policy is conditioned on."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from py123d.datatypes.metadata.route_metadata import RouteMetadata

from py123d_garage.datatypes.numerics import NonNegativeFloat
from py123d_garage.py123d_help.misc import provided_route_source_info

ROUTE_RESOLUTION_M = 0.1

_PROGRESS_SEARCH_M = 20.0


class Route:
    """
    The driven route, and how far along it the ego has come.

    Rebuilds the ``route_position`` modality the training logs carry, so
    evaluation conditions the policy through the same code path training does.
    """

    def __init__(
        self,
        world_route: list[tuple[Any, Any]],
        extension_m: NonNegativeFloat,
    ) -> None:
        """
        Resamples the leaderboard's dense route into a uniform 123D polyline.

        Args:
            world_route: the leaderboard's (transform, road option) route, in the
                CARLA world frame.
            extension_m: how far to run the polyline past the destination along
                its final heading, so target points stay servable to the end.
        """
        vertices = np.array(
            [[transform.location.x, -transform.location.y, transform.location.z] for transform, _ in world_route],
            dtype=np.float64,
        )
        steps = np.linalg.norm(np.diff(vertices[:, :2], axis=0), axis=1)
        vertices = vertices[np.concatenate([[True], steps > 0.0])]

        if extension_m > 0.0:
            heading = vertices[-1] - vertices[-2]
            vertices = np.concatenate(
                [vertices, [vertices[-1] + heading / np.linalg.norm(heading[:2]) * extension_m]],
            )

        arc_m = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(vertices[:, :2], axis=0), axis=1))],
        )
        num_vertices = int(arc_m[-1] / ROUTE_RESOLUTION_M) + 1
        sample_arc_m = np.arange(num_vertices, dtype=np.float64) * ROUTE_RESOLUTION_M
        polyline = np.stack(
            [np.interp(sample_arc_m, arc_m, vertices[:, axis]) for axis in range(3)],
            axis=1,
        )

        self._metadata = RouteMetadata(
            resolution_m=ROUTE_RESOLUTION_M,
            total_arc_m=float(sample_arc_m[-1]),
            polyline_x=polyline[:, 0].tolist(),
            polyline_y=polyline[:, 1].tolist(),
            polyline_z=polyline[:, 2].tolist(),
            cache_source_info=provided_route_source_info("carla leaderboard route"),
            source="provided",
        )
        self._polyline_xy = polyline[:, :2]
        self._progress_index = 0

    @property
    def metadata(self) -> RouteMetadata:
        """The polyline as the route modality of a 123D log carries it."""
        return self._metadata

    def advance(self, position_xy: npt.NDArray[np.float64]) -> NonNegativeFloat:
        """
        Advances the ego's position along the route and returns its arc length.

        Progress only ever moves forward, so a route that crosses itself and an
        ego that is pushed backwards both keep their place.

        Args:
            position_xy: the ego rear-axle position in the ISO world frame.

        Returns:
            the ego's arc-length position on the route in meters.
        """
        window = self._polyline_xy[
            self._progress_index : self._progress_index + int(_PROGRESS_SEARCH_M / ROUTE_RESOLUTION_M)
        ]
        self._progress_index += int(np.argmin(np.linalg.norm(window - position_xy, axis=1)))
        return self._progress_index * ROUTE_RESOLUTION_M
