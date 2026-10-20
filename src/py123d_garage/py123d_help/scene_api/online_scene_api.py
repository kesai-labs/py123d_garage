# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""A SceneAPI over timestamped modality bundles that arrive at runtime."""

from __future__ import annotations

import threading
from bisect import insort
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api import MapAPI, SceneAPI
from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes import (
    BaseModalityMetadata,
    LogMetadata,
    MapMetadata,
    ModalityType,
    SceneMetadata,
    Timestamp,
)
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import (
    BaseModality,
    get_modality_key,
)
from typing_extensions import override

from py123d_garage.datatypes.numerics import NonNegativeFloat, NonNegativeInt, PositiveInt
from py123d_garage.py123d_help.misc import provided_route_source_info

# The cameras of one frame end their exposure at slightly different times, so even a no-history scene keeps this much.
_MIN_SENSOR_RETENTION_US: Final[int] = 250_000

# Ego poses are tiny and arrive in bursts, so they are kept far beyond the sensor window.
_EGO_RETENTION_US: Final[int] = 60_000_000

_EMPTY_ROUTE = RouteMetadata(
    resolution_m=1.0,
    total_arc_m=0.0,
    polyline_x=[0.0],
    polyline_y=[0.0],
    polyline_z=[0.0],
    cache_source_info=provided_route_source_info("empty placeholder until a route is submitted"),
    source="provided",
)


@dataclass(order=True)
class _Bundle:
    """One synchronized observation bundle, ordered by time."""

    # Observation time on the simulator's clock, the only field bundles are ordered by; a backfilled history may
    # precede the clock's start.
    timestamp_us: int
    # The observed modalities by modality key.
    modalities: dict[str, BaseModality] = field(compare=False)
    # The ego's arc-length position on the route at this time.
    route_progress_m: NonNegativeFloat = field(compare=False)


class OnlineSceneAPI(SceneAPI):
    """
    A live scene over observation bundles arriving at runtime: iteration -i maps to the bundle
    nearest anchor - i * iteration_interval_us within half an interval, None on a miss.
    """

    def __init__(
        self,
        log_metadata: LogMetadata,
        modality_metadatas: dict[str, BaseModalityMetadata],
        iteration_interval_us: PositiveInt,
        num_history_iterations: NonNegativeInt,
    ) -> None:
        """
        Creates an empty scene.

        Args:
            log_metadata: the metadata the scene reports for its log.
            modality_metadatas: the metadata of every modality served, by modality key.
            iteration_interval_us: spacing of the iteration grid.
            num_history_iterations: how many past iterations the scene serves.
        """
        self._log_metadata = log_metadata
        self._modality_metadatas = modality_metadatas
        self._iteration_interval_us = iteration_interval_us
        self._num_history_iterations = num_history_iterations
        self._route_metadata = _EMPTY_ROUTE
        self._anchor_us: int | None = None
        self._bundles: list[_Bundle] = []
        self._lock = threading.Lock()

    def append(
        self,
        timestamp_us: int,
        modalities: dict[str, BaseModality],
        route_progress_m: NonNegativeFloat = 0.0,
    ) -> None:
        """
        Adds one observation bundle; a bundle already at the timestamp absorbs the modalities and keeps its progress.

        Args:
            timestamp_us: the observation time on the simulator's clock; a backfilled history may precede its start.
            modalities: the observed modalities by modality key.
            route_progress_m: the ego's arc-length position on the route.
        """
        with self._lock:
            for bundle in reversed(self._bundles):
                if bundle.timestamp_us == timestamp_us:
                    bundle.modalities.update(modalities)
                    return
            insort(self._bundles, _Bundle(timestamp_us, dict(modalities), route_progress_m))
            newest_us = self._bundles[-1].timestamp_us
            horizon_us = newest_us - _EGO_RETENTION_US
            while len(self._bundles) > 1 and self._bundles[0].timestamp_us < horizon_us:
                self._bundles.pop(0)
            sensor_horizon_us = newest_us - max(
                self._num_history_iterations * self._iteration_interval_us + self._iteration_interval_us // 2,
                _MIN_SENSOR_RETENTION_US,
            )
            ego_key = get_modality_key(ModalityType.EGO_STATE_SE3)
            for bundle in self._bundles:
                if bundle.timestamp_us >= sensor_horizon_us:
                    break
                if len(bundle.modalities) > 1 or ego_key not in bundle.modalities:
                    bundle.modalities = {key: modality for key, modality in bundle.modalities.items() if key == ego_key}

    def set_route(self, route_metadata: RouteMetadata | None) -> None:
        """
        Replaces the route the scene serves.

        Args:
            route_metadata: the new route; None serves an empty placeholder of no length.
        """
        with self._lock:
            self._route_metadata = _EMPTY_ROUTE if route_metadata is None else route_metadata

    def snapshot(self, anchor_us: int | None = None) -> OnlineSceneAPI:
        """
        Freezes a view for one planning cycle, which later appends cannot change.

        Args:
            anchor_us: the time of iteration 0; None anchors at the newest bundle.

        Returns:
            the frozen view.
        """
        with self._lock:
            assert self._bundles, "an empty scene has no anchor to snapshot at."
            view = OnlineSceneAPI(
                log_metadata=self._log_metadata,
                modality_metadatas=self._modality_metadatas,
                iteration_interval_us=self._iteration_interval_us,
                num_history_iterations=self._num_history_iterations,
            )
            view._route_metadata = self._route_metadata
            view._anchor_us = self._bundles[-1].timestamp_us if anchor_us is None else anchor_us
            view._bundles = list(self._bundles)
            return view

    @property
    def iteration_interval_us(self) -> PositiveInt:
        """Spacing of the iteration grid the scene serves."""
        return self._iteration_interval_us

    @property
    def num_history_iterations(self) -> NonNegativeInt:
        """How many past iterations the scene keeps."""
        return self._num_history_iterations

    @property
    def newest_timestamp_us(self) -> int | None:
        """The newest bundle's timestamp, or None while the scene is empty."""
        with self._lock:
            return self._bundles[-1].timestamp_us if self._bundles else None

    def _resolved_anchor_us(self) -> int:
        """
        Resolves the time of iteration 0.

        Returns:
            the snapshot's anchor, or the newest bundle's time on a live scene.
        """
        return self._anchor_us if self._anchor_us is not None else self._bundles[-1].timestamp_us

    def _bundle_near(self, target_us: int) -> _Bundle | None:
        """
        Finds the bundle nearest a target time within half an interval.

        Args:
            target_us: the target time.

        Returns:
            the nearest bundle, or None when none is close enough.
        """
        if not self._bundles:
            return None
        nearest = min(self._bundles, key=lambda bundle: abs(bundle.timestamp_us - target_us))
        if abs(nearest.timestamp_us - target_us) > self._iteration_interval_us // 2:
            return None
        return nearest

    def _iteration_target_us(self, iteration: int) -> int:
        """
        Maps an iteration to the time it serves.

        Args:
            iteration: a history iteration, from -num_history_iterations to 0.

        Returns:
            the iteration's time.
        """
        assert -self._num_history_iterations <= iteration <= 0, (
            f"iteration {iteration} is outside the {self._num_history_iterations} history iterations this scene keeps."
        )
        return self._resolved_anchor_us() + iteration * self._iteration_interval_us

    @override
    def get_scene_metadata(self) -> SceneMetadata:
        """Inherited, see superclass."""
        return SceneMetadata(
            dataset=self._log_metadata.dataset,
            split="",
            initial_uuid="",
            initial_idx=0,
            num_future_iterations=0,
            num_history_iterations=self._num_history_iterations,
            future_duration_s=0.0,
            history_duration_s=self._num_history_iterations * self._iteration_interval_us / 1e6,
            iteration_duration_s=self._iteration_interval_us / 1e6,
        )

    @override
    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return self._log_metadata

    @override
    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        """Inherited, see superclass."""
        return Timestamp.from_us(self._iteration_target_us(iteration))

    @override
    def get_all_iteration_timestamps(self, include_history: bool = False) -> list[Timestamp]:
        """Inherited, see superclass."""
        first_iteration = -self._num_history_iterations if include_history else 0
        return [Timestamp.from_us(self._iteration_target_us(iteration)) for iteration in range(first_iteration, 1)]

    @override
    def get_scene_timestamp_boundaries(
        self,
        include_history: bool = False,
    ) -> tuple[Timestamp, Timestamp]:
        """Inherited, see superclass."""
        timestamps = self.get_all_iteration_timestamps(include_history)
        return timestamps[0], timestamps[-1]

    @override
    def get_route(
        self,
    ) -> tuple[
        RouteMetadata,
        jt.Float64[npt.NDArray[np.float64], " num_vertices"],
        jt.Float64[npt.NDArray[np.float64], "num_vertices 3"],
    ]:
        """Inherited, see superclass."""
        return (
            self._route_metadata,
            self._route_metadata.polyline_arc_m,
            self._route_metadata.polyline_xyz,
        )

    @override
    def get_route_progress_at_iteration(self, iteration: int) -> NonNegativeFloat:
        """Inherited, see superclass."""
        bundle = self._bundle_near(self._iteration_target_us(iteration))
        assert bundle is not None, f"no bundle serves iteration {iteration}."
        return bundle.route_progress_m

    @override
    def get_map_metadata(self) -> MapMetadata | None:
        """Inherited, see superclass."""
        raise NotImplementedError("this live scene serves no 123D map; a sensor-track policy must not read it")

    @override
    def get_map_api(self) -> MapAPI | None:
        """Inherited, see superclass."""
        raise NotImplementedError("this live scene serves no 123D map; a sensor-track policy must not read it")

    @override
    def get_all_modality_metadatas(self) -> dict[str, BaseModalityMetadata]:
        """Inherited, see superclass."""
        return self._modality_metadatas

    @override
    def get_modality_metadata(
        self,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
    ) -> BaseModalityMetadata | None:
        """Inherited, see superclass."""
        return self._modality_metadatas.get(get_modality_key(modality_type, modality_id))

    @override
    def get_all_modality_timestamps(
        self,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
        include_history: bool = False,
    ) -> list[Timestamp]:
        """Inherited, see superclass."""
        key = get_modality_key(modality_type, modality_id)
        if include_history:
            bundles: list[_Bundle] = self._bundles
        else:
            anchor_bundle = self._bundle_near(self._resolved_anchor_us())
            bundles = [anchor_bundle] if anchor_bundle is not None else []
        return [Timestamp.from_us(bundle.timestamp_us) for bundle in bundles if key in bundle.modalities]

    @override
    def get_modality_at_iteration(
        self,
        iteration: int,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
        **kwargs: object,
    ) -> BaseModality | None:
        """Inherited, see superclass."""
        del kwargs
        bundle = self._bundle_near(self._iteration_target_us(iteration))
        if bundle is None:
            return None
        return bundle.modalities.get(get_modality_key(modality_type, modality_id))

    @override
    def get_modality_at_timestamp(
        self,
        timestamp: Timestamp | int,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        **kwargs: object,
    ) -> BaseModality | None:
        """Inherited, see superclass."""
        del kwargs
        time_us = timestamp.time_us if isinstance(timestamp, Timestamp) else timestamp
        key = get_modality_key(modality_type, modality_id)
        candidates = [
            (bundle.timestamp_us, bundle.modalities[key]) for bundle in self._bundles if key in bundle.modalities
        ]
        if criteria == "exact":
            candidates = [candidate for candidate in candidates if candidate[0] == time_us]
        elif criteria == "forward":
            candidates = [candidate for candidate in candidates if candidate[0] >= time_us]
        elif criteria == "backward":
            candidates = [candidate for candidate in candidates if candidate[0] <= time_us]
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: abs(candidate[0] - time_us))[1]

    @override
    def get_modality_between_timestamps(
        self,
        start_timestamp: Timestamp | int,
        end_timestamp: Timestamp | int,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
        inclusive: Literal["left", "right", "both", "neither"] = "left",
        **kwargs: object,
    ) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        del kwargs
        start_us = start_timestamp.time_us if isinstance(start_timestamp, Timestamp) else start_timestamp
        end_us = end_timestamp.time_us if isinstance(end_timestamp, Timestamp) else end_timestamp
        include_start = inclusive in ("left", "both")
        include_end = inclusive in ("right", "both")
        key = get_modality_key(modality_type, modality_id)
        for bundle in self._bundles:
            time_us = bundle.timestamp_us
            after_start = time_us >= start_us if include_start else time_us > start_us
            before_end = time_us <= end_us if include_end else time_us < end_us
            modality = bundle.modalities.get(key)
            if after_start and before_end and modality is not None:
                yield modality
