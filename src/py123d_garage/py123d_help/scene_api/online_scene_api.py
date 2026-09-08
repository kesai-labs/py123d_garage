"""A SceneAPI over timestamped modality bundles that arrive at runtime."""

from __future__ import annotations

import threading
from bisect import insort
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

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

    timestamp_us: NonNegativeInt
    modalities: dict[str, BaseModality] = field(compare=False)
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
        """Creates an empty scene."""
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
        timestamp_us: NonNegativeInt,
        modalities: dict[str, BaseModality],
        route_progress_m: NonNegativeFloat = 0.0,
    ) -> None:
        """
        Adds one observation bundle; a bundle already at the timestamp absorbs the modalities
        and keeps its original route progress.
        """
        with self._lock:
            for bundle in reversed(self._bundles):
                if bundle.timestamp_us == timestamp_us:
                    bundle.modalities.update(modalities)
                    return
            insort(self._bundles, _Bundle(timestamp_us, dict(modalities), route_progress_m))
            retention_us = self._num_history_iterations * self._iteration_interval_us + self._iteration_interval_us // 2
            horizon_us = self._bundles[-1].timestamp_us - retention_us
            while len(self._bundles) > 1 and self._bundles[0].timestamp_us < horizon_us:
                self._bundles.pop(0)

    def set_route(self, route_metadata: RouteMetadata) -> None:
        """Replaces the route the scene serves."""
        with self._lock:
            self._route_metadata = route_metadata

    def snapshot(self, anchor_us: NonNegativeInt | None = None) -> OnlineSceneAPI:
        """
        A frozen view for one planning cycle, anchored at anchor_us (None = the newest bundle);
        later appends cannot change it.
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
    def newest_timestamp_us(self) -> NonNegativeInt | None:
        """The newest bundle's timestamp, or None while the scene is empty."""
        with self._lock:
            return self._bundles[-1].timestamp_us if self._bundles else None

    def _resolved_anchor_us(self) -> int:
        return self._anchor_us if self._anchor_us is not None else self._bundles[-1].timestamp_us

    def _bundle_near(self, target_us: NonNegativeInt) -> _Bundle | None:
        """The bundle nearest a target within half an interval, or None."""
        if not self._bundles:
            return None
        nearest = min(self._bundles, key=lambda bundle: abs(bundle.timestamp_us - target_us))
        if abs(nearest.timestamp_us - target_us) > self._iteration_interval_us // 2:
            return None
        return nearest

    def _iteration_target_us(self, iteration: int) -> int:
        assert -self._num_history_iterations <= iteration <= 0, (
            f"iteration {iteration} is outside the {self._num_history_iterations} history iterations this scene keeps."
        )
        return self._resolved_anchor_us() + iteration * self._iteration_interval_us

    @override
    def get_scene_metadata(self) -> SceneMetadata:
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
        return self._log_metadata

    @override
    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        return Timestamp.from_us(self._iteration_target_us(iteration))

    @override
    def get_all_iteration_timestamps(self, include_history: bool = False) -> list[Timestamp]:
        first_iteration = -self._num_history_iterations if include_history else 0
        return [Timestamp.from_us(self._iteration_target_us(iteration)) for iteration in range(first_iteration, 1)]

    @override
    def get_scene_timestamp_boundaries(
        self,
        include_history: bool = False,
    ) -> tuple[Timestamp, Timestamp]:
        timestamps = self.get_all_iteration_timestamps(include_history)
        return timestamps[0], timestamps[-1]

    @override
    def get_route(
        self,
    ) -> tuple[RouteMetadata, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return (
            self._route_metadata,
            self._route_metadata.polyline_arc_m,
            self._route_metadata.polyline_xyz,
        )

    @override
    def get_route_progress_at_iteration(self, iteration: int) -> NonNegativeFloat:
        bundle = self._bundle_near(self._iteration_target_us(iteration))
        assert bundle is not None, f"no bundle serves iteration {iteration}."
        return bundle.route_progress_m

    @override
    def get_map_metadata(self) -> MapMetadata | None:
        raise NotImplementedError("this live scene serves no 123D map; a sensor-track policy must not read it")

    @override
    def get_map_api(self) -> MapAPI | None:
        raise NotImplementedError("this live scene serves no 123D map; a sensor-track policy must not read it")

    @override
    def get_all_modality_metadatas(self) -> dict[str, BaseModalityMetadata]:
        return self._modality_metadatas

    @override
    def get_modality_metadata(
        self,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
    ) -> BaseModalityMetadata | None:
        return self._modality_metadatas.get(get_modality_key(modality_type, modality_id))

    @override
    def get_all_modality_timestamps(
        self,
        modality_type: str | ModalityType,
        modality_id: str | SerialIntEnum | None = None,
        include_history: bool = False,
    ) -> list[Timestamp]:
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
