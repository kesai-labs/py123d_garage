# Mirrors py123d.geometry.occupancy_map; the bare `np.ndarray` parameters of
# query/query_nearest make the inline types partially unknown for pyright.
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry

class OccupancyMap2D:
    def __init__(
        self,
        geometries: Sequence[BaseGeometry],
        ids: Sequence[str] | Sequence[int] | None = None,
        node_capacity: int = 10,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls,
        geometry_dict: dict[str, BaseGeometry] | dict[int, BaseGeometry],
        node_capacity: int = 10,
    ) -> OccupancyMap2D: ...
    def __getitem__(self, id: str | int) -> BaseGeometry: ...
    def __len__(self) -> int: ...
    @property
    def ids(self) -> list[str] | list[int]: ...
    @property
    def geometries(self) -> Sequence[BaseGeometry]: ...
    @property
    def id_to_idx(self) -> dict[int | str, int]: ...
    def intersects(self, geometry: BaseGeometry) -> list[str] | list[int]: ...
    def query(
        self,
        geometry: BaseGeometry | npt.NDArray[Any],
        predicate: Literal[
            "intersects",
            "within",
            "dwithin",
            "contains",
            "overlaps",
            "crosses",
            "touches",
            "covers",
            "covered_by",
        ]
        | None = None,
        distance: float | None = None,
    ) -> npt.NDArray[np.int64]: ...
    def query_nearest(
        self,
        geometry: BaseGeometry | npt.NDArray[Any],
        max_distance: float | None = None,
        return_distance: bool = False,
        exclusive: bool = False,
        all_matches: bool = True,
    ) -> (
        npt.NDArray[np.int64]
        | tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
    ): ...
    def contains_points_2d(
        self,
        points_2d: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_]: ...
