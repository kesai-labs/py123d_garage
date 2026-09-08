from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat, PositiveInt
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


@dataclass
class AbstractOfflineDataSourceConfig:
    """
    Configuration for an offline data source: 123D root, scene selection, and stream rates.

    Can be used both for training and open-loop evaluation.
    """

    # Any 123D directory holding logs/ and maps/.
    data_root: str = MISSING

    # Scene filter for the data source; empty filter = all scenes.
    garage_scene_filter: GarageSceneFilter = field(
        default_factory=GarageSceneFilter,
    )

    # Feature cache store of this source; None = no training cache.
    cache_root: str | None = None

    # Time interval the source's logs record past ego states on.
    served_ego_state_interval_us: PositiveInt = MISSING

    # Time interval the source's logs record camera frames on.
    served_camera_interval_us: PositiveInt = MISSING

    # Time interval the source's logs record lidar on; None = no lidar recorded.
    served_lidar_interval_us: PositiveInt | None = MISSING

    # Nearest distance the source's route data can place a target point at; None = no limit.
    min_served_target_point_distance_m: NonNegativeFloat | None = None

    # Farthest distance the source's route data can place a target point at; None = no limit.
    max_served_target_point_distance_m: PositiveFloat | None = None


@dataclass
class OfflineTrainingDataSourceConfig(AbstractOfflineDataSourceConfig):
    """One dataset a training run reads, plus its cache store and mixture weight."""

    # Training requires a store, e.g. "$PY123D_GARAGE_DATA_ROOT/nuplan/py123d_garage_cache/transfuser_nuplan_train".
    cache_root: str = MISSING  # pyright: ignore[reportIncompatibleVariableOverride]

    # Per-sample sampling weight in [0, inf), relative across sources; 0
    # excludes the source, all 1.0 = plain concatenation.
    source_weight: float = 1.0


@dataclass
class OfflineEvaluationDataSourceConfig(AbstractOfflineDataSourceConfig):
    """One dataset a benchmark scores; the scene filter selects the scored frames."""
