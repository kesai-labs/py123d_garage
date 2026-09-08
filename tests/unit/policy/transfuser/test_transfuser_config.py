"""TransfuserConfig invariants: derived geometry, signature fields, loss gating."""

from __future__ import annotations

import pytest
import yaml

from py123d_garage.api.abstract_policy import AbstractPolicy
from py123d_garage.common.config_help import build_from_string, load_config
from py123d_garage.config.presets.python.policy.transfuser import tf_carla
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    BevSemanticConfig,
    BoxDetectionConfig,
    CameraConfig,
    LidarConfig,
    PlanningConfig,
    TransfuserConfig,
)
from py123d_garage.config.schema.training.training_config import TrainingConfig
from py123d_garage.py123d_help.misc import FRONT_CAMERAS


def test_default_bev_geometry_is_forward_biased() -> None:
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
    )
    assert (
        config.lidar_config.bev_width_pixel,
        config.lidar_config.bev_height_pixel,
    ) == (384, 320)
    assert (
        config.lidar_config.lidar_bev_grid_cols,
        config.lidar_config.lidar_bev_grid_rows,
    ) == (12, 10)
    assert (
        config.camera_config.img_horz_anchors,
        config.camera_config.img_vert_anchors,
    ) == (32, 8)


def _config(**overrides: object) -> TransfuserConfig:
    """A config whose cameras are configured, as every cached store's is."""
    return TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(input_cameras=FRONT_CAMERAS),
        **overrides,  # pyright: ignore[reportArgumentType]
    )


def test_signature_fields_exist_and_stringify() -> None:
    """
    A renamed config field must not silently drop out of the signature.

    Computing the signature getattrs every declared name, so a stale name raises here.
    """
    signature = build_from_string(
        PolicyConfig(transfuser_config=_config()),
        AbstractPolicy,
    ).cache_signature("nuplan")
    assert signature
    assert all(isinstance(value, str) for fields in signature.values() for value in fields.values())


def test_signature_tracks_geometry_changes() -> None:
    changed = _config(lidar_config=LidarConfig(bev_pixels_per_meter=2.0))
    assert build_from_string(
        PolicyConfig(transfuser_config=changed),
        AbstractPolicy,
    ).cache_signature("nuplan") != build_from_string(
        PolicyConfig(transfuser_config=_config()),
        AbstractPolicy,
    ).cache_signature("nuplan")


def test_signature_narrows_to_the_dataset_cameras() -> None:
    """A store holds one dataset, so only that dataset's cameras reach its signature."""
    policy = build_from_string(PolicyConfig(transfuser_config=_config()), AbstractPolicy)
    assert policy.cache_signature("nuplan")["camera_feature"]["input_cameras"] == str(
        ["PCAM_L0", "PCAM_F0", "PCAM_R0"],
    )
    assert policy.cache_signature("kesai")["camera_feature"]["input_cameras"] == str(
        ["FTCAM_L0", "FTCAM_F0", "FTCAM_R0"],
    )


def _loss_weights(config: TransfuserConfig) -> dict[str, float]:
    return build_from_string(
        PolicyConfig(transfuser_config=config),
        AbstractPolicy,
    ).loss_weights(0)


def test_loss_weights_zero_for_disabled_heads() -> None:
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        bev_semantic_config=BevSemanticConfig(use_bev_semantic=False),
        box_detection_config=BoxDetectionConfig(detect_boxes=False),
        planning_config=PlanningConfig(use_planning_decoder=True),
        lidar_config=LidarConfig(),
    )
    weights = _loss_weights(config)
    assert weights["loss_bev_semantic"] == 0.0
    assert weights["loss_center_net_heatmap"] == 0.0
    assert weights["loss_trajectory"] > 0.0
    # Velocity needs sweep history; anchor-only rasters cannot learn it.
    assert weights["loss_center_net_velocity"] == 0.0
    assert not config.box_detection_config.predict_box_velocity


def test_box_velocity_is_an_explicit_switch() -> None:
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
        lidar_config=LidarConfig(
            lidar_interval_us=100_000,
            lidar_horizon_us=200_000,
        ),
    )
    assert _loss_weights(config)["loss_center_net_velocity"] > 0.0
    # The head needs motion cues; enabling it without sweep history is rejected.
    with pytest.raises(ValueError, match="motion cues"):
        TransfuserConfig(
            trajectory_horizon_us=4_000_000,
            trajectory_interval_us=500_000,
            required_target_point_distances_m=[45.0],
            box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
            lidar_config=LidarConfig(),
        )


def test_sweeps_require_a_declared_interval() -> None:
    with pytest.raises(ValueError, match="needs an interval"):
        TransfuserConfig(
            trajectory_horizon_us=4_000_000,
            trajectory_interval_us=500_000,
            required_target_point_distances_m=[45.0],
            lidar_config=LidarConfig(lidar_horizon_us=200_000),
        )
    with pytest.raises(ValueError, match="not a multiple"):
        TransfuserConfig(
            trajectory_horizon_us=4_000_000,
            trajectory_interval_us=500_000,
            required_target_point_distances_m=[45.0],
            lidar_config=LidarConfig(
                lidar_interval_us=100_000,
                lidar_horizon_us=250_000,
            ),
        )


def test_sweep_window_sets_the_required_history() -> None:
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        lidar_config=LidarConfig(
            lidar_interval_us=100_000,
            lidar_horizon_us=400_000,
        ),
    )
    assert config.required_history_duration_us == 400_000
    assert config.required_past_lidar_interval_us == 100_000


_CARLA_TRAINING_PRESET = "py123d_garage.config.presets.python.training.transfuser:tf_carla_train"


@pytest.fixture(autouse=True)
def _data_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY123D_GARAGE_DATA_ROOT", "/data")


def _load_training(overrides: list[str] | None = None) -> TrainingConfig:
    return load_config(
        TrainingConfig,
        "train",
        args=[f"training={_CARLA_TRAINING_PRESET}", *(overrides or [])],
    )


def test_loaded_policy_config_is_the_dataclass() -> None:
    """The loaded policy_config.transfuser_config node round-trips to the exact preset dataclass."""
    assert _load_training().policy_config.transfuser_config == tf_carla()


@pytest.mark.parametrize(
    "override",
    [
        "policy_config.transfuser_config.planning_config.use_planning_decoder=false",
        "policy_config.transfuser_config.bev_semantic_config.use_bev_semantic=false",
        "policy_config.transfuser_config.box_detection_config.detect_boxes=false",
        "policy_config.transfuser_config.backbone_config.freeze_backbone=true",
        "policy_config.transfuser_config.planning_conditioning_config.use_velocity=false",
        "policy_config.transfuser_config.backbone_config.latent=true",
        "policy_config.transfuser_config.lidar_config.bev_pixels_per_meter=2.0",
        "policy_config.transfuser_config.lidar_config.bev_min_x_m=-32",
        "policy_config.transfuser_config.lidar_config.bev_max_x_m=32",
        "policy_config.transfuser_config.lidar_config.bev_min_y_m=-32",
        "policy_config.transfuser_config.lidar_config.bev_max_y_m=32",
    ],
)
def test_e2e_overrides_load(override: str) -> None:
    """Every policy override the e2e workflow passes must resolve at load time."""
    config = _load_training(overrides=[override])
    key, _, value = override.partition("=")
    node: object = config
    for part in key.split("."):
        node = getattr(node, part)
    assert node == yaml.safe_load(value)
