from __future__ import annotations

from py123d_garage.config.presets.python.offline_data_sources.carla import carla
from py123d_garage.config.presets.python.offline_data_sources.kesai import kesai_train
from py123d_garage.config.presets.python.offline_data_sources.nuplan import nuplan_test, nuplan_train, nuplan_val
from py123d_garage.config.presets.python.offline_data_sources.physical_ai_av import (
    physical_ai_av_test,
    physical_ai_av_train,
    physical_ai_av_val,
)
from py123d_garage.config.presets.python.policy.transfuser import (
    tf_carla,
    tf_kesai,
    tf_nuplan,
)
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import LightningTrainerConfig, TrainingConfig


def tf_nuplan_trainval() -> TrainingConfig:
    """TransFuser on the sensor-bearing nuplan train+val logs."""
    architecture = tf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="transfuser_nuplan_val",
                timestamp_threshold_s=1.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(
            max_epochs=63,
            # The focal heatmap loss spikes on hard frames
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        ),
        # The fully annealed model of every cosine warm-restart cycle.
        epochs_to_keep_checkpoints_for=[0, 2, 6, 14, 30, 62],
    )


def tf_nuplan_trainvaltest() -> TrainingConfig:
    """TransFuser on the nuplan train+val+test logs; nuplan_test carries the navtest samples, contaminating navsim scores."""
    architecture = tf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="transfuser_nuplan_val",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_test": nuplan_test(
                architecture,
                cache_store="transfuser_nuplan_test",
                timestamp_threshold_s=1.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(
            max_epochs=63,
            # The focal heatmap loss spikes on hard frames
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        ),
        # The fully annealed model of every cosine warm-restart cycle.
        epochs_to_keep_checkpoints_for=[0, 2, 6, 14, 30, 62],
    )


def tf_carla_train() -> TrainingConfig:
    """TransFuser on the CARLA logs, with the 2 s horizon."""
    architecture = tf_carla()
    return TrainingConfig(
        offline_data_sources={
            "carla": carla(
                architecture,
                cache_store="transfuser_carla",
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
    )


def tf_physical_ai_av_trainval() -> TrainingConfig:
    """TransFuser on the physical-ai-av train+val logs."""
    architecture = tf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="transfuser_physical_ai_av_val",
                timestamp_threshold_s=2.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(
            # Ends at a cosine warm-restart cycle end, like the nuplan 63.
            max_epochs=15,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        ),
    )


def tf_physical_ai_av_trainvaltest() -> TrainingConfig:
    """TransFuser on the physical-ai-av train+val+test logs."""
    architecture = tf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="transfuser_physical_ai_av_val",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_test": physical_ai_av_test(
                architecture,
                cache_store="transfuser_physical_ai_av_test",
                timestamp_threshold_s=2.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(
            # Ends at a cosine warm-restart cycle end, like the nuplan 63.
            max_epochs=15,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        ),
    )


def tf_kesai_train() -> TrainingConfig:
    """TransFuser on the kesai train logs; planning only — the logs carry no perception labels."""
    architecture = tf_kesai()
    return TrainingConfig(
        offline_data_sources={
            "kesai_train": kesai_train(
                architecture,
                cache_store="transfuser_kesai_train",
                timestamp_threshold_s=1.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
    )


def tf_mixed_nuplan_physical_ai_av_trainval() -> TrainingConfig:
    """TransFuser on the nuplan + physical-ai-av train+val logs."""
    architecture = tf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="transfuser_nuplan_val",
                timestamp_threshold_s=1.0,
            ),
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="transfuser_physical_ai_av_val",
                timestamp_threshold_s=2.0,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        lightning_trainer_config=LightningTrainerConfig(
            # Ends at a cosine warm-restart cycle end, like the physical-ai-av 15.
            max_epochs=15,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        ),
    )
