from __future__ import annotations

from py123d_garage.config.presets.python.offline_data_sources.carla import carla
from py123d_garage.config.presets.python.offline_data_sources.kesai import kesai_train
from py123d_garage.config.presets.python.offline_data_sources.nuplan import nuplan_test, nuplan_train, nuplan_val
from py123d_garage.config.presets.python.offline_data_sources.physical_ai_av import (
    physical_ai_av_test,
    physical_ai_av_train,
    physical_ai_av_val,
)
from py123d_garage.config.presets.python.policy.latent_transfuser import (
    ltf_carla,
    ltf_kesai,
    ltf_nuplan,
)
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import LightningTrainerConfig, OptimizerConfig, TrainingConfig


def ltf_nuplan_trainval() -> TrainingConfig:
    """Camera-only latent TransFuser on the camera-bearing nuplan train+val logs."""
    architecture = ltf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="latent_transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="latent_transfuser_nuplan_val",
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
    )


def ltf_nuplan_trainvaltest() -> TrainingConfig:
    """Camera-only latent TransFuser on the nuplan train+val+test logs; nuplan_test carries the navtest samples, contaminating navsim scores."""
    architecture = ltf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="latent_transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="latent_transfuser_nuplan_val",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_test": nuplan_test(
                architecture,
                cache_store="latent_transfuser_nuplan_test",
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
    )


def ltf_carla_train() -> TrainingConfig:
    """Camera-only latent TransFuser on the CARLA logs, with the 2 s horizon."""
    architecture = ltf_carla()
    return TrainingConfig(
        offline_data_sources={
            "carla": carla(
                architecture,
                cache_store="latent_transfuser_carla",
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
        # No gradient clipping here, so the fused kernel is allowed.
        optimizer_config=OptimizerConfig(fused=True),
    )


def ltf_physical_ai_av_trainval() -> TrainingConfig:
    """Camera-only latent TransFuser on the physical-ai-av train+val logs."""
    architecture = ltf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_val",
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


def ltf_physical_ai_av_trainvaltest() -> TrainingConfig:
    """Camera-only latent TransFuser on the physical-ai-av train+val+test logs."""
    architecture = ltf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_val",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_test": physical_ai_av_test(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_test",
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


def ltf_kesai_train() -> TrainingConfig:
    """Camera-only latent TransFuser on the kesai train logs."""
    architecture = ltf_kesai()
    return TrainingConfig(
        offline_data_sources={
            "kesai_train": kesai_train(
                architecture,
                cache_store="latent_transfuser_kesai_train",
                timestamp_threshold_s=0.25,
            ),
        },
        policy_config=PolicyConfig(transfuser_config=architecture),
    )


def ltf_mixed_nuplan_physical_ai_av_trainval() -> TrainingConfig:
    """Camera-only latent TransFuser on the nuplan + physical-ai-av train+val logs."""
    architecture = ltf_nuplan()
    return TrainingConfig(
        offline_data_sources={
            "nuplan_train": nuplan_train(
                architecture,
                cache_store="latent_transfuser_nuplan_train",
                timestamp_threshold_s=1.0,
            ),
            "nuplan_val": nuplan_val(
                architecture,
                cache_store="latent_transfuser_nuplan_val",
                timestamp_threshold_s=1.0,
            ),
            "physical_ai_av_train": physical_ai_av_train(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_train",
                timestamp_threshold_s=2.0,
            ),
            "physical_ai_av_val": physical_ai_av_val(
                architecture,
                cache_store="latent_transfuser_physical_ai_av_val",
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
