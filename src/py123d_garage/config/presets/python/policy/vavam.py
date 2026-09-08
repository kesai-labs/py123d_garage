from __future__ import annotations

from py123d_garage.config.schema.policy.vavam_config import VavamConfig


def vavam() -> VavamConfig:
    """The pretrained VaVAM video-action model, evaluated as released."""
    return VavamConfig()
