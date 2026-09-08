from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING

from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.config.schema.policy.vavam_config import VavamConfig


@dataclass
class PolicyConfig:
    """
    The policy to build; each policy reads its own block, constructed by
    whatever selects the target.
    """

    # -- Config objects --

    # TransFuser's block, read when target names the TransfuserPolicy.
    transfuser_config: TransfuserConfig | None = None

    # VaVAM's block, read when target names the VavamPolicy.
    vavam_config: VavamConfig | None = None

    # -- Atomic settings --

    # module:Class path of the AbstractPolicy implementation.
    target: str = "py123d_garage.policy.transfuser.transfuser_policy:TransfuserPolicy"


@dataclass
class EvaluationPolicyConfig(PolicyConfig):
    """The policy under evaluation; training builds a plain PolicyConfig, which carries neither file."""

    # Evaluation checkpoint file; the policy loads its weights from here.
    evaluation_checkpoint_file: str = MISSING

    # YAML file path of the sensor rig; the benchmark can build its served sensor rig from this file.
    # Trained policies save those files next to the checkpoint; each file belongs to a data source.
    # Those YAML files can be edited to evaluate the policy with a different sensor rig than it was trained with.
    evaluation_sensor_rig_file: str | None = None
