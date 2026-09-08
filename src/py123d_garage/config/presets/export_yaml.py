"""Writes the yaml siblings of the Python training presets: python -m py123d_garage.config.presets.export_yaml."""

from __future__ import annotations

import dataclasses
import enum
import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import yaml
from omegaconf import MISSING
from typing_extensions import override

from py123d_garage.config.presets.python.policy import latent_transfuser as latent_transfuser_policy_presets
from py123d_garage.config.presets.python.policy import transfuser as transfuser_policy_presets
from py123d_garage.config.presets.python.training import latent_transfuser as latent_transfuser_training_presets
from py123d_garage.config.presets.python.training import transfuser as transfuser_training_presets
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.training.training_config import TrainingConfig

YAML_DIR = Path(__file__).parent / "yaml"
POLICY_PRESET_MODULES = (transfuser_policy_presets, latent_transfuser_policy_presets)
TRAINING_PRESET_MODULES = (transfuser_training_presets, latent_transfuser_training_presets)


class _PlainDumper(yaml.SafeDumper):
    @override
    def ignore_aliases(self, data: Any) -> bool:
        return True


def preset_factories(module: ModuleType) -> dict[str, Callable[[], Any]]:
    """The public factories a preset module defines, by name."""
    return {
        name: factory
        for name, factory in inspect.getmembers(module, inspect.isfunction)
        if factory.__module__ == module.__name__ and not name.startswith("_")
    }


def trimmed(instance: Any) -> dict[str, Any]:
    """The fields differing from the dataclass defaults, nested; enums as their names."""
    values: dict[str, Any] = {}
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = MISSING
        if dataclasses.is_dataclass(value):
            block = trimmed(value)
            if block or default is None:
                values[field.name] = block
        elif value != default or default is MISSING:
            values[field.name] = _plain(value)
    return values


def _plain(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in cast("dict[Any, Any]", value).items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in cast("list[Any]", value)]
    return value


def _write(path: Path, header_lines: list[str], body: dict[str, Any]) -> None:
    text = yaml.dump(body, Dumper=_PlainDumper, sort_keys=False, width=120) if body else ""
    if path.exists() and path.read_text().split("\n", len(header_lines))[-1] != text:
        raise ValueError(f"{path} is written twice with different content; name the sources apart")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in header_lines) + text)


def main() -> None:
    for subdir in ("policy", "offline_data_sources", "training"):
        for stale in (YAML_DIR / subdir).rglob("*.yaml"):
            stale.unlink()
    policies: dict[str, PolicyConfig] = {}
    for module in POLICY_PRESET_MODULES:
        for name, factory in preset_factories(module).items():
            policies[name] = PolicyConfig(transfuser_config=factory())
            _write(
                YAML_DIR / "policy" / f"{name}.yaml",
                [f"# Mirrors {module.__name__.removeprefix('py123d_garage.')}:{name}."],
                trimmed(policies[name]),
            )
    for module in TRAINING_PRESET_MODULES:
        for name, factory in preset_factories(module).items():
            training: TrainingConfig = factory()
            (policy_name,) = [policy for policy, config in policies.items() if config == training.policy_config]
            defaults: list[Any] = [{"/policy@policy_config": policy_name}]
            for source_name, source in training.offline_data_sources.items():
                _write(YAML_DIR / "offline_data_sources" / policy_name / f"{source_name}.yaml", [], trimmed(source))
                defaults.append(
                    {f"/offline_data_sources/{policy_name}@offline_data_sources.{source_name}": source_name},
                )
            body = trimmed(training)
            del body["policy_config"], body["offline_data_sources"]
            _write(
                YAML_DIR / "training" / f"{name}.yaml",
                [
                    "# @package _global_",
                    f"# Mirrors {module.__name__.removeprefix('py123d_garage.')}:{name}; every omitted key keeps its schema default.",
                ],
                {"defaults": [*defaults, "_self_"], **body},
            )


if __name__ == "__main__":
    main()
