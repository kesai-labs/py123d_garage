# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The Python presets as Hydra config options: <group>=<module:function> composes what the factory builds."""

from __future__ import annotations

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.object_type import ObjectType
from hydra.plugins.config_source import ConfigResult, ConfigSource
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf
from typing_extensions import override

from py123d_garage.common.config_help.preset_handling import preset_values

# Hydra asks whether the left side of an override names a group, so a key like seed must not answer yes.
PRESET_GROUPS = frozenset({"preset", "training", "evaluation"})


class PresetConfigSource(ConfigSource):
    """Serves any <group>/<module:function> by importing and calling the factory."""

    @staticmethod
    @override
    def scheme() -> str:
        """The URL scheme of the preset search path."""
        return "presets"

    @override
    def load_config(self, config_path: str) -> ConfigResult:
        """Builds the config of the preset factory named by the last path element."""
        target = config_path.rpartition("/")[2].removesuffix(".yaml")
        return ConfigResult(
            config=OmegaConf.create(preset_values(target)),
            path=self.full_path(),
            provider=self.provider,
            header={"package": "_global_"},
        )

    @override
    def available(self) -> bool:
        """The preset source is always available."""
        return True

    @override
    def is_group(self, config_path: str) -> bool:
        """Whether the path names one of the preset groups."""
        return config_path.rstrip("/") in PRESET_GROUPS

    @override
    def is_config(self, config_path: str) -> bool:
        """Whether the last path element is a module:function preset."""
        return ":" in config_path.rpartition("/")[2]

    @override
    def list(self, config_path: str, results_filter: ObjectType | None) -> list[str]:
        """Lists nothing, since any importable factory is a valid preset."""
        return []


class PresetSearchPathPlugin(SearchPathPlugin):
    """Puts the presets on every search path, so no primary config has to list them."""

    @override
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Appends the preset source to the search path."""
        search_path.append(provider="py123d_garage", path="presets://")
