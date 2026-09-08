"""The Python presets as Hydra config options: <group>=<module:function> composes what the factory builds."""

from __future__ import annotations

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.object_type import ObjectType
from hydra.plugins.config_source import ConfigResult, ConfigSource
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf
from typing_extensions import override

from py123d_garage.common.config_help.preset_handling import preset_values


class PresetConfigSource(ConfigSource):
    """Serves any <group>/<module:function> by importing and calling the factory; the yaml sources are searched first."""

    @staticmethod
    @override
    def scheme() -> str:
        return "presets"

    @override
    def load_config(self, config_path: str) -> ConfigResult:
        target = config_path.rpartition("/")[2].removesuffix(".yaml")
        return ConfigResult(
            config=OmegaConf.create(preset_values(target)),
            path=self.full_path(),
            provider=self.provider,
            header={"package": "_global_"},
        )

    @override
    def available(self) -> bool:
        return True

    @override
    def is_group(self, config_path: str) -> bool:
        # The one group no yaml dir provides; every other group is a dir of a yaml source.
        return config_path.rstrip("/") == "preset"

    @override
    def is_config(self, config_path: str) -> bool:
        return ":" in config_path.rpartition("/")[2]

    @override
    def list(self, config_path: str, results_filter: ObjectType | None) -> list[str]:
        return []


class PresetSearchPathPlugin(SearchPathPlugin):
    """Puts the presets after the yaml sources on every search path, so no primary config has to list them."""

    @override
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="py123d_garage", path="presets://")
