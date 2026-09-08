"""py123d's SceneFilter as an OmegaConf-compatible config node, and its resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
from py123d.api.scene.scene_filter import AnchorFilterContext, SceneFilter

from py123d_garage.common.config_help import import_string


@dataclass
class GarageSceneFilter(SceneFilter):
    """py123d's SceneFilter, but types edited so it can be directly usable as an OmegaConf config node."""

    custom_filter_fns: Any | None = None
    custom_anchor_filter_fns: Any | None = None

    def to_py123d_scene_filter(self) -> SceneFilter:
        """
        A copy with configured module:function paths replaced by their functions.

        The config carries paths because functions survive neither OmegaConf
        nor the saved config.yaml.

        Returns:
            a copy with resolved functions, ready for get_scenes.

        Raises:
            TypeError: if a path names a non-callable.
        """
        targets: list[str] = self.custom_anchor_filter_fns or []
        filter_fns: list[Callable[[AnchorFilterContext], npt.NDArray[np.bool_]]] = []
        for target in targets:
            filter_fn: Callable[[AnchorFilterContext], npt.NDArray[np.bool_]] = import_string(target)
            if not callable(filter_fn):
                raise TypeError(f"custom anchor filter fn {target!r} is not callable")
            filter_fns.append(filter_fn)
        return replace(self, custom_anchor_filter_fns=filter_fns or None)
