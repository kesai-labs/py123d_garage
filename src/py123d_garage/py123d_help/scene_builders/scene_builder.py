"""Scene-builder construction and log discovery over a 123D dir."""

from __future__ import annotations

import os
from pathlib import Path

from py123d.api.scene.arrow.arrow_scene_builder import (
    ArrowSceneBuilder,
    LazyArrowSceneBuilder,
)


def build_scene_builder(data_root: str, lazy: bool = True) -> ArrowSceneBuilder:
    """
    Builds the py123d scene builder over one 123D dir.

    Args:
        data_root: the dir holding logs/ and maps/.
        lazy: defer sensor loading to the DataLoader workers (training and
            caching); evaluation loads eagerly.

    Returns:
        the scene builder.
    """
    builder_class = LazyArrowSceneBuilder if lazy else ArrowSceneBuilder
    return builder_class(
        logs_root=f"{data_root}/logs",
        maps_root=f"{data_root}/maps",
    )


def find_log_names(data_root: str, split_names: list[str] | None) -> list[str]:
    """
    Lists log names without reading any log: a log is any directory holding a
    sync.arrow, which is how py123d finds them too.
    """
    logs_root = Path(data_root) / "logs"
    search_roots = [logs_root] if split_names is None else [logs_root / split_name for split_name in split_names]
    log_names: list[str] = []
    for search_root in search_roots:
        # followlinks: symlinked log dirs must shard like py123d discovers them.
        for walk_dir, directory_names, file_names in os.walk(
            search_root,
            followlinks=True,
        ):
            if "sync.arrow" in file_names:
                log_names.append(Path(walk_dir).name)
                directory_names[:] = []  # logs are never nested
    if not log_names:
        raise RuntimeError(
            f"no logs under {logs_root} for splits {split_names}; check the data root",
        )
    return sorted(log_names)
