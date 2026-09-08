from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    scene_uuids_to_binary,  # pyright: ignore[reportUnknownVariableType]
)
from py123d.api.scene.scene_filter import AnchorFilterContext


@lru_cache(maxsize=1)
def _navtest_scene_uuids() -> tuple[str, ...]:
    text = (files("py123d_garage.py123d_help.scene_filters") / "assets" / "navtest.ids").read_text()
    return tuple(text.split())


@lru_cache(maxsize=1)
def _navtest_uuids_binary() -> pa.FixedSizeBinaryArray:
    return cast(
        "pa.FixedSizeBinaryArray",
        scene_uuids_to_binary(list(_navtest_scene_uuids())),
    )


def is_navtest_scene(context: AnchorFilterContext) -> npt.NDArray[np.bool_]:
    """
    Whether each candidate scene's anchor frame is a navtest sample.

    Args:
        context: the log's candidate anchors.

    Returns:
        True where the anchor frame's uuid is in the packaged navtest list.
    """
    uuid_column = context.sync_table["sync.uuid"].combine_chunks()
    if isinstance(uuid_column.type, pa.BaseExtensionType):
        uuid_column = uuid_column.cast(uuid_column.type.storage_type)
    anchor_uuids = uuid_column.take(pa.array(cast("list[int]", context.anchors.tolist())))
    mask: pa.BooleanArray = pc.is_in(  # pyright: ignore[reportUnknownMemberType]
        anchor_uuids,
        value_set=_navtest_uuids_binary(),
    )
    return cast(
        "npt.NDArray[np.bool_]",
        mask.to_numpy(zero_copy_only=False).astype(bool),  # pyright: ignore[reportUnknownMemberType]
    )
