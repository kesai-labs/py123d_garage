# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import yaml
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    scene_uuids_to_binary,  # pyright: ignore[reportUnknownVariableType]
)
from py123d.api.scene.scene_filter import AnchorFilterContext


@cache
def _packaged_ids(asset_name: str) -> frozenset[str]:
    """
    Reads a packaged id list.

    Args:
        asset_name: the yaml file under scene_filters/assets.

    Returns:
        the ids.
    """
    text = (files("py123d_garage.py123d_help.scene_filters") / "assets" / asset_name).read_text()
    return frozenset(cast("list[str]", yaml.safe_load(text)))


@cache
def _packaged_uuids_binary(asset_name: str) -> pa.FixedSizeBinaryArray:
    """
    Reads a packaged uuid list as the binary array the sync table stores uuids in.

    Args:
        asset_name: the yaml file under scene_filters/assets.

    Returns:
        the sorted uuids as binary.
    """
    return cast(
        "pa.FixedSizeBinaryArray",
        scene_uuids_to_binary(sorted(_packaged_ids(asset_name))),
    )


def _anchors_in(context: AnchorFilterContext, asset_name: str) -> jt.Bool[npt.NDArray[np.bool_], " num_anchors"]:
    """
    Checks which candidate anchors have their frame uuid in a packaged list.

    Args:
        context: the log's candidate anchors.
        asset_name: the yaml file under scene_filters/assets.

    Returns:
        True where the anchor frame's uuid is in the list.
    """
    uuid_column = context.sync_table["sync.uuid"].combine_chunks()
    if isinstance(uuid_column.type, pa.BaseExtensionType):
        uuid_column = uuid_column.cast(uuid_column.type.storage_type)
    anchor_uuids = uuid_column.take(pa.array(cast("list[int]", context.anchors.tolist())))
    mask: pa.BooleanArray = pc.is_in(  # pyright: ignore[reportUnknownMemberType]
        anchor_uuids,
        value_set=_packaged_uuids_binary(asset_name),
    )
    return cast(
        "npt.NDArray[np.bool_]",
        mask.to_numpy(zero_copy_only=False).astype(bool),  # pyright: ignore[reportUnknownMemberType]
    )


def num_navtest_scenes() -> int:
    """
    Counts the navtest samples.

    Returns:
        the number of frame uuids in the packaged navtest list.
    """
    return len(_packaged_ids("navtest_frame_ids.yaml"))


def is_navtest_scene(context: AnchorFilterContext) -> jt.Bool[npt.NDArray[np.bool_], " num_anchors"]:
    """
    Whether each candidate scene's anchor frame is a navtest sample.

    Args:
        context: the log's candidate anchors.

    Returns:
        True where the anchor frame's uuid is in the packaged navtest list.
    """
    return _anchors_in(context, "navtest_frame_ids.yaml")


def is_nurec_hard_scene(context: AnchorFilterContext) -> jt.Bool[npt.NDArray[np.bool_], " num_anchors"]:
    """
    Whether each candidate scene's anchor frame is in the nurec_val hard split.

    The split keeps 2 Hz nurec_val anchors where a constant-velocity rollout scores PDMS <= 0.5 and the human >= 0.8.

    Args:
        context: the log's candidate anchors.

    Returns:
        True where the anchor frame's uuid is in the packaged nurec_hard list.
    """
    return _anchors_in(context, "nurec_hard_frame_ids.yaml")


def is_not_nurec_clip(context: AnchorFilterContext) -> jt.Bool[npt.NDArray[np.bool_], " num_anchors"]:
    """
    Whether each candidate scene belongs to a log NuRec does not reconstruct.

    The AlpaSim nurec suite drives the real clips of these names, so training on them contaminates its scores.

    Args:
        context: the log's candidate anchors.

    Returns:
        all True when the log name is outside the packaged NuRec clip list, all False otherwise.
    """
    keep = context.log_dir.name not in _packaged_ids("nurec_log_ids.yaml")
    return np.full(len(context.anchors), keep, dtype=np.bool_)


def is_not_reversing_log(context: AnchorFilterContext) -> jt.Bool[npt.NDArray[np.bool_], " num_anchors"]:
    """
    Whether each candidate scene belongs to a Physical AI log that never drives backwards.

    The packaged list holds every log whose ego reverses at 0.3 m/s or faster anywhere, since the route spans the log.

    Args:
        context: the log's candidate anchors.

    Returns:
        all True when the log name is outside the packaged reversing list, all False otherwise.
    """
    keep = context.log_dir.name not in _packaged_ids("physical_ai_av_reversing_log_ids.yaml")
    return np.full(len(context.anchors), keep, dtype=np.bool_)
