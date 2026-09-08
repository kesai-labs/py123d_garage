"""Debug grid of the raw CenterNet and BEV-semantic label and prediction rasters."""

from __future__ import annotations

from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch

from py123d_garage.policy.transfuser.features import TransfuserFeatures
from py123d_garage.policy.transfuser.labels import TransfuserLabels
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.visualization._drawing_primitives import (
    to_display,
)

_PANEL_WIDTH_PIXEL = 160
_PANEL_HEIGHT_PIXEL = 192
_TITLE_BAR_PIXEL = 16
_BORDER_PIXEL = 4
_GRID_COLUMNS = 4


def render_bev_maps(
    features: TransfuserFeatures,
    labels: TransfuserLabels,
    predictions: TransfuserPredictions,
    bev_semantic_classes: jt.Int[torch.Tensor, "batch bev_height bev_width"] | None = None,
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Renders the raw BEV-map rasters of the first sample as a titled grid.

    Label and prediction panels of the same head sit next to each other; heads
    without tensors in the bundles are skipped.

    Args:
        features: batched feature bundle, as consumed by the policy's forward
        labels: batched label bundle
        predictions: batched prediction bundle, as returned by forward
        bev_semantic_classes: predicted BEV class raster, as the label would draw it,
            or None to skip the panel

    Returns:
        RGB image of the panel grid
    """

    def raster(
        tensor: torch.Tensor | None,
    ) -> npt.NDArray[np.float32] | None:
        if tensor is None:
            return None
        return cast(
            npt.NDArray[np.float32],
            tensor[0].detach().cpu().float().numpy(),  # pyright: ignore[reportUnknownMemberType]
        )

    def channel_norm(
        array: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32] | None:
        return None if array is None else np.linalg.norm(array, axis=0)

    def first_channel(
        array: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32] | None:
        return None if array is None else array[0]

    def argmax_channel(
        array: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32] | None:
        return None if array is None else array.argmax(0).astype(np.float32)

    def max_channel(
        array: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32] | None:
        return None if array is None else array.max(0)

    center_net = labels.center_net
    boxes = predictions.boxes
    panel_sources: list[tuple[str, npt.NDArray[np.float32] | None]] = [
        (
            "heatmap gt",
            max_channel(
                raster(center_net.center_net_heatmap if center_net else None),
            ),
        ),
        (
            "heatmap pred",
            max_channel(raster(boxes.center_heatmap_pred if boxes else None)),
        ),
        (
            "wh gt",
            channel_norm(
                raster(center_net.center_net_wh if center_net else None),
            ),
        ),
        ("wh pred", channel_norm(raster(boxes.wh_pred if boxes else None))),
        (
            "offset gt",
            channel_norm(
                raster(center_net.center_net_offset if center_net else None),
            ),
        ),
        (
            "offset pred",
            channel_norm(raster(boxes.offset_pred if boxes else None)),
        ),
        (
            "yaw class gt",
            raster(center_net.center_net_yaw_class if center_net else None),
        ),
        (
            "yaw class pred",
            argmax_channel(raster(boxes.yaw_class_pred if boxes else None)),
        ),
        (
            "yaw res gt",
            first_channel(
                raster(center_net.center_net_yaw_res if center_net else None),
            ),
        ),
        (
            "yaw res pred",
            first_channel(raster(boxes.yaw_res_pred if boxes else None)),
        ),
        (
            "velocity gt",
            first_channel(
                raster(center_net.center_net_velocity if center_net else None),
            ),
        ),
        (
            "velocity pred",
            first_channel(raster(boxes.velocity_pred if boxes else None)),
        ),
        (
            "pixel weight",
            first_channel(
                raster(
                    center_net.center_net_pixel_weight if center_net else None,
                ),
            ),
        ),
        (
            "bev semantic gt",
            raster(labels.bev_semantic.bev_semantic if labels.bev_semantic else None),
        ),
        (
            "bev semantic pred",
            raster(bev_semantic_classes),
        ),
        ("lidar input", first_channel(raster(features.lidar_feature))),
    ]
    panels = [_render_panel(title, array) for title, array in panel_sources if array is not None]
    assert panels, "No BEV-map rasters found in labels or predictions."
    return _tile_panels(panels)


def _render_panel(
    title: str,
    raster: jt.Float32[npt.NDArray[np.float32], "height width"],
) -> jt.UInt8[npt.NDArray[np.uint8], "panel_height panel_width 3"]:
    """One titled, colormapped panel in display orientation."""
    display = np.ascontiguousarray(to_display(raster)).astype(np.float32)
    low, high = float(display.min()), float(display.max())
    normalized = (display - low) / (high - low) if high > low else display * 0.0
    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = cv2.resize(
        colored,
        (_PANEL_WIDTH_PIXEL, _PANEL_HEIGHT_PIXEL),
        interpolation=cv2.INTER_NEAREST,
    )

    panel = np.zeros(
        (_TITLE_BAR_PIXEL + _PANEL_HEIGHT_PIXEL, _PANEL_WIDTH_PIXEL, 3),
        dtype=np.uint8,
    )
    panel[_TITLE_BAR_PIXEL:] = colored
    cv2.putText(
        panel,
        title,
        (2, _TITLE_BAR_PIXEL - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return panel


def _tile_panels(
    panels: list[jt.UInt8[npt.NDArray[np.uint8], "panel_height panel_width 3"]],
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """Arranges equally sized panels into a bordered grid."""
    panel_height, panel_width = panels[0].shape[:2]
    rows = -(-len(panels) // _GRID_COLUMNS)
    height = rows * panel_height + (rows + 1) * _BORDER_PIXEL
    width = _GRID_COLUMNS * panel_width + (_GRID_COLUMNS + 1) * _BORDER_PIXEL
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        top = _BORDER_PIXEL + (index // _GRID_COLUMNS) * (panel_height + _BORDER_PIXEL)
        left = _BORDER_PIXEL + (index % _GRID_COLUMNS) * (panel_width + _BORDER_PIXEL)
        canvas[top : top + panel_height, left : left + panel_width] = panel
    return canvas
