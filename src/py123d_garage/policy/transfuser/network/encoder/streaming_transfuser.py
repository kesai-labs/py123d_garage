# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""TransFuser backbone whose image branch takes no fusion add-back, so a frame's image tokens can be cached and reused."""

from __future__ import annotations

from typing import TypeAlias

import jaxtyping as jt
import torch
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.policy.transfuser.network.encoder.transfuser import (
    EvenAdaptiveAvgPool2d,
    TransfuserBackbone,
    _resample_to,  # pyright: ignore[reportPrivateUsage]
    _return_layers,  # pyright: ignore[reportPrivateUsage]
    normalize_imagenet,
)

# One frame's pooled image tokens, one [B, C_stage, row_vert_anchors, img_horz_anchors] grid per fusion stage.
FrameTokens: TypeAlias = list[torch.Tensor]


class StreamingTransfuserBackbone(TransfuserBackbone):
    """
    Latent TransFuser backbone that encodes every canvas row on its own and adds the fusion back into the BEV only.

    A frame's image tokens depend on that frame alone, so a stream can encode each frame once and reuse its tokens.
    """

    def __init__(self, config: TransfuserConfig, image_encoder_pretrained: bool):
        """
        Initializes the backbone with the image tokens pooled per canvas row.

        Args:
            config: global config dataclass of TransFuser.
            image_encoder_pretrained: whether timm fetches pretrained image encoder weights.
        """
        super().__init__(config, image_encoder_pretrained)
        camera = config.camera_config
        self.num_frames = 1 + camera.num_past_camera_rows if camera.past_camera_offset_us else 1
        if not config.backbone_config.latent:
            raise ValueError(f"{type(self).__name__} needs latent mode: the LiDAR raster has no past frames.")
        if camera.img_vert_anchors % self.num_frames:
            raise ValueError(
                f"image_height {camera.image_height} does not split into {self.num_frames} rows of whole image tokens.",
            )
        if self.num_frames > 1 and (config.perspective_config.use_semantic or config.perspective_config.use_depth):
            raise ValueError("the perspective heads read the current row only, but their labels span the canvas.")
        self.row_height = camera.image_height // self.num_frames
        row_vert_anchors = camera.img_vert_anchors // self.num_frames
        self.avgpool_img = EvenAdaptiveAvgPool2d((row_vert_anchors, camera.img_horz_anchors))

    def split_canvas_into_frames(
        self,
        image: jt.Float[torch.Tensor, "batch 3 height width"],
    ) -> jt.Float[torch.Tensor, "batch frames 3 row_height width"]:
        """
        Splits the camera canvas into its frames.

        Args:
            image: the camera canvas, past rows oldest on top above the current row.

        Returns:
            the frames, oldest first.
        """
        batch_size, channels, _, width = image.shape
        return image.reshape(batch_size, channels, self.num_frames, self.row_height, width).transpose(1, 2)

    def encode_frame(
        self,
        image: jt.Float[torch.Tensor, "batch 3 row_height width"],
    ) -> tuple[FrameTokens, jt.Float[torch.Tensor, "batch image_channels feat_height feat_width"]]:
        """
        Runs the image encoder over one frame.

        Args:
            image: the frame, values in [0, 255].

        Returns:
            the frame's tokens and its last image feature map.
        """
        image_features = normalize_imagenet(image)
        image_layers = iter(self.image_encoder.items())
        image_return_layers = _return_layers(self.image_encoder)
        if len(image_return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, image_return_layers, image_features)
        frame_tokens: FrameTokens = []
        for _ in range(4):
            image_features = self.forward_layer_block(image_layers, image_return_layers, image_features)
            frame_tokens.append(self.avgpool_img(image_features))
        return frame_tokens, image_features

    def fuse_stage(
        self,
        stage_index: int,
        image_tokens: jt.Float[torch.Tensor, "batch img_channels row_vert img_horz"],
        past_image_tokens: list[jt.Float[torch.Tensor, "batch img_channels row_vert img_horz"]],
        lidar_features: jt.Float[torch.Tensor, "lidar_batch bev_channels bev_height bev_width"],
    ) -> tuple[
        jt.Float[torch.Tensor, "batch bev_channels bev_rows bev_cols"],
        jt.Float[torch.Tensor, "batch registers img_channels"] | None,
    ]:
        """
        Fuses one stage's BEV tokens with the frames' image tokens, stacked oldest on top like the canvas rows.

        Args:
            stage_index: the fusion stage.
            image_tokens: the current frame's pooled image tokens.
            past_image_tokens: the past frames' pooled image tokens nearest first; the oldest one repeats for missing
                frames.
            lidar_features: the BEV branch's feature map.

        Returns:
            the fused BEV tokens in BEV channels, and the stage's registers or None.
        """
        frames = [image_tokens, *past_image_tokens]
        frames += [frames[-1]] * (self.num_frames - len(frames))
        _, lidar_fused, registers = self.transformers[stage_index](
            torch.cat(frames[::-1], dim=2),
            self.lidar_channel_to_img[stage_index](self.avgpool_lidar(lidar_features)),
        )
        return self.img_channel_to_lidar[stage_index](lidar_fused), registers

    def fuse(
        self,
        frame_tokens: FrameTokens,
        past_frame_tokens: list[FrameTokens],
    ) -> tuple[
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
        jt.Float[torch.Tensor, "batch registers image_channels"] | None,
    ]:
        """
        Runs the BEV branch, each stage fusing the BEV tokens with the current and past frames' image tokens.

        Args:
            frame_tokens: the current frame's tokens.
            past_frame_tokens: the past frames' tokens nearest first, filled as fuse_stage does.

        Returns:
            the BEV features and the last stage's registers.
        """
        lidar_features = self._latent_lidar_grid(frame_tokens[0])
        lidar_layers = iter(self.lidar_encoder.items())
        lidar_return_layers = _return_layers(self.lidar_encoder)
        if len(lidar_return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, lidar_return_layers, lidar_features)

        fusion_registers: torch.Tensor | None = None
        mode = self._config.backbone_config.upsample_mode
        for i in range(4):
            lidar_features = self.forward_layer_block(lidar_layers, lidar_return_layers, lidar_features)
            lidar_fused, stage_registers = self.fuse_stage(
                i,
                frame_tokens[i],
                [past[i] for past in past_frame_tokens],
                lidar_features,
            )
            lidar_features = lidar_features + _resample_to(lidar_fused, lidar_features.shape[2:], mode)
            if stage_registers is not None:
                fusion_registers = stage_registers
        return lidar_features, fusion_registers

    def stream(
        self,
        image: jt.Float[torch.Tensor, "batch 3 row_height width"],
        past_frame_tokens: list[FrameTokens],
        lidar: jt.Float[torch.Tensor, "batch lidar_channels lidar_height lidar_width"] | None,
    ) -> tuple[
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
        jt.Float[torch.Tensor, "batch image_channels feat_height feat_width"],
        jt.Float[torch.Tensor, "batch registers image_channels"] | None,
        FrameTokens,
    ]:
        """
        Encodes one frame and fuses it with the tokens the past frames returned.

        Args:
            image: the current frame, values in [0, 255].
            past_frame_tokens: the past frames' tokens nearest first, one past_camera_offset_us apart, possibly fewer
                than num_past_camera_rows.
            lidar: ignored, latent mode reads no LiDAR.

        Returns:
            the BEV features, the current frame's image features, the last stage's registers, and the current frame's
            tokens for the later frames.
        """
        del lidar
        frame_tokens, image_features = self.encode_frame(image)
        bev_features, fusion_registers = self.fuse(frame_tokens, past_frame_tokens)
        return bev_features, image_features, fusion_registers, frame_tokens

    @override
    def forward(
        self,
        image: jt.Float[torch.Tensor, "batch 3 height width"],
        lidar: jt.Float[torch.Tensor, "batch lidar_channels lidar_height lidar_width"] | None,
    ) -> tuple[
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
        jt.Float[torch.Tensor, "batch image_channels feat_height feat_width"],
        jt.Float[torch.Tensor, "batch registers image_channels"] | None,
    ]:
        """
        Encodes all canvas rows in one batch, then fuses them.

        Args:
            image: the camera canvas, values in [0, 255].
            lidar: ignored, latent mode reads no LiDAR.

        Returns:
            the BEV features, the current frame's image features and the last stage's registers.
        """
        del lidar
        frames = self.split_canvas_into_frames(image)
        batch_size = frames.shape[0]
        frame_tokens, image_features = self.encode_frame(frames.flatten(0, 1))
        frame_tokens_by_frame = [
            stage_tokens.reshape(batch_size, self.num_frames, *stage_tokens.shape[1:]).unbind(1)
            for stage_tokens in frame_tokens
        ]
        frames_newest_first = [
            [stage_tokens[frame_index] for stage_tokens in frame_tokens_by_frame]
            for frame_index in reversed(range(self.num_frames))
        ]
        bev_features, fusion_registers = self.fuse(frames_newest_first[0], frames_newest_first[1:])
        current_image_features = image_features.reshape(batch_size, self.num_frames, *image_features.shape[1:])[:, -1]
        return bev_features, current_image_features, fusion_registers
