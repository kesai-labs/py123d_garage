# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import jaxtyping as jt
import timm
import torch
import torch.nn.functional as F
from timm.models import FeatureListNet
from torch import nn
from typing_extensions import override

from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)


def normalize_imagenet(
    x: jt.Float[torch.Tensor, "batch 3 height width"],
) -> jt.Float[torch.Tensor, "batch 3 height width"]:
    """
    Normalize input images according to ImageNet standards.

    Args:
        x: input image batch with values in [0, 255].

    Returns:
        normalized image batch.
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


def _resample_to(
    features: jt.Float[torch.Tensor, "batch channels in_height in_width"],
    size: torch.Size,
    mode: str,
) -> jt.Float[torch.Tensor, "batch channels height width"]:
    """
    Resample a feature map to size, passing through when it already matches.

    Args:
        features: the map to resample, shape [B, C, H, W].
        size: target spatial size.
        mode: interpolation mode, e.g. "bilinear" or "nearest".

    Returns:
        the resampled map, or features unchanged.
    """
    if features.shape[2:] == size:
        return features
    # align_corners is an error with "nearest"
    if mode == "nearest":
        return cast(
            torch.Tensor,
            F.interpolate(features, size=tuple(size), mode=mode),  # pyright: ignore[reportUnknownMemberType]
        )
    return cast(
        torch.Tensor,
        F.interpolate(  # pyright: ignore[reportUnknownMemberType]
            features,
            size=tuple(size),
            mode=mode,
            align_corners=False,
        ),
    )


def _feature_info_int(encoder: FeatureListNet, index: int, key: str) -> int:
    """An integer property of the encoder's index-th feature level, e.g. "num_chs"."""
    return cast(int, encoder.feature_info.info[index][key])  # pyright: ignore[reportUnknownMemberType]


def _return_layers(encoder: FeatureListNet) -> dict[str, str]:
    """The encoder's feature-returning layer names."""
    return cast("dict[str, str]", encoder.return_layers)  # pyright: ignore[reportUnknownMemberType]


class TransfuserBackbone(nn.Module):
    """
    TransFuser backbone network for multi-modal sensor fusion.

    Implements the TransFuser architecture that fuses RGB image and LiDAR features
    using transformer-based attention mechanisms across multiple resolution levels.
    """

    def __init__(self, config: TransfuserConfig, image_encoder_pretrained: bool):
        """
        Initializes the TransFuser backbone with dual encoder branches and fusion modules.

        Args:
            config: global config dataclass of TransFuser.
            image_encoder_pretrained: whether timm fetches pretrained image encoder weights.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config

        # Image branch
        self.image_encoder = cast(
            FeatureListNet,
            timm.create_model(
                config.backbone_config.image_architecture,
                pretrained=image_encoder_pretrained,
                features_only=True,
            ),
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (
                config.camera_config.img_vert_anchors,
                config.camera_config.img_horz_anchors,
            ),
        )
        image_start_index = 0
        if len(_return_layers(self.image_encoder)) > 4:
            image_start_index += 1
        self.num_image_features: int = _feature_info_int(
            self.image_encoder,
            image_start_index + 3,
            "num_chs",
        )

        # LiDAR branch. In latent mode the raster is replaced by a fixed 2-channel
        # positional-encoding grid, so the branch always has two input channels.
        self.lidar_encoder = cast(
            FeatureListNet,
            timm.create_model(
                config.backbone_config.lidar_architecture,
                pretrained=False,
                in_chans=2 if config.backbone_config.latent else config.lidar_config.lidar_in_channels,
                features_only=True,
            ),
        )
        lidar_start_index = 0
        if len(_return_layers(self.lidar_encoder)) > 4:
            lidar_start_index += 1
        self.num_lidar_features: int = _feature_info_int(
            self.lidar_encoder,
            lidar_start_index + 3,
            "num_chs",
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    _feature_info_int(
                        self.lidar_encoder,
                        lidar_start_index + i,
                        "num_chs",
                    ),
                    _feature_info_int(
                        self.image_encoder,
                        image_start_index + i,
                        "num_chs",
                    ),
                    kernel_size=1,
                )
                for i in range(4)
            ],
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    _feature_info_int(
                        self.image_encoder,
                        image_start_index + i,
                        "num_chs",
                    ),
                    _feature_info_int(
                        self.lidar_encoder,
                        lidar_start_index + i,
                        "num_chs",
                    ),
                    kernel_size=1,
                )
                for i in range(4)
            ],
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (
                config.lidar_config.lidar_bev_grid_rows,
                config.lidar_config.lidar_bev_grid_cols,
            ),
        )

        # Fusion transformers
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=_feature_info_int(
                        self.image_encoder,
                        image_start_index + i,
                        "num_chs",
                    ),
                    config=config,
                )
                for i in range(4)
            ],
        )

        # Post-fusion convs
        self.perspective_upsample_factor: int = (
            _feature_info_int(
                self.image_encoder,
                image_start_index + 3,
                "reduction",
            )
            // config.perspective_config.perspective_downsample_factor
        )

        if config.box_detection_config.detect_boxes or config.bev_semantic_config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=config.bev_semantic_config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    config.lidar_config.bev_height_pixel // config.bev_semantic_config.bev_downsample_factor,
                    config.lidar_config.bev_width_pixel // config.bev_semantic_config.bev_downsample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )
            self.up_conv5 = nn.Conv2d(
                config.bev_semantic_config.bev_feature_channels,
                config.bev_semantic_config.bev_feature_channels,
                (3, 3),
                padding=1,
            )
            self.up_conv4 = nn.Conv2d(
                config.bev_semantic_config.bev_feature_channels,
                config.bev_semantic_config.bev_feature_channels,
                (3, 3),
                padding=1,
            )
            self.c5_conv = nn.Conv2d(
                self.num_lidar_features,
                config.bev_semantic_config.bev_feature_channels,
                (1, 1),
            )

    def _latent_lidar_grid(
        self,
        image: jt.Float[torch.Tensor, "batch 3 height width"],
    ) -> jt.Float[torch.Tensor, "batch 2 lidar_height lidar_width"]:
        """
        The positional-encoding grid that stands in for the LiDAR raster in latent mode.

        Args:
            image: the camera batch, used for batch size, device and dtype.

        Returns:
            grid of shape [B, 2, bev_height_pixel, bev_width_pixel].
        """
        config = self._config
        x = torch.linspace(
            0,
            1,
            config.lidar_config.bev_width_pixel,
            device=image.device,
        )
        y = torch.linspace(
            0,
            1,
            config.lidar_config.bev_height_pixel,
            device=image.device,
        )
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        lidar = torch.zeros(
            (
                image.shape[0],
                2,
                config.lidar_config.bev_height_pixel,
                config.lidar_config.bev_width_pixel,
            ),
            device=image.device,
            dtype=image.dtype,
        )
        lidar[:, 0] = y_grid.unsqueeze(0)  # Top down positional encoding
        lidar[:, 1] = x_grid.unsqueeze(0)  # Left right positional encoding
        return lidar

    def top_down(
        self,
        x: jt.Float[torch.Tensor, "batch channels height width"],
    ) -> jt.Float[torch.Tensor, "batch bev_channels up_height up_width"]:
        """
        Applies top-down feature pyramid processing to BEV features.

        Progressively upsamples and refines features through multiple resolution levels
        to create a higher-resolution bird's-eye-view representation.

        Args:
            x: input BEV feature tensor from the LiDAR encoder, shape [B, C, H, W].

        Returns:
            upsampled and refined BEV feature grid at the BEV head resolution.
        """
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
        return F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)

    @override
    def forward(
        self,
        image: jt.Float[torch.Tensor, "batch 3 height width"],
        lidar: jt.Float[
            torch.Tensor,
            "batch lidar_channels lidar_height lidar_width",
        ]
        | None,
    ) -> tuple[
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
        jt.Float[torch.Tensor, "batch image_channels feat_height feat_width"],
    ]:
        """
        Image + LiDAR feature fusion using transformers.

        Args:
            image: RGB image of shape [B, 3, image_height, image_width], values in [0, 255].
            lidar: rasterized LiDAR of shape [B, lidar_in_channels, bev_height_pixel,
                bev_width_pixel]; ignored (may be None) in latent mode.

        Returns:
            tuple of (lidar_features, image_features):
            - lidar_features: BEV feature map for planning tasks [B, C_lidar, H, W].
            - image_features: image feature map for perception tasks [B, C_img, H, W].
        """
        lidar = self._latent_lidar_grid(image) if self._config.backbone_config.latent else lidar
        assert lidar is not None, "LiDAR raster is required outside of latent mode!"

        image_features = normalize_imagenet(image)
        lidar_features = lidar

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())
        image_return_layers = _return_layers(self.image_encoder)
        lidar_return_layers = _return_layers(self.lidar_encoder)

        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(image_return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers,
                image_return_layers,
                image_features,
            )
        if len(lidar_return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers,
                lidar_return_layers,
                lidar_features,
            )

        # Loop through the 4 blocks of the network.
        for i in range(4):
            # Branch-specific forward pass
            image_features = self.forward_layer_block(
                image_layers,
                image_return_layers,
                image_features,
            )
            lidar_features = self.forward_layer_block(
                lidar_layers,
                lidar_return_layers,
                lidar_features,
            )
            image_features, lidar_features = self.fuse_features(
                image_features,
                lidar_features,
                i,
            )

        return lidar_features, image_features

    def forward_layer_block(
        self,
        layers: Iterator[tuple[str, Any]],
        return_layers: dict[str, str],
        features: jt.Float[
            torch.Tensor,
            "batch in_channels in_height in_width",
        ],
    ) -> jt.Float[torch.Tensor, "batch out_channels out_height out_width"]:
        """
        Runs one forward pass to a block of layers from a timm neural network.

        Advances the whole network by just one block.

        Args:
            layers: iterator starting at the current layer block of the target network.
            return_layers: timm dictionary describing at which intermediate layers features are returned.
            features: input features.

        Returns:
            processed features.
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(
        self,
        image_features: jt.Float[
            torch.Tensor,
            "batch img_channels img_height img_width",
        ],
        lidar_features: jt.Float[
            torch.Tensor,
            "batch bev_channels bev_height bev_width",
        ],
        layer_idx: int,
    ) -> tuple[
        jt.Float[torch.Tensor, "batch img_channels img_height img_width"],
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
    ]:
        """
        Performs a TransFuser feature fusion block using a Transformer module.

        Args:
            image_features: features from the image branch, shape [B, C, H, W].
            lidar_features: features from the LiDAR branch, shape [B, C2, H2, W2].
            layer_idx: transformer layer index.

        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](
            lidar_embd_layer,
        )

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer,
        )
        mode = self._config.backbone_config.upsample_mode
        image_features_layer = _resample_to(
            image_features_layer,
            image_features.shape[2:],
            mode,
        )
        lidar_features_layer = _resample_to(
            lidar_features_layer,
            lidar_features.shape[2:],
            mode,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """
    GPT-style transformer module for cross-modal feature fusion.

    Implements a transformer that fuses image and LiDAR features using learned
    positional embeddings and multi-head self-attention across both modalities.
    """

    def __init__(self, n_embd: int, config: TransfuserConfig):
        """
        Initializes the GPT fusion transformer.

        Args:
            n_embd: embedding dimension (number of feature channels).
            config: global config dataclass of TransFuser.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.n_embd = n_embd
        self._config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                config.camera_config.img_vert_anchors * config.camera_config.img_horz_anchors
                + config.lidar_config.lidar_bev_grid_rows * config.lidar_config.lidar_bev_grid_cols,
                n_embd,
            ),
        )
        self.drop = nn.Dropout(config.backbone_config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.backbone_config.n_head,
                    config.backbone_config.block_exp,
                    config.backbone_config.attn_pdrop,
                    config.backbone_config.resid_pdrop,
                )
                for _ in range(config.backbone_config.n_layer)
            ],
        )
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights for linear and layer norm modules.

        Args:
            module: torch module to initialize (Linear or LayerNorm).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self._config.backbone_config.gpt_linear_layer_init_mean,
                std=self._config.backbone_config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(
                self._config.backbone_config.gpt_layer_norm_init_weight,
            )

    @override
    def forward(
        self,
        image_tensor: jt.Float[
            torch.Tensor,
            "batch channels img_vert img_horz",
        ],
        lidar_tensor: jt.Float[
            torch.Tensor,
            "batch channels bev_rows bev_cols",
        ],
    ) -> tuple[
        jt.Float[torch.Tensor, "batch channels img_vert img_horz"],
        jt.Float[torch.Tensor, "batch channels bev_rows bev_cols"],
    ]:
        """
        Fusion transformer forward pass.

        Args:
            image_tensor: image tensor of shape [B, C, img_vert_anchors, img_horz_anchors].
            lidar_tensor: LiDAR tensor of shape [B, C, lidar_bev_grid_rows, lidar_bev_grid_cols].

        Returns:
            tuple of (fused image tensor, fused LiDAR tensor) with input shapes.
        """
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        num_img_tokens = self._config.camera_config.img_vert_anchors * self._config.camera_config.img_horz_anchors
        image_tensor_out = x[:, :num_img_tokens, :].view(bz, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()
        lidar_tensor_out = x[:, num_img_tokens:, :].view(bz, lidar_h, lidar_w, -1).permute(0, 3, 1, 2).contiguous()

        return image_tensor_out, lidar_tensor_out


class Block(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.

    Implements a standard transformer block with pre-normalization,
    multi-head self-attention, and an MLP with residual connections.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        """
        Initializes a transformer block.

        Args:
            n_embd: embedding dimension (feature channels).
            n_head: number of attention heads.
            block_exp: expansion factor for MLP hidden dimension.
            attn_pdrop: dropout probability for attention weights.
            resid_pdrop: dropout probability for residual connections.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies transformer block with attention and feed-forward processing.

        Uses pre-normalization and residual connections for stable training.

        Args:
            x: input tensor of shape [B, T, n_embd].

        Returns:
            output tensor of same shape as input with attention and MLP applied.
        """
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SelfAttention(nn.Module):
    """
    Multi-head self-attention module.

    Implements scaled dot-product attention across multiple heads with
    learnable query, key, and value projections.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        """
        Initializes multi-head self-attention.

        Args:
            n_embd: embedding dimension (must be divisible by n_head).
            n_head: number of attention heads.
            attn_pdrop: dropout probability for attention weights.
            resid_pdrop: dropout probability for output projection.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-head self-attention.

        Projects input to queries, keys, and values, applies scaled dot-product
        attention independently for each head, then concatenates and projects
        the results.

        Args:
            x: input tensor of shape [B, T, n_embd].

        Returns:
            attention output tensor of shape [B, T, n_embd].
        """
        b, t, c = x.size()
        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        return self.resid_drop(self.proj(y))
