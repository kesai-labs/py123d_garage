# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    Normalizes images with the ImageNet mean and std.

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


def _bilinear_matrix(
    out_size: int,
    in_size: int,
    device: torch.device,
) -> jt.Float[torch.Tensor, "out_size in_size"]:
    """
    Builds the weights of F.interpolate(mode="bilinear", align_corners=False) along one axis as a matrix.

    Args:
        out_size: output length of the axis.
        in_size: input length of the axis.
        device: device of the matrix.

    Returns:
        the [out_size, in_size] interpolation matrix, rows summing to one.
    """
    # PyTorch's source index is clamped into the input, so the edge outputs put all their weight on the edge pixel.
    source = ((torch.arange(out_size, device=device, dtype=torch.float64) + 0.5) * (in_size / out_size) - 0.5).clamp(
        0.0,
        in_size - 1,
    )
    inputs = torch.arange(in_size, device=device, dtype=torch.float64)
    distance = (source[:, None] - inputs[None, :]).abs()
    return torch.clamp(-distance + 1.0, min=0.0)


def _resample_to(
    features: jt.Float[torch.Tensor, "batch channels in_height in_width"],
    size: torch.Size,
    mode: str,
) -> jt.Float[torch.Tensor, "batch channels height width"]:
    """
    Resamples a feature map to size, passing through when it already matches.

    Args:
        features: the map to resample.
        size: target spatial size.
        mode: "bilinear", "nearest", or "bilinear_matmul", which is bilinear written as two matrix products.

    Returns:
        the resampled map, or features unchanged.
    """
    if features.shape[2:] == size:
        return features
    if mode == "bilinear_matmul":
        height_matrix = _bilinear_matrix(size[0], features.shape[2], features.device).to(features.dtype)
        width_matrix = _bilinear_matrix(size[1], features.shape[3], features.device).to(features.dtype)
        if features.is_contiguous(memory_format=torch.channels_last) and not features.is_contiguous():
            # On the NHWC layout both products run on contiguous views, so neither side is transposed to NCHW.
            batch, channels, in_height, in_width = features.shape
            rows = features.permute(0, 2, 3, 1).reshape(batch * in_height, in_width, channels)
            over_width = torch.matmul(width_matrix, rows).reshape(batch, in_height, size[1] * channels)
            over_height = torch.matmul(height_matrix, over_width).reshape(batch, size[0], size[1], channels)
            return over_height.permute(0, 3, 1, 2)
        return height_matrix @ features @ width_matrix.T
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


class BilinearMatmulUpsample(nn.Module):
    """nn.Upsample(mode="bilinear", align_corners=False) written as two matrix products, so the backward avoids a scatter."""

    def __init__(self, size: tuple[int, int] | None = None, scale_factor: int | None = None) -> None:
        """
        Initializes the up-sample with its output size or its scale factor, one of the two.

        Args:
            size: output height and width.
            scale_factor: the factor both input dimensions grow by.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert (size is None) != (scale_factor is None), "give the output size or the scale factor"
        self.size = size
        self.scale_factor = scale_factor

    @override
    def forward(
        self,
        features: jt.Float[torch.Tensor, "batch channels in_height in_width"],
    ) -> jt.Float[torch.Tensor, "batch channels height width"]:
        """
        Up-samples the map.

        Args:
            features: the map to up-sample.

        Returns:
            the up-sampled map.
        """
        if self.size is not None:
            size = self.size
        else:
            assert self.scale_factor is not None
            size = (features.shape[2] * self.scale_factor, features.shape[3] * self.scale_factor)
        return _resample_to(features, torch.Size(size), "bilinear_matmul")


class EvenAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """AdaptiveAvgPool2d that runs avg_pool2d when the output size divides the input, whose backward inductor can fuse."""

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pools the input to output_size."""
        height, width = input.shape[-2:]
        size = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)
        out_height = height if size[0] is None else size[0]
        out_width = width if size[1] is None else size[1]
        if height % out_height or width % out_width:
            return super().forward(input)
        window = (height // out_height, width // out_width)
        return F.avg_pool2d(input, kernel_size=window, stride=window)


def _feature_info_int(encoder: FeatureListNet, index: int, key: str) -> int:
    """
    Reads an integer property of one feature level of a timm encoder.

    Args:
        encoder: the timm feature encoder.
        index: index of the feature level.
        key: name of the property, e.g. "num_chs".

    Returns:
        the property's value.
    """
    return cast(int, encoder.feature_info.info[index][key])  # pyright: ignore[reportUnknownMemberType]


def _return_layers(encoder: FeatureListNet) -> dict[str, str]:
    """
    Reads the names of a timm encoder's feature-returning layers.

    Args:
        encoder: the timm feature encoder.

    Returns:
        the returned layer names, mapped to their output names.
    """
    return cast("dict[str, str]", encoder.return_layers)  # pyright: ignore[reportUnknownMemberType]


# (features, grid) to features + the grid up-sampled bilinearly onto them, or None where it does not apply; a backbone
# kernel patch's fused up-sample and residual add.
FusedUpsampleAdd = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]


class TransfuserBackbone(nn.Module):
    """TransFuser backbone that fuses image and LiDAR features with transformers at four resolutions."""

    fused_upsample_add: FusedUpsampleAdd | None = None

    def __init__(self, config: TransfuserConfig, image_encoder_pretrained: bool):
        """
        Initializes the TransFuser backbone with dual encoder branches and fusion modules.

        Args:
            config: global config dataclass of TransFuser.
            image_encoder_pretrained: whether timm fetches pretrained image encoder weights.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._config = config

        self.image_encoder = cast(
            FeatureListNet,
            timm.create_model(
                config.backbone_config.image_architecture,
                pretrained=image_encoder_pretrained,
                features_only=True,
            ),
        )
        self.avgpool_img = EvenAdaptiveAvgPool2d(
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

        image_channels = [_feature_info_int(self.image_encoder, image_start_index + i, "num_chs") for i in range(4)]
        bev_grid = (config.lidar_config.lidar_bev_grid_rows, config.lidar_config.lidar_bev_grid_cols)
        # In latent mode a fixed 2-channel positional grid replaces the LiDAR raster.
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
        lidar_channels = [_feature_info_int(self.lidar_encoder, lidar_start_index + i, "num_chs") for i in range(4)]
        self.lidar_channel_to_img = nn.ModuleList(
            [nn.Conv2d(lidar_channels[i], image_channels[i], kernel_size=1) for i in range(4)],
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [nn.Conv2d(image_channels[i], lidar_channels[i], kernel_size=1) for i in range(4)],
        )
        self.avgpool_lidar = EvenAdaptiveAvgPool2d(bev_grid)
        self.num_lidar_features: int = lidar_channels[3]
        self.num_register_features: int = self.num_image_features

        backbone_config = config.backbone_config
        n_head_per_stage = backbone_config.n_head_per_stage or [backbone_config.n_head] * 4
        n_layer_per_stage = backbone_config.n_layer_per_stage or [backbone_config.n_layer] * 4
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=_feature_info_int(
                        self.image_encoder,
                        image_start_index + i,
                        "num_chs",
                    ),
                    n_head=n_head_per_stage[i],
                    n_layer=n_layer_per_stage[i],
                    config=config,
                    num_registers=backbone_config.num_fusion_registers if i == 3 else 0,
                )
                for i in range(4)
            ],
        )
        self.perspective_upsample_factor: int = (
            _feature_info_int(
                self.image_encoder,
                image_start_index + 3,
                "reduction",
            )
            // config.perspective_config.perspective_downsample_factor
        )

        if config.box_detection_config.detect_boxes or config.bev_semantic_config.use_bev_semantic:
            self.upsample = BilinearMatmulUpsample(scale_factor=config.bev_semantic_config.bev_upsample_factor)
            self.upsample2 = BilinearMatmulUpsample(
                size=(
                    config.lidar_config.bev_height_pixel // config.bev_semantic_config.bev_downsample_factor,
                    config.lidar_config.bev_width_pixel // config.bev_semantic_config.bev_downsample_factor,
                ),
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
        image: jt.Float[torch.Tensor, "batch ..."],
    ) -> jt.Float[torch.Tensor, "batch 2 lidar_height lidar_width"]:
        """
        Builds the positional-encoding grid that stands in for the LiDAR raster in latent mode.

        Args:
            image: the camera batch, or any tensor of the batch, used for batch size, device and dtype.

        Returns:
            the grid of row and column coordinates in [0, 1] at the LiDAR raster size.
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
        lidar[:, 0] = y_grid.unsqueeze(0)
        lidar[:, 1] = x_grid.unsqueeze(0)
        return lidar

    def top_down(
        self,
        x: jt.Float[torch.Tensor, "batch channels height width"],
    ) -> jt.Float[torch.Tensor, "batch bev_channels up_height up_width"]:
        """
        Up-samples the BEV features to the BEV head resolution with a small feature pyramid.

        Args:
            x: BEV features from the LiDAR encoder.

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
        jt.Float[torch.Tensor, "batch registers image_channels"] | None,
    ]:
        """
        Encodes the image and the LiDAR, fusing the two branches after each of the four stages.

        Args:
            image: RGB image, values in [0, 255].
            lidar: rasterized LiDAR; ignored and may be None in latent mode.

        Returns:
            the BEV features, the image features and the last stage's register tokens, None without registers.
        """
        image_features = normalize_imagenet(image)
        image_layers = iter(self.image_encoder.items())
        image_return_layers = _return_layers(self.image_encoder)
        lidar = self._latent_lidar_grid(image) if self._config.backbone_config.latent else lidar
        assert lidar is not None, "LiDAR raster is required outside of latent mode!"
        lidar_features = lidar
        lidar_layers = iter(self.lidar_encoder.items())
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

        fusion_registers: torch.Tensor | None = None
        for i in range(4):
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
            image_features, lidar_features, stage_registers = self.fuse_features(
                image_features,
                lidar_features,
                i,
            )
            if stage_registers is not None:
                fusion_registers = stage_registers
        return lidar_features, image_features, fusion_registers

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
        Runs a timm encoder from its current layer up to its next feature-returning layer.

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
            "lidar_batch bev_channels bev_height bev_width",
        ],
        layer_idx: int,
    ) -> tuple[
        jt.Float[torch.Tensor, "batch img_channels img_height img_width"],
        jt.Float[torch.Tensor, "batch bev_channels bev_height bev_width"],
        jt.Float[torch.Tensor, "batch registers img_channels"] | None,
    ]:
        """
        Fuses the image and LiDAR features of one stage with its fusion transformer.

        Args:
            image_features: features from the image branch.
            lidar_features: features from the LiDAR branch.
            layer_idx: index of the fusion stage.

        Returns:
            image_features and lidar_features with added features from the other branch, and the stage's register
            tokens, None for a stage without registers.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](
            lidar_embd_layer,
        )

        image_features_layer, lidar_features_layer, registers = self.transformers[layer_idx](
            image_embd_layer,
            lidar_embd_layer,
        )

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer,
        )
        mode = self._config.backbone_config.upsample_mode
        fused_upsample_add = self.fused_upsample_add

        def add_resampled(features: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
            """
            Adds a fused grid to a feature map, resampled to the map's size.

            Args:
                features: the feature map.
                layer: the fusion transformer's grid for it.

            Returns:
                the feature map plus the resampled grid.
            """
            if fused_upsample_add is not None and mode.startswith("bilinear") and layer.shape[2:] != features.shape[2:]:
                fused = fused_upsample_add(features, layer)
                if fused is not None:
                    return fused
            return features + _resample_to(layer, features.shape[2:], mode)

        return (
            add_resampled(image_features, image_features_layer),
            add_resampled(lidar_features, lidar_features_layer),
            registers,
        )


class GPT(nn.Module):
    """GPT-style transformer that fuses image and LiDAR tokens with self-attention over both."""

    def __init__(self, n_embd: int, n_head: int, n_layer: int, config: TransfuserConfig, num_registers: int = 0):
        """
        Initializes the GPT fusion transformer.

        Args:
            n_embd: embedding dimension (number of feature channels).
            n_head: attention heads per transformer layer.
            n_layer: transformer layers.
            config: global config dataclass of TransFuser.
            num_registers: learned register tokens appended to the image and LiDAR tokens.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.n_embd = n_embd
        self._config = config

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                config.camera_config.img_vert_anchors * config.camera_config.img_horz_anchors
                + config.lidar_config.lidar_bev_grid_rows * config.lidar_config.lidar_bev_grid_cols,
                n_embd,
            ),
        )
        self.drop = nn.Dropout(config.backbone_config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    config.backbone_config.block_exp,
                    config.backbone_config.attn_pdrop,
                    config.backbone_config.resid_pdrop,
                )
                for _ in range(n_layer)
            ],
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)
        # No positional embedding: a learned register already is its own identity.
        self.registers = nn.Parameter(torch.randn(1, num_registers, n_embd) * 0.02) if num_registers else None

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
        jt.Float[torch.Tensor, "batch registers channels"] | None,
    ]:
        """
        Fuses the pooled image and LiDAR grids.

        Args:
            image_tensor: pooled image grid.
            lidar_tensor: pooled LiDAR grid.

        Returns:
            the fused image and LiDAR grids, and the registers after fusion or None.
        """
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.pos_emb + token_embeddings
        if self.registers is not None:
            x = torch.cat((x, self.registers.expand(bz, -1, -1)), dim=1)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        num_img_tokens = self._config.camera_config.img_vert_anchors * self._config.camera_config.img_horz_anchors
        num_lidar_tokens = lidar_h * lidar_w
        image_tensor_out = x[:, :num_img_tokens, :].view(bz, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()
        lidar_tensor_out = (
            x[:, num_img_tokens : num_img_tokens + num_lidar_tokens, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        registers_out = x[:, num_img_tokens + num_lidar_tokens :, :] if self.registers is not None else None

        return image_tensor_out, lidar_tensor_out, registers_out


class Block(nn.Module):
    """Pre-norm transformer block with self-attention and an MLP."""

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
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    @override
    def forward(
        self,
        x: jt.Float[torch.Tensor, "batch tokens channels"],
    ) -> jt.Float[torch.Tensor, "batch tokens channels"]:
        """
        Applies self-attention and the MLP, each with a residual connection.

        Args:
            x: input tokens.

        Returns:
            the output tokens.
        """
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    fused_qkv: bool = False

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
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    @override
    def forward(
        self,
        x: jt.Float[torch.Tensor, "batch tokens channels"],
    ) -> jt.Float[torch.Tensor, "batch tokens channels"]:
        """
        Computes multi-head self-attention.

        Args:
            x: input tokens.

        Returns:
            the attention output.
        """
        b, t, c = x.size()
        if self.fused_qkv:
            # One GEMM over the stacked key, query and value weights; the per-step stack costs 3 c^2 elements.
            weight = torch.cat([self.key.weight, self.query.weight, self.value.weight])
            bias = torch.cat([self.key.bias, self.query.bias, self.value.bias])
            kqv = F.linear(x, weight, bias).view(b, t, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
            k, q, v = kqv.unbind(0)  # (b, nh, t, hs) each
        else:
            k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
            q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
            v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


def fuse_fusion_qkv(backbone: nn.Module) -> None:
    """
    Patches every fusion transformer's self-attention inside backbone to project keys, queries and values with one GEMM.

    Parameters and state_dict keys stay the same.

    Args:
        backbone: the module to patch.
    """
    for module in backbone.modules():
        if isinstance(module, SelfAttention):
            module.fused_qkv = True


def share_latent_bev_stem(backbone: nn.Module) -> None:
    """
    Patches a latent TransfuserBackbone in place to run its BEV stem on one sample and expand it at the first fusion.

    The latent BEV input is the same for every sample, so outputs and gradients are unchanged; only BatchNorm's running
    variance differs slightly.

    Args:
        backbone: a latent TransfuserBackbone.
    """
    assert isinstance(backbone, TransfuserBackbone), f"needs a TransfuserBackbone, got {type(backbone).__name__}"
    assert backbone._config.backbone_config.latent, "share_latent_bev_stem needs latent mode"
    batch_grid: Callable[[torch.Tensor], torch.Tensor] = backbone._latent_lidar_grid

    def single_grid(image: torch.Tensor) -> torch.Tensor:
        """
        Builds the latent BEV grid for the first sample only.

        Args:
            image: the camera batch.

        Returns:
            the grid with a batch of one.
        """
        return batch_grid(image[:1])

    first_fusion = backbone.transformers[0]
    fuse: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = (
        first_fusion.forward
    )

    def fuse_expanded(
        image_tensor: torch.Tensor,
        lidar_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Runs the first fusion with the LiDAR grid expanded to the image batch.

        Args:
            image_tensor: pooled image grid.
            lidar_tensor: pooled LiDAR grid, possibly with a batch of one.

        Returns:
            the outputs of the first fusion transformer.
        """
        return fuse(image_tensor, lidar_tensor.expand(image_tensor.shape[0], *lidar_tensor.shape[1:]))

    backbone._latent_lidar_grid = single_grid  # pyright: ignore[reportAttributeAccessIssue]
    first_fusion.forward = fuse_expanded  # pyright: ignore[reportAttributeAccessIssue]
