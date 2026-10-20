# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""Flattens visual token grids into sequences with their spatial and temporal position ids."""

from typing import Dict

import torch
from einops import rearrange, repeat
from torch import Tensor


def compute_position_indices(batch_size: int, num_frames: int, height: int, width: int) -> Dict[str, Tensor]:
    """
    Computes the spatial and temporal position id of every token in a flattened frame sequence.

    Args:
        batch_size: number of samples in the batch.
        num_frames: number of frames in the sequence.
        height: height of each token frame.
        width: width of each token frame.

    Returns:
        spatial_positions and temporal_positions, each shaped [batch, frames * height * width], on CUDA when available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_visual_tokens = height * width
    total_tokens_per_frame = num_visual_tokens

    spatial_positions = torch.arange(height * width, device=device)
    spatial_positions = repeat(spatial_positions, "s -> b (t s)", b=batch_size, t=num_frames)

    temporal_positions = torch.arange(num_frames, device=device)
    temporal_positions = repeat(temporal_positions, "t -> b (t s)", b=batch_size, s=total_tokens_per_frame)

    return {"spatial_positions": spatial_positions, "temporal_positions": temporal_positions}


def prepare_token_sequence(visual_tokens: Tensor) -> Dict[str, Tensor]:
    """
    Flattens a token grid into a sequence with its position ids.

    Args:
        visual_tokens: token ids shaped [batch, frames, height, width].

    Returns:
        token_sequence, spatial_positions and temporal_positions, each shaped [batch, frames * height * width].
    """
    position_indices = compute_position_indices(*visual_tokens.shape)

    visual_tokens = rearrange(visual_tokens, "b t h w -> b (t h w)")

    return {"token_sequence": visual_tokens, **position_indices}


def prepare_AR_token_sequences(visual_tokens: Tensor) -> Dict[str, Tensor]:
    """
    Splits a token grid into next-token inputs and targets, shifted by one token.

    Args:
        visual_tokens: token ids shaped [batch, frames, height, width].

    Returns:
        the input sequence with its position ids, and the target sequence.
    """
    sequence_data = prepare_token_sequence(visual_tokens)

    input_data = {
        "token_sequence": sequence_data["token_sequence"][:, :-1],
        "spatial_positions": sequence_data["spatial_positions"][:, :-1],
        "temporal_positions": sequence_data["temporal_positions"][:, :-1],
    }

    target_data = {
        "token_sequence": sequence_data["token_sequence"][:, 1:],
    }

    return input_data, target_data
