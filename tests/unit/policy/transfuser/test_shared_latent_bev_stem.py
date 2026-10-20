# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The shared latent BEV stem gives the unpatched backbone's outputs, gradients and BatchNorm statistics."""

from __future__ import annotations

import copy

import pytest
import torch

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.common.config_help import import_string
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, CameraConfig, TransfuserConfig
from py123d_garage.policy.transfuser.network.encoder.transfuser import TransfuserBackbone, share_latent_bev_stem

_ENCODER = "py123d_garage.policy.transfuser.network.encoder"
# The canvas backbone, and the per-frame streaming backbone with one past camera row.
BACKBONES = [
    (f"{_ENCODER}.transfuser:TransfuserBackbone", 0),
    (f"{_ENCODER}.streaming_transfuser:StreamingTransfuserBackbone", 1),
]


def _backbone(target: str = BACKBONES[0][0], past_rows: int = 0) -> TransfuserBackbone:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        camera_config=CameraConfig(
            image_width=256,
            image_height=128 * (1 + past_rows),
            past_camera_offset_us=500_000 if past_rows else 0,
            num_past_camera_rows=max(1, past_rows),
        ),
        backbone_config=BackboneConfig(image_encoder_pretrained=False, latent=True, backbone_target=target),
    )
    torch.manual_seed(0)
    return import_string(target)(config, image_encoder_pretrained=False).double()


@pytest.mark.parametrize(("target", "past_rows"), BACKBONES)
def test_matches_the_unpatched_backbone(target: str, past_rows: int) -> None:
    reference = _backbone(target, past_rows)
    patched = copy.deepcopy(reference)
    share_latent_bev_stem(patched)
    assert patched.state_dict().keys() == reference.state_dict().keys()
    torch.manual_seed(1)
    image = torch.rand(3, 3, 128 * (1 + past_rows), 256, dtype=torch.float64) * 255.0
    # The fusion transformers' dropout draws the same masks from the same seed.
    torch.manual_seed(3)
    outputs = patched(image, None)
    torch.manual_seed(3)
    reference_outputs = reference(image, None)
    torch.manual_seed(2)
    weights = [torch.randn_like(out) for out in reference_outputs if out is not None]
    loss = torch.stack([(out * w).sum() for out, w in zip([o for o in outputs if o is not None], weights, strict=True)])
    reference_loss = torch.stack(
        [(out * w).sum() for out, w in zip([o for o in reference_outputs if o is not None], weights, strict=True)],
    )
    for out, reference_out in zip(outputs, reference_outputs, strict=True):
        if reference_out is not None:
            assert out is not None
            assert out.shape == reference_out.shape
            torch.testing.assert_close(out, reference_out, rtol=1e-10, atol=1e-10)
    loss.sum().backward()
    reference_loss.sum().backward()
    # Absolute tolerance from the largest gradient: some gradients are zero up to rounding (biases before a softmax's
    # keys or a BatchNorm), where any summation order differs relatively.
    scale = max(p.grad.abs().max().item() for p in reference.parameters() if p.grad is not None)
    for (name, parameter), reference_parameter in zip(patched.named_parameters(), reference.parameters(), strict=True):
        if reference_parameter.grad is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-9, atol=1e-12 * scale, msg=name)
    for (name, buffer), reference_buffer in zip(patched.named_buffers(), reference.buffers(), strict=True):
        # running_var's n / (n - 1) correction counts one sample instead of three.
        torch.testing.assert_close(buffer, reference_buffer, rtol=1e-3, atol=0, msg=name)


def test_eval_matches_the_unpatched_backbone() -> None:
    reference = _backbone().eval()
    patched = copy.deepcopy(reference)
    share_latent_bev_stem(patched)
    image = torch.rand(2, 3, 128, 256, dtype=torch.float64) * 255.0
    with torch.no_grad():
        for out, reference_out in zip(patched(image, None), reference(image, None), strict=True):
            if reference_out is not None:
                assert out is not None
                torch.testing.assert_close(out, reference_out, rtol=1e-10, atol=1e-10)
