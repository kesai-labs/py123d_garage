# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""VaVAM video-action model: flow matching of driving actions on top of a GPT2 video model."""

import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import BoolTensor, LongTensor, Tensor
from tqdm import tqdm

from py123d_garage.policy.vavam.network.joint_model import JointModel
from py123d_garage.policy.vavam.network.mup_action_expert import MupActionExpert
from py123d_garage.policy.vavam.network.mup_gpt2 import MupGPT2
from py123d_garage.policy.vavam.network.mup_readout import checkpoint_pickle_module, set_base_shapes

logger = logging.getLogger(__name__)


def instantiate(config: OmegaConf) -> nn.Module:
    """
    Builds a MupGPT2 or MupActionExpert from a Hydra-style config without importing its _target_.

    Args:
        config: config whose _target_ names the class and whose other keys are its arguments.

    Returns:
        the built module.
    """
    targets = {"MupGPT2": MupGPT2, "MupActionExpert": MupActionExpert}
    kwargs = {key: value for key, value in dict(config).items() if not key.startswith("_")}
    return targets[str(config["_target_"]).rsplit(".", 1)[-1]](**kwargs)

mupShapes = str | Dict[str, Tuple[int, ...]]


class VideoActionModel(nn.Module):
    """Wraps a JointModel with the flow-matching loss and the Euler sampler for actions."""

    def __init__(
        self,
        gpt_config: OmegaConf,
        gpt_mup_base_shapes: mupShapes | None,
        gpt_checkpoint_path: str | None,
        action_config: OmegaConf,
        action_mup_base_shapes: mupShapes | None,
        action_checkpoint_path: str | None,
        finetuning_timesteps: int = 8,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        final_action_clip_value: float | None = None,
        action_scaling: float = 1.0,
    ) -> None:
        """
        Builds the video model, the action expert and the joint model around them.

        Args:
            gpt_config: config of the GPT2 video model.
            gpt_mup_base_shapes: muP base shapes of the video model.
            gpt_checkpoint_path: checkpoint to load the frozen video model from; None initializes it.
            action_config: config of the action expert.
            action_mup_base_shapes: muP base shapes of the action expert.
            action_checkpoint_path: checkpoint to load the frozen action expert from; None initializes it.
            finetuning_timesteps: number of context frames in training.
            num_inference_steps: number of Euler steps at inference.
            flow_sig_min: minimum noise level of the flow path.
            final_action_clip_value: bound on the sampled actions before rescaling; None disables clipping.
            action_scaling: factor between model-space and real actions.
        """
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.final_action_clip_value = final_action_clip_value
        self.context_length = finetuning_timesteps
        self.action_scaling = action_scaling

        self.gpt: MupGPT2 = instantiate(gpt_config)
        self.gpt_mup_base_shapes = gpt_mup_base_shapes
        if gpt_checkpoint_path is not None:
            logger.info(f"Loading GPT2 checkpoint from {gpt_checkpoint_path}")
            gpt_state_dict = self._load_ckpt(gpt_checkpoint_path, key="network.")
            self.gpt.load_state_dict(gpt_state_dict)
            logger.info(f"Setting base shapes for GPT2 with mup {gpt_mup_base_shapes}")
            set_base_shapes(self.gpt, gpt_mup_base_shapes, rescale_params=False)
            self.gpt.requires_grad_(False)
            print("-" * 80)
        else:
            logger.info(f"Initializing GPT2 from scratch, with mup {gpt_mup_base_shapes}")
            set_base_shapes(self.gpt, gpt_mup_base_shapes)
            self.gpt.apply(self.gpt._init_weights)  # re-initialize after set_base_shapes
        self.action_expert: MupActionExpert = instantiate(action_config)
        self.action_mup_base_shapes = action_mup_base_shapes
        if action_checkpoint_path is not None:
            logger.info(f"Loading Action Expert checkpoint from {action_checkpoint_path}")
            action_state_dict = self._load_ckpt(action_checkpoint_path)
            self.action_expert.load_state_dict(action_state_dict)
            logger.info(f"Setting base shapes for Action Expert with mup {action_mup_base_shapes}")
            set_base_shapes(self.action_expert, action_mup_base_shapes, rescale_params=False)
            self.action_expert.requires_grad_(False)
            print("-" * 80)
        else:
            logger.info(f"Initializing Action Expert from scratch, with mup {action_mup_base_shapes}")
            set_base_shapes(self.action_expert, action_mup_base_shapes)
            self.action_expert.apply(self.action_expert._init_weights)  # re-initialize after set_base_shapes
        self.joint_model = JointModel(self.gpt, self.action_expert)

        self.nb_tokens_per_timestep = self.gpt.nb_tokens_per_timestep

        self.action_dim = self.action_expert.action_dim
        self.action_horizon = self.action_expert.action_horizon
        self.action_hidden_dim = self.action_expert.embedding_dim

        self.build_attention_mask()

    def _load_ckpt(self, ckpt: str, key: str | None) -> OrderedDict:
        """
        Reads a checkpoint's state dict, stripping a key prefix when given.

        Args:
            ckpt: checkpoint path; ~ and environment variables are expanded.
            key: prefix to remove from every key, or None to keep the keys.

        Returns:
            the state dict.
        """
        ckpt = torch.load(os.path.expanduser(os.path.expandvars(ckpt)), map_location="cpu")
        if key is not None:
            state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                state_dict[k.replace(key, "")] = v
        else:
            state_dict = ckpt["state_dict"]
        return state_dict

    def build_attention_mask(self) -> None:
        """
        Registers the training mask: causal over visual tokens; each frame's action chunk sees itself and the frames up to it.
        """
        context_length = self.context_length
        action_horizon = self.action_horizon
        nb_tokens_per_timestep = self.nb_tokens_per_timestep

        visual_seqlen = self.nb_tokens_per_timestep * self.context_length
        action_seq_len = self.action_horizon * self.context_length
        seqlen = visual_seqlen + action_seq_len
        attn_mask = torch.zeros(seqlen, seqlen, dtype=torch.bool)

        for i in range(visual_seqlen):
            attn_mask[i, : i + 1] = True

        for i in range(context_length):
            attn_mask[
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
            ] = True

        for i in range(context_length):
            attn_mask[
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
                : nb_tokens_per_timestep * (i + 1),
            ] = True

        self.register_buffer("attn_mask", attn_mask)

    def build_inference_attention_mask(self, context_length: int, device: torch.device | str) -> BoolTensor:
        """
        Builds the inference mask: causal over visual tokens, and one action chunk that sees everything.

        Args:
            context_length: number of context frames.
            device: device of the mask.

        Returns:
            the boolean mask, True where attention is allowed.
        """
        visual_seqlen = self.nb_tokens_per_timestep * context_length
        action_seq_len = self.action_horizon
        seqlen = visual_seqlen + action_seq_len
        attn_mask = torch.zeros(seqlen, seqlen, dtype=torch.bool, device=device)

        for i in range(visual_seqlen):
            attn_mask[i, : i + 1] = True

        attn_mask[visual_seqlen:] = True

        return attn_mask

    def psi_t(
        self,
        x: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Interpolates between noise and target actions along the conditional flow path.

        Args:
            x: noise shaped [batch, context, horizon, action_dim].
            x1: target actions with the same shape.
            t: flow time shaped [batch, context].

        Returns:
            the noisy actions at time t.
        """
        t = t[:, :, None, None]
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        actions: Tensor,
        t: Tensor,
        return_gpt: bool = False,
    ) -> Tensor:
        """
        Computes the flow-matching loss of the action expert.

        Args:
            visual_tokens: token ids shaped [batch, context, height, width].
            high_level_command: command ids shaped [batch, context].
            actions: target actions shaped [batch, context, horizon, action_dim].
            t: flow time shaped [batch, context].
            return_gpt: whether the joint model also computes the GPT2 outputs.

        Returns:
            the mean squared error between predicted and true velocities.
        """
        x0 = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        x1 = actions / self.action_scaling
        psi_t = self.psi_t(x0, x1, t.type_as(x1))

        self.joint_model.train()
        v_psi = self.joint_model(
            attention_mask=self.attn_mask,
            inputs_all={
                "visual_tokens": visual_tokens,
                "noisy_actions": psi_t,
                "high_level_command": high_level_command,
                "diffusion_step": t,
            },
            return_gpt=return_gpt,
        )["actions"]

        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)

    def forward_inference(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        dtype: torch.dtype,
        verbose: bool = False,
        num_inference_steps: Optional[int] = None,
        final_action_clip_value: Optional[float] = None,
        store_intermediate_states: bool = False,
    ) -> Tensor:
        """
        Samples the actions of the last context frame by Euler integration of the flow from random noise.

        Args:
            visual_tokens: token ids shaped [batch, context, height, width].
            high_level_command: command ids shaped [batch, 1].
            dtype: dtype of the noise and flow time.
            verbose: whether to show a progress bar.
            num_inference_steps: number of Euler steps; None uses the model default.
            final_action_clip_value: bound on the actions before rescaling; None uses the model default.
            store_intermediate_states: whether to also return the actions after every step.

        Returns:
            the actions shaped [batch, 1, horizon, action_dim], plus the per-step actions when asked.
        """
        device = visual_tokens.device
        bsz, context_length, *_ = visual_tokens.size()
        assert context_length <= self.context_length
        num_inference_steps = num_inference_steps or self.num_inference_steps
        final_action_clip_value = final_action_clip_value or self.final_action_clip_value

        def _post_process_traj(action: Tensor) -> Tensor:
            """Clips the actions when configured and undoes the action scaling."""
            if final_action_clip_value is not None:
                action = torch.clamp(
                    action,
                    -final_action_clip_value,
                    final_action_clip_value,
                )
            return action * self.action_scaling

        action = torch.randn((bsz, 1, self.action_horizon, self.action_dim), device=device, dtype=dtype)
        intermediate_states = []

        attn_mask = self.build_inference_attention_mask(context_length, device=device)

        self.joint_model.eval()
        self.joint_model.init_kv_cache()

        delta_t = 1.0 / num_inference_steps
        t = torch.zeros((bsz, 1), device=device, dtype=dtype)
        for _ in tqdm(range(num_inference_steps), "Euler integration", disable=not verbose):
            action_vel = self.joint_model(
                attention_mask=attn_mask,
                inputs_all={
                    "visual_tokens": visual_tokens,
                    "noisy_actions": action,
                    "high_level_command": high_level_command,
                    "diffusion_step": t,
                },
            )["actions"]
            action += delta_t * action_vel
            t += delta_t

            if store_intermediate_states:
                intermediate_states.append(_post_process_traj(action.clone()))

        self.joint_model.cleanup_kv_cache()

        if store_intermediate_states:
            return _post_process_traj(action), intermediate_states

        return _post_process_traj(action)


class VideoActionModelInference(VideoActionModel):
    """VideoActionModel whose forward samples actions instead of computing the loss."""

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Samples actions, see forward_inference.

        Args:
            *args: positional arguments of forward_inference.
            **kwargs: keyword arguments of forward_inference.

        Returns:
            the sampled actions.
        """
        return super().forward_inference(*args, **kwargs)


def load_inference_VAM(
    checkpoint_path: str, device: torch.device | str = "cuda", tempdir: Optional[str] = None,
) -> VideoActionModelInference:
    """
    Loads a frozen VideoActionModelInference in eval mode from a Lightning or DeepSpeed checkpoint.

    Args:
        checkpoint_path: checkpoint file, or a DeepSpeed checkpoint directory.
        device: device to move the model to.
        tempdir: parent directory for the temporary fused DeepSpeed checkpoint.

    Returns:
        the loaded model.
    """
    if os.path.isdir(checkpoint_path):
        # A directory is a DeepSpeed checkpoint.
        import tempfile

        from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

        with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
            ckpt = convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_path,
                os.path.join(tmpdirname, "fused_ckpt.pt"),
            )
    else:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            pickle_module=checkpoint_pickle_module,
        )

    config = ckpt["hyper_parameters"]["vam_conf"].copy()
    config.pop("_target_")
    config.pop("_recursive_")
    config.gpt_checkpoint_path = None
    config.action_checkpoint_path = None
    config.gpt_mup_base_shapes = None
    config.action_mup_base_shapes = None
    vam = VideoActionModelInference(**config)
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("vam.", "")] = v

    vam.load_state_dict(state_dict)
    set_base_shapes(vam.gpt, ckpt["gpt_mup_base_shapes"], rescale_params=False)
    set_base_shapes(vam.action_expert, ckpt["action_mup_base_shapess"], rescale_params=False)

    vam.requires_grad_(False)
    vam.eval()
    vam.to(device)
    return vam
