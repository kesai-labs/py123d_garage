#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

# Train pretrain, midtrain and posttrain on all the training data, one after the other, into
# outputs/training/all/<time>/<stage>. Arguments like key=value override that setting in every stage.
# To start at a later stage, name the stages to train and the finished run to start from:
#   STAGES="midtrain posttrain" PRETRAIN_RUN_DIR=<pretrain dir> scripts/training/train_all.sh
#   STAGES=posttrain MIDTRAIN_RUN_DIR=<midtrain dir> scripts/training/train_all.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00

set -euo pipefail

# Important environment variables for py123d_garage for good training performance
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"
export PY123D_GARAGE_RUNTIME_TYPE_CHECKING=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup batch size per GPU
_global_batch_size=256
_num_nodes="${SLURM_JOB_NUM_NODES:-1}"
_gpus_per_node="${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

# On slurm, use srun to launch the training script, otherwise just use python
_launch=(python)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	_launch=(srun --kill-on-bad-exit=1 python)
fi

# Optional speed-ups with the same math, uncomment to use; the py123d_garage_cuda patches need the kernels built:
# pip install --no-build-isolation ./cuda
_speed_overrides=(
	# Copy the permanent cache to a fast node-local disk on first read; the stages share the copy
	# cache_mirror_root=/scratch
	# compile_mode=max-autotune
	# compile_ahead=true
	# cudnn_attention=true
	# symmetric_gradient_memory=true
	# "policy_config.transfuser_config.backbone_config.kernel_patches=['py123d_garage_cuda.transfuser.resnet:fuse_stem','py123d_garage.policy.transfuser.network.encoder.transfuser:share_latent_bev_stem','py123d_garage_cuda.transfuser.resnet:pin_conv_engines','py123d_garage_cuda.transfuser.linear_convs:patch_pointwise_convs','py123d_garage.policy.transfuser.network.encoder.transfuser:fuse_fusion_qkv','py123d_garage_cuda.transfuser.fusion:patch_fusion_upsample_add']"
	# "policy_config.transfuser_config.bev_semantic_config.kernel_patches=['py123d_garage_cuda.transfuser.resnet:pin_conv_engines','py123d_garage_cuda.transfuser.linear_convs:patch_pointwise_convs']"
	# "policy_config.transfuser_config.box_detection_config.kernel_patches=['py123d_garage_cuda.transfuser.resnet:pin_conv_engines','py123d_garage_cuda.transfuser.linear_convs:patch_pointwise_convs']"
	# Applies to midtrain and posttrain only, since pretrain has no motion planner
	# "policy_config.transfuser_config.planning_config.kernel_patches=['py123d_garage_cuda.transfuser.planner:patch_planner_attention','py123d_garage_cuda.transfuser.planner:patch_planner_feed_forward','py123d_garage_cuda.transfuser.adaptive_norm:patch_planner_adaptive_norm']"
)

_run_dir="outputs/training/all/$(date +%Y.%m.%d.%H.%M.%S)"
_stages=" ${STAGES:-pretrain midtrain posttrain} "
_pretrain_run_dir="${PRETRAIN_RUN_DIR:-${_run_dir}/pretrain}"
_midtrain_run_dir="${MIDTRAIN_RUN_DIR:-${_run_dir}/midtrain}"

# Train one stage of the recipe into <run dir>/<stage>
# Usage: _train <stage> [overrides...]
_stage_overrides=()
_train() {
	local stage="$1"
	shift
	_check_kernels
	"${_launch[@]}" -m py123d_garage.training.train \
		training="py123d_garage.config.presets.training.latent_transfuser:${stage}" \
		hydra.run.dir="${_run_dir}/${stage}" \
		lightning_trainer_config.devices="${_gpus_per_node}" \
		lightning_trainer_config.num_nodes="${_num_nodes}" \
		dataloader_config.batch_size="$((_global_batch_size / (_gpus_per_node * _num_nodes)))" \
		dataloader_config.num_workers="${SLURM_CPUS_PER_TASK:-16}" \
		"$@" \
		"${_stage_overrides[@]}" \
		"${_speed_overrides[@]}" \
		"${_overrides[@]}"
	# The later stages read the same cache stores and a subset of the scenes, which the first stage checked
	_stage_overrides=(check_cache_coverage=false)
}
_overrides=("$@")

# Run the kernel tests on one GPU before every stage that uses the kernel patches; a failing suite ends the run here
_check_kernels() {
	[[ "${_speed_overrides[*]} ${_overrides[*]}" == *kernel_patches=* ]] || return 0
	python -c 'import py123d_garage_cuda'
	local test=(python -m pytest tests/cuda_unit -q)
	[[ -z "${SLURM_JOB_ID:-}" ]] || test=(srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 "${test[@]}")
	"${test[@]}"
}

# Every seed's weights after the last epoch of a finished stage
# Usage: _weights <run dir>
_weights() {
	local seed_0_files=("$1"/model_0_*.pth)
	local epoch="${seed_0_files[-1]##*_}"
	epoch="${epoch%.pth}"
	echo "initial_weights_files=[$1/model_0_${epoch}.pth,$1/model_1_${epoch}.pth,$1/model_2_${epoch}.pth]"
}

# While a stage trains, fill the next stage's session cache on the CPUs the training leaves idle, so its first epoch
# runs at full speed; without cache_mirror_root there is no session cache and this does nothing
# Usage: _fill <stage>
_fill_pid=""
_fill() {
	nice -n 19 python -m py123d_garage.training.fill_session_cache \
		training="py123d_garage.config.presets.training.latent_transfuser:${1}" \
		hydra.run.dir="${_run_dir}/${1}/session_cache_fill" \
		check_cache_coverage=false \
		"${_speed_overrides[@]}" \
		"${_overrides[@]}" &
	_fill_pid=$!
}
_wait_for_fill() {
	if [[ -n "${_fill_pid}" ]]; then
		wait "${_fill_pid}" || echo "The session cache fill failed; the stage fills its session cache itself" >&2
		_fill_pid=""
	fi
}
trap '[[ -z "${_fill_pid}" ]] || kill "${_fill_pid}" 2>/dev/null || true' EXIT

# Stage 1 of 3: robust perception features
if [[ "${_stages}" == *" pretrain "* ]]; then
	[[ "${_stages}" != *" midtrain "* ]] || _fill midtrain
	_train pretrain
fi
# Stage 2 of 3: general driving capability, from each seed's pretrain weights
if [[ "${_stages}" == *" midtrain "* ]]; then
	_wait_for_fill
	[[ "${_stages}" != *" posttrain "* ]] || _fill posttrain
	_train midtrain "$(_weights "${_pretrain_run_dir}")"
fi
# Stage 3 of 3: instruction following, from each seed's midtrain weights. The planner kernels stay, the backbone and
# head kernels go: on the frozen backbone the box head's gradients drift past the kernel check with them together.
if [[ "${_stages}" == *" posttrain "* ]]; then
	_wait_for_fill
	_planner_only=()
	for _override in "${_speed_overrides[@]}"; do
		[[ "${_override}" == *_config.kernel_patches=* && "${_override}" != *planning_config.kernel_patches=* ]] \
			|| _planner_only+=("${_override}")
	done
	_speed_overrides=("${_planner_only[@]}")
	_train posttrain "$(_weights "${_midtrain_run_dir}")"
fi
