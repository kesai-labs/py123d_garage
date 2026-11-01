#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# HUGSIM drives the scenarios of the tree that the HUGSIM checkout's prepare_benchmark.py built; 16 of the nuScenes
# scenarios also read the HD map of the raw nuScenes under NUSCENES_RAW, the directory with v1.0-trainval and maps.
: "${HUGSIM_DATA:?set HUGSIM_DATA to the tree prepare_benchmark.py built}"
: "${NUSCENES_RAW:?set NUSCENES_RAW to the raw nuScenes}"

# Declare input and output directories relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/hugsim/$(date +%F_%H-%M-%S)}")"
# GPU indices as nvidia-smi lists them, and episodes at once on each of them.
_gpus="${GPUS:-0}"
_slots=()
for _gpu in ${_gpus//,/ }; do
	_slots+=("${_gpu}:${RUNS_PER_GPU:-1}")
done

# The launcher runs the policy with this checkpoint and python, next to the checkout of lib/hugsim/tools/setup_hugsim.sh.
CHECKPOINT_FILE="$(realpath "${_checkpoint_file}")"
PY123D_GARAGE_PYTHON="$(command -v python)"
HUGSIM_ROOT="${PWD}/lib/hugsim/hugsim"
HUGSIM_AD_PY123D_GARAGE="${PWD}/src/py123d_garage/evaluation/hugsim/launch.sh"
export CHECKPOINT_FILE PY123D_GARAGE_PYTHON HUGSIM_ROOT HUGSIM_AD_PY123D_GARAGE
_hugsim_python="${HUGSIM_ROOT}/.pixi/envs/default/bin/python"

# Run the first LIMIT scenarios of each of the four datasets (0: all) of the scenario set SET, the 436 of the current
# release (official436) or the 345 of the pre-challenge release (prechallenge345), rendering the cameras only, then print
# the scores.
_set="${SET:-official436}"
cd "${HUGSIM_ROOT}"
"${_hugsim_python}" run_benchmark.py \
	--agents py123d_garage \
	--set "${_set}" \
	--slots "${_slots[@]}" \
	--limit "${LIMIT:-1}" \
	--data "${HUGSIM_DATA}" \
	--out "${_out_dir}" \
	-- --obs_channels rgb "$@"
"${_hugsim_python}" aggregate_benchmark.py --agents py123d_garage --set "${_set}" --out-root "${_out_dir}" \
	--data "${HUGSIM_DATA}" --json "${_out_dir}/summary.json"
