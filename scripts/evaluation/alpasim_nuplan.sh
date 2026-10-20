#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# The nuPlan track drives MTGS scenes under ALPASIM_NUPLAN_ROOT with the rig the run wrote for its nuPlan sources.
# CAMERAS=nuplan_1cam renders only the front camera, enough for a policy that reads no other.
_preset="+e2e_challenge_nuplan=dev +cameras=${CAMERAS:-nuplan_3cam}"
_scene_ids="[2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78]"
_sensor_rig_dataset=nuplan

# Declare input and output directories relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }
_checkpoint_file="$(realpath "${_checkpoint_file}")"
_sensor_rig_file="$(realpath "${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_file}")/sensor_rig_${_sensor_rig_dataset}.yaml}")"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/alpasim/nuplan/$(date +%F_%H-%M-%S)}")"

# Declare the port for the driver to listen on, defaulting to 6789 if not set.
export ALPASIM_DRIVER_PORT="${ALPASIM_DRIVER_PORT:-6789}"

# Start the driver in the background; DRIVER_OVERRIDES holds space-separated driver settings, e.g. "route_memory=true".
# shellcheck disable=SC2086
uv run --project lib/alpasim/alpasim python -m py123d_garage.evaluation.alpasim.evaluate \
	hydra.run.dir="${_out_dir}/driver" \
	policy_config.evaluation_checkpoint_files="[${_checkpoint_file}]" \
	policy_config.evaluation_sensor_rig_file="${_sensor_rig_file}" \
	port="${ALPASIM_DRIVER_PORT}" \
	${DRIVER_OVERRIDES:-} &

# Stop the driver when the wizard is done.
_driver_pid=$!
trap 'kill "${_driver_pid}" 2>/dev/null || true' EXIT

# Run the evaluation wizard, which will connect to the driver and run the simulation.
# shellcheck disable=SC2086
uv run --project lib/alpasim/alpasim alpasim_wizard ${PRESET:-${_preset}} \
	driver=garage_transfuser driver_source=garage_host \
	driver.model.checkpoint_path="${_checkpoint_file}" \
	driver.output_dir="${_out_dir}/alpasim_driver" \
	scenes.scene_ids="${SCENE_IDS:-${_scene_ids}}" \
	wizard.log_dir="${_out_dir}" \
	"$@"
