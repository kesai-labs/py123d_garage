#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# The Physical-AI-AV track drives NuRec scenes, rendered without the harmonizer like the challenge, with the rig the
# run wrote for its NuRec sources. The README's numbers render with the harmonizer, as AlpaSim's Alpamayo 1 reference
# run does: PRESET="deploy=local topology=1gpu +garage_renderer=nre_harmonized".
_preset="deploy=local topology=1gpu +garage_renderer=nre"
_scene_ids="[clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6]"
_sensor_rig_dataset=nurec

# Declare input and output directories relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }
_checkpoint_file="$(realpath "${_checkpoint_file}")"
_sensor_rig_file="$(realpath "${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_file}")/sensor_rig_${_sensor_rig_dataset}.yaml}")"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/alpasim/pai/$(date +%F_%H-%M-%S)}")"

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
