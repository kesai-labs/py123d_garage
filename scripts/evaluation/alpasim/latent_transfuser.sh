#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Declare input and output directories relative to py123d_garage
_checkpoint_file="$(realpath "${CHECKPOINT_FILE:-outputs/checkpoints/pai_latent_transfuser/model.pth}")"
_sensor_rig_file="$(realpath "${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_file}")/sensor_rig_0.yaml}")"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/alpasim/latent_transfuser/$(date +%F_%H-%M-%S)}")"

# Declare the port for the driver to listen on, defaulting to 6789 if not set.
export ALPASIM_DRIVER_PORT="${ALPASIM_DRIVER_PORT:-6789}"

# Start the driver in the background.
uv run --project lib/alpasim/alpasim python -m py123d_garage.evaluation.alpasim.evaluate \
	hydra.run.dir="${_out_dir}/driver" \
	policy_config.evaluation_checkpoint_file="${_checkpoint_file}" \
	policy_config.evaluation_sensor_rig_file="${_sensor_rig_file}" \
	port="${ALPASIM_DRIVER_PORT}" &

# Stop the driver when the wizard is done.
_driver_pid=$!
trap 'kill "${_driver_pid}" 2>/dev/null || true' EXIT

# Run the evaluation wizard, which will connect to the driver and run the simulation.
# shellcheck disable=SC2086
uv run --project lib/alpasim/alpasim alpasim_wizard ${PRESET:-deploy=local topology=1gpu} \
	driver=garage_transfuser driver_source=garage_host \
	driver.model.checkpoint_path="${_checkpoint_file}" \
	driver.output_dir="${_out_dir}/alpasim_driver" \
	scenes.scene_ids="${SCENE_IDS:-[clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6]}" \
	wizard.log_dir="${_out_dir}" \
	"$@"
