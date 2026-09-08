#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the image, the track (pai or nuplan, the latter needs ALPASIM_NUPLAN_ROOT) and where the run writes
_image="py123d-garage-alpasim:${TAG:-dev}"
_container="${CONTAINER_NAME:-py123d-garage-alpasim}"
_preset="+e2e_challenge=dev"
[[ "${TRACK:-pai}" == nuplan ]] && _preset="+e2e_challenge_nuplan=dev"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/alpasim_submission/smoke_test_image/$(date +%F_%H-%M-%S)}")"
mkdir -p "${_out_dir}/driver"
export ALPASIM_DRIVER_PORT="${ALPASIM_DRIVER_PORT:-6789}"

# Run the image as the driver in the background
docker run --rm --gpus all --name "${_container}" -p "${ALPASIM_DRIVER_PORT}:6789" \
	-v "${_out_dir}/driver:/tmp/alpasim_driver" "${_image}" >"${_out_dir}/driver.log" 2>&1 &

# Stop the driver when the wizard is done.
trap 'docker rm -f "${_container}" >/dev/null 2>&1 || true' EXIT

# Start the simulator
uv run --project lib/alpasim/alpasim alpasim_wizard "${_preset}" wizard.log_dir="${_out_dir}" "$@"
