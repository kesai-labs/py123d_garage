#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the checkpoints and the sensor rig baked into the image, the image tag, and the driver settings it bakes in.
# DRIVER_OVERRIDES holds space-separated driver settings, e.g. "route_memory=true extrapolate_route=true".
# CHECKPOINT_FILES holds several space-separated checkpoints of one training recipe, driven as an ensemble.
# The checkpoint defaults to seed 0 of the newest posttrain run, and the rig to the one that run wrote for the TRACK's
# dataset: NuRec for pai, nuPlan for nuplan.
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
read -ra _checkpoint_files <<<"${CHECKPOINT_FILES:-${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}}"
_sensor_rig_dataset=nurec
[[ "${TRACK:-pai}" == nuplan ]] && _sensor_rig_dataset=nuplan
_sensor_rig_file="$(realpath "${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_files[0]}")/sensor_rig_${_sensor_rig_dataset}.yaml}")"
_tag="${TAG:-dev}"

# Stage the checkpoints as model_<i>.pth next to the first one's config.yaml, and build the alpasim gRPC wheel the image installs
_stage="$(mktemp -d)"
_wheels="$(mktemp -d)"
trap 'rm -rf "${_stage}" "${_wheels}"' EXIT
_image_checkpoint_files=()
for _index in "${!_checkpoint_files[@]}"; do
	cp "$(realpath "${_checkpoint_files[${_index}]}")" "${_stage}/model_${_index}.pth"
	_image_checkpoint_files+=("/opt/checkpoint/model_${_index}.pth")
done
cp "$(dirname "$(realpath "${_checkpoint_files[0]}")")/config.yaml" "${_stage}/config.yaml"
(cd lib/alpasim/alpasim/src/grpc && uv build --wheel --out-dir "${_wheels}")

# Build the image py123d-garage-ltfv6:TAG from this checkout, the checkpoints and the rig
docker buildx build \
	-f lib/alpasim/tools/submission.Dockerfile \
	--build-context run="${_stage}" \
	--build-context rig="$(dirname "${_sensor_rig_file}")" \
	--build-context wheels="${_wheels}" \
	--build-arg CHECKPOINT_FILES="[$(IFS=,; echo "${_image_checkpoint_files[*]}")]" \
	--build-arg SENSOR_RIG_FILE="$(basename "${_sensor_rig_file}")" \
	--build-arg GIT_HASH="$(git rev-parse --short HEAD)" \
	--build-arg DRIVER_OVERRIDES="${DRIVER_OVERRIDES:-}" \
	--label org.opencontainers.image.source=https://github.com/kesai-labs/py123d_garage \
	-t "py123d-garage-ltfv6:${_tag}" \
	.
