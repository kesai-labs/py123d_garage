#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the checkpoint and the sensor rig baked into the image, and the image tag
_checkpoint_file="$(realpath "${CHECKPOINT_FILE:-outputs/checkpoints/pai_latent_transfuser/model.pth}")"
_sensor_rig_file="$(realpath "${SENSOR_RIG_FILE:-$(dirname "${_checkpoint_file}")/sensor_rig_0.yaml}")"
_tag="${TAG:-dev}"

# Build the alpasim gRPC wheel the image installs
_wheels="$(mktemp -d)"
trap 'rm -rf "${_wheels}"' EXIT
(cd lib/alpasim/alpasim/src/grpc && uv build --wheel --out-dir "${_wheels}")

# Build the image py123d-garage-alpasim:TAG from this checkout, the checkpoint and the rig
docker buildx build \
	-f lib/alpasim/tools/submission.Dockerfile \
	--build-context run="$(dirname "${_checkpoint_file}")" \
	--build-context rig="$(dirname "${_sensor_rig_file}")" \
	--build-context wheels="${_wheels}" \
	--build-arg CHECKPOINT_FILE="$(basename "${_checkpoint_file}")" \
	--build-arg SENSOR_RIG_FILE="$(basename "${_sensor_rig_file}")" \
	--build-arg GIT_HASH="$(git rev-parse --short HEAD)" \
	-t "py123d-garage-alpasim:${_tag}" \
	.
