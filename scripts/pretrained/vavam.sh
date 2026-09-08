#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Assemble the checkpoint in the layout of the published checkpoints: model.pth, config.yaml, sensor_rig_0.yaml
_out_dir="outputs/checkpoints/vavam"
_release="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0"
mkdir -p "${_out_dir}"

# Download the video action model and its tokenizer from the VaVAM release
curl -L --fail --continue-at - -o "${_out_dir}/VAM_width_1024_pretrained_139k.pt" \
	"${_release}/VAM_width_1024_pretrained_139k.pt"
curl -L --fail --continue-at - -o "${_out_dir}/VQ_ds16_16384_llamagen_encoder.jit" \
	"${_release}/VQ_ds16_16384_llamagen_encoder.jit"

# Bundle the model and tokenizer into one model.pth, the file the VaVAM policy loads, and drop the downloads
python3 - "${_out_dir}" <<'EOF'
import sys
import zipfile
from pathlib import Path

out_dir = Path(sys.argv[1])
with zipfile.ZipFile(out_dir / "model.pth", "w") as bundle:
    bundle.write(out_dir / "VAM_width_1024_pretrained_139k.pt", "model.pt")
    bundle.write(out_dir / "VQ_ds16_16384_llamagen_encoder.jit", "tokenizer.jit")
EOF
rm "${_out_dir}/VAM_width_1024_pretrained_139k.pt" "${_out_dir}/VQ_ds16_16384_llamagen_encoder.jit"

# Write the policy config
cat >"${_out_dir}/config.yaml" <<EOF
policy_config:
  target: py123d_garage.policy.vavam.vavam_policy:VavamPolicy
  vavam_config: {}
EOF

# Write the sensor rig
cat >"${_out_dir}/sensor_rig_0.yaml" <<EOF
modalities:
  camera.ftcam_f0:
    camera_model: ftheta
    camera_name: camera_front_wide_120fov
    camera_id: 12
    intrinsics:
    - 961.27124
    - 753.274353
    - 0.0
    - 931.7612940673102
    - 32.987987112086635
    - -72.82956582342041
    - 22.720089068156213
    - 0.0
    - 0.0
    - 0.0010741659390590083
    - -4.5748863088807144e-08
    - 1.0254764446258745e-10
    - -3.2366948806555124e-14
    - 0.0
    width: 1920
    height: 1080
    camera_to_imu_se3:
    - 1.801815152168274
    - -0.042615413665771484
    - 1.2806285619735718
    - -0.49909161198285273
    - 0.4954226054863875
    - -0.5023466779326323
    - 0.5031021963691722
EOF
