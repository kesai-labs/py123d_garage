#!/bin/bash
# Installs CARLA 0.9.15 with its additional maps into the given directory (default: lib/carla/simulator/0915).
set -e

_target="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/simulator/0915}"
mkdir -p "${_target}/Import"
cd "${_target}"

_bucket=https://carla-releases.s3.us-east-005.backblazeb2.com/Linux
wget -O CARLA_0915.tar.gz "${_bucket}/CARLA_0.9.15.tar.gz"
tar -xzf CARLA_0915.tar.gz
wget -O Import/AdditionalMaps_0.9.15.tar.gz "${_bucket}/AdditionalMaps_0.9.15.tar.gz"
# ImportAssets.sh extracts every tarball in Import/ into this directory with `tar --keep-newer-files`,
# which exits 2 over a tree a cancelled run left behind. The town check below is the real test.
bash ImportAssets.sh || true
test -d CarlaUE4/Content/Carla/Maps/Town12
