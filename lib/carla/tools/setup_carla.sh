#!/bin/bash
# Clones the leaderboards into lib/carla/benchmark at pinned commits and applies our patches (see the header of each),
# then installs CARLA 0.9.15 with its additional maps into the given directory (default: lib/carla/simulator/0915).
set -e

_carla_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_target="${1:-${_carla_dir}/simulator/0915}"

# Usage: _clone <url> <commit> <dir under benchmark/> [patch]; a patch already in place stays.
_clone() {
  local _dir="${_carla_dir}/benchmark/$3"
  if [ ! -d "${_dir}/.git" ]; then
    git clone --quiet "$1" "${_dir}"
  fi
  git -C "${_dir}" fetch --quiet origin
  git -C "${_dir}" checkout --quiet --detach "$2"
  git -C "${_dir}" --no-pager log -1 --format='%h %ad %s' --date=short
  if [ -n "$4" ] && ! git -C "${_dir}" apply --check --reverse "$4" 2>/dev/null; then
    git -C "${_dir}" apply "$4"
  fi
}

_clone https://github.com/Thinklab-SJTU/Bench2Drive.git 7ec25d1c9f7522d923ce5f3420986cef1cb2d956 bench2drive \
  "${_carla_dir}/tools/patches/bench2drive.patch"
_clone https://github.com/carla-simulator/leaderboard.git aec81319de6bec57372f25aa94083c9ab0bd4223 standard/leaderboard
_clone https://github.com/carla-simulator/scenario_runner.git 94ff3b8af752bad2b9d464ad5105868906aa34c0 \
  standard/scenario_runner
# The leaderboard imports every module in srunner/scenarios. The OpenSCENARIO ones drive no route, and osc2 needs
# antlr4 4.10 where omegaconf pins 4.9.
rm -f "${_carla_dir}"/benchmark/standard/scenario_runner/srunner/scenarios/{open_scenario,osc2_scenario}.py

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
