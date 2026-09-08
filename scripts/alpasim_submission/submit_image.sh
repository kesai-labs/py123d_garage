#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the tag to submit under. The challenge rejects a reused tag, so VERSION makes every submission unique.
_tag="${TAG:-dev}-${VERSION:?set VERSION, e.g. v1}"

# Look up the team's registry namespace, which needs the login from login_challenge.sh
_team_id="$(uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py me | python3 -c 'import json, sys; print(json.load(sys.stdin)["registration"]["team_id"])')"
_uri="696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/${_team_id}:${_tag}"

# Push the image under its versioned tag and submit it to the track
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
docker tag "py123d-garage-alpasim:${TAG:-dev}" "${_uri}"
docker push "${_uri}"
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py submit --track "${TRACK:-pai}" "${_uri}"
