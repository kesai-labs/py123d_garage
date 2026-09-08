#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Log in to the challenge in the browser and store the token, which expires after 12 hours
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py auth-url
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py configure-token
uv run --project lib/alpasim/alpasim lib/alpasim/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py me
