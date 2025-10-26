#!/usr/bin/env bash
set -euo pipefail

# Clone or update the Pico-Banana-400K metadata repository.
# The repo contains manifests and download helpers that fetch the full dataset
# from Apple's public CDN. Images/videos are NOT stored in the Git repository.
#
# Usage:
#   scripts/setup_pico_banana.sh [--branch main]
#
# After cloning, read the upstream README for instructions on downloading
# assets. Place downloaded files under data/pico_banana/ (ignored by git).

BRANCH="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      shift
      BRANCH="${1:-main}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift || true
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PB_ROOT="${REPO_ROOT}/third_party/pico-banana-400k"

if [[ -d "${PB_ROOT}/.git" ]]; then
  echo "Updating existing pico-banana-400k checkout..."
  git -C "${PB_ROOT}" fetch origin
  git -C "${PB_ROOT}" checkout "${BRANCH}"
  git -C "${PB_ROOT}" pull --ff-only origin "${BRANCH}"
else
  echo "Cloning pico-banana-400k..."
  git clone --depth 1 --branch "${BRANCH}" https://github.com/apple/pico-banana-400k.git "${PB_ROOT}"
fi

echo "Pico-Banana metadata repository is ready at ${PB_ROOT}."
echo "Follow the upstream README to download media assets to data/pico_banana/."
