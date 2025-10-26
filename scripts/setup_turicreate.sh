#!/usr/bin/env bash
set -euo pipefail

# This script clones the official Turicreate repository and installs it in the
# current Python environment. Run it from the repo root after activating a
# Python 3.8 x86_64 (Rosetta) environment.
#
# Usage:
#   scripts/setup_turicreate.sh [--branch main]

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
TC_ROOT="${REPO_ROOT}/third_party/turicreate"

if [[ -d "${TC_ROOT}/.git" ]]; then
  echo "Updating existing Turicreate checkout..."
  git -C "${TC_ROOT}" fetch origin
  git -C "${TC_ROOT}" checkout "${BRANCH}"
  git -C "${TC_ROOT}" pull --ff-only origin "${BRANCH}"
else
  echo "Cloning Turicreate..."
  git clone --depth 1 --branch "${BRANCH}" https://github.com/apple/turicreate.git "${TC_ROOT}"
fi

echo "Installing Turicreate in editable mode..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "${TC_ROOT}/requirements.txt"
python3 -m pip install -e "${TC_ROOT}[visualization]"

echo "Turicreate setup complete."
