# Turicreate Setup Guide

> **Note:** Turicreate has been archived by Apple. PicoTuri-EditJudge ships with a default histogram-based similarity fallback, so installing Turicreate is optional. Use this guide only if you need the legacy feature parity or want to reproduce historical experiments.

Turicreate powers the image-similarity signal used in PicoTuri-EditJudge. Because the project relies on a macOS-only binary, we install it from source inside a Python 3.8 x86_64 environment (Rosetta on Apple Silicon).

## Prerequisites

- macOS 12 or later.
- Xcode command line tools: `xcode-select --install`.
- Python 3.8 interpreter (Rosetta build recommended for Apple Silicon).
- Virtual environment or conda environment created with `CONDA_SUBDIR=osx-64`.

## Installation Steps

1. Activate your Python 3.8 x86_64 environment.
2. From the repo root, run:
   ```bash
   scripts/setup_turicreate.sh --branch main
   ```
   This clones `https://github.com/apple/turicreate` into `third_party/turicreate/` and installs it in editable mode.
3. Verify the installation:
   ```bash
   python -c "import turicreate as tc; print(tc.__version__)"
   ```

## Upgrading

- Re-run `scripts/setup_turicreate.sh` to fetch the latest changes.
- Inspect Turicreate release notes for API changes, then re-run the smoke tests:
  ```bash
  pytest tests/test_smoke.py -k train_baseline
  ```

## Troubleshooting

- **`ImportError: dlopen(...)`** – Ensure your shell is running in Rosetta mode (`arch -x86_64 zsh`).
- **Compiler errors (`clang: error:`)** – Install Xcode command line tools and reboot your terminal session.
- **Wheel install failures** – Installing from source avoids the deprecated binary wheels. Ensure you remove any prior `pip install turicreate` attempts before re-running the script.
