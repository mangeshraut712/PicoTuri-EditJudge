# Data scripts

- `make_pairs.py` – normalize manifests or Hugging Face datasets (including Pico-Banana-400K) into the baseline `(instruction, original, edited, label)` CSV format.
- `setup_turicreate.sh` – clone and install Turicreate from source (Rosetta environment).
- `setup_pico_banana.sh` – clone the Pico-Banana metadata repository so you can download assets from Apple's CDN.
- Additional helpers (dataset sync, schema validation) live here as they are added.

Contributors should keep large datasets out of the repository and respect the CC BY-NC-ND 4.0 terms.
