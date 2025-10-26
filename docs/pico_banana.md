# Working with Pico-Banana-400K

Pico-Banana-400K provides instruction-driven image edits paired with quality annotations. The dataset is released under **CC BY-NC-ND 4.0**, so review and accept the license before downloading any artifacts.

## Official Repository

- **Dataset Repository**: https://github.com/apple-ondevice-research/pico-banana-400k

- **Documentation**: See the dataset repository for detailed information and usage guidelines

## Access

1. Clone the public metadata repository:
   ```bash
   scripts/setup_pico_banana.sh --branch main
   ```
   This pulls manifests and download helpers into `third_party/pico-banana-400k/`.
2. Review `third_party/pico-banana-400k/README.md` and run the provided scripts to fetch media from Apple's CDN. Downloaded images/videos should live under `data/pico_banana/` (ignored by git).

## Converting to PicoTuri pairs

Use the enhanced `scripts/make_pairs.py` utility to normalize Pico-Banana records into the `pair_id,instruction,original_image,edited_image,label` schema consumed by the baseline pipeline. Manifests shipped in the GitHub repository are under `third_party/pico-banana-400k/manifests/`.

### From an existing manifest

```bash
python scripts/make_pairs.py \
  --input third_party/pico-banana-400k/manifests/train_manifest.jsonl \
  --output data/manifests/pico_banana_pairs.csv \
  --sample-size 1000 \
  --overwrite
```



Notes:

- Images are materialized as PNG files beneath `data/pico_banana/images`. Ensure you have sufficient disk space.
- Override column names with `--instruction-key`, `--original-key`, `--edited-key`, `--label-key` if the dataset uses alternative field names.
- The script requires `datasets`, `Pillow`, and `tqdm` (included in `requirements-dev.txt`).

## Computing similarity scores

Once the dataset is materialized locally, you can recompute image similarity signals during training. The pipeline will use Turicreate if installed, otherwise it falls back to the built-in histogram similarity:

```bash
python -m src.train.baseline \
  --pairs data/manifests/pico_banana_pairs.csv \
  --model-path artifacts/pico_banana_baseline.joblib \
  --compute-similarity \
  --image-root "$(pwd)/data/pico_banana/images" \
  --seed 23
```

Similarity routines expect local file paths, so keep the manifest paths relative to `--image-root` or absolute on disk.

## Best Practices

- Do not commit raw Pico-Banana images or large manifests to the repository.
- Store metadata under `data/manifests/` (≤50 rows) for reproducible samples.
- Track dataset versions and splits in `configs/models/model.yaml` when exporting models.
- Respect the non-commercial license and document downstream uses in your model card.
