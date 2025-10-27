#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for building a lightweight (original, instruction, edited, label) CSV
compatible with PicoTuri-EditJudge.

Notes:
    • This script never downloads media. It only reshapes manifest metadata.
    • The Pico-Banana-400K dataset is released under CC BY-NC-ND 4.0 – ensure
      you accepted the terms before mirroring manifests locally.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, cast

import pandas as pd  # type: ignore[import]

OUTPUT_COLUMNS = [
    "pair_id",
    "instruction",
    "original_image",
    "edited_image",
    "image_similarity",
    "label",
    "notes",
]


def read_manifest(path: Path) -> pd.DataFrame:
    """Load a manifest from CSV, JSON, or JSONL into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) if suffix == ".json" else [json.loads(line) for line in handle]
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported manifest format: {path}")

    logging.debug("Loaded %d records from %s", len(df), path)
    return df


def normalize_records(
    df: pd.DataFrame,
    start_index: int = 0,
    instruction_key: str = "instruction",
    original_key: str = "original_image",
    edited_key: str = "edited_image",
    label_key: str = "label",
) -> pd.DataFrame:
    """Map manifest columns into the expected output schema."""
    required = {instruction_key, original_key, edited_key}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    records = []
    for i, (idx, row) in enumerate(df.iterrows()):
        pair_id = f"{start_index + i:04d}"
        image_similarity = float(row.get("image_similarity", row.get("similarity", 0.0)))
        label = int(row.get(label_key, row.get("is_good_edit", 1)))  # Default to 1 for successful edits from Apple
        notes = row.get("notes", row.get("edit_type", ""))
        records.append(
            {
                "pair_id": pair_id,
                "instruction": row[instruction_key],
                "original_image": row[original_key],
                "edited_image": row[edited_key],
                "image_similarity": image_similarity,
                "label": label,
                "notes": notes,
            }
        )

    normalized = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    logging.debug("Normalized manifest to %d rows", len(normalized))
    return normalized


def combine_manifests(
    paths: Sequence[Path],
    instruction_key: str = "instruction",
    original_key: str = "original_image",
    edited_key: str = "edited_image",
    label_key: str = "label",
) -> pd.DataFrame:
    """Concatenate multiple manifests with stable pair IDs."""
    frames: List[pd.DataFrame] = []
    start = 0
    for path in paths:
        frame = read_manifest(path)
        normalized = normalize_records(
            frame,
            start_index=start,
            instruction_key=instruction_key,
            original_key=original_key,
            edited_key=edited_key,
            label_key=label_key,
        )
        frames.append(normalized)
        start += len(normalized)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    logging.info("Combined %d manifests into %d rows", len(paths), len(combined))
    return combined


def sample_pairs(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Optionally subsample the manifest while preserving label balance."""
    if sample_size is None or sample_size >= len(df):
        return df
    frac = sample_size / len(df)

    def per_group_sample(group: pd.DataFrame) -> pd.DataFrame:
        target = max(1, int(round(len(group) * frac)))
        target = min(target, len(group))
        return group.sample(n=target, random_state=seed, replace=False)

    balanced = cast(pd.DataFrame, df.groupby("label", group_keys=False).apply(per_group_sample))
    balanced = balanced.reset_index(drop=True)
    if len(balanced) > sample_size:
        balanced = balanced.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    logging.info("Sampled %d rows from %d total", len(balanced), len(df))
    return balanced


def parse_args(raw_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PicoTuri edit pairs CSV from Pico-Banana manifests.")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Path(s) to manifest files (CSV / JSON / JSONL).")
    parser.add_argument("--output", "-o", required=True, help="Destination CSV path.")
    parser.add_argument("--sample-size", type=int, help="Optional number of rows to sample for quick experiments.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational logging.")
    parser.add_argument("--instruction-key", default="instruction", help="Instruction column name in the dataset.")
    parser.add_argument("--original-key", default="original_image", help="Original image column name in the dataset.")
    parser.add_argument("--edited-key", default="edited_image", help="Edited image column name in the dataset.")
    parser.add_argument("--label-key", default="label", help="Label column name in the dataset (optional).")
    return parser.parse_args(raw_args)


def main(raw_args: Sequence[str] | None = None) -> None:
    args = parse_args(raw_args)

    logging.basicConfig(
        level=logging.DEBUG if not args.quiet else logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_path = Path(args.output).expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}. Pass --overwrite to continue.")

    input_paths = [Path(path).expanduser().resolve() for path in args.input]
    combined = combine_manifests(
        input_paths,
        instruction_key=args.instruction_key,
        original_key=args.original_key,
        edited_key=args.edited_key,
        label_key=args.label_key,
    )
    sampled = sample_pairs(combined, args.sample_size, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)
    logging.info("Wrote %d rows to %s", len(sampled), output_path)


if __name__ == "__main__":
    main()
