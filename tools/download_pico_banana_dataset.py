#!/usr/bin/env python3
"""Utility helpers for working with the Pico-Banana-400K dataset.

The goal of this script is to offer a clean, type-safe reference implementation
that can:

1. Download the public manifest files Apple provides for the dataset.
2. Print basic statistics about local manifests.
3. Outline (but not automatically perform) large image downloads.

It intentionally avoids auto-downloading the full 500 GB of source images, but
it gives guidance and directory scaffolding for developers who have access.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

MANIFEST_URLS: Dict[str, str] = {
    "sft": "https://docs-assets.developer.apple.com/ml-research/datasets/pico-banana/manifest_sft.jsonl",
    "preference": "https://docs-assets.developer.apple.com/ml-research/datasets/pico-banana/manifest_preference.jsonl",
    "multiturn": "https://docs-assets.developer.apple.com/ml-research/datasets/pico-banana/manifest_multiturn.jsonl",
}

OPEN_IMAGES_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"


class PicoBananaDatasetDownloader:
    """Download helper for manifests and (optionally) Open Images assets."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_dir = root / "manifests"
        self.images_dir = root / "images"
        self.source_images_dir = root / "openimage_source_images"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.source_images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def download_manifests(self) -> bool:
        """Download manifest JSONL files from Apple's CDN."""
        print("\n🖼️  STEP 1: Downloading Pico-Banana manifests")
        print("=" * 60)

        success = 0
        for name, url in MANIFEST_URLS.items():
            destination = self.manifest_dir / f"manifest_{name}.jsonl"
            if destination.exists():
                print(f"✅ {name} manifest already present")
                success += 1
                continue

            print(f"📥 Fetching {name} manifest from {url}")
            try:
                urllib.request.urlretrieve(url, destination)
                with destination.open("r", encoding="utf-8") as handle:
                    num_examples = sum(1 for _ in handle)
                print(f"   ✓ Downloaded {num_examples:,} examples")
                success += 1
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"   ❌ Failed to download: {exc}")
                destination.unlink(missing_ok=True)

        print(f"\n📊 Manifest download summary: {success}/3 successful")
        if success == len(MANIFEST_URLS):
            print("✅ All manifests available.")
            return True
        print("⚠️  Some manifests are missing. Ensure you have access to Apple's CDN.")
        return False

    # ------------------------------------------------------------------
    def show_statistics(self) -> None:
        """Print sample statistics for the manifests available on disk."""
        print("\n📊 DATASET STATISTICS")
        print("=" * 60)

        total_examples = 0
        for manifest_type in MANIFEST_URLS.keys():
            path = self.manifest_dir / f"manifest_{manifest_type}.jsonl"
            if not path.exists():
                print(f"🗂️  {manifest_type}: manifest missing")
                continue

            try:
                with path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for _ in handle)
                print(f"🗂️  {manifest_type}: {count:,} examples")
                total_examples += count
            except Exception as exc:  # pragma: no cover - file issues
                print(f"🗂️  {manifest_type}: error reading file ({exc})")

        print(f"\nTotal examples across available manifests: {total_examples:,}")

    # ------------------------------------------------------------------
    def download_open_images_metadata(self) -> bool:
        """Download the Open Images metadata CSV to help map image IDs."""
        destination = self.root / "train-images-boxable-with-rotation.csv"
        if destination.exists():
            print("✅ Open Images metadata already present")
            return True

        print("📄 Downloading Open Images metadata CSV...")
        try:
            urllib.request.urlretrieve(OPEN_IMAGES_METADATA_URL, destination)
            print("   ✓ Metadata downloaded")
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"   ❌ Failed to download metadata: {exc}")
            return False

    # ------------------------------------------------------------------
    def outline_full_download(self) -> None:
        """Print instructions for downloading the full Open Images dataset."""
        print("\n⚠️  Full image download requires >500GB. Steps to follow:")
        print(
            """
1. Install AWS CLI:   pip install awscli
2. Create directory:   mkdir -p open_images_v7 && cd open_images_v7
3. Download archives:
   for i in {0..9}; do
       aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${i}.tar.gz .
   done
4. Extract archives:   for file in *.tar.gz; do tar -xzf "$file"; done
5. Map manifests → Open Images paths using a helper script.
"""
        )

    # ------------------------------------------------------------------
    def check_aws_cli(self) -> bool:
        """Return True if aws CLI is available."""
        result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        return result.returncode == 0


# =============================================================================
# CLI UTILITIES
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pico-Banana dataset utilities")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("./pico_banana_dataset"),
        help="Destination directory for manifests and assets.",
    )
    parser.add_argument(
        "--download-manifests",
        action="store_true",
        help="Download manifest files from Apple's CDN.",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show statistics for available manifests.",
    )
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download Open Images metadata CSV (helpful for mapping IDs).",
    )
    parser.add_argument(
        "--detail-open-images",
        action="store_true",
        help="Print instructions for downloading full Open Images archives.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    downloader = PicoBananaDatasetDownloader(root=args.download_dir)

    if args.download_manifests:
        downloader.download_manifests()

    if args.show_stats:
        downloader.show_statistics()

    if args.download_metadata:
        downloader.download_open_images_metadata()

    if args.detail_open_images:
        downloader.outline_full_download()

    if not any(
        [
            args.download_manifests,
            args.show_stats,
            args.download_metadata,
            args.detail_open_images,
        ]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
