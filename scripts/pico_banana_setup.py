"""
PRACTICAL SETUP GUIDE: Getting Started with Apple's Modern ML Pipeline
----------------------------------------------------------------------

Step-by-step helper for downloading, organising, and exploring the
Pico-Banana-400K dataset plus bootstrapping an Apple-aligned development
environment. The functions are designed to be invoked from the command line:

    python scripts/pico_banana_setup.py

They print detailed instructions rather than attempting large downloads during
execution, ensuring the script remains safe to run in restricted environments.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================================


def setup_environment() -> None:
    """Display environment prerequisites and recommended install commands."""
    print("=" * 70)
    print("STEP 1: ENVIRONMENT SETUP")
    print("=" * 70)

    python_version = sys.version_info
    print(
        f"\n✓ Python version detected: {python_version.major}."
        f"{python_version.minor}.{python_version.micro}"
    )

    if python_version < (3, 9):
        print("⚠️  Warning: Python 3.9 or newer is recommended for this workflow.")

    install_commands = r"""
    # Create conda environment (recommended for Apple Silicon)
    conda create -n apple_ml python=3.12 -y
    conda activate apple_ml

    # Install PyTorch with GPU/MPS support
    pip install torch torchvision torchaudio

    # Install Core ML tools
    pip install coremltools>=8.0

    # Hugging Face + diffusion ecosystem
    pip install transformers diffusers accelerate datasets huggingface-hub

    # Image processing
    pip install pillow opencv-python albumentations

    # Quality metrics
    pip install lpips pytorch-fid
    pip install git+https://github.com/openai/CLIP.git

    # Utilities & dev tooling
    pip install pandas numpy tqdm wandb
    pip install pytest black flake8

    # Verify installation
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import coremltools; print(f'CoreML: {coremltools.__version__}')"
    """

    print("\n📦 Recommended installation commands:")
    print(install_commands)

    try:
        import torch  # type: ignore[import]

        if torch.backends.mps.is_available():
            print("\n✓ Apple Silicon GPU (MPS) detected.")
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"\n✓ CUDA GPU detected: {device_name}")
        else:
            print("\n⚠️  No GPU detected; training will run on CPU.")
    except ModuleNotFoundError:
        print("\n⚠️  PyTorch not installed yet – install it using the commands above.")


# ============================================================================
# STEP 2: DOWNLOAD PICO-BANANA-400K DATASET
# ============================================================================


class PicoBananaDownloader:
    """Utility for downloading and inspecting Pico-Banana-400K manifests."""

    BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/pico-banana"
    MANIFEST_URLS = {
        "sft": f"{BASE_URL}/manifest_sft.jsonl",
        "preference": f"{BASE_URL}/manifest_preference.jsonl",
        "multiturn": f"{BASE_URL}/manifest_multiturn.jsonl",
    }

    def __init__(self, download_dir: str = "./pico_banana_dataset") -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_manifests(self) -> None:
        """Download all manifest files if they are not present locally."""
        print("\n" + "=" * 70)
        print("STEP 2: DOWNLOADING PICO-BANANA-400K MANIFESTS")
        print("=" * 70)

        for name, url in self.MANIFEST_URLS.items():
            output_path = self.download_dir / f"manifest_{name}.jsonl"
            if output_path.exists():
                print(f"\n✓ Manifest already present: {output_path}")
                continue

            print(f"\n📥 Downloading {name} manifest from {url}")
            try:
                urllib.request.urlretrieve(url, output_path)
                with output_path.open("r", encoding="utf-8") as handle:
                    num_examples = sum(1 for _ in handle)
                print(f"   ✓ Downloaded {num_examples:,} examples to {output_path}")
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"   ❌ Failed to download {url}: {exc}")
                print(f"   Please download manually and place the file at {output_path}")

    def parse_manifest(self, manifest_type: str = "sft") -> List[Dict]:
        manifest_path = self.download_dir / f"manifest_{manifest_type}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        examples: List[Dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json.loads(line))
        return examples

    def print_statistics(self) -> None:
        """Display high-level statistics for each manifest (sampled)."""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)

        for name in ["sft", "preference", "multiturn"]:
            manifest_path = self.download_dir / f"manifest_{name}.jsonl"
            if not manifest_path.exists():
                print(f"\n{name.upper()} manifest missing – download skipped.")
                continue

            examples = self.parse_manifest(name)
            print(f"\n{name.upper()} DATASET:")
            print(f"  Total examples: {len(examples):,}")

            if name == "sft":
                edit_types: Dict[str, int] = {}
                for ex in examples[:1000]:
                    edit_type = ex.get("edit_type", "unknown")
                    edit_types[edit_type] = edit_types.get(edit_type, 0) + 1
                print("  Sample edit-type distribution (first 1,000 examples):")
                for edit_type, count in sorted(edit_types.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                    print(f"    - {edit_type}: {count}")
            elif name == "preference":
                print("  Contains success/failure edit comparisons for DPO.")
            elif name == "multiturn":
                seq_lengths = [len(ex.get("turns", [])) for ex in examples[:1000] if "turns" in ex]
                if seq_lengths:
                    avg_length = sum(seq_lengths) / len(seq_lengths)
                    print(f"  Avg sequence length (sample): {avg_length:.1f} turns")
                    print(f"  Max sequence length (sample): {max(seq_lengths)} turns")

    def download_images_sample(self, num_examples: int = 100) -> Path:
        """
        Provide guidance and directory scaffolding for downloading sample images.
        Full dataset downloads are explained but not executed automatically.
        """
        print("\n" + "=" * 70)
        print(f"DOWNLOADING IMAGE SAMPLE ({num_examples} examples)")
        print("=" * 70)

        print("⚠️  Full Pico-Banana-400K image data requires ~500GB of storage.")
        print("   This script only prepares directories and outlines download steps.")

        images_dir = self.download_dir / "images" / "sample"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Sample images can be stored in: {images_dir}")

        instructions = r"""
FOR FULL DATASET DOWNLOAD:
──────────────────────────

1. Install AWS CLI:
   pip install awscli

2. Download Open Images V7 (source dataset):
   mkdir -p open_images_v7
   cd open_images_v7
   for i in {0..9}; do
       aws s3 --no-sign-request cp \
           s3://open-images-dataset/tar/train_${i}.tar.gz .
   done
   for file in *.tar.gz; do
       tar -xzf "$file"
   done

3. Map Pico-Banana examples to Open Images:
   python map_pico_to_openimages.py \
       --manifest manifest_sft.jsonl \
       --images_dir open_images_v7/

ALTERNATIVE (Recommended to start):
───────────────────────────────────
Use Hugging Face streaming:
   from datasets import load_dataset
   dataset = load_dataset("apple/pico-banana-400k", streaming=True)
"""
        print(instructions)
        return images_dir


# ============================================================================
# STEP 3: DATA LOADER CONVENIENCE CLASS
# ============================================================================


class PicoBananaDataLoader:
    """Lightweight helper for inspecting manifest entries."""

    def __init__(self, manifest_path: str, images_dir: str, mode: str = "sft") -> None:
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir)
        self.mode = mode
        self.examples = self._load_manifest()
        print(f"\n✓ Loaded {len(self.examples):,} examples from {self.manifest_path}")

    def _load_manifest(self) -> List[Dict]:
        examples: List[Dict] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json.loads(line))
        return examples

    def get_example(self, idx: int) -> Dict:
        example = self.examples[idx]
        return {
            "source_image_id": example.get("source_image_id"),
            "edited_image_id": example.get("edited_image_id"),
            "instruction": example.get("instruction"),
            "edit_type": example.get("edit_type"),
            "quality_score": example.get("quality_score"),
            "metadata": example.get("metadata", {}),
        }

    def create_pytorch_dataset_snippet(self) -> str:
        """Return a ready-to-use PyTorch dataset code snippet."""
        return r"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image  # type: ignore[import]


class PicoBananaDataset(Dataset):
    def __init__(self, manifest_path, images_dir, transform=None):
        self.examples = []
        with open(manifest_path, "r") as handle:
            for line in handle:
                self.examples.append(json.loads(line))

        self.images_dir = Path(images_dir)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source_path = self.images_dir / example["source_image_id"]
        edited_path = self.images_dir / example["edited_image_id"]

        source_img = Image.open(source_path).convert("RGB")
        edited_img = Image.open(edited_path).convert("RGB")

        source_tensor = self.transform(source_img)
        edited_tensor = self.transform(edited_img)

        return {
            "source": source_tensor,
            "edited": edited_tensor,
            "instruction": example["instruction"],
            "edit_type": example.get("edit_type"),
        }


# Usage example
# dataset = PicoBananaDataset("manifest_sft.jsonl", "images/")
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
"""


# ============================================================================
# STEP 4: QUICK START TRAINING SCRIPT GENERATOR
# ============================================================================


def generate_quick_start_script() -> str:
    """Return a template training script aligned with the roadmap."""
    return '''#!/usr/bin/env python3
"""
Quick Start Training Script for Pico-Banana-400K
Apple-style ML Pipeline
"""

from pathlib import Path

import torch


class Config:
    data_dir = Path("./pico_banana_dataset")
    output_dir = Path("./outputs")
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    image_size = 512
    quality_threshold = 0.7
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    use_wandb = False
    project_name = "pico-banana-training"


def train() -> None:
    config = Config()
    print("🚀 Starting Pico-Banana-400K Training")
    print(f"Device: {config.device}")

    # TODO:
    #   1. Instantiate dataset & dataloader (see scripts/pico_banana_setup.py).
    #   2. Build instruction-conditioned diffusion model.
    #   3. Apply quality-aware filtering (≥0.7 weighted score).
    #   4. Implement DPO fine-tuning loop.
    #   5. Export final model to Core ML.

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs} – implement training loop here.")

    print("✅ Training complete!")


if __name__ == "__main__":
    train()
'''


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main() -> None:
    """Run the complete practical setup workflow."""
    print(
        """
╔═══════════════════════════════════════════════════════════════════╗
║  PRACTICAL SETUP GUIDE: Apple's Modern ML Pipeline                ║
║  From Turi Create → PyTorch + Core ML + Pico-Banana-400K          ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    )

    setup_environment()

    downloader = PicoBananaDownloader(download_dir="./pico_banana_dataset")
    downloader.download_manifests()
    downloader.print_statistics()
    downloader.download_images_sample(num_examples=100)

    print("\n" + "=" * 70)
    print("STEP 3: DATA LOADER SETUP")
    print("=" * 70)

    manifest_path = "./pico_banana_dataset/manifest_sft.jsonl"
    if Path(manifest_path).exists():
        loader = PicoBananaDataLoader(
            manifest_path=manifest_path,
            images_dir="./pico_banana_dataset/images",
            mode="sft",
        )

        if loader.examples:
            example = loader.get_example(0)
            print("\n📝 Example entry:")
            print(f"   Instruction: {example['instruction']}")
            print(f"   Edit Type:   {example['edit_type']}")
            print(f"   Quality:     {example.get('quality_score', 'N/A')}")

        print("\n📦 PyTorch dataset scaffold:\n")
        print(loader.create_pytorch_dataset_snippet())
    else:
        print("⚠️  manifest_sft.jsonl not found yet; download step may have failed.")

    print("\n" + "=" * 70)
    print("STEP 4: TRAINING SCRIPT GENERATION")
    print("=" * 70)

    script_text = generate_quick_start_script()
    script_path = Path("./train_pico_banana.py")
    script_path.write_text(script_text, encoding="utf-8")
    print(f"\n✓ Wrote quick-start script to {script_path}.\n   Run it with: python {script_path}")

    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE - FINAL CHECKLIST")
    print("=" * 70)
    print(
        """
ENVIRONMENT:
☐ Python 3.9+ with PyTorch + Core ML tools installed
☐ Hugging Face ecosystem available

DATASET:
☐ Manifests downloaded (`manifest_*.jsonl`)
☐ Sample images prepared or streaming pipeline configured

CODE:
☐ Advanced algorithms scaffold ready (`src/algorithms/deep_dive.py`)
☐ Quality scorer & DPO loss implemented
☐ Training script template created (`train_pico_banana.py`)

NEXT STEPS:
1. Validate data loader on a small sample.
2. Train on a subset (e.g., 1,000 examples) to confirm pipeline.
3. Scale training + implement DPO/multi-turn workflows.
4. Export to Core ML and profile on device.
"""
    )

    print("\nSetup guide finished. Happy building! 🚀🎨")


if __name__ == "__main__":
    main()
