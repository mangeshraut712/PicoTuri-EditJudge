#!/usr/bin/env python3
"""Compact end-to-end pipeline showcase for PicoTuri-EditJudge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import sys


@dataclass
class ComponentStatus:
    completed: bool
    notes: str


def check_components() -> Dict[str, ComponentStatus]:
    """Return a status map for the major pipeline components."""
    components = {
        "Environment": ComponentStatus(
            False, "Python 3.12+ with torch/coremltools"
        ),
        "Manifests": ComponentStatus(
            Path("pico_banana_dataset/manifest_sft.jsonl").exists(),
            "Sample manifests available"
        ),
        "SFT Model": ComponentStatus(
            Path("outputs/sft_models/latest_model.pt").exists(),
            "Trained baseline model checkpoint"
        ),
        "Core ML": ComponentStatus(
            Path("coreml_output/model.pt").exists(),
            "TorchScript export ready for conversion"
        ),
        "Documentation": ComponentStatus(
            Path("docs/modern_roadmap.md").exists(),
            "Roadmap + guides"
        ),
    }

    # Environment check
    if sys.version_info >= (3, 12):
        components["Environment"].completed = True
        components["Environment"].notes = "Python 3.12+ detected"

    return components


def display_summary() -> None:
    print("🌟 PicoTuri-EditJudge: Complete Pipeline Overview")
    print("=" * 80)

    status_map = check_components()
    print("\n📋 Component Checklist")
    print("-" * 40)
    for name, status in status_map.items():
        mark = "✅" if status.completed else "❌"
        print(f" {mark} {name:12s} – {status.notes}")

    if Path("pico_banana_dataset/manifest_sft.jsonl").exists():
        with Path("pico_banana_dataset/manifest_sft.jsonl").open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        print(f"\n📊 Sample manifest entries: {line_count}")

    print("\n📚 Key Resources")
    print("-" * 40)
    print(" • docs/modern_roadmap.md – Research & upgrade roadmap")
    print(" • scripts/pico_banana_setup.py – Practical setup guide")
    print(" • src/algorithms/deep_dive.py – Advanced algorithm scaffold")


def main() -> None:
    display_summary()


if __name__ == "__main__":
    main()
