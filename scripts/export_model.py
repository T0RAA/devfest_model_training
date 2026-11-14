#!/usr/bin/env python3
"""Export trained checkpoint, frozen classes.json, calibration.json, and model_card.md into models/v1/ (§7.3)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.models.classifier import load_checkpoint


def export_model_bundle(
    checkpoint_path: Path,
    export_dir: Path,
    model_card_path: Path = ROOT / "reports" / "model_card.md",
) -> dict:
    model, payload = load_checkpoint(checkpoint_path)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    target_model = export_dir / "model.pt"
    shutil.copy2(checkpoint_path, target_model)
    sha256_hex = hashlib.sha256(target_model.read_bytes()).hexdigest()
    (export_dir / "model.pt.sha256").write_text(
        f"{sha256_hex}  model.pt\n", encoding="utf-8"
    )

    classes_out = export_dir / "classes.json"
    classes_out.write_text(json.dumps(model.class_map, indent=2), encoding="utf-8")

    cal_out = export_dir / "calibration.json"
    cal_payload = {
        "calibration_version": "temp-v1",
        "temperature": float(payload.get("temperature", 1.0)),
        "max_prob_threshold": 0.80,
        "margin_threshold": 0.15,
        "energy_threshold": -1.20,
    }
    cal_out.write_text(json.dumps(cal_payload, indent=2), encoding="utf-8")

    if model_card_path.exists():
        shutil.copy2(model_card_path, export_dir / "model_card.md")

    manifest_meta = {
        "model_version": payload.get("model_version", "species-v1.0"),
        "calibration_version": "temp-v1",
        "rating_version": "quality-v1",
        "argon_model_version": "gemini-4-argon",
        "checkpoint_sha256": sha256_hex,
        "num_classes": 20,
        "exported_files": ["model.pt", "classes.json", "calibration.json", "model_card.md"],
    }
    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest_meta, indent=2), encoding="utf-8"
    )
    return manifest_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "efficientnet_b0_v1" / "best.ckpt",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "models" / "v1",
    )
    args = parser.parse_args()

    meta = export_model_bundle(args.checkpoint, args.export_dir)
    print(f"Exported model bundle to {args.export_dir}: sha256={meta['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
