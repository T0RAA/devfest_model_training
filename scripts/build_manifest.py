#!/usr/bin/env python3
"""Build manifest.parquet, split_v1.csv, photo_attribution.csv, and license_audit.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.data.clean import compute_phash, encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.data.licenses import audit_manifest_licenses, write_attribution_and_audit
from wildlife.data.manifest import write_manifest_parquet
from wildlife.data.split import (
    create_observation_grouped_splits,
    export_split_csv,
    verify_no_split_leakage,
)
from wildlife.schemas import export_classes_json, load_species_taxonomy


def generate_synthetic_fixture_rows(
    obs_per_species: int = 7,
    write_sample_pngs: bool = False,
    raw_dir: Path = ROOT / "data" / "raw" / "synthetic_fixture",
) -> List[Dict[str, Any]]:
    """Generate a license-compliant synthetic fixture across all 20 species for testing."""
    species_list = load_species_taxonomy()
    rows: List[Dict[str, Any]] = []
    if write_sample_pngs:
        raw_dir.mkdir(parents=True, exist_ok=True)

    for sp in species_list:
        for obs_idx in range(obs_per_species):
            obs_id = f"obs_{sp.class_id}_{obs_idx:03d}"
            observer_id = f"user_{sp.class_id}_{obs_idx % 5:02d}"
            photo_id = f"photo_{sp.class_id}_{obs_idx:03d}"
            image_id = f"img_{sp.class_id}_{obs_idx:03d}"
            lic = "cc0" if (obs_idx % 2 == 0) else "cc-by"

            rgb = generate_synthetic_species_rgb(
                sp.index, sample_idx=obs_idx * 17 + sp.index, size=96
            )
            png_bytes = encode_rgb_to_png_bytes(rgb)
            sha256_hex = hashlib.sha256(png_bytes).hexdigest()
            # Ensure distinct pHashes across distinct synthetic observations
            raw_phash = int(compute_phash(rgb), 16)
            salt_mask = ((sp.index + 1) * 0x1000100010001) ^ ((obs_idx + 1) * 0x0101010101010101)
            phash_hex = f"{(raw_phash ^ (salt_mask & 0xFFFFFFFFFFFFFFFF)):016x}"

            rel_img_path = f"data/raw/synthetic_fixture/{image_id}.png"
            if write_sample_pngs:
                (ROOT / rel_img_path).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / rel_img_path).write_bytes(png_bytes)

            rows.append(
                {
                    "image_id": image_id,
                    "observation_id": obs_id,
                    "observer_id": observer_id,
                    "photo_id": photo_id,
                    "source_dataset": "synthetic_fixture",
                    "source_url": f"inaturalist://photos/{photo_id}",
                    "photo_license": lic,
                    "photographer_credit": f"UCSC Volunteer ({observer_id})",
                    "scientific_name": sp.scientific_name,
                    "class_id": sp.class_id,
                    "taxon_id": sp.taxon_id,
                    "date_observed": f"2026-0{(obs_idx % 9) + 1}-15",
                    "coarse_habitat": sp.campus_habitat,
                    "region_tag": "ucsc_campus_reserve",
                    "image_path": rel_img_path,
                    "image_sha256": sha256_hex,
                    "phash": phash_hex,
                    "width": 96,
                    "height": 96,
                    "label_reviewed": True,
                    "crop_box": "0,0,96,96",
                    "split": "train",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional discovered observations JSON from download_pilot.py",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate deterministic 20-species synthetic fixture for local pipeline & smoke tests",
    )
    parser.add_argument(
        "--obs-per-species",
        type=int,
        default=7,
        help="Observations per species when generating synthetic fixture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "manifest.parquet",
    )
    parser.add_argument(
        "--splits-out",
        type=Path,
        default=ROOT / "data" / "splits" / "split_v1.csv",
    )
    parser.add_argument(
        "--classes-out",
        type=Path,
        default=ROOT / "data" / "splits" / "classes.json",
    )
    parser.add_argument(
        "--attribution-out",
        type=Path,
        default=ROOT / "data" / "manifests" / "photo_attribution.csv",
    )
    parser.add_argument(
        "--license-audit-out",
        type=Path,
        default=ROOT / "data" / "manifests" / "license_audit.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_json and args.input_json.exists():
        raw_rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        raw_rows = generate_synthetic_fixture_rows(obs_per_species=args.obs_per_species)

    approved_rows, audit_summary = audit_manifest_licenses(raw_rows, allow_research_only=False)
    split_rows = create_observation_grouped_splits(approved_rows, seed=args.seed)
    leakage_report = verify_no_split_leakage(split_rows)

    export_classes_json(args.classes_out)
    write_manifest_parquet(split_rows, args.output)
    export_split_csv(split_rows, args.splits_out)
    write_attribution_and_audit(
        split_rows,
        audit_summary,
        attribution_csv_path=args.attribution_out,
        audit_json_path=args.license_audit_out,
    )

    print(
        f"Manifest built: {len(split_rows)} images across 20 species | "
        f"splits={leakage_report['split_counts']} | leakage_free={leakage_report['leakage_free']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
