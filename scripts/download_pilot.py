#!/usr/bin/env python3
"""Discover pilot research-grade CC0/CC-BY observations from iNaturalist API (§6.3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.data.collect import build_s3_open_data_duckdb_query, get_pilot_observations
from wildlife.schemas import load_species_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-per-species", type=int, default=1)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "manifests" / "pilot_discovered.json",
    )
    parser.add_argument(
        "--export-duckdb-sql",
        type=Path,
        default=ROOT / "data" / "manifests" / "s3_open_data_query.sql",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Execute live HTTP queries against api.inaturalist.org (requires network)",
    )
    args = parser.parse_args()

    species_list = load_species_taxonomy()
    sql_query = build_s3_open_data_duckdb_query(species_list)
    args.export_duckdb_sql.parent.mkdir(parents=True, exist_ok=True)
    args.export_duckdb_sql.write_text(sql_query, encoding="utf-8")
    print(f"Exported S3 Open Data DuckDB SQL query to {args.export_duckdb_sql}")

    if not args.online:
        print(
            "Offline mode: skipped live iNaturalist API calls (no fabricated remote counts). "
            "Pass --online when network access is enabled."
        )
        return 0

    discovered: List[Dict[str, Any]] = []
    for sp in species_list:
        ckpt = args.output_json.parent / f".ckpt_{sp.class_id}.json"
        rows = get_pilot_observations(
            sp.scientific_name,
            pages=args.pages_per_species,
            per_page=args.per_page,
            checkpoint_path=ckpt,
        )
        for r in rows:
            r["class_id"] = sp.class_id
            r["coarse_habitat"] = sp.campus_habitat
        discovered.extend(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(discovered, indent=2), encoding="utf-8")
    print(f"Saved {len(discovered)} license-approved pilot records to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
