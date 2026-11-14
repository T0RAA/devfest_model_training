#!/usr/bin/env python3
"""Audit iNaturalist observation frequency for the 20 UCSC Wildlife starter species."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.schemas import SpeciesRecord, load_species_taxonomy

SCHEME = "https"
INAT_API_HOST = "api.inaturalist.org"

AUDIT_CSV_FIELDS = [
    "class_id",
    "taxon_id",
    "scientific_name",
    "common_name",
    "group",
    "research_grade_photographed_observations",
    "all_observations",
    "unique_observers",
    "season",
    "snapshot_date",
    "audit_status",
    "search_url",
]


def query_inaturalist_counts(
    scientific_name: str,
    place_id: int,
    user_agent: str = "UCSC-Wildlife-Vision/2.1 (academic-audit)",
) -> Dict[str, Any]:
    """Query iNaturalist observations and observers count for a scientific_name and place_id."""
    base = f"{SCHEME}://{INAT_API_HOST}/v1/observations"
    rg_params = urllib.parse.urlencode(
        {
            "taxon_name": scientific_name,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 1,
        }
    )
    all_params = urllib.parse.urlencode(
        {
            "taxon_name": scientific_name,
            "place_id": place_id,
            "per_page": 1,
        }
    )

    def _get_json(url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    rg_data = _get_json(f"{base}?{rg_params}")
    time.sleep(0.35)
    all_data = _get_json(f"{base}?{all_params}")
    time.sleep(0.35)
    obs_data = _get_json(f"{base}/observers?{rg_params}")
    time.sleep(0.35)

    return {
        "research_grade_photographed_observations": int(rg_data.get("total_results", 0)),
        "all_observations": int(all_data.get("total_results", 0)),
        "unique_observers": int(obs_data.get("total_results", 0)),
    }


def generate_audit_rows(
    species_list: List[SpeciesRecord],
    online: bool = False,
    place_id: Optional[int] = None,
    snapshot_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    snap = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows: List[Dict[str, Any]] = []
    for sp in species_list:
        if online and place_id is not None:
            counts = query_inaturalist_counts(sp.scientific_name, place_id)
            status = f"verified_place_{place_id}"
            rg_obs = str(counts["research_grade_photographed_observations"])
            all_obs = str(counts["all_observations"])
            uniq_obs = str(counts["unique_observers"])
        else:
            status = "unverified_starter_set"
            rg_obs = "UNVERIFIED_OFFLINE"
            all_obs = "UNVERIFIED_OFFLINE"
            uniq_obs = "UNVERIFIED_OFFLINE"

        rows.append(
            {
                "class_id": sp.class_id,
                "taxon_id": sp.taxon_id,
                "scientific_name": sp.scientific_name,
                "common_name": sp.common_name,
                "group": sp.group,
                "research_grade_photographed_observations": rg_obs,
                "all_observations": all_obs,
                "unique_observers": uniq_obs,
                "season": "all_seasons",
                "snapshot_date": snap,
                "audit_status": status,
                "search_url": sp.inaturalist_search_url,
            }
        )
    return rows


def write_audit_outputs(
    rows: List[Dict[str, Any]],
    csv_path: Path,
    md_path: Path,
    online: bool = False,
    place_id: Optional[int] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mode_str = (
        f"Verified against iNaturalist place_id={place_id} (UC Santa Cruz Campus)"
        if (online and place_id is not None)
        else "Unverified offline starter-set audit (no fabricated counts)"
    )

    lines = [
        "# UCSC Wildlife 20 Starter Species — Campus Observation Frequency Audit",
        "",
        "- **Dataset Title:** UCSC Wildlife 20 Starter Species",
        f"- **Audit Status:** `{mode_str}`",
        f"- **Snapshot Date:** `{rows[0]['snapshot_date'] if rows else '2026-11-14'}`",
        "- **Note:** Reporting frequency on iNaturalist reflects observer activity and detectability within the UCSC campus boundary (`place_id=94596`), not biological population abundance.",
        "",
        "| Class ID | Taxon ID | Common Name | Scientific Name | Group | RG Photographed Obs | All Obs | Unique Observers | Audit Status |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['class_id']}` | {r['taxon_id']} | {r['common_name']} | "
            f"`{r['scientific_name']}` | `{r['group']}` | "
            f"`{r['research_grade_photographed_observations']}` | "
            f"`{r['all_observations']}` | `{r['unique_observers']}` | `{r['audit_status']}` |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "manifests" / "campus_frequency_audit.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "data" / "manifests" / "campus_frequency_audit.md",
    )
    parser.add_argument("--online", action="store_true", help="Query live iNaturalist API")
    parser.add_argument(
        "--place-id",
        type=int,
        default=94596,
        help="Confirmed UCSC Campus iNaturalist place_id (94596 = UC Santa Cruz Campus, 3519 = UCSC Campus Natural Reserve)",
    )
    parser.add_argument("--snapshot-date", type=str, default="2026-11-14")
    args = parser.parse_args()

    species = load_species_taxonomy()
    rows = generate_audit_rows(
        species,
        online=args.online,
        place_id=args.place_id,
        snapshot_date=args.snapshot_date,
    )
    write_audit_outputs(
        rows,
        csv_path=args.output_csv,
        md_path=args.output_md,
        online=args.online,
        place_id=args.place_id,
    )
    print(f"Wrote {len(rows)} species audit records to {args.output_csv} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
