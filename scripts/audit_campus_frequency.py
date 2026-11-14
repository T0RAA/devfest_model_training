#!/usr/bin/env python3
"""Audit iNaturalist observation frequency for the 20 UCSC Wildlife starter species.

Per SPEC.md §3:
- Never invent campus observation counts or frequency rankings.
- In offline mode (or without a verified UCSC campus place_id), emit an explicit
  `unverified_offline` audit table for the 20 documented starter species with the
  honest title "UCSC Wildlife 20 Starter Species".
- When `--online` and `--place-id` are provided, query the iNaturalist v1 API
  respecting the 1 req/sec rate limit and record actual observation and observer counts.
"""

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
    taxon_id: int,
    place_id: int,
    user_agent: str = "UCSC-Wildlife-Vision/1.0 (academic-audit)",
) -> Dict[str, Any]:
    """Query iNaturalist observations and observers count for a taxon and place_id."""
    base = "https://api.inaturalist.org/v1/observations"
    rg_params = urllib.parse.urlencode(
        {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 1,
        }
    )
    all_params = urllib.parse.urlencode(
        {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "per_page": 1,
        }
    )
    obs_params = urllib.parse.urlencode(
        {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 1,
        }
    )

    def _get_json(url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    rg_data = _get_json(f"{base}?{rg_params}")
    time.sleep(1.1)
    all_data = _get_json(f"{base}?{all_params}")
    time.sleep(1.1)
    obs_data = _get_json(f"{base}/observers?{obs_params}")
    time.sleep(1.1)

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
            counts = query_inaturalist_counts(sp.taxon_id, place_id)
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
        f"Verified against iNaturalist place_id={place_id}"
        if (online and place_id is not None)
        else "Unverified offline starter-set audit (no fabricated counts)"
    )

    lines = [
        "# UCSC Wildlife 20 Starter Species — Campus Observation Frequency Audit",
        "",
        f"- **Dataset Title:** UCSC Wildlife 20 Starter Species",
        f"- **Audit Status:** `{mode_str}`",
        f"- **Snapshot Date:** `{rows[0]['snapshot_date'] if rows else '2026-11-14'}`",
        "- **Scope Caveat:** The 20 species below are validated against the UCSC Campus Natural Reserve mammal and bird presence inventories. Per `SPEC.md` §3, no observation totals or frequency ranks are invented when offline. To populate live iNaturalist counts for a verified campus polygon/place ID, run:",
        "  ```bash",
        "  python3 scripts/audit_campus_frequency.py --online --place-id <UCSC_CAMPUS_PLACE_ID>",
        "  ```",
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
        default=None,
        help="Confirmed UCSC Campus Natural Reserve iNaturalist place_id (never substitute county/island)",
    )
    parser.add_argument("--snapshot-date", type=str, default="2026-11-14")
    args = parser.parse_args()

    if args.online and args.place_id is None:
        raise SystemExit(
            "Error: --online requires an explicit --place-id for the UCSC Campus Reserve footprint."
        )

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
