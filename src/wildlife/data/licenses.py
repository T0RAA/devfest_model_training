"""License validation, attribution export, and commercial-release guardrails."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ALLOWED_PHOTO_LICENSES: Set[str] = {"cc0", "cc-by"}

NON_COMMERCIAL_OR_RESTRICTED: Set[str] = {
    "cc-by-nc",
    "cc-by-nc-sa",
    "cc-by-nc-nd",
    "cc-by-nd",
    "cc-by-sa",
    "all-rights-reserved",
    "copyright",
    "unknown",
    "",
}

RESEARCH_ONLY_SOURCES: Set[str] = {"cub_200_2011", "cub-200-2011", "cub"}


class LicenseViolationError(ValueError):
    """Raised when an image or dataset source violates licensing rules."""


def normalize_license_code(code: Any) -> str:
    if code is None:
        return ""
    norm = str(code).strip().lower().replace("_", "-")
    aliases = {
        "cc0-1.0": "cc0",
        "public-domain": "cc0",
        "pdm": "cc0",
        "cc-by-4.0": "cc-by",
        "cc-by-3.0": "cc-by",
        "cc-by-2.0": "cc-by",
    }
    return aliases.get(norm, norm)


def validate_photo_license(
    license_code: Any,
    source_dataset: str = "inaturalist_open_data",
    allow_research_only: bool = False,
    allowed_licenses: Set[str] = ALLOWED_PHOTO_LICENSES,
) -> str:
    """Validate photo license and dataset source before ingestion."""
    src_norm = (source_dataset or "").strip().lower()
    if src_norm in RESEARCH_ONLY_SOURCES and not allow_research_only:
        raise LicenseViolationError(
            f"Research-only dataset '{source_dataset}' cannot enter commercial-eligible manifest "
            "without explicit allow_research_only=True."
        )

    norm = normalize_license_code(license_code)
    if not norm:
        raise LicenseViolationError("Missing or empty photo_license is not permitted.")

    if norm not in allowed_licenses:
        if allow_research_only and norm in {"cc-by-nc", "cc-by-sa", "research-only"}:
            return norm
        raise LicenseViolationError(
            f"Unapproved photo license '{license_code}' (normalized='{norm}'). "
            f"Permitted licenses: {sorted(allowed_licenses)}"
        )
    return norm


def audit_manifest_licenses(
    rows: Iterable[Dict[str, Any]],
    allow_research_only: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts_by_license: Dict[str, int] = {}
    counts_by_source: Dict[str, int] = {}

    for row in rows:
        lic = row.get("photo_license")
        src = str(row.get("source_dataset", "unknown"))
        try:
            norm_lic = validate_photo_license(
                lic, source_dataset=src, allow_research_only=allow_research_only
            )
            clean_row = dict(row)
            clean_row["photo_license"] = norm_lic
            approved.append(clean_row)
            counts_by_license[norm_lic] = counts_by_license.get(norm_lic, 0) + 1
            counts_by_source[src] = counts_by_source.get(src, 0) + 1
        except LicenseViolationError as exc:
            rejected.append(
                {
                    "image_id": row.get("image_id"),
                    "photo_id": row.get("photo_id"),
                    "source_dataset": src,
                    "photo_license": lic,
                    "reason": str(exc),
                }
            )

    summary = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "allowed_licenses": sorted(ALLOWED_PHOTO_LICENSES),
        "allow_research_only": allow_research_only,
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "counts_by_license": counts_by_license,
        "counts_by_source": counts_by_source,
        "rejected_records": rejected,
    }
    return approved, summary


def write_attribution_and_audit(
    approved_rows: Iterable[Dict[str, Any]],
    audit_summary: Dict[str, Any],
    attribution_csv_path: Path,
    audit_json_path: Path,
) -> None:
    attribution_csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_id",
        "photo_id",
        "observation_id",
        "class_id",
        "scientific_name",
        "photo_license",
        "photographer_credit",
        "source_dataset",
        "source_url",
    ]
    with attribution_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in approved_rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    with audit_json_path.open("w", encoding="utf-8") as fh:
        json.dump(audit_summary, fh, indent=2, sort_keys=False)
