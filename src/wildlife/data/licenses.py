"""License validation, attribution export, and commercial-release guardrails (v1 + v2 Decision D1)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ALLOWED_PHOTO_LICENSES: Set[str] = {"cc0", "cc-by"}

# Decision D1 (Resolved): CDLA-Permissive-1.0 is allowed ONLY for detector fine-tuning
# and detector evaluation, NEVER in classifier training manifests.
DETECTOR_ONLY_LICENSES: Set[str] = {"cdla-permissive-1.0", "cdla-permissive"}
LILA_DETECTOR_SOURCES: Set[str] = {
    "lila_cct",
    "lila_cct_empty",
    "lila_ena24",
    "lila_channel_islands",
    "lila_missouri",
}

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
        "cdla-permissive": "cdla-permissive-1.0",
    }
    return aliases.get(norm, norm)


def validate_photo_license(
    license_code: Any,
    source_dataset: str = "inat_open_data",
    allow_research_only: bool = False,
    for_detector: bool = False,
    allowed_licenses: Set[str] = ALLOWED_PHOTO_LICENSES,
) -> str:
    """Validate photo license and dataset source before ingestion (SPEC.md §2.3, v2 Decision D1)."""
    src_norm = (source_dataset or "").strip().lower()
    if src_norm in RESEARCH_ONLY_SOURCES and not allow_research_only:
        raise LicenseViolationError(
            f"Research-only dataset '{source_dataset}' cannot enter commercial-eligible manifest "
            "without explicit allow_research_only=True."
        )

    norm = normalize_license_code(license_code)
    if not norm:
        raise LicenseViolationError("Missing or empty photo_license is not permitted.")

    if norm in allowed_licenses:
        if src_norm in LILA_DETECTOR_SOURCES and not for_detector:
            raise LicenseViolationError(
                f"LILA dataset '{source_dataset}' is restricted to detector use only (Decision D1)."
            )
        return norm

    if for_detector and norm in DETECTOR_ONLY_LICENSES and src_norm in LILA_DETECTOR_SOURCES:
        return norm

    if allow_research_only and norm in {"cc-by-nc", "cc-by-sa", "research-only"}:
        return norm

    raise LicenseViolationError(
        f"Unapproved photo license '{license_code}' (normalized='{norm}', source='{source_dataset}', "
        f"for_detector={for_detector}). Classifier allowed licenses: {sorted(allowed_licenses)}"
    )


def audit_manifest_licenses(
    rows: Iterable[Dict[str, Any]],
    allow_research_only: bool = False,
    for_detector: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts_by_license: Dict[str, int] = {}
    counts_by_source: Dict[str, int] = {}

    for row in rows:
        lic = row.get("photo_license")
        src = str(row.get("source_dataset", "unknown"))
        row_for_det = bool(for_detector or row.get("for_detector_only", False))
        try:
            norm_lic = validate_photo_license(
                lic,
                source_dataset=src,
                allow_research_only=allow_research_only,
                for_detector=row_for_det,
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
        "allowed_photo_licenses": sorted(ALLOWED_PHOTO_LICENSES),
        "detector_only_licenses": sorted(DETECTOR_ONLY_LICENSES),
        "allow_research_only": allow_research_only,
        "for_detector": for_detector,
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
