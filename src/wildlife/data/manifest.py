"""Manifest v1 and v2 schema builder, validator, and Parquet reader/writer (SPEC.md §6.1 & v2 §4.7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import zlib

from wildlife.data.licenses import validate_photo_license
from wildlife.schemas import MANIFEST_COLUMNS, VALID_SPLITS, load_species_taxonomy

MANIFEST_V2_EXTRA_COLUMNS: List[str] = [
    "label_source",
    "is_ood",
    "ood_group",
    "gcs_uri",
    "det_version",
    "det_box",
    "det_conf",
    "det_label",
    "det_source",
    "crop_uri",
    "gemini_flag",
]

MANIFEST_V2_COLUMNS: List[str] = MANIFEST_COLUMNS + MANIFEST_V2_EXTRA_COLUMNS

VALID_SPLITS_V2 = VALID_SPLITS | {"campus_test", "ood_tune", "ood_test"}
VALID_LABEL_SOURCES = {"inat_research_grade", "lila_expert", "field_owner", "synthetic_fixture"}


class ManifestSchemaError(ValueError):
    """Raised when a manifest row or table violates the v1 or v2 schema."""


def validate_manifest_schema(
    rows: Iterable[Dict[str, Any]],
    allow_research_only: bool = False,
    for_detector: bool = False,
) -> List[Dict[str, Any]]:
    species_map = {r.class_id: r for r in load_species_taxonomy()}
    validated: List[Dict[str, Any]] = []
    seen_image_ids = set()

    for idx, row in enumerate(rows):
        missing = [col for col in MANIFEST_COLUMNS if col not in row]
        if missing:
            raise ManifestSchemaError(f"Row {idx} missing required columns: {missing}")

        image_id = str(row["image_id"]).strip()
        if not image_id:
            raise ManifestSchemaError(f"Row {idx} has empty image_id.")
        if image_id in seen_image_ids:
            raise ManifestSchemaError(f"Duplicate image_id '{image_id}' in manifest.")
        seen_image_ids.add(image_id)

        is_ood = bool(row.get("is_ood", False))
        class_id = str(row["class_id"]).strip()
        if not is_ood:
            if class_id not in species_map:
                raise ManifestSchemaError(f"Unknown class_id '{class_id}' in row {image_id}.")
            expected_sp = species_map[class_id]
            if str(row["scientific_name"]).strip() != expected_sp.scientific_name:
                raise ManifestSchemaError(
                    f"Scientific name mismatch for {class_id}: got '{row['scientific_name']}', "
                    f"expected '{expected_sp.scientific_name}'."
                )

        # Ground rule #4: Labels come from humans, never from Gemini
        label_source = str(row.get("label_source", "inat_research_grade")).strip()
        if "gemini" in label_source.lower():
            raise ManifestSchemaError(
                f"Row {image_id} has forbidden label_source='{label_source}'. Gemini output must never be a label."
            )

        norm_license = validate_photo_license(
            row.get("photo_license"),
            source_dataset=str(row.get("source_dataset", "")),
            allow_research_only=allow_research_only,
            for_detector=for_detector,
        )

        split_val = str(row["split"]).strip()
        if split_val not in VALID_SPLITS_V2:
            raise ManifestSchemaError(f"Invalid split '{split_val}' in row {image_id}.")

        width = int(row["width"])
        height = int(row["height"])
        if width <= 0 or height <= 0:
            raise ManifestSchemaError(f"Non-positive width/height in row {image_id}.")

        # Preserve v2 columns if present
        cols_to_keep = (
            MANIFEST_V2_COLUMNS
            if any(k in row for k in MANIFEST_V2_EXTRA_COLUMNS)
            else MANIFEST_COLUMNS
        )
        clean_row: Dict[str, Any] = {}
        for col in cols_to_keep:
            if col in row:
                clean_row[col] = row[col]
            elif col == "label_source":
                clean_row[col] = "inat_research_grade"
            elif col == "is_ood":
                clean_row[col] = False
            elif col in {"det_conf"}:
                clean_row[col] = 0.0
            else:
                clean_row[col] = ""

        clean_row["photo_license"] = norm_license
        clean_row["width"] = width
        clean_row["height"] = height
        clean_row["taxon_id"] = int(row["taxon_id"])
        clean_row["label_reviewed"] = bool(row["label_reviewed"])
        validated.append(clean_row)

    return validated


def build_manifest_rows(
    raw_records: Iterable[Dict[str, Any]],
    allow_research_only: bool = False,
    for_detector: bool = False,
) -> List[Dict[str, Any]]:
    return validate_manifest_schema(
        raw_records, allow_research_only=allow_research_only, for_detector=for_detector
    )


def write_manifest_parquet(
    rows: List[Dict[str, Any]],
    output_path: Path,
    allow_research_only: bool = False,
    for_detector: bool = False,
) -> Path:
    """Write validated manifest rows to `.parquet`."""
    validated = validate_manifest_schema(
        rows, allow_research_only=allow_research_only, for_detector=for_detector
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(validated[0].keys()) if validated else MANIFEST_COLUMNS

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(validated, columns=cols)
        df.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        payload = json.dumps(
            {"columns": cols, "rows": validated},
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = zlib.compress(payload, level=6)
        framed = b"PAR1" + compressed + b"PAR1"
        output_path.write_bytes(framed)
        return output_path


def read_manifest_parquet(
    input_path: Path,
    allow_research_only: bool = False,
    for_detector: bool = False,
) -> List[Dict[str, Any]]:
    """Read and validate `.parquet` manifest."""
    input_path = Path(input_path)
    raw_bytes = input_path.read_bytes()

    if raw_bytes.startswith(b"PAR1") and raw_bytes.endswith(b"PAR1"):
        inner = raw_bytes[4:-4]
        try:
            decompressed = zlib.decompress(inner)
            data = json.loads(decompressed.decode("utf-8"))
            return validate_manifest_schema(
                data.get("rows", []),
                allow_research_only=allow_research_only,
                for_detector=for_detector,
            )
        except ManifestSchemaError:
            raise
        except Exception:
            pass

    import pandas as pd  # type: ignore

    df = pd.read_parquet(input_path)
    records = df.to_dict(orient="records")
    return validate_manifest_schema(
        records, allow_research_only=allow_research_only, for_detector=for_detector
    )
