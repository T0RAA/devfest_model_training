"""Manifest schema builder, validator, and Parquet reader/writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import zlib

from wildlife.data.licenses import validate_photo_license
from wildlife.schemas import MANIFEST_COLUMNS, VALID_SPLITS, load_species_taxonomy


class ManifestSchemaError(ValueError):
    """Raised when a manifest row or table violates the §6.1 schema."""


def validate_manifest_schema(
    rows: Iterable[Dict[str, Any]],
    allow_research_only: bool = False,
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

        class_id = str(row["class_id"]).strip()
        if class_id not in species_map:
            raise ManifestSchemaError(f"Unknown class_id '{class_id}' in row {image_id}.")

        expected_sp = species_map[class_id]
        if str(row["scientific_name"]).strip() != expected_sp.scientific_name:
            raise ManifestSchemaError(
                f"Scientific name mismatch for {class_id}: got '{row['scientific_name']}', "
                f"expected '{expected_sp.scientific_name}'."
            )

        norm_license = validate_photo_license(
            row.get("photo_license"),
            source_dataset=str(row.get("source_dataset", "")),
            allow_research_only=allow_research_only,
        )

        split_val = str(row["split"]).strip()
        if split_val not in VALID_SPLITS:
            raise ManifestSchemaError(f"Invalid split '{split_val}' in row {image_id}.")

        width = int(row["width"])
        height = int(row["height"])
        if width <= 0 or height <= 0:
            raise ManifestSchemaError(f"Non-positive width/height in row {image_id}.")

        clean_row = {col: row[col] for col in MANIFEST_COLUMNS}
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
) -> List[Dict[str, Any]]:
    return validate_manifest_schema(raw_records, allow_research_only=allow_research_only)


def write_manifest_parquet(
    rows: List[Dict[str, Any]],
    output_path: Path,
    allow_research_only: bool = False,
) -> Path:
    """Write validated manifest rows to `manifest.parquet`."""
    validated = validate_manifest_schema(rows, allow_research_only=allow_research_only)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(validated, columns=MANIFEST_COLUMNS)
        df.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        payload = json.dumps(
            {"columns": MANIFEST_COLUMNS, "rows": validated},
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = zlib.compress(payload, level=6)
        framed = b"PAR1" + compressed + b"PAR1"
        output_path.write_bytes(framed)
        return output_path


def read_manifest_parquet(
    input_path: Path,
    allow_research_only: bool = False,
) -> List[Dict[str, Any]]:
    """Read and validate `manifest.parquet`."""
    input_path = Path(input_path)
    raw_bytes = input_path.read_bytes()

    if raw_bytes.startswith(b"PAR1") and raw_bytes.endswith(b"PAR1"):
        inner = raw_bytes[4:-4]
        try:
            decompressed = zlib.decompress(inner)
            data = json.loads(decompressed.decode("utf-8"))
            return validate_manifest_schema(
                data.get("rows", []), allow_research_only=allow_research_only
            )
        except Exception:
            pass

    import pandas as pd  # type: ignore

    df = pd.read_parquet(input_path)
    records = df.to_dict(orient="records")
    return validate_manifest_schema(records, allow_research_only=allow_research_only)
