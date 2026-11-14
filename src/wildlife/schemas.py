"""Canonical schemas and taxonomy loader for UCSC Wildlife Vision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import urllib.parse
import yaml


MANIFEST_COLUMNS: List[str] = [
    "image_id",
    "observation_id",
    "observer_id",
    "photo_id",
    "source_dataset",
    "source_url",
    "photo_license",
    "photographer_credit",
    "scientific_name",
    "class_id",
    "taxon_id",
    "date_observed",
    "coarse_habitat",
    "region_tag",
    "image_path",
    "image_sha256",
    "phash",
    "width",
    "height",
    "label_reviewed",
    "crop_box",
    "split",
]

VALID_GROUPS = {"large_mammal", "small_mammal", "bird"}
VALID_STATUSES = {"identified", "uncertain", "unsupported_animal", "no_animal"}
VALID_SPLITS = {"train", "validation", "test", "calibration", "field_holdout", "ood"}


@dataclass(frozen=True)
class SpeciesRecord:
    class_id: str
    index: int
    common_name: str
    scientific_name: str
    group: str
    inventory_group: str
    taxon_id: int
    aliases: List[str] = field(default_factory=list)
    campus_habitat: str = ""
    primary_source: str = ""
    subspecies_filter: Optional[str] = None
    confusion_pair: Optional[str] = None
    hard_negatives: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def inaturalist_search_url(self) -> str:
        encoded = urllib.parse.quote(self.scientific_name)
        return (
            f"https://www.inaturalist.org/observations"
            f"?taxon_name={encoded}&quality_grade=research&photos=true"
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "index": self.index,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "group": self.group,
            "taxon_id": self.taxon_id,
            "aliases": list(self.aliases),
            "campus_habitat": self.campus_habitat,
            "inaturalist_search_url": self.inaturalist_search_url,
        }


@dataclass
class ManifestRow:
    image_id: str
    observation_id: str
    observer_id: str
    photo_id: str
    source_dataset: str
    source_url: str
    photo_license: str
    photographer_credit: str
    scientific_name: str
    class_id: str
    taxon_id: int
    date_observed: str
    coarse_habitat: str
    region_tag: str
    image_path: str
    image_sha256: str
    phash: str
    width: int
    height: int
    label_reviewed: bool
    crop_box: str
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictedSpecies:
    class_id: str
    common_name: str
    scientific_name: str
    group: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TopCandidate:
    class_id: str
    probability: float
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    group: Optional[str] = None

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "class_id": self.class_id,
            "probability": round(float(self.probability), 4),
        }
        if not minimal:
            if self.common_name:
                base["common_name"] = self.common_name
            if self.scientific_name:
                base["scientific_name"] = self.scientific_name
            if self.group:
                base["group"] = self.group
        return base


@dataclass
class PhotoQualityScores:
    overall: int
    sharpness: int
    visibility: int
    exposure: int
    resolution: int
    composition: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "overall": int(self.overall),
            "sharpness": int(self.sharpness),
            "visibility": int(self.visibility),
            "exposure": int(self.exposure),
            "resolution": int(self.resolution),
            "composition": int(self.composition),
        }


@dataclass
class PredictionResponse:
    status: str
    predicted_species: Optional[PredictedSpecies]
    confidence_calibrated: Optional[float]
    top_3: List[TopCandidate]
    photo_quality: PhotoQualityScores
    identification_ready_rating: Optional[int]
    explanation: str
    model_version: str = "species-v1.0"
    calibration_version: str = "temp-v1"
    rating_version: str = "quality-v1"
    argon_model_version: str = "gemini-4-argon"
    coarse_group_fallback: Optional[str] = None
    crop_used: bool = False
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status,
            "predicted_species": (
                self.predicted_species.to_dict() if self.predicted_species else None
            ),
            "confidence_calibrated": (
                round(float(self.confidence_calibrated), 4)
                if self.confidence_calibrated is not None
                else None
            ),
            "top_3": [c.to_dict(minimal=True) for c in self.top_3[:3]],
            "photo_quality": self.photo_quality.to_dict(),
            "identification_ready_rating": self.identification_ready_rating,
            "explanation": self.explanation,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "rating_version": self.rating_version,
            "argon_model_version": self.argon_model_version,
            "crop_used": self.crop_used,
        }
        if self.coarse_group_fallback:
            payload["coarse_group_fallback"] = self.coarse_group_fallback
        if self.note:
            payload["note"] = self.note
        return payload


def default_species_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "species.yaml"


def load_species_taxonomy(config_path: Optional[Path] = None) -> List[SpeciesRecord]:
    path = Path(config_path) if config_path else default_species_config_path()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    records: List[SpeciesRecord] = []
    seen_ids = set()
    seen_indices = set()
    for item in raw.get("species", []):
        rec = SpeciesRecord(
            class_id=str(item["class_id"]),
            index=int(item["index"]),
            common_name=str(item["common_name"]),
            scientific_name=str(item["scientific_name"]),
            group=str(item["group"]),
            inventory_group=str(item.get("inventory_group", "")),
            taxon_id=int(item["taxon_id"]),
            aliases=list(item.get("aliases", [])),
            campus_habitat=str(item.get("campus_habitat", "")),
            primary_source=str(item.get("primary_source", "")),
            subspecies_filter=item.get("subspecies_filter"),
            confusion_pair=item.get("confusion_pair"),
            hard_negatives=list(item.get("hard_negatives", [])),
            notes=str(item.get("notes", "")),
        )
        if rec.group not in VALID_GROUPS:
            raise ValueError(f"Unsupported group '{rec.group}' for {rec.class_id}")
        if rec.class_id in seen_ids:
            raise ValueError(f"Duplicate class_id '{rec.class_id}'")
        if rec.index in seen_indices:
            raise ValueError(f"Duplicate index '{rec.index}'")
        seen_ids.add(rec.class_id)
        seen_indices.add(rec.index)
        records.append(rec)
    records.sort(key=lambda r: r.index)
    if len(records) != 20 or [r.index for r in records] != list(range(20)):
        raise ValueError("Expected exactly 20 contiguous species indices 0..19")
    return records


def build_classes_Index_map(config_path: Optional[Path] = None) -> Dict[str, Any]:
    records = load_species_taxonomy(config_path)
    return {
        "version": "1.0.0",
        "num_classes": len(records),
        "class_ids": [r.class_id for r in records],
        "class_to_idx": {r.class_id: r.index for r in records},
        "idx_to_class": {str(r.index): r.class_id for r in records},
        "species_by_id": {
            r.class_id: {
                "index": r.index,
                "common_name": r.common_name,
                "scientific_name": r.scientific_name,
                "group": r.group,
                "taxon_id": r.taxon_id,
            }
            for r in records
        },
    }


def export_classes_json(output_path: Path, config_path: Optional[Path] = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_classes_Index_map(config_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return output_path


def resolve_species_by_name_or_alias(
    query: str, records: Optional[List[SpeciesRecord]] = None
) -> Optional[SpeciesRecord]:
    species_list = records if records is not None else load_species_taxonomy()
    norm = query.strip().lower()
    for rec in species_list:
        if (
            rec.class_id.lower() == norm
            or rec.scientific_name.lower() == norm
            or rec.common_name.lower() == norm
            or any(alias.lower() == norm for alias in rec.aliases)
        ):
            return rec
    return None
