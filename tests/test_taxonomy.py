"""Unit tests for the 20-species UCSC taxonomy and frequency audit (Milestone 0)."""

from pathlib import Path
import json
import pytest

from wildlife.schemas import (
    VALID_GROUPS,
    build_classes_Index_map,
    export_classes_json,
    load_species_taxonomy,
    resolve_species_by_name_or_alias,
)


def test_20_species_taxonomy_completeness():
    records = load_species_taxonomy()
    assert len(records) == 20
    class_ids = [r.class_id for r in records]
    expected_ids = [f"M{i:02d}" for i in range(1, 11)] + [f"B{i:02d}" for i in range(1, 11)]
    assert class_ids == expected_ids
    assert [r.index for r in records] == list(range(20))

    for rec in records:
        assert rec.group in VALID_GROUPS
        assert rec.scientific_name
        assert rec.common_name
        assert rec.taxon_id > 0
        assert "taxon_name=" in rec.inaturalist_search_url
        assert "quality_grade=research" in rec.inaturalist_search_url
        assert "photos=true" in rec.inaturalist_search_url


def test_mammal_and_bird_group_mappings():
    records = {r.class_id: r for r in load_species_taxonomy()}
    for cid in ["M01", "M02", "M03", "M04"]:
        assert records[cid].group == "large_mammal"
    # Per SPEC.md §2.1, raccoons (M05) and skunks (M06) are mapped to small_mammal in the 3-way UI
    for cid in ["M05", "M06", "M07", "M08", "M09", "M10"]:
        assert records[cid].group == "small_mammal"
    for cid in [f"B{i:02d}" for i in range(1, 11)]:
        assert records[cid].group == "bird"


def test_western_scrub_jay_alias_and_black_tailed_deer_subspecies():
    scrub_jay = resolve_species_by_name_or_alias("Western scrub-jay")
    assert scrub_jay is not None
    assert scrub_jay.class_id == "B04"
    assert scrub_jay.scientific_name == "Aphelocoma californica"

    deer = resolve_species_by_name_or_alias("Columbian black-tailed deer")
    assert deer is not None
    assert deer.class_id == "M01"
    assert deer.scientific_name == "Odocoileus hemionus"
    assert deer.subspecies_filter == "Odocoileus hemionus columbianus"


def test_export_classes_json(tmp_path: Path):
    out_file = tmp_path / "classes.json"
    export_classes_json(out_file)
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["num_classes"] == 20
    assert data["class_to_idx"]["M01"] == 0
    assert data["class_to_idx"]["B10"] == 19
    assert data["idx_to_class"]["12"] == "B03"
