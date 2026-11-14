"""Unit tests for v2 licensing rules (Decision D1), OOD disjointness, and label provenance (Milestone 6)."""

from pathlib import Path
import pytest
import yaml

from wildlife.data.licenses import (
    LicenseViolationError,
    audit_manifest_licenses,
    validate_photo_license,
)
from wildlife.data.manifest import ManifestSchemaError, validate_manifest_schema
from wildlife.data.split import SplitLeakageError, verify_no_split_leakage


def test_decision_d1_cdla_permissive_allowed_for_detector_only():
    # Decision D1: cdla-permissive-1.0 is allowed ONLY when for_detector=True and source is LILA
    assert (
        validate_photo_license(
            "cdla-permissive-1.0", source_dataset="lila_cct", for_detector=True
        )
        == "cdla-permissive-1.0"
    )
    assert (
        validate_photo_license(
            "cdla-permissive", source_dataset="lila_ena24", for_detector=True
        )
        == "cdla-permissive-1.0"
    )

    # Must fail when for_detector=False (classifier training manifest)
    with pytest.raises(LicenseViolationError):
        validate_photo_license(
            "cdla-permissive-1.0", source_dataset="lila_cct", for_detector=False
        )

    # Even if license is marked cc0, LILA source is blocked from classifier manifest
    with pytest.raises(LicenseViolationError):
        validate_photo_license("cc0", source_dataset="lila_ena24", for_detector=False)


def test_gemini_output_rejected_as_label_source():
    # Ground Rule #4: Labels come from humans, not from Gemini
    row = {
        "image_id": "img_M01_1",
        "observation_id": "obs_1",
        "observer_id": "u_1",
        "photo_id": "p_1",
        "source_dataset": "inat_open_data",
        "source_url": "inaturalist://photos/1",
        "photo_license": "cc-by",
        "photographer_credit": "Observer 1",
        "scientific_name": "Odocoileus hemionus",
        "class_id": "M01",
        "taxon_id": 42158,
        "date_observed": "2026-10-01",
        "coarse_habitat": "Meadows",
        "region_tag": "US-CA",
        "image_path": "data/raw/inat/42158/1.jpg",
        "image_sha256": "a" * 64,
        "phash": "0123456789abcdef",
        "width": 1024,
        "height": 768,
        "label_reviewed": True,
        "crop_box": "0,0,1024,768",
        "split": "train",
        "label_source": "gemini-4-argon",
    }
    with pytest.raises(ManifestSchemaError, match="Gemini output must never be a label"):
        validate_manifest_schema([row])


def test_ood_taxa_yaml_disjoint_between_ood_tune_and_ood_test():
    ood_path = Path(__file__).resolve().parents[1] / "configs" / "ood_taxa.yaml"
    cfg = yaml.safe_load(ood_path.read_text(encoding="utf-8"))
    taxa = cfg["ood_taxa"]
    assert len(taxa) >= 18

    tune_ids = {t["taxon_id"] for t in taxa if t["ood_group"] == "ood_tune"}
    test_ids = {t["taxon_id"] for t in taxa if t["ood_group"] == "ood_test"}
    assert len(tune_ids) > 0 and len(test_ids) > 0
    assert tune_ids.isdisjoint(test_ids), "OOD taxon IDs must be strictly disjoint between ood_tune and ood_test"

    # Also verify split leakage check catches an OOD taxon appearing in both ood_tune and ood_test
    bad_ood_rows = [
        {
            "image_id": "ood_1",
            "observation_id": "obs_ood_1",
            "observer_id": "u1",
            "class_id": "OOD_8021",
            "taxon_id": 8021,
            "is_ood": True,
            "split": "ood_tune",
            "image_sha256": "sha_ood_1",
            "phash": "1111111111111111",
        },
        {
            "image_id": "ood_2",
            "observation_id": "obs_ood_2",
            "observer_id": "u2",
            "class_id": "OOD_8021",
            "taxon_id": 8021,
            "is_ood": True,
            "split": "ood_test",
            "image_sha256": "sha_ood_2",
            "phash": "eeeeeeeeeeeeeeee",
        },
    ]
    with pytest.raises(SplitLeakageError, match="OOD taxon ID appeared in both"):
        verify_no_split_leakage(bad_ood_rows)
