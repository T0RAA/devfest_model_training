"""Unit tests for detector_version and crop_version parity enforcement on checkpoint load (Milestone 11, §2.3 #4 & §9.2)."""

from pathlib import Path
import pytest

from wildlife.models.backbones import (
    VersionParityError,
    load_v2_checkpoint,
    save_v2_checkpoint,
)
from wildlife.models.classifier import WildlifeClassifier


def test_checkpoint_with_mismatched_detector_or_crop_version_refuses_to_load(tmp_path: Path):
    model = WildlifeClassifier(num_classes=20, seed=42, use_torch_if_available=False)
    ckpt_path = tmp_path / "v2_model.ckpt"

    save_v2_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        tier="t1",
        backbone="hf-hub:imageomics/bioclip-2",
        input_size=224,
        detector_version="md6-apa-rtdetr-e@apache2.0",
        crop_version="crop-v1",
        train_manifest_sha256="abc123",
        split_version="split_v2",
        git_commit="deadbeef",
        epoch=1,
        config={},
        metrics={},
    )

    # Matching versions load cleanly
    loaded, payload = load_v2_checkpoint(
        ckpt_path,
        expected_detector_version="md6-apa-rtdetr-e@apache2.0",
        expected_crop_version="crop-v1",
    )
    assert loaded.num_classes == 20
    assert payload["tier"] == "t1"

    # Mismatched detector_version must raise VersionParityError
    with pytest.raises(VersionParityError, match="detector_version"):
        load_v2_checkpoint(
            ckpt_path,
            expected_detector_version="md6-mit-yolov9-c@other",
            expected_crop_version="crop-v1",
        )

    # Mismatched crop_version must raise VersionParityError
    with pytest.raises(VersionParityError, match="crop_version"):
        load_v2_checkpoint(
            ckpt_path,
            expected_detector_version="md6-apa-rtdetr-e@apache2.0",
            expected_crop_version="crop-v99-incompatible",
        )
