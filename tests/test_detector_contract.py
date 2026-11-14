"""Unit tests for the v2 AnimalDetector contract, MegaDetector license guard, and crop-v1 (§5.2 & §6)."""

import numpy as np
import pytest

from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.detect import AnimalDetector, Detection
from wildlife.detect.crops import (
    compute_square_padded_crop_box,
    extract_crop_rgb,
    select_inference_boxes,
    select_training_boxes,
)
from wildlife.detect.megadetector import (
    AGPLLicenseError,
    MegaDetectorV6Detector,
    nms_detections,
    validate_megadetector_variant_license,
)


def test_megadetector_agpl_variants_rejected():
    # §5.1: Do not use any yolov9/yolov10 variant without the mit/apa prefix
    assert validate_megadetector_variant_license("MDV6-apa-rtdetr-e") == "MDV6-apa-rtdetr-e"
    assert validate_megadetector_variant_license("MDV6-mit-yolov9-c") == "MDV6-mit-yolov9-c"

    for agpl_variant in ["MDV6-yolov9-e", "MDV6-yolov10-x", "MegaDetectorV6", ""]:
        with pytest.raises(AGPLLicenseError):
            validate_megadetector_variant_license(agpl_variant)


def test_detection_dataclass_invariants_and_iou():
    d1 = Detection(x1=10, y1=20, x2=110, y2=120, label="animal", confidence=0.9, source="megadetector_v6")
    d2 = Detection(x1=10, y1=20, x2=110, y2=120, label="animal", confidence=0.8, source="owlv2")
    assert d1.width == 100 and d1.height == 100 and d1.area == 10000
    assert abs(d1.iou(d2) - 1.0) < 1e-6

    kept = nms_detections([d1, d2], iou_threshold=0.5)
    assert len(kept) == 1 and kept[0].confidence == 0.9

    with pytest.raises(ValueError, match="x2 > x1"):
        Detection(x1=50, y1=50, x2=40, y2=80, label="animal", confidence=0.5, source="manual")


def test_detector_protocol_and_require_torch_flag():
    det = MegaDetectorV6Detector(variant="MDV6-apa-rtdetr-e", test_mode=True)
    assert isinstance(det, AnimalDetector)
    rgb = generate_synthetic_species_rgb(0, sample_idx=1, size=128)
    boxes = det.detect(rgb)
    assert len(boxes) >= 1
    assert boxes[0].label == "animal"

    # Empty uniform frame returns no boxes
    empty_rgb = np.full((128, 128, 3), 120, dtype=np.uint8)
    assert det.detect(empty_rgb) == []


def test_crop_v1_selection_and_square_padding():
    # §6.1.1: Training selects the largest animal box within 0.1 confidence of top box
    small_high = Detection(x1=10, y1=10, x2=50, y2=50, label="animal", confidence=0.92, source="megadetector_v6")
    large_near = Detection(x1=60, y1=60, x2=200, y2=200, label="animal", confidence=0.85, source="megadetector_v6")
    chosen = select_training_boxes([small_high, large_near], tau_det=0.25)
    assert chosen == large_near

    inf_boxes = select_inference_boxes([small_high, large_near], tau_det=0.25, max_boxes=5)
    assert inf_boxes[0] == small_high and inf_boxes[1] == large_near

    # §6.1.2-4: 15% padding, square expansion, clamping inward at edges, 512px output
    rgb = generate_synthetic_species_rgb(12, sample_idx=2, size=300)
    crop_512, meta = extract_crop_rgb(rgb, box=(5, 20, 85, 140), out_size=512, pad_fraction=0.15)
    assert crop_512.shape == (512, 512, 3)
    sx1, sy1, sx2, sy2 = meta["square_crop_box"]
    assert (sx2 - sx1) == (sy2 - sy1)
    assert sx1 >= 0 and sy1 >= 0 and sx2 <= 300 and sy2 <= 300
