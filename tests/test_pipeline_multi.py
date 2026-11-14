"""Unit tests for multi-animal detection, primary box selection, and IoU > 0.5 disagreement (Milestone 11, §9.1.3 & §9.2)."""

import numpy as np

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.detect import Detection
from wildlife.detect.megadetector import MegaDetectorV6Detector
from wildlife.infer import WildlifePredictor


def test_pipeline_two_boxes_returns_two_results_and_selects_primary_by_conf_times_area():
    detector = MegaDetectorV6Detector(variant="MDV6-apa-rtdetr-e", tau_det=0.25, test_mode=True)
    # Box 0: smaller box (50x50 = 2500 px), conf=0.92 -> score = 2300
    # Box 1: larger box (100x100 = 10000 px), conf=0.85 -> score = 8500 (should be primary_index=1)
    detector.set_mock_detections(
        [
            Detection(x1=10, y1=10, x2=60, y2=60, label="animal", confidence=0.92, source="megadetector_v6"),
            Detection(x1=90, y1=90, x2=190, y2=190, label="animal", confidence=0.85, source="megadetector_v6"),
        ]
    )
    predictor = WildlifePredictor(checkpoint_path=None, detector=detector)

    rgb = generate_synthetic_species_rgb(0, sample_idx=2, size=220)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    # Override logits per box: box 0 -> B03 (idx 12), box 1 -> M01 (idx 0)
    logits_2x20 = np.full((2, 20), -2.5, dtype=np.float64)
    logits_2x20[0, 12] = 5.0
    logits_2x20[1, 0] = 5.2

    res = predictor.predict_bytes(png_bytes, logits_override=logits_2x20)
    v2 = res.to_v2_dict()

    assert len(v2["detections"]) == 2
    assert v2["detections"][0]["species"]["class_id"] == "B03"
    assert v2["detections"][1]["species"]["class_id"] == "M01"
    assert v2["primary_index"] == 1
    assert v2["primary"]["species"]["class_id"] == "M01"
    assert v2["predicted_species"]["class_id"] == "M01"


def test_overlapping_boxes_with_disagreeing_species_become_uncertain():
    # §9.1.3: Keep `uncertain` when two boxes overlap with IoU > 0.5 but disagree on species
    detector = MegaDetectorV6Detector(
        variant="MDV6-apa-rtdetr-e", tau_det=0.25, nms_iou=0.95, test_mode=True
    )
    detector.set_mock_detections(
        [
            Detection(x1=20, y1=20, x2=140, y2=140, label="animal", confidence=0.88, source="megadetector_v6"),
            Detection(x1=25, y1=25, x2=145, y2=145, label="animal", confidence=0.84, source="megadetector_v6"),
        ]
    )
    predictor = WildlifePredictor(checkpoint_path=None, detector=detector)
    rgb = generate_synthetic_species_rgb(12, sample_idx=1, size=180)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    logits_2x20 = np.full((2, 20), -2.5, dtype=np.float64)
    logits_2x20[0, 12] = 5.0  # B03
    logits_2x20[1, 13] = 5.0  # B04

    res = predictor.predict_bytes(png_bytes, logits_override=logits_2x20)
    assert res.status == "uncertain"
    assert res.predicted_species is None
    assert res.identification_ready_rating is None
