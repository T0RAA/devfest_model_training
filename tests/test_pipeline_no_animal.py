"""Unit tests for empty landscape frames returning `no_animal` (Milestone 11, §9.2)."""

import numpy as np

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.detect.megadetector import MegaDetectorV6Detector
from wildlife.infer import WildlifePredictor


def test_pipeline_landscape_returns_no_animal():
    detector = MegaDetectorV6Detector(variant="MDV6-apa-rtdetr-e", tau_det=0.25, test_mode=True)
    detector.set_mock_detections([])
    predictor = WildlifePredictor(checkpoint_path=None, detector=detector)

    # Empty landscape image
    rgb = np.full((160, 200, 3), 110, dtype=np.uint8)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    res = predictor.predict_bytes(png_bytes)
    assert res.status == "no_animal"
    assert res.predicted_species is None
    assert res.confidence_calibrated is None
    assert res.identification_ready_rating is None
    assert res.detections == []
    assert res.primary_index is None
