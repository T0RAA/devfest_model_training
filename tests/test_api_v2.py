"""Integration tests for `/v2/predict` and backwards-compatible `/v1/predict` wrapper (Milestone 11, §9.1.5 & §9.2)."""

from pathlib import Path
import numpy as np

from api.main import LocalAPIClient, WildlifeAPIService
from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.detect import Detection
from wildlife.detect.megadetector import MegaDetectorV6Detector


def test_v2_predict_returns_detections_array_and_versions(tmp_path: Path):
    detector = MegaDetectorV6Detector(variant="MDV6-apa-rtdetr-e", tau_det=0.25, test_mode=True)
    detector.set_mock_detections(
        [
            Detection(x1=15, y1=20, x2=140, y2=145, label="animal", confidence=0.91, source="megadetector_v6")
        ]
    )
    svc = WildlifeAPIService(
        checkpoint_path=None,
        detector=detector,
        feedback_queue_path=tmp_path / "fb.jsonl",
    )
    client = LocalAPIClient(service=svc)

    rgb = generate_synthetic_species_rgb(12, sample_idx=1, size=160)
    png_bytes = encode_rgb_to_png_bytes(rgb)
    logits = np.full((20,), -2.5, dtype=np.float64)
    logits[12] = 4.9

    r_v2 = client.post(
        "/v2/predict",
        files={"file": ("jay.png", png_bytes, "image/png")},
        data={"logits_override": logits},
    )
    assert r_v2.status_code == 200
    b2 = r_v2.json()
    assert b2["detector_version"] == "md6-apa-rtdetr-e@apache2.0"
    assert b2["crop_version"] == "crop-v1"
    assert b2["primary_index"] == 0
    assert len(b2["detections"]) == 1
    det0 = b2["detections"][0]
    assert det0["box"] == [15, 20, 140, 145]
    assert det0["box_source"] == "megadetector_v6"
    assert det0["detector_confidence"] == 0.91
    assert det0["status"] == "identified"
    assert det0["species"]["class_id"] == "B03"
    assert det0["rating"] is not None

    # /v1/predict wrapper returns primary only
    r_v1 = client.post(
        "/v1/predict",
        files={"file": ("jay.png", png_bytes, "image/png")},
        data={"logits_override": logits},
    )
    assert r_v1.status_code == 200
    b1 = r_v1.json()
    assert b1["status"] == "identified"
    assert b1["predicted_species"]["class_id"] == "B03"
    assert "detections" not in b1
