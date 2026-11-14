"""Unit tests for person/vehicle-only frames returning `no_animal` with no species output (Milestone 11, §9.2)."""

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.detect import Detection
from wildlife.detect.megadetector import MegaDetectorV6Detector
from wildlife.infer import WildlifePredictor


def test_pipeline_person_only_returns_no_animal_and_no_species():
    detector = MegaDetectorV6Detector(variant="MDV6-apa-rtdetr-e", tau_det=0.25, test_mode=True)
    detector.set_mock_detections(
        [
            Detection(
                x1=20,
                y1=15,
                x2=120,
                y2=150,
                label="person",
                confidence=0.94,
                source="megadetector_v6",
            )
        ]
    )
    predictor = WildlifePredictor(checkpoint_path=None, detector=detector)

    rgb = generate_synthetic_species_rgb(0, sample_idx=1, size=160)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    res = predictor.predict_bytes(png_bytes)
    assert res.status == "no_animal"
    assert res.predicted_species is None
    assert res.confidence_calibrated is None
    assert res.identification_ready_rating is None
    assert res.top_3 == []
    assert res.detections == []
    assert res.primary_index is None
