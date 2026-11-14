"""Unit and integration tests for FastAPI endpoints and operational safeguards (Milestone 5)."""

from pathlib import Path
import struct
import numpy as np
import pytest

from api.main import LocalAPIClient, WildlifeAPIService
from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb


def _make_client(tmp_path: Path) -> LocalAPIClient:
    svc = WildlifeAPIService(
        checkpoint_path=None,
        feedback_queue_path=tmp_path / "feedback_queue.jsonl",
        rate_limit_per_minute=5,
    )
    return LocalAPIClient(service=svc)


def test_healthz_and_classes_endpoints(tmp_path: Path):
    client = _make_client(tmp_path)

    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["status"] == "ok"
    assert h["model_version"] == "species-v1.0"
    assert h["calibration_version"] == "temp-v1"
    assert h["rating_version"] == "quality-v1"
    assert h["argon_model_version"] == "gemini-4-argon"
    assert h["num_classes"] == 20

    r_classes = client.get("/v1/classes")
    assert r_classes.status_code == 200
    c = r_classes.json()
    assert len(c["classes"]) == 20
    assert set(c["supported_groups"]) == {"bird", "large_mammal", "small_mammal"}


def test_predict_endpoint_schema_and_accepted_identification(tmp_path: Path):
    client = _make_client(tmp_path)
    rgb = generate_synthetic_species_rgb(12, sample_idx=1, size=160)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    logits = np.full((20,), -2.5, dtype=np.float64)
    logits[12] = 4.8

    resp = client.post(
        "/v1/predict",
        files={"file": ("steller.png", png_bytes, "image/png")},
        data={"logits_override": logits},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "identified"
    assert body["predicted_species"] == {
        "class_id": "B03",
        "common_name": "Steller's jay",
        "scientific_name": "Cyanocitta stelleri",
        "group": "bird",
    }
    assert 0.80 <= body["confidence_calibrated"] <= 1.0
    assert len(body["top_3"]) == 3
    assert body["top_3"][0]["class_id"] == "B03"
    assert (
        body["top_3"][0]["probability"]
        >= body["top_3"][1]["probability"]
        >= body["top_3"][2]["probability"]
    )
    q = body["photo_quality"]
    for key in ["overall", "sharpness", "visibility", "exposure", "resolution", "composition"]:
        assert 0 <= q[key] <= 100
    assert body["identification_ready_rating"] == round(
        0.65 * q["overall"] + 0.35 * (100.0 * body["confidence_calibrated"])
    )
    assert body["model_version"] == "species-v1.0"
    assert body["calibration_version"] == "temp-v1"
    assert body["rating_version"] == "quality-v1"
    assert "Steller's jay" in body["explanation"]


def test_predict_multiple_animals_without_crop_prompts_crop(tmp_path: Path):
    client = _make_client(tmp_path)
    rgb = generate_synthetic_species_rgb(0, sample_idx=1, size=128)
    png_bytes = encode_rgb_to_png_bytes(rgb)

    resp = client.post(
        "/v1/predict",
        files={"file": ("herd.png", png_bytes, "image/png")},
        data={"multiple_animals": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "uncertain"
    assert body["predicted_species"] is None
    assert body["identification_ready_rating"] is None
    assert "Multiple animals" in body["explanation"]


def test_predict_security_ssrf_and_malformed_uploads(tmp_path: Path):
    client = _make_client(tmp_path)

    r_ssrf = client.post(
        "/v1/predict",
        data={"image_url": "remote-fetch-target-metadata"},
    )
    assert r_ssrf.status_code == 400
    assert r_ssrf.json()["error"] == "ssrf_prohibited"

    r_bad_type = client.post(
        "/v1/predict",
        files={"file": ("script.sh", b"#!/bin/bash\necho hi", "text/plain")},
    )
    assert r_bad_type.status_code == 415

    ihdr = struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0)
    bomb_png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
    r_bomb = client.post(
        "/v1/predict",
        files={"file": ("bomb.png", bomb_png, "image/png")},
    )
    assert r_bomb.status_code == 413


def test_feedback_endpoint_stores_for_review_and_rate_limits(tmp_path: Path):
    client = _make_client(tmp_path)

    r_no_consent = client.post(
        "/v1/feedback",
        json_body={"predicted_class_id": "B03", "corrected_class_id": "B04", "consent": False},
    )
    assert r_no_consent.status_code == 400

    r_ok = client.post(
        "/v1/feedback",
        json_body={
            "predicted_class_id": "B03",
            "corrected_class_id": "B04",
            "consent": True,
            "notes": "No crest visible, blue and gray back.",
        },
    )
    assert r_ok.status_code == 202
    assert r_ok.json()["status"] == "queued_for_review"
    assert r_ok.json()["auto_retrain"] is False
    assert (tmp_path / "feedback_queue.jsonl").exists()
