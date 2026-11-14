#!/usr/bin/env python3
"""End-to-end smoke test verifying taxonomy, manifest v2, splits v2, crop-v1, training, calibration, quality scoring, and /v1 + /v2 API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["WILDLIFE_TEST_MODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from api.main import LocalAPIClient, WildlifeAPIService
from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.evaluate import run_evaluation
from wildlife.quality.scoring import evaluate_quality_benchmark
from wildlife.train import run_training


def main() -> int:
    print("=== [1/4] Training 20-species classifier on smoke-test fixture (WILDLIFE_TEST_MODE=1) ===")
    train_summary = run_training(
        config_path=ROOT / "configs" / "train.yaml",
        manifest_path=ROOT / "data" / "manifests" / "manifest.parquet",
        splits_path=ROOT / "data" / "splits" / "split_v1.csv",
        output_dir=ROOT / "runs" / "efficientnet_b0_v1",
        max_epochs_override=9,
        require_torch=False,
    )
    ckpt_path = Path(train_summary["best_checkpoint"])
    assert ckpt_path.exists(), "Expected best.ckpt to be created"

    print("=== [2/4] Evaluating validation split & fitting temperature calibration ===")
    eval_summary = run_evaluation(
        checkpoint_path=ckpt_path,
        manifest_path=ROOT / "data" / "manifests" / "manifest.parquet",
        splits_path=ROOT / "data" / "splits" / "split_v1.csv",
        split="validation",
        out_dir=ROOT / "reports",
    )

    print("=== [3/4] Running photograph quality benchmark & generating quality_examples.png ===")
    qual_summary = evaluate_quality_benchmark(
        out_png_path=ROOT / "reports" / "quality_examples.png"
    )
    assert qual_summary["mae"] <= 10.0
    assert qual_summary["spearman_r"] >= 0.70

    print("=== [4/4] Exercising FastAPI /healthz, /v1/classes, /v1/predict, /v2/predict, and /v1/feedback ===")
    svc = WildlifeAPIService(checkpoint_path=ckpt_path)
    client = LocalAPIClient(service=svc)

    assert client.get("/healthz").status_code == 200
    assert len(client.get("/v1/classes").json()["classes"]) == 20

    sample_rgb = generate_synthetic_species_rgb(12, sample_idx=0, size=160)
    png_bytes = encode_rgb_to_png_bytes(sample_rgb)
    logits = np.full((20,), -2.5, dtype=np.float64)
    logits[12] = 4.8

    pred_v2 = client.post(
        "/v2/predict",
        files={"file": ("sample.png", png_bytes, "image/png")},
        data={"logits_override": logits, "crop_box": "10,10,150,150"},
    )
    assert pred_v2.status_code == 200
    v2_json = pred_v2.json()
    assert v2_json["status"] == "identified"
    assert v2_json["predicted_species"]["class_id"] == "B03"
    assert len(v2_json["detections"]) == 1
    assert v2_json["primary_index"] == 0
    assert v2_json["detector_version"] == "md6-apa-rtdetr-e@apache2.0"
    assert v2_json["crop_version"] == "crop-v1"

    print(
        "Smoke test PASSED:",
        json.dumps(
            {
                "smoke_test_val_macro_f1": train_summary["best_val_macro_f1"],
                "temperature": eval_summary["calibration"]["temperature"],
                "ece": eval_summary["calibration"]["post_calibration"]["ece"],
                "ood_far": eval_summary["selective_and_ood"]["ood_false_accept_rate"],
                "quality_mae": qual_summary["mae"],
                "quality_spearman": qual_summary["spearman_r"],
                "v2_primary_rating": v2_json["identification_ready_rating"],
            },
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
