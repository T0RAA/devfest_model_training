"""Unit tests for temperature scaling, calibration metrics, and checkpoint class-order checks."""

from pathlib import Path
import pickle
import numpy as np
import pytest

from wildlife.models.calibration import (
    TemperatureScaler,
    calibrated_softmax,
    compute_brier_score,
    compute_ece,
    compute_nll,
)
from wildlife.models.classifier import (
    CheckpointClassOrderError,
    WildlifeClassifier,
    load_checkpoint,
    save_checkpoint,
)


def test_temperature_scaling_reduces_nll_on_overconfident_logits():
    rng = np.random.RandomState(42)
    n, k = 120, 20
    labels = rng.randint(0, k, size=n)
    # Construct overconfident logits where ~70% are correct but scaled by 6x
    logits = rng.randn(n, k) * 0.5
    for i in range(n):
        if i % 10 < 7:
            logits[i, labels[i]] += 2.2
        else:
            wrong = (labels[i] + 1) % k
            logits[i, wrong] += 2.0
    overconfident_logits = logits * 5.0

    scaler = TemperatureScaler(initial_temperature=1.0)
    report = scaler.fit(overconfident_logits, labels)
    assert scaler.temperature > 1.0
    assert report["post_calibration"]["nll"] <= report["pre_calibration"]["nll"]
    assert 0.0 <= report["post_calibration"]["ece"] <= 1.0
    assert 0.0 <= report["post_calibration"]["brier"] <= 2.0

    probs = scaler.transform(overconfident_logits)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)


def test_checkpoint_class_mismatch_or_reordered_classes_fails_loudly(tmp_path: Path):
    # Failure case #3: Class mismatch or changed class-order checkpoint -> load must fail loudly
    model = WildlifeClassifier(num_classes=20, seed=42, use_torch_if_available=False)
    ckpt_path = tmp_path / "valid.ckpt"
    save_checkpoint(model, ckpt_path, epoch=1, config={}, metrics={})

    loaded_model, payload = load_checkpoint(ckpt_path)
    assert loaded_model.num_classes == 20

    # Tamper with class order (swap M01 and M02)
    tampered = dict(payload)
    reordered = list(tampered["class_ids"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    tampered["class_ids"] = reordered

    bad_ckpt = tmp_path / "reordered.ckpt"
    bad_ckpt.write_bytes(pickle.dumps(tampered))
    with pytest.raises(CheckpointClassOrderError, match="class list/order mismatch"):
        load_checkpoint(bad_ckpt)
