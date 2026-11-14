"""Post-hoc temperature scaling and calibration metrics (ECE, NLL, Brier score) (§9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def calibrated_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute calibrated probabilities `p = softmax(logits / T)` for `T > 0`."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature T must be strictly positive, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    probs = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
    return probs.astype(np.float64)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean negative log-likelihood on held-out examples."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    eps = 1e-12
    chosen = np.clip(p[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multi-class Brier score: mean squared error between probability vector and one-hot label."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def compute_reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Tuple_ECE:
    """Compute Expected Calibration Error (ECE) and reliability diagram bin statistics."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if n == 0:
        raise ValueError("Cannot compute ECE on empty dataset.")

    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins_out: List[Dict[str, Any]] = []

    for b in range(num_bins):
        low, high = float(bin_boundaries[b]), float(bin_boundaries[b + 1])
        if b == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / float(n)) * gap
        else:
            avg_conf = 0.5 * (low + high)
            avg_acc = 0.0
            gap = 0.0

        bins_out.append(
            {
                "bin_index": b,
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "calibration_gap": round(gap, 4),
            }
        )

    return float(ece), bins_out


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    ece, _ = compute_reliability_bins(probs, labels, num_bins=num_bins)
    return ece


class TemperatureScaler:
    """Single-parameter post-hoc temperature scaling fit via NLL minimization on calibration split."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        if initial_temperature <= 0.0:
            raise ValueError("Initial temperature must be > 0.")
        self.temperature: float = float(initial_temperature)
        self.version: str = "temp-v1"

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        t_min: float = 0.05,
        t_max: float = 10.0,
    ) -> Dict[str, Any]:
        """Minimize negative log-likelihood over `T in [t_min, t_max]` using coarse-to-fine grid + Newton refinement."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)

        pre_probs = calibrated_softmax(logits_arr, 1.0)
        pre_nll = compute_nll(pre_probs, labels_arr)
        pre_ece = compute_ece(pre_probs, labels_arr)
        pre_brier = compute_brier_score(pre_probs, labels_arr)

        # Log-spaced coarse grid search followed by fine local search around optimum
        coarse_grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), 200))
        best_t = 1.0
        best_nll = pre_nll
        for t in coarse_grid:
            p = calibrated_softmax(logits_arr, float(t))
            nll = compute_nll(p, labels_arr)
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        fine_low = max(t_min, best_t * 0.75)
        fine_high = min(t_max, best_t * 1.25)
        fine_grid = np.linspace(fine_low, fine_high, 200)
        for t in fine_grid:
            p = calibrated_softmax(logits_arr, float(t))
            nll = compute_nll(p, labels_arr)
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = round(float(best_t), 4)
        post_probs = calibrated_softmax(logits_arr, self.temperature)
        post_nll = compute_nll(post_probs, labels_arr)
        post_ece = compute_ece(post_probs, labels_arr)
        post_brier = compute_brier_score(post_probs, labels_arr)

        return {
            "calibration_version": self.version,
            "temperature": self.temperature,
            "pre_calibration": {
                "nll": round(pre_nll, 4),
                "ece": round(pre_ece, 4),
                "brier": round(pre_brier, 4),
            },
            "post_calibration": {
                "nll": round(post_nll, 4),
                "ece": round(post_ece, 4),
                "brier": round(post_brier, 4),
            },
        }

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return calibrated_softmax(logits, self.temperature)

    def save(self, path: Path, extra_metadata: Dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "calibration_version": self.version,
            "temperature": self.temperature,
        }
        if extra_metadata:
            payload.update(extra_metadata)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(initial_temperature=float(data.get("temperature", 1.0)))
        scaler.version = str(data.get("calibration_version", "temp-v1"))
        return scaler
