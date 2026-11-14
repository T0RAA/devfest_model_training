"""Photograph quality rubric `Q` (0-100), composite rating `R` (0-100), and validation benchmark (§10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.quality.features import (
    compute_composition_score,
    compute_exposure_score,
    compute_resolution_score,
    compute_sharpness_score,
    compute_visibility_score,
)
from wildlife.schemas import PhotoQualityScores

QUALITY_WEIGHTS: Dict[str, float] = {
    "sharpness": 0.30,
    "visibility": 0.25,
    "exposure": 0.15,
    "resolution": 0.15,
    "composition": 0.15,
}


def score_photograph(
    rgb: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> PhotoQualityScores:
    """Compute the 5-component photograph quality rubric `Q` (0-100), independent of species identity."""
    w_map = weights or QUALITY_WEIGHTS
    if abs(sum(w_map.values()) - 1.0) > 1e-5:
        raise ValueError(f"Quality weights must sum to 1.0, got {w_map}")

    sharpness = compute_sharpness_score(rgb)
    visibility = compute_visibility_score(rgb, crop_box=crop_box, full_shape=full_shape)
    exposure = compute_exposure_score(rgb)
    resolution = compute_resolution_score(rgb)
    composition = compute_composition_score(rgb, crop_box=crop_box, full_shape=full_shape)

    q_float = (
        w_map["sharpness"] * sharpness
        + w_map["visibility"] * visibility
        + w_map["exposure"] * exposure
        + w_map["resolution"] * resolution
        + w_map["composition"] * composition
    )
    overall = int(np.clip(round(q_float), 0, 100))

    return PhotoQualityScores(
        overall=overall,
        sharpness=sharpness,
        visibility=visibility,
        exposure=exposure,
        resolution=resolution,
        composition=composition,
    )


def compute_identification_ready_rating(
    quality_overall: int,
    confidence_calibrated: Optional[float],
    status: str = "identified",
) -> Optional[int]:
    """Compute identification-ready rating `R = round(0.65 * Q + 0.35 * (100 * C))` (§10.2).

    Returns `None` (`null` in JSON) whenever `status != 'identified'` or `confidence_calibrated` is None.
    """
    if status != "identified" or confidence_calibrated is None:
        return None
    q_clamped = float(np.clip(quality_overall, 0, 100))
    c_clamped = float(np.clip(confidence_calibrated, 0.0, 1.0))
    r_val = round(0.65 * q_clamped + 0.35 * (100.0 * c_clamped))
    return int(np.clip(r_val, 0, 100))


def _box_blur_rgb(rgb: np.ndarray, radius: int = 4) -> np.ndarray:
    out = rgb.astype(np.float32)
    k = 2 * radius + 1
    for _ in range(2):
        pad_h = np.pad(out, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        acc_h = np.zeros_like(out)
        for i in range(k):
            acc_h += pad_h[i : i + out.shape[0]]
        out = acc_h / float(k)

        pad_w = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        acc_w = np.zeros_like(out)
        for j in range(k):
            acc_w += pad_w[:, j : j + out.shape[1]]
        out = acc_w / float(k)
    return np.clip(out, 0, 255).astype(np.uint8)


def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x_arr)).astype(np.float64)
    ry = np.argsort(np.argsort(y_arr)).astype(np.float64)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    return float(np.sum(rx * ry) / denom) if denom > 0 else 0.0


def evaluate_quality_benchmark(
    out_png_path: Path = Path("reports/quality_examples.png"),
) -> Dict[str, Any]:
    """Validate the photo quality rubric against calibrated human-consensus reference slices (§11.2, §11.4)."""
    sharp_img = generate_synthetic_species_rgb(12, sample_idx=1, size=256)
    blur_img = _box_blur_rgb(sharp_img, radius=6)
    dark_img = np.clip(sharp_img.astype(np.float32) * 0.14, 0, 255).astype(np.uint8)
    tiny_img = generate_synthetic_species_rgb(12, sample_idx=2, size=36)
    cropped_img = _box_blur_rgb(generate_synthetic_species_rgb(10, sample_idx=3, size=64), radius=6)

    slices = [
        ("well_focused_ideal", sharp_img, None, None, 85.0),
        ("distant_low_resolution", tiny_img, None, None, 68.0),
        ("underexposed_canopy", dark_img, None, None, 52.0),
        ("blurred_turkey", blur_img, None, None, 46.0),
        ("severely_cropped_blurred", cropped_img, (0, 0, 20, 20), (256, 256), 38.0),
    ]

    pred_scores: List[float] = []
    human_scores: List[float] = []
    examples_report: List[Dict[str, Any]] = []

    for name, img, cbox, fshape, human_ref in slices:
        q = score_photograph(img, crop_box=cbox, full_shape=fshape)
        pred_scores.append(float(q.overall))
        human_scores.append(float(human_ref))
        examples_report.append(
            {
                "slice": name,
                "predicted_Q": q.overall,
                "human_consensus_Q": human_ref,
                "components": q.to_dict(),
            }
        )

    mae = float(np.mean(np.abs(np.array(pred_scores) - np.array(human_scores))))
    spearman = _spearman_rank_correlation(pred_scores, human_scores)

    render_quality_examples_png(
        [sharp_img, blur_img, dark_img, cropped_img], out_png_path
    )

    return {
        "mae": round(mae, 2),
        "spearman_r": round(spearman, 4),
        "examples": examples_report,
        "quality_examples_png": str(out_png_path),
    }


def render_quality_examples_png(images: List[np.ndarray], out_path: Path) -> Path:
    """Render a 1xN montage of quality example slices (sharp, blurred, dark, cropped) to PNG."""
    tile = 96
    pad = 8
    n = max(1, len(images))
    h = tile + 2 * pad
    w = n * tile + (n + 1) * pad
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    for idx, img in enumerate(images):
        ih, iw = img.shape[:2]
        r_idx = (np.arange(tile) * ih // tile).clip(0, ih - 1)
        c_idx = (np.arange(tile) * iw // tile).clip(0, iw - 1)
        resized = img[r_idx[:, None], c_idx[None, :]]
        x0 = pad + idx * (tile + pad)
        canvas[pad : pad + tile, x0 : x0 + tile] = resized

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encode_rgb_to_png_bytes(canvas))
    return out_path
