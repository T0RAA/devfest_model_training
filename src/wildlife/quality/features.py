"""Photographic quality feature extractors (sharpness, visibility, exposure, resolution, composition) (§10.1)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def rgb_to_grayscale_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 RGB image to HxW uint8 grayscale."""
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb.shape}")
    lum = (
        0.299 * rgb[:, :, 0].astype(np.float64)
        + 0.587 * rgb[:, :, 1].astype(np.float64)
        + 0.114 * rgb[:, :, 2].astype(np.float64)
    )
    return np.clip(np.round(lum), 0, 255).astype(np.uint8)


def laplacian_focus(gray_uint8: np.ndarray) -> float:
    """Compute raw Laplacian variance focus measure (SPEC.md §10.1).

    Uses `cv2.Laplacian(gray_uint8, cv2.CV_64F).var()` when OpenCV is installed,
    and falls back to the exact 4-neighbor discrete Laplacian kernel
    `[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` in NumPy otherwise.
    """
    if gray_uint8.ndim != 2 or gray_uint8.size == 0:
        raise ValueError("Expected non-empty grayscale image")

    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(gray_uint8, cv2.CV_64F).var())
    except Exception:
        g = gray_uint8.astype(np.float64)
        if g.shape[0] < 3 or g.shape[1] < 3:
            return float(np.var(g))
        padded = np.pad(g, pad_width=1, mode="reflect")
        lap = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * padded[1:-1, 1:-1]
        )
        return float(np.var(lap))


def compute_sharpness_score(rgb: np.ndarray) -> int:
    """Map raw Laplacian variance + resolution scale to a bounded [0, 100] sharpness score."""
    gray = rgb_to_grayscale_uint8(rgb)
    raw_var = laplacian_focus(gray)
    h, w = gray.shape
    scale_norm = math.sqrt(max(1.0, float(h * w)) / (224.0 * 224.0))
    adj_var = raw_var / max(0.35, min(2.0, scale_norm))

    # Log-logistic calibration mapping calibrated on sharp vs blurred wildlife crops
    if adj_var <= 1.0:
        return 0
    log_v = math.log10(max(1.0, adj_var))
    # log10(5) ~ 0.70 -> ~15, log10(80) ~ 1.90 -> ~65, log10(500) ~ 2.70 -> ~92
    score = 100.0 * (log_v - 0.30) / (2.90 - 0.30)
    return int(np.clip(round(score), 0, 100))


def compute_exposure_score(rgb: np.ndarray) -> int:
    """Evaluate luminance balance, dynamic range, and highlight/shadow clipping [0, 100]."""
    gray = rgb_to_grayscale_uint8(rgb).astype(np.float64)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    shadow_clip = float(np.mean(gray <= 8.0))
    highlight_clip = float(np.mean(gray >= 247.0))

    # Ideal mean luminance ~122; penalize extreme darkness or blown highlights
    mean_penalty = abs(mean_lum - 122.0) / 122.0 * 48.0
    clip_penalty = (shadow_clip + highlight_clip) * 95.0
    contrast_bonus = min(18.0, (std_lum / 48.0) * 18.0)

    raw_score = 88.0 - mean_penalty - clip_penalty + (contrast_bonus - 10.0)
    return int(np.clip(round(raw_score), 0, 100))


def compute_resolution_score(
    rgb: np.ndarray,
    min_side_px: int = 48,
    ideal_side_px: int = 384,
) -> int:
    """Score useful animal-crop resolution [0, 100], penalizing tiny or upsampled crops."""
    h, w = rgb.shape[:2]
    short_side = float(min(h, w))
    if short_side <= min_side_px:
        base = 15.0 * (short_side / float(max(1, min_side_px)))
    elif short_side >= ideal_side_px:
        base = 92.0 + 8.0 * min(1.0, (short_side - ideal_side_px) / float(ideal_side_px))
    else:
        ratio = (short_side - min_side_px) / float(ideal_side_px - min_side_px)
        base = 15.0 + 77.0 * math.sqrt(ratio)
    return int(np.clip(round(base), 0, 100))


def compute_visibility_score(
    rgb: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None,
) -> int:
    """Estimate subject visibility and border-truncation proxy [0, 100].

    Per SPEC.md §10.1, treats V1 visibility as manual crop + edge truncation / foreground
    contrast proxy rather than claiming a bounding box alone measures occlusion.
    """
    gray = rgb_to_grayscale_uint8(rgb).astype(np.float64)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 10

    # Measure gradient energy in interior vs outer 8% border (detects severe edge cutoffs)
    gy, gx = np.gradient(gray)
    grad_mag = np.hypot(gx, gy)
    total_energy = float(np.mean(grad_mag)) + 1e-6

    bh = max(1, int(round(h * 0.10)))
    bw = max(1, int(round(w * 0.10)))
    interior = grad_mag[bh : h - bh, bw : w - bw]
    interior_energy = float(np.mean(interior)) if interior.size > 0 else total_energy

    border_mask = np.ones((h, w), dtype=bool)
    border_mask[bh : h - bh, bw : w - bw] = False
    border_energy = float(np.mean(grad_mag[border_mask])) if np.any(border_mask) else 0.0

    # High interior detail relative to border truncation indicates well-contained subject
    containment_ratio = interior_energy / max(1e-6, interior_energy + 0.85 * border_energy)
    score = 25.0 + 68.0 * containment_ratio

    # Penalize extreme crop boxes touching full-image boundaries or tiny crop fractions
    if crop_box is not None and full_shape is not None:
        x1, y1, x2, y2 = crop_box
        fh, fw = full_shape
        crop_area_ratio = max(0, (x2 - x1) * (y2 - y1)) / float(max(1, fh * fw))
        if crop_area_ratio < 0.04:
            score -= 25.0
        touches_edge = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= fw - 1) + int(y2 >= fh - 1)
        if touches_edge >= 2 and crop_area_ratio < 0.50:
            score -= 18.0

    if total_energy < 2.5:
        score = min(score, 22.0)

    return int(np.clip(round(score), 0, 100))


def compute_composition_score(
    rgb: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None,
) -> int:
    """Score subject placement (center / rule-of-thirds) and aspect framing [0, 100]."""
    gray = rgb_to_grayscale_uint8(rgb).astype(np.float64)
    h, w = gray.shape
    gy, gx = np.gradient(gray)
    salience = np.hypot(gx, gy) + 1e-6

    yy, xx = np.mgrid[0:h, 0:w]
    cy = float(np.sum(yy * salience) / np.sum(salience)) / float(max(1, h))
    cx = float(np.sum(xx * salience) / np.sum(salience)) / float(max(1, w))

    # Distance to center (0.5, 0.5) or nearest rule-of-thirds intersection (1/3, 2/3)
    anchors = [
        (0.5, 0.5),
        (1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ]
    min_dist = min(math.hypot(cx - ax, cy - ay) for ax, ay in anchors)
    placement_score = max(0.0, 100.0 - min_dist * 210.0)

    aspect = float(max(h, w)) / float(max(1, min(h, w)))
    aspect_penalty = max(0.0, (aspect - 1.8) * 20.0)

    raw = 0.80 * placement_score + 20.0 - aspect_penalty
    return int(np.clip(round(raw), 0, 100))
