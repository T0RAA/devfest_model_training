"""Photograph quality feature extraction and composite rating rubric (§10)."""

from wildlife.quality.features import (
    compute_composition_score,
    compute_exposure_score,
    compute_resolution_score,
    compute_sharpness_score,
    compute_visibility_score,
    laplacian_focus,
    rgb_to_grayscale_uint8,
)
from wildlife.quality.scoring import (
    QUALITY_WEIGHTS,
    compute_identification_ready_rating,
    evaluate_quality_benchmark,
    render_quality_examples_png,
    score_photograph,
)

__all__ = [
    "QUALITY_WEIGHTS",
    "compute_composition_score",
    "compute_exposure_score",
    "compute_identification_ready_rating",
    "compute_resolution_score",
    "compute_sharpness_score",
    "compute_visibility_score",
    "evaluate_quality_benchmark",
    "laplacian_focus",
    "render_quality_examples_png",
    "rgb_to_grayscale_uint8",
    "score_photograph",
]
