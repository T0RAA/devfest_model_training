"""Unit tests for photograph quality rubric `Q` and composite rating `R` (Milestone 4)."""

import numpy as np
import pytest

from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.quality.features import laplacian_focus
from wildlife.quality.scoring import (
    _box_blur_rgb,
    compute_identification_ready_rating,
    evaluate_quality_benchmark,
    score_photograph,
)


def test_laplacian_focus_validates_input_and_detects_blur():
    with pytest.raises(ValueError, match="Expected non-empty grayscale image"):
        laplacian_focus(np.zeros((10, 10, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected non-empty grayscale image"):
        laplacian_focus(np.array([], dtype=np.uint8).reshape(0, 0))

    sharp_rgb = generate_synthetic_species_rgb(12, sample_idx=0, size=160)
    blurred_rgb = _box_blur_rgb(sharp_rgb, radius=5)

    q_sharp = score_photograph(sharp_rgb)
    q_blur = score_photograph(blurred_rgb)
    assert q_sharp.sharpness > q_blur.sharpness + 25
    assert q_sharp.overall > q_blur.overall


def test_Composite_rating_spec_formula_and_failure_cases():
    # SPEC.md §10.2 exact example: Q=82, C=0.91 -> R=85
    r_val = compute_identification_ready_rating(82, 0.91, status="identified")
    assert r_val == 85

    # Failure case #6: A sharp, bright, beautiful unsupported animal -> high Q is possible;
    # species remains uncertain/unsupported, R is null (None).
    sharp_unsupported = generate_synthetic_species_rgb(3, sample_idx=5, size=256)
    q_unsupported = score_photograph(sharp_unsupported)
    assert q_unsupported.overall >= 65
    assert compute_identification_ready_rating(q_unsupported.overall, None, status="uncertain") is None
    assert (
        compute_identification_ready_rating(q_unsupported.overall, 0.92, status="unsupported_animal")
        is None
    )

    # Failure case #5: A bird with a severe crop/blur -> low visibility/sharpness
    severely_blurred = _box_blur_rgb(sharp_unsupported[:32, :32], radius=6)
    q_severe = score_photograph(
        severely_blurred, crop_box=(0, 0, 20, 20), full_shape=(256, 256)
    )
    assert q_severe.sharpness < 35
    assert q_severe.visibility < 60
    assert q_severe.overall < 50

    # Failure case #7: A blurry but otherwise recognizable turkey -> confidence may remain high
    # but photo-quality score must remain low.
    r_blurry_turkey = compute_identification_ready_rating(q_severe.overall, 0.92, status="identified")
    assert q_severe.overall < 50
    assert r_blurry_turkey is not None and r_blurry_turkey < 70


def test_quality_benchmark_mae_and_spearman(tmp_path):
    out_png = tmp_path / "quality_examples.png"
    bench = evaluate_quality_benchmark(out_png_path=out_png)
    assert out_png.exists()
    assert bench["mae"] <= 10.0
    assert bench["spearman_r"] >= 0.70
