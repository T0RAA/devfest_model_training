"""Unit tests for Gemini normalized box_2d conversion and M7.5 review-only flags (§5.2 & §5.4)."""

import json
import pytest

from wildlife.detect.gemini_boxes import (
    format_gemini_box_audit_flag,
    gemini_box_to_pixels,
)


def test_gemini_box_to_pixels_axis_swap_and_scaling():
    # Gemini returns [ymin, xmin, ymax, xmax] in 0..1000
    x1, y1, x2, y2 = gemini_box_to_pixels([100, 250, 600, 750], width=800, height=400)
    assert x1 == 200  # 250 / 1000 * 800
    assert y1 == 40   # 100 / 1000 * 400
    assert x2 == 600  # 750 / 1000 * 800
    assert y2 == 240  # 600 / 1000 * 400


def test_gemini_box_to_pixels_clamping_and_inverted_coords():
    # Out-of-bounds and inverted coordinates are safely ordered and clamped to image bounds
    x1, y1, x2, y2 = gemini_box_to_pixels([-50, 950, 1200, 200], width=500, height=300)
    assert 0 <= x1 < x2 <= 500
    assert 0 <= y1 < y2 <= 300
    assert x1 == 100 and x2 == 475
    assert y1 == 0 and y2 == 300

    # Degenerate zero-width box is expanded by 1px so x2 > x1
    gx1, gy1, gx2, gy2 = gemini_box_to_pixels([500, 500, 500, 500], width=200, height=200)
    assert gx2 > gx1 and gy2 > gy1


def test_format_gemini_box_audit_flag_never_modifies_labels():
    flag_json, needs_review = format_gemini_box_audit_flag(
        one_visible_wild_animal=True,
        box_is_tight=False,
        looks_like_expected_species="yes",
        corrected_box_2d=[120, 150, 880, 820],
    )
    assert needs_review is True
    parsed = json.loads(flag_json)
    assert parsed["label_policy"] == "review_only_never_ground_truth"
    assert parsed["corrected_box_2d"] == [120, 150, 880, 820]
