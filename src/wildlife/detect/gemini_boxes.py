"""Gemini box coordinate conversion (`[ymin, xmin, ymax, xmax]` 0..1000 -> pixel `[x1, y1, x2, y2]`) and M7.5 audit."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence, Tuple


def gemini_box_to_pixels(
    box_2d: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Convert Gemini `[ymin, xmin, ymax, xmax]` normalized 0..1000 box to pixel `(x1, y1, x2, y2)` (§5.2).

    Swaps axis order from `[ymin, xmin, ymax, xmax]` to `(x1, y1, x2, y2)`, converts with
    `x = round(v / 1000 * width)` and `y = round(v / 1000 * height)`, clamps to `[0, width]`
    and `[0, height]`, and enforces `x2 > x1` and `y2 > y1`.
    """
    if len(box_2d) != 4:
        raise ValueError(f"Expected 4-element box_2d [ymin, xmin, ymax, xmax], got {box_2d}")
    if width <= 1 or height <= 1:
        raise ValueError(f"Invalid image dimensions width={width}, height={height}")

    ymin_norm, xmin_norm, ymax_norm, xmax_norm = [float(v) for v in box_2d]
    if ymin_norm > ymax_norm:
        ymin_norm, ymax_norm = ymax_norm, ymin_norm
    if xmin_norm > xmax_norm:
        xmin_norm, xmax_norm = xmax_norm, xmin_norm

    x1 = int(round(xmin_norm / 1000.0 * width))
    y1 = int(round(ymin_norm / 1000.0 * height))
    x2 = int(round(xmax_norm / 1000.0 * width))
    y2 = int(round(ymax_norm / 1000.0 * height))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1:
        if x1 < width - 1:
            x2 = x1 + 1
        else:
            x1 = max(0, width - 2)
            x2 = width
    if y2 <= y1:
        if y1 < height - 1:
            y2 = y1 + 1
        else:
            y1 = max(0, height - 2)
            y2 = height

    return (int(x1), int(y1), int(x2), int(y2))


def format_gemini_box_audit_flag(
    one_visible_wild_animal: bool,
    box_is_tight: bool,
    looks_like_expected_species: str,
    corrected_box_2d: Sequence[float] | None = None,
) -> Tuple[str, bool]:
    """Serialize M7.5 Gemini box audit response into `gemini_flag` JSON and flag whether human review is needed.

    Per Ground Rule #4 and §5.4:
    - Never writes to a species label column.
    - Flags rows where Gemini says 'no' for `data/manifests/review_queue.csv`.
    """
    if looks_like_expected_species not in {"yes", "no", "cannot tell"}:
        raise ValueError("looks_like_expected_species must be 'yes', 'no', or 'cannot tell'")

    needs_human_review = (
        (not one_visible_wild_animal)
        or (not box_is_tight)
        or (looks_like_expected_species == "no")
    )
    payload: Dict[str, Any] = {
        "one_visible_wild_animal": bool(one_visible_wild_animal),
        "box_is_tight": bool(box_is_tight),
        "looks_like_expected_species": looks_like_expected_species,
        "corrected_box_2d": list(corrected_box_2d) if corrected_box_2d else None,
        "needs_human_review": needs_human_review,
        "label_policy": "review_only_never_ground_truth",
    }
    return json.dumps(payload, separators=(",", ":")), needs_human_review
