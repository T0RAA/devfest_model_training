"""Crop dataset builder (`crop-v1`), 15% square context expansion, train jitter, and WebDataset shard writer (§6)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wildlife.data.clean import decode_image_bytes, encode_rgb_to_png_bytes
from wildlife.detect import CROP_VERSION_DEFAULT, Detection


def select_training_boxes(
    detections: Sequence[Detection],
    tau_det: float = 0.25,
) -> Optional[Detection]:
    """Select the training animal box per §6.1.1:
    Take the largest `animal` box whose confidence is within 0.1 of the top animal box >= tau_det.
    """
    animals = [
        d for d in detections if d.label == "animal" and d.confidence >= tau_det
    ]
    if not animals:
        return None
    top_conf = max(d.confidence for d in animals)
    candidates = [d for d in animals if (top_conf - d.confidence) <= 0.10 + 1e-6]
    candidates.sort(key=lambda d: (d.area, d.confidence), reverse=True)
    return candidates[0]


def select_inference_boxes(
    detections: Sequence[Detection],
    tau_det: float = 0.25,
    max_boxes: int = 5,
) -> List[Detection]:
    """Select up to `max_boxes` animal detections >= tau_det ordered by confidence."""
    animals = [
        d for d in detections if d.label == "animal" and d.confidence >= tau_det
    ]
    animals.sort(key=lambda d: d.confidence, reverse=True)
    return animals[:max_boxes]


def compute_square_padded_crop_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_fraction: float = 0.15,
    jitter_rng: Optional[np.random.RandomState] = None,
) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
    """Compute a 15%-padded square crop box using real image context (shifting inward at edges, §6.1).

    Also supports train-time box jitter (scale 0.9..1.25, center shift +-5% of box size, §6.3)
    and returns geometry metadata (`inside_frame_ratio`, `touches_edge`) for Photo Quality `visibility`.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    if jitter_rng is not None:
        scale = float(jitter_rng.uniform(0.90, 1.25))
        dx = float(jitter_rng.uniform(-0.05, 0.05)) * bw
        dy = float(jitter_rng.uniform(-0.05, 0.05)) * bh
        cx += dx
        cy += dy
        bw *= scale
        bh *= scale

    # Pad 15% of box width and height on each side (total 1 + 2*0.15 = 1.30x)
    padded_w = bw * (1.0 + 2.0 * pad_fraction)
    padded_h = bh * (1.0 + 2.0 * pad_fraction)

    # Expand shorter side to make the region square
    side = max(padded_w, padded_h)
    unclamped_x1 = cx - 0.5 * side
    unclamped_y1 = cy - 0.5 * side
    unclamped_x2 = cx + 0.5 * side
    unclamped_y2 = cy + 0.5 * side

    # Measure how much of the ideal padded square lies inside the frame before clamping
    inter_x1 = max(0.0, unclamped_x1)
    inter_y1 = max(0.0, unclamped_y1)
    inter_x2 = min(float(img_w), unclamped_x2)
    inter_y2 = min(float(img_h), unclamped_y2)
    ideal_area = max(1.0, (unclamped_x2 - unclamped_x1) * (unclamped_y2 - unclamped_y1))
    inside_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    inside_frame_ratio = float(np.clip(inside_area / ideal_area, 0.0, 1.0))

    # Clamp side to max square that fits in the image, and shift inward if hitting an edge
    side_int = max(16, min(int(round(side)), int(min(img_w, img_h))))
    sx1 = int(round(cx - 0.5 * side_int))
    sy1 = int(round(cy - 0.5 * side_int))

    if sx1 < 0:
        sx1 = 0
    elif sx1 + side_int > img_w:
        sx1 = max(0, img_w - side_int)

    if sy1 < 0:
        sy1 = 0
    elif sy1 + side_int > img_h:
        sy1 = max(0, img_h - side_int)

    sx2 = min(img_w, sx1 + side_int)
    sy2 = min(img_h, sy1 + side_int)

    touches_edge = bool(
        box[0] <= 2 or box[1] <= 2 or box[2] >= img_w - 2 or box[3] >= img_h - 2
    )
    meta = {
        "crop_version": CROP_VERSION_DEFAULT,
        "raw_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
        "square_crop_box": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "inside_frame_ratio": round(inside_frame_ratio, 4),
        "touches_edge": touches_edge,
    }
    return (int(sx1), int(sy1), int(sx2), int(sy2)), meta


def resize_rgb_square(rgb: np.ndarray, out_size: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == out_size and w == out_size:
        return rgb.copy()
    r_idx = (np.arange(out_size) * h // out_size).clip(0, h - 1)
    c_idx = (np.arange(out_size) * w // out_size).clip(0, w - 1)
    return rgb[r_idx[:, None], c_idx[None, :]].astype(np.uint8)


def extract_crop_rgb(
    rgb: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    out_size: int = 512,
    pad_fraction: float = 0.15,
    jitter_rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract a 15%-padded square context crop and resize to `out_size` (default 512 px, §6.1)."""
    h, w = rgb.shape[:2]
    if box is None:
        # Full-image square center crop fallback (crop_mode = 'full', §6.2)
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = rgb[y1 : y1 + side, x1 : x1 + side]
        resized = resize_rgb_square(crop, out_size=out_size)
        return resized, {
            "crop_version": CROP_VERSION_DEFAULT,
            "crop_mode": "full",
            "raw_box": [0, 0, w, h],
            "square_crop_box": [x1, y1, x1 + side, y1 + side],
            "inside_frame_ratio": 1.0,
            "touches_edge": False,
        }

    (sx1, sy1, sx2, sy2), meta = compute_square_padded_crop_box(
        box=box,
        img_w=w,
        img_h=h,
        pad_fraction=pad_fraction,
        jitter_rng=jitter_rng,
    )
    meta["crop_mode"] = "detector_crop"
    crop = rgb[sy1:sy2, sx1:sx2]
    resized = resize_rgb_square(crop, out_size=out_size)
    return resized, meta


def write_webdataset_shard(
    samples: Iterable[Dict[str, Any]],
    shard_tar_path: Path,
) -> int:
    """Write a WebDataset `.tar` shard holding `<key>.png`, `<key>.cls`, and `<key>.json` (§6.4)."""
    shard_tar_path = Path(shard_tar_path)
    shard_tar_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(shard_tar_path, "w") as tar:
        for sample in samples:
            key = str(sample["image_id"])
            img_bytes: bytes = sample["image_bytes"]
            cls_idx = int(sample.get("class_idx", -1))
            meta_json = json.dumps(sample["metadata"], separators=(",", ":")).encode("utf-8")
            cls_bytes = f"{cls_idx}".encode("utf-8")

            for ext, payload in [("png", img_bytes), ("cls", cls_bytes), ("json", meta_json)]:
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            count += 1
    return count
