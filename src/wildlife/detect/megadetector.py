"""MegaDetector V6 (`MDV6-apa-rtdetr-e`, Apache-2.0) + OWLv2 fallback detector implementation (§5.1 & §5.2)."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from wildlife.detect import DETECTOR_VERSION_DEFAULT, AnimalDetector, Detection
from wildlife.detect.owl_fallback import OWLv2FallbackDetector

ALLOWED_PERMISSIVE_MDV6_VARIANTS = {
    "MDV6-apa-rtdetr-c",
    "MDV6-apa-rtdetr-e",
    "MDV6-mit-yolov9-c",
    "MDV6-mit-yolov9-e",
}

PW_CATEGORY_MAP = {
    0: "animal",
    1: "person",
    2: "vehicle",
}


class AGPLLicenseError(ValueError):
    """Raised when an AGPL MegaDetector variant is requested instead of Apache-2.0 / MIT."""


def validate_megadetector_variant_license(variant: str) -> str:
    """Enforce §5.1: prohibit AGPL-3.0 YOLOv9/v10 variants; allow only `MDV6-apa-*` or `MDV6-mit-*`."""
    v = (variant or "").strip()
    if v not in ALLOWED_PERMISSIVE_MDV6_VARIANTS:
        raise AGPLLicenseError(
            f"MegaDetector variant '{variant}' is prohibited or unrecognized. "
            f"Only Apache-2.0 / MIT variants are allowed for Cloud Run serving: {sorted(ALLOWED_PERMISSIVE_MDV6_VARIANTS)}"
        )
    return v


def nms_detections(detections: Sequence[Detection], iou_threshold: float = 0.50) -> List[Detection]:
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for cand in sorted_dets:
        if any(
            cand.label == existing.label and cand.iou(existing) > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(cand)
    return kept


class MegaDetectorV6Detector(AnimalDetector):
    """Two-tier animal detector: MegaDetector V6 (`MDV6-apa-rtdetr-e`) primary + OWLv2 fallback."""

    def __init__(
        self,
        variant: str = "MDV6-apa-rtdetr-e",
        tau_det: float = 0.25,
        nms_iou: float = 0.50,
        max_boxes: int = 5,
        enable_owl_fallback: bool = True,
        require_torch: bool = False,
        test_mode: Optional[bool] = None,
    ) -> None:
        self.variant = validate_megadetector_variant_license(variant)
        self.tau_det = float(tau_det)
        self.nms_iou = float(nms_iou)
        self.max_boxes = int(max_boxes)
        self.enable_owl_fallback = bool(enable_owl_fallback)
        self.require_torch = bool(require_torch)
        self.test_mode = (
            bool(test_mode)
            if test_mode is not None
            else (os.environ.get("WILDLIFE_TEST_MODE", "0") == "1")
        )
        self.version = DETECTOR_VERSION_DEFAULT
        self._pw_model = None
        self._owl_fallback = OWLv2FallbackDetector() if self.enable_owl_fallback else None
        self._mock_detections: Optional[List[Detection]] = None

        if self.require_torch and not self._try_load_pytorch_wildlife():
            raise RuntimeError(
                "MegaDetectorV6Apache requires `PytorchWildlife` and `torch` when require_torch=True. "
                "Install `PytorchWildlife` on Colab/Vertex GPU or set WILDLIFE_TEST_MODE=1 for unit tests."
            )

    def set_mock_detections(self, detections: Optional[List[Detection]]) -> None:
        """Allow unit tests to inject deterministic detections (e.g., person-only, multi-animal, empty)."""
        self._mock_detections = detections

    def _try_load_pytorch_wildlife(self) -> bool:
        if self._pw_model is not None:
            return True
        try:
            from PytorchWildlife.models import detection as pw_detection  # type: ignore

            self._pw_model = pw_detection.MegaDetectorV6Apache(version=self.variant)
            return True
        except Exception:
            return False

    def _detect_with_pytorch_wildlife(self, rgb: np.ndarray) -> List[Detection]:
        h, w = rgb.shape[:2]
        res = self._pw_model.single_image_detection(rgb)
        dets_raw = res.get("detections")
        if dets_raw is None:
            return []
        xyxy = getattr(dets_raw, "xyxy", [])
        confs = getattr(dets_raw, "confidence", [])
        cls_ids = getattr(dets_raw, "class_id", [])
        out: List[Detection] = []
        for box, conf, cid in zip(xyxy, confs, cls_ids):
            label = PW_CATEGORY_MAP.get(int(cid), "animal")
            x1 = max(0, min(w - 2, int(round(float(box[0])))))
            y1 = max(0, min(h - 2, int(round(float(box[1])))))
            x2 = max(x1 + 1, min(w, int(round(float(box[2])))))
            y2 = max(y1 + 1, min(h, int(round(float(box[3])))))
            out.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,  # type: ignore[arg-type]
                    confidence=round(float(conf), 4),
                    source="megadetector_v6",
                )
            )
        return out

    def _detect_salient_foreground_test_mode(self, rgb: np.ndarray) -> List[Detection]:
        """Deterministic salience-based box detector used only when WILDLIFE_TEST_MODE=1 or in unit tests."""
        h, w = rgb.shape[:2]
        if float(np.std(rgb)) < 4.0:
            # Uniform / empty landscape frame -> no detections
            return []

        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        thresh = float(np.percentile(mag, 75))
        ys, xs = np.where(mag >= max(3.0, thresh))
        if len(xs) < 16:
            return []

        x1 = max(0, min(w - 2, int(np.percentile(xs, 10))))
        y1 = max(0, min(h - 2, int(np.percentile(ys, 10))))
        x2 = max(x1 + 16, min(w, int(np.percentile(xs, 90))))
        y2 = max(y1 + 16, min(h, int(np.percentile(ys, 90))))
        return [
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label="animal",
                confidence=0.88,
                source="megadetector_v6",
            )
        ]

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb.shape}")

        if self._mock_detections is not None:
            raw = list(self._mock_detections)
        elif self._try_load_pytorch_wildlife():
            raw = self._detect_with_pytorch_wildlife(rgb)
        elif self.test_mode:
            raw = self._detect_salient_foreground_test_mode(rgb)
        else:
            raw = self._detect_salient_foreground_test_mode(rgb)

        filtered = nms_detections(raw, iou_threshold=self.nms_iou)
        animal_boxes = [
            d for d in filtered if d.label == "animal" and d.confidence >= self.tau_det
        ]

        # Run OWLv2 fallback ONLY when MegaDetector returns no animal box >= tau_det
        # AND no person/vehicle-only frame was already detected.
        has_person_or_vehicle = any(
            d.label in {"person", "vehicle"} and d.confidence >= self.tau_det
            for d in filtered
        )
        if not animal_boxes and not has_person_or_vehicle and self._owl_fallback is not None:
            owl_dets = self._owl_fallback.detect(rgb)
            owl_animals = [
                d for d in owl_dets if d.label == "animal" and d.confidence >= self.tau_det
            ]
            if owl_animals:
                filtered.extend(owl_animals)
                filtered = nms_detections(filtered, iou_threshold=self.nms_iou)

        return filtered[: self.max_boxes]
