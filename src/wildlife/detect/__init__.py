"""Two-stage detector contract (`Detection`, `AnimalDetector`), MegaDetector V6, OWLv2 fallback, and crop-v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Protocol, runtime_checkable

import numpy as np

DetectionLabel = Literal["animal", "person", "vehicle"]
DetectionSource = Literal["megadetector_v6", "owlv2", "manual"]

DETECTOR_VERSION_DEFAULT = "md6-apa-rtdetr-e@apache2.0"
CROP_VERSION_DEFAULT = "crop-v1"


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: DetectionLabel
    confidence: float
    source: DetectionSource

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Detection box must have x2 > x1 and y2 > y1, got ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
        if self.label not in {"animal", "person", "vehicle"}:
            raise ValueError(f"Invalid detection label '{self.label}'")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Detection confidence must be in [0, 1], got {self.confidence}")
        if self.source not in {"megadetector_v6", "owlv2", "manual"}:
            raise ValueError(f"Invalid detection source '{self.source}'")

    @property
    def width(self) -> int:
        return int(self.x2 - self.x1)

    @property
    def height(self) -> int:
        return int(self.y2 - self.y1)

    @property
    def area(self) -> int:
        return int(self.width * self.height)

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        union = float(self.area + other.area) - inter
        return inter / union if union > 0.0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class AnimalDetector(Protocol):
    version: str

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        ...
