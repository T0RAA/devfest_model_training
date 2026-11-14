"""OWLv2 zero-shot open-vocabulary fallback detector (`google/owlv2-base-patch16-ensemble`, Apache-2.0) (§5.1)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from wildlife.detect import Detection


class OWLv2FallbackDetector:
    """Zero-shot open-vocabulary animal detector invoked only when MegaDetector V6 finds no animal >= tau_det."""

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        queries: Sequence[str] = ("a photo of a bird", "a photo of a mammal"),
        min_confidence: float = 0.18,
    ) -> None:
        self.model_id = model_id
        self.queries = list(queries)
        self.min_confidence = float(min_confidence)
        self.version = f"owlv2@{model_id.split('/')[-1]}"
        self._processor = None
        self._model = None

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore

            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
            self._model.eval()
            return True
        except Exception:
            return False

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        if not self._lazy_load():
            return []

        import torch  # type: ignore
        from PIL import Image  # type: ignore

        h, w = rgb.shape[:2]
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        inputs = self._processor(text=[self.queries], images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.Tensor([[h, w]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.min_confidence
        )[0]

        detections: List[Detection] = []
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        for box, score in zip(boxes, scores):
            x1 = max(0, min(w - 2, int(round(float(box[0])))))
            y1 = max(0, min(h - 2, int(round(float(box[1])))))
            x2 = max(x1 + 1, min(w, int(round(float(box[2])))))
            y2 = max(y1 + 1, min(h, int(round(float(box[3])))))
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label="animal",
                    confidence=round(float(score), 4),
                    source="owlv2",
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
