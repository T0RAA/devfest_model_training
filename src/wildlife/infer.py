"""End-to-end wildlife inference pipeline combining classifier, calibration, OOD gate, photo rating, and Argon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from wildlife.data.clean import strip_exif_and_validate
from wildlife.data.dataset import preprocess_rgb_for_model
from wildlife.models.argon import ArgonMultimodalAssistant
from wildlife.models.classifier import WildlifeClassifier, load_checkpoint
from wildlife.models.unknown import UnknownSpeciesGate
from wildlife.quality.scoring import (
    compute_identification_ready_rating,
    score_photograph,
)
from wildlife.schemas import (
    PredictedSpecies,
    PredictionResponse,
    TopCandidate,
    load_species_taxonomy,
)


def parse_crop_box(
    crop_str_or_tuple: Optional[Any],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if crop_str_or_tuple is None or crop_str_or_tuple == "":
        return None
    if isinstance(crop_str_or_tuple, str):
        parts = [int(float(x.strip())) for x in crop_str_or_tuple.split(",")]
    else:
        parts = [int(float(x)) for x in crop_str_or_tuple]
    if len(parts) != 4:
        raise ValueError("crop_box must contain 4 coordinates: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        raise ValueError("Crop region must be at least 16x16 pixels.")
    return (x1, y1, x2, y2)


class WildlifePredictor:
    """Stateful inference engine for the 20-species UCSC Wildlife Vision service."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        enable_live_argon: bool = False,
    ) -> None:
        self.model_version = "species-v1.0"
        self.calibration_version = "temp-v1"
        self.rating_version = "quality-v1"
        self.argon_model_version = "gemini-4-argon"
        self.temperature = 1.0
        self.image_size = 96

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, payload = load_checkpoint(Path(checkpoint_path))
            self.temperature = float(payload.get("temperature", 1.0))
            cfg = payload.get("config", {})
            self.image_size = int(cfg.get("image_size", 96))
        else:
            self.model = WildlifeClassifier(
                num_classes=20, seed=42, pretrained=False, use_torch_if_available=False
            )

        self.gate = UnknownSpeciesGate(
            max_prob_threshold=0.80,
            margin_threshold=0.15,
            energy_threshold=-1.20,
        )
        self.argon = ArgonMultimodalAssistant(enable_live_api=enable_live_argon)
        self.species_records = load_species_taxonomy()

    def predict_bytes(
        self,
        raw_bytes: bytes,
        crop_box: Optional[Any] = None,
        multiple_animals_reported: bool = False,
        scene_hint: Optional[str] = None,
        logits_override: Optional[np.ndarray] = None,
    ) -> PredictionResponse:
        """Validate uploaded image bytes, strip EXIF, optionally crop, classify, and score quality."""
        safe_png, full_rgb, meta = strip_exif_and_validate(raw_bytes)
        full_h, full_w = int(meta["height"]), int(meta["width"])

        parsed_crop = parse_crop_box(crop_box, width=full_w, height=full_h)
        crop_used = parsed_crop is not None
        if parsed_crop is not None:
            x1, y1, x2, y2 = parsed_crop
            eval_rgb = full_rgb[y1:y2, x1:x2]
        else:
            eval_rgb = full_rgb

        quality = score_photograph(
            eval_rgb,
            crop_box=parsed_crop,
            full_shape=(full_h, full_w),
        )

        if multiple_animals_reported and not crop_used:
            top_cands = [
                TopCandidate(
                    class_id=r.class_id,
                    probability=round(1.0 / 20.0, 4),
                    common_name=r.common_name,
                    scientific_name=r.scientific_name,
                    group=r.group,
                )
                for r in self.species_records[:3]
            ]
            return PredictionResponse(
                status="uncertain",
                predicted_species=None,
                confidence_calibrated=None,
                top_3=top_cands,
                photo_quality=quality,
                identification_ready_rating=None,
                explanation=(
                    "Multiple animals reported in frame; V1 requires a manual crop around one "
                    "individual animal before assigning a single species identification."
                ),
                model_version=self.model_version,
                calibration_version=self.calibration_version,
                rating_version=self.rating_version,
                argon_model_version=self.argon_model_version,
                crop_used=False,
                note="Please provide crop_box coordinates isolating a single animal.",
            )

        if logits_override is not None:
            logits_1d = np.asarray(logits_override, dtype=np.float64)
        else:
            chw = preprocess_rgb_for_model(
                eval_rgb, image_size=self.image_size, is_train=False
            )
            logits_1d = self.model.forward_logits(chw)[0]

        if scene_hint is None and float(np.std(eval_rgb)) < 3.0:
            scene_hint = "no_animal"

        gate_res = self.gate.evaluate_single(
            logits_1d, temperature=self.temperature, scene_hint=scene_hint
        )
        status = str(gate_res["status"])
        pred_dict = gate_res["predicted_species"]
        pred_sp = (
            PredictedSpecies(
                class_id=pred_dict["class_id"],
                common_name=pred_dict["common_name"],
                scientific_name=pred_dict["scientific_name"],
                group=pred_dict["group"],
            )
            if pred_dict
            else None
        )
        conf_cal = gate_res["confidence_calibrated"]
        top_3_objs = [
            TopCandidate(
                class_id=item["class_id"],
                probability=item["probability"],
                common_name=item.get("common_name"),
                scientific_name=item.get("scientific_name"),
                group=item.get("group"),
            )
            for item in gate_res["top_3"]
        ]

        r_rating = compute_identification_ready_rating(
            quality_overall=quality.overall,
            confidence_calibrated=conf_cal,
            status=status,
        )

        explanation = self.argon.explain_prediction(
            status=status,
            predicted_species=pred_dict,
            top_3=gate_res["top_3"],
            photo_quality=quality.to_dict(),
            coarse_group_fallback=gate_res.get("coarse_group_fallback"),
            crop_used=crop_used,
            image_bytes=safe_png,
        )

        return PredictionResponse(
            status=status,
            predicted_species=pred_sp,
            confidence_calibrated=conf_cal,
            top_3=top_3_objs,
            photo_quality=quality,
            identification_ready_rating=r_rating,
            explanation=explanation,
            model_version=self.model_version,
            calibration_version=self.calibration_version,
            rating_version=self.rating_version,
            argon_model_version=self.argon_model_version,
            coarse_group_fallback=gate_res.get("coarse_group_fallback"),
            crop_used=crop_used,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/efficientnet_b0_v1/best.ckpt"),
    )
    parser.add_argument("--crop", type=str, default=None, help="x1,y1,x2,y2")
    args = parser.parse_args()

    predictor = WildlifePredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict_bytes(args.image.read_bytes(), crop_box=args.crop)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
