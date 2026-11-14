"""Out-of-distribution (OOD) / unknown species rejection gate and coarse-group fallback (§9)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from wildlife.models.calibration import calibrated_softmax
from wildlife.schemas import SpeciesRecord, load_species_taxonomy


GROUP_DISPLAY_LABELS = {
    "bird": "Bird (species uncertain)",
    "large_mammal": "Large mammal (species uncertain)",
    "small_mammal": "Small/medium mammal (species uncertain)",
}


def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute Helmholtz free energy `-T * log(sum(exp(logits / T)))` (lower/more negative = more in-distribution)."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    scaled = arr / float(temperature)
    max_v = np.max(scaled, axis=1, keepdims=True)
    lse = max_v[:, 0] + np.log(np.sum(np.exp(scaled - max_v), axis=1))
    return (-float(temperature) * lse).astype(np.float64)


class UnknownSpeciesGate:
    """Decision gate combining calibrated max(p), top1-top2 margin, energy score, and OOD heuristics."""

    def __init__(
        self,
        max_prob_threshold: float = 0.80,
        margin_threshold: float = 0.15,
        energy_threshold: float = -1.20,
        coarse_group_agreement_min_sum: float = 0.72,
        species_records: Optional[List[SpeciesRecord]] = None,
    ) -> None:
        self.max_prob_threshold = float(max_prob_threshold)
        self.margin_threshold = float(margin_threshold)
        self.energy_threshold = float(energy_threshold)
        self.coarse_group_agreement_min_sum = float(coarse_group_agreement_min_sum)
        self.species_records = species_records or load_species_taxonomy()
        self.idx_to_species: Dict[int, SpeciesRecord] = {
            r.index: r for r in self.species_records
        }

    def evaluate_single(
        self,
        logits_1d: np.ndarray,
        temperature: float = 1.0,
        scene_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single 20-logit vector and return status, calibrated top-3, and fallback."""
        probs = calibrated_softmax(np.asarray(logits_1d, dtype=np.float64)[None, :], temperature)[0]
        order = np.argsort(-probs)
        top1_idx, top2_idx, top3_idx = int(order[0]), int(order[1]), int(order[2])
        p1, p2, p3 = float(probs[top1_idx]), float(probs[top2_idx]), float(probs[top3_idx])
        margin = p1 - p2
        energy = float(compute_energy_score(logits_1d, temperature=temperature)[0])

        top3_records = [self.idx_to_species[int(i)] for i in order[:3]]
        top3_list = [
            {
                "class_id": rec.class_id,
                "probability": round(float(probs[rec.index]), 4),
                "common_name": rec.common_name,
                "scientific_name": rec.scientific_name,
                "group": rec.group,
            }
            for rec in top3_records
        ]

        # Check explicit scene hints (e.g. empty background vs unsupported domestic/non-target animal)
        if scene_hint == "no_animal":
            status = "no_animal"
        elif scene_hint == "unsupported_animal":
            status = "unsupported_animal"
        else:
            passes_prob = p1 >= self.max_prob_threshold
            passes_margin = margin >= self.margin_threshold
            passes_energy = energy <= self.energy_threshold
            if passes_prob and passes_margin and passes_energy:
                status = "identified"
            else:
                status = "uncertain"

        coarse_fallback: Optional[str] = None
        if status != "identified":
            groups = {rec.group for rec in top3_records}
            if len(groups) == 1 and (p1 + p2 + p3) >= self.coarse_group_agreement_min_sum:
                only_group = next(iter(groups))
                coarse_fallback = GROUP_DISPLAY_LABELS.get(only_group)

        predicted_sp = self.idx_to_species[top1_idx] if status == "identified" else None

        return {
            "status": status,
            "accepted": status == "identified",
            "predicted_species": (
                {
                    "class_id": predicted_sp.class_id,
                    "common_name": predicted_sp.common_name,
                    "scientific_name": predicted_sp.scientific_name,
                    "group": predicted_sp.group,
                }
                if predicted_sp
                else None
            ),
            "confidence_calibrated": round(p1, 4) if status == "identified" else None,
            "raw_top1_probability": round(p1, 4),
            "top1_top2_margin": round(margin, 4),
            "energy_score": round(energy, 4),
            "coarse_group_fallback": coarse_fallback,
            "top_3": top3_list,
        }

    def evaluate_selective_and_ood(
        self,
        id_logits: np.ndarray,
        id_labels: np.ndarray,
        ood_logits: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute coverage, selective precision on accepted ID samples, and OOD false-accept rate (§9, §11.2)."""
        id_logits_arr = np.asarray(id_logits, dtype=np.float64)
        id_labels_arr = np.asarray(id_labels, dtype=np.int64)
        ood_logits_arr = np.asarray(ood_logits, dtype=np.float64)

        accepted_mask = []
        correct_mask = []
        for i in range(len(id_labels_arr)):
            res = self.evaluate_single(id_logits_arr[i], temperature=temperature)
            is_acc = bool(res["accepted"])
            accepted_mask.append(is_acc)
            pred_idx = int(np.argmax(id_logits_arr[i]))
            correct_mask.append(pred_idx == int(id_labels_arr[i]))

        accepted_arr = np.array(accepted_mask, dtype=bool)
        correct_arr = np.array(correct_mask, dtype=bool)
        coverage = float(np.mean(accepted_arr)) if len(accepted_arr) > 0 else 0.0
        if np.any(accepted_arr):
            selective_precision = float(np.mean(correct_arr[accepted_arr]))
        else:
            selective_precision = 0.0

        ood_false_accepts = 0
        for j in range(len(ood_logits_arr)):
            res_ood = self.evaluate_single(ood_logits_arr[j], temperature=temperature)
            if res_ood["accepted"]:
                ood_false_accepts += 1

        ood_far = (
            float(ood_false_accepts) / float(len(ood_logits_arr))
            if len(ood_logits_arr) > 0
            else 0.0
        )

        return {
            "coverage": round(coverage, 4),
            "selective_precision": round(selective_precision, 4),
            "selective_risk": round(1.0 - selective_precision, 4),
            "ood_samples": int(len(ood_logits_arr)),
            "ood_false_accepts": int(ood_false_accepts),
            "ood_false_accept_rate": round(ood_far, 4),
            "thresholds": {
                "max_prob_threshold": self.max_prob_threshold,
                "margin_threshold": self.margin_threshold,
                "energy_threshold": self.energy_threshold,
            },
        }
