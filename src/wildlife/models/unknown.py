"""Out-of-distribution (OOD) / unknown species rejection gate, threshold grid search, and full-distribution group rollup (§8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

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


def default_rating_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "rating.yaml"


class UnknownSpeciesGate:
    """Decision gate combining calibrated max(p), top1-top2 margin, energy score, full group rollup, and OWLv2 cap (§8)."""

    def __init__(
        self,
        max_prob_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        energy_threshold: Optional[float] = None,
        coarse_group_agreement_min_sum: Optional[float] = None,
        tau_det: float = 0.25,
        owlv2_uncertain_margin: float = 0.10,
        config_path: Optional[Path] = None,
        species_records: Optional[List[SpeciesRecord]] = None,
    ) -> None:
        cfg_file = Path(config_path) if config_path else default_rating_config_path()
        gate_cfg: Dict[str, Any] = {}
        if cfg_file.exists():
            raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            gate_cfg = raw.get("acceptance_gate", {})

        self.max_prob_threshold = float(
            max_prob_threshold
            if max_prob_threshold is not None
            else gate_cfg.get("max_prob_threshold", 0.78)
        )
        self.margin_threshold = float(
            margin_threshold
            if margin_threshold is not None
            else gate_cfg.get("margin_threshold", 0.15)
        )
        self.energy_threshold = float(
            energy_threshold
            if energy_threshold is not None
            else gate_cfg.get("energy_threshold", -1.20)
        )
        self.coarse_group_agreement_min_sum = float(
            coarse_group_agreement_min_sum
            if coarse_group_agreement_min_sum is not None
            else gate_cfg.get("coarse_group_agreement_min_sum", 0.75)
        )
        self.tau_det = float(tau_det)
        self.owlv2_uncertain_margin = float(owlv2_uncertain_margin)
        self.species_records = species_records or load_species_taxonomy()
        self.idx_to_species: Dict[int, SpeciesRecord] = {
            r.index: r for r in self.species_records
        }

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "UnknownSpeciesGate":
        return cls(config_path=config_path)

    def evaluate_single(
        self,
        logits_1d: np.ndarray,
        temperature: float = 1.0,
        scene_hint: Optional[str] = None,
        box_source: str = "megadetector_v6",
        detector_confidence: float = 0.90,
    ) -> Dict[str, Any]:
        """Evaluate a single 20-logit vector with temperature calibration, group rollup, and OWLv2 confidence cap (§8.3-8.4)."""
        probs = calibrated_softmax(
            np.asarray(logits_1d, dtype=np.float64)[None, :], temperature
        )[0]
        order = np.argsort(-probs)
        top1_idx, top2_idx = int(order[0]), int(order[1])
        p1, p2 = float(probs[top1_idx]), float(probs[top2_idx])
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

        # Compute full 20-class probability sums within each coarse group (§8.3)
        group_sums: Dict[str, float] = {"large_mammal": 0.0, "small_mammal": 0.0, "bird": 0.0}
        for idx, rec in self.idx_to_species.items():
            group_sums[rec.group] = group_sums.get(rec.group, 0.0) + float(probs[idx])

        # §8.4: If selected box came from OWLv2 fallback and confidence < tau_det + 0.10, cap at uncertain
        owlv2_capped = bool(
            box_source == "owlv2"
            and float(detector_confidence) < (self.tau_det + self.owlv2_uncertain_margin)
        )

        if scene_hint == "no_animal":
            status = "no_animal"
        elif scene_hint == "unsupported_animal":
            status = "unsupported_animal"
        elif owlv2_capped:
            status = "uncertain"
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
            best_group, best_group_sum = max(group_sums.items(), key=lambda kv: kv[1])
            if best_group_sum >= self.coarse_group_agreement_min_sum:
                coarse_fallback = GROUP_DISPLAY_LABELS.get(best_group)

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
            "group_probabilities": {k: round(v, 4) for k, v in group_sums.items()},
            "coarse_group_fallback": coarse_fallback,
            "owlv2_capped": owlv2_capped,
            "top_3": top3_list,
        }

    def evaluate_selective_and_ood(
        self,
        id_logits: np.ndarray,
        id_labels: np.ndarray,
        ood_logits: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute coverage, selective precision on accepted ID samples, and OOD false-accept rate (§8 & §10)."""
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


def tune_gate_thresholds_grid(
    id_logits: np.ndarray,
    id_labels: np.ndarray,
    ood_tune_logits: np.ndarray,
    temperature: float = 1.0,
    far_target: float = 0.05,
) -> Dict[str, Any]:
    """Grid-search `max_prob_threshold`, `margin_threshold`, and `energy_threshold` (§8.2).

    Maximizes in-distribution coverage subject to `ood_false_accept_rate <= far_target`.
    """
    best_cfg = {
        "max_prob_threshold": 0.78,
        "margin_threshold": 0.15,
        "energy_threshold": -1.20,
        "coverage": -1.0,
        "selective_precision": 0.0,
        "ood_false_accept_rate": 1.0,
    }
    for p_thresh in [0.65, 0.72, 0.78, 0.80, 0.85]:
        for m_thresh in [0.10, 0.15, 0.20]:
            for e_thresh in [-0.80, -1.20, -1.50]:
                gate = UnknownSpeciesGate(
                    max_prob_threshold=p_thresh,
                    margin_threshold=m_thresh,
                    energy_threshold=e_thresh,
                )
                rep = gate.evaluate_selective_and_ood(
                    id_logits=id_logits,
                    id_labels=id_labels,
                    ood_logits=ood_tune_logits,
                    temperature=temperature,
                )
                if rep["ood_false_accept_rate"] <= far_target:
                    if (
                        rep["coverage"] > best_cfg["coverage"]
                        or (
                            rep["coverage"] == best_cfg["coverage"]
                            and rep["selective_precision"] > best_cfg["selective_precision"]
                        )
                    ):
                        best_cfg = {
                            "max_prob_threshold": p_thresh,
                            "margin_threshold": m_thresh,
                            "energy_threshold": e_thresh,
                            "coverage": rep["coverage"],
                            "selective_precision": rep["selective_precision"],
                            "ood_false_accept_rate": rep["ood_false_accept_rate"],
                        }
    return best_cfg
