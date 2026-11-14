"""Unit tests for Milestone 10 config-driven unknown/OOD gate, full group rollup, OWLv2 cap, and threshold tuner."""

import numpy as np
import pytest

from wildlife.models.unknown import (
    UnknownSpeciesGate,
    compute_energy_score,
    tune_gate_thresholds_grid,
)


def test_high_confidence_in_distribution_is_accepted():
    gate = UnknownSpeciesGate.from_config()
    logits = np.full((20,), -2.0, dtype=np.float64)
    logits[12] = 5.0  # B03 Steller's jay
    res = gate.evaluate_single(logits, temperature=1.0)
    assert res["status"] == "identified"
    assert res["accepted"] is True
    assert res["predicted_species"]["class_id"] == "B03"
    assert res["confidence_calibrated"] >= gate.max_prob_threshold
    assert len(res["top_3"]) == 3


def test_ood_and_ambiguous_inputs_are_rejected():
    gate = UnknownSpeciesGate.from_config()
    flat_logits = np.zeros((20,), dtype=np.float64)
    flat_logits[0] = 0.2
    res_flat = gate.evaluate_single(flat_logits, temperature=1.0)
    assert res_flat["status"] == "uncertain"
    assert res_flat["accepted"] is False
    assert res_flat["predicted_species"] is None
    assert res_flat["confidence_calibrated"] is None

    res_unsup = gate.evaluate_single(
        flat_logits, temperature=1.0, scene_hint="unsupported_animal"
    )
    assert res_unsup["status"] == "unsupported_animal"
    assert res_unsup["predicted_species"] is None


def test_full_distribution_group_rollup_and_owlv2_confidence_cap():
    gate = UnknownSpeciesGate.from_config()
    # Distribute probability across 5 bird classes so top-1 < 0.78 but total bird sum >= 0.75 (§8.3)
    logits = np.full((20,), -5.0, dtype=np.float64)
    for bird_idx in [10, 11, 12, 13, 14]:
        logits[bird_idx] = 1.5
    res_rollup = gate.evaluate_single(logits, temperature=1.0)
    assert res_rollup["status"] == "uncertain"
    assert res_rollup["group_probabilities"]["bird"] >= 0.75
    assert res_rollup["coarse_group_fallback"] == "Bird (species uncertain)"

    # §8.4: High classifier confidence but OWLv2 box confidence < tau_det + 0.10 (0.25 + 0.10 = 0.35) -> capped at uncertain
    strong_logits = np.full((20,), -2.5, dtype=np.float64)
    strong_logits[12] = 5.5
    res_owl = gate.evaluate_single(
        strong_logits,
        temperature=1.0,
        box_source="owlv2",
        detector_confidence=0.29,
    )
    assert res_owl["status"] == "uncertain"
    assert res_owl["owlv2_capped"] is True
    assert res_owl["predicted_species"] is None


def test_tune_gate_thresholds_grid_respects_far_target():
    rng = np.random.RandomState(42)
    id_labels = rng.randint(0, 20, size=40)
    id_logits = rng.randn(40, 20) * 0.4
    for i, y in enumerate(id_labels):
        id_logits[i, y] += 4.2
    ood_logits = rng.uniform(-0.3, 0.3, size=(30, 20))

    best = tune_gate_thresholds_grid(
        id_logits=id_logits,
        id_labels=id_labels,
        ood_tune_logits=ood_logits,
        temperature=1.0,
        far_target=0.05,
    )
    assert best["ood_false_accept_rate"] <= 0.05
    assert best["coverage"] >= 0.80
