"""Unit tests for unknown/OOD rejection gate and coarse-group fallback (Milestone 3)."""

import numpy as np
import pytest

from wildlife.models.unknown import UnknownSpeciesGate, compute_energy_score


def test_high_confidence_in_distribution_is_accepted():
    gate = UnknownSpeciesGate(
        max_prob_threshold=0.80,
        margin_threshold=0.15,
        energy_threshold=-1.20,
    )
    logits = np.full((20,), -2.0, dtype=np.float64)
    logits[12] = 5.0  # B03 Steller's jay
    res = gate.evaluate_single(logits, temperature=1.0)
    assert res["status"] == "identified"
    assert res["accepted"] is True
    assert res["predicted_species"]["class_id"] == "B03"
    assert res["confidence_calibrated"] >= 0.80
    assert len(res["top_3"]) == 3


def test_ood_and_ambiguous_inputs_are_rejected():
    # Failure case #4: A cat/dog/banana slug/empty image cannot be force-labeled as a supported species
    gate = UnknownSpeciesGate(
        max_prob_threshold=0.80,
        margin_threshold=0.15,
        energy_threshold=-1.20,
    )
    # Flat / low-confidence logits typical of OOD or empty background
    flat_logits = np.zeros((20,), dtype=np.float64)
    flat_logits[0] = 0.2
    res_flat = gate.evaluate_single(flat_logits, temperature=1.0)
    assert res_flat["status"] == "uncertain"
    assert res_flat["accepted"] is False
    assert res_flat["predicted_species"] is None
    assert res_flat["confidence_calibrated"] is None

    # Explicit unsupported animal hint
    res_unsup = gate.evaluate_single(flat_logits, temperature=1.0, scene_hint="unsupported_animal")
    assert res_unsup["status"] == "unsupported_animal"
    assert res_unsup["predicted_species"] is None


def test_coarse_group_fallback_when_all_top3_are_birds():
    # SPEC.md §9: if species confidence is low but all leading candidates share a coarse group,
    # display "Bird (species uncertain)" instead of a false species ID.
    gate = UnknownSpeciesGate(
        max_prob_threshold=0.80,
        margin_threshold=0.15,
        energy_threshold=-1.20,
        coarse_group_agreement_min_sum=0.70,
    )
    logits = np.full((20,), -4.0, dtype=np.float64)
    # Close tie between B03 (Steller's jay, idx 12), B04 (Scrub-jay, idx 13), and B05 (Raven, idx 14)
    logits[12] = 2.1
    logits[13] = 2.0
    logits[14] = 1.6
    res = gate.evaluate_single(logits, temperature=1.0)
    assert res["status"] == "uncertain"
    assert res["predicted_species"] is None
    assert res["coarse_group_fallback"] == "Bird (species uncertain)"
