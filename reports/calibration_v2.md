# Calibration & OOD Gate Report v2 (`temp-v2`)

- **Calibration Version:** `temp-v2`
- **Method:** Single-parameter post-hoc temperature scaling (`p = softmax(logits / T)`) minimizing negative log-likelihood (NLL), evaluated with 15-bin Expected Calibration Error (ECE) and Brier score.
- **Config-Driven Gate (`configs/rating.yaml`):**
  - `max_prob_threshold: 0.78`
  - `margin_threshold: 0.15`
  - `energy_threshold: -1.20`
  - `coarse_group_agreement_min_sum: 0.75` (sums full 20-class probabilities within `large_mammal`, `small_mammal`, `bird`)
  - `owlv2_uncertain_margin: 0.10` (caps OWLv2 fallback detections with confidence `< tau_det + 0.10` at `uncertain`)

## Gate & Execution Status (Ground Rule #1 Compliance)

| Step / Check | Target | Measured Result / Status | Notes |
|---|---|---|---|
| Config-driven threshold loading (`UnknownSpeciesGate.from_config`) | No hard-coded thresholds in `WildlifePredictor` | `PASS` | Verified in `tests/test_unknown.py` |
| Full 20-class coarse-group rollup (`>= 0.75` sum) | Roll up to `"Bird (species uncertain)"` etc. | `PASS` | Verified in `tests/test_unknown.py` |
| OWLv2 low-confidence cap (`conf < tau_det + 0.10`) | Force `status = uncertain` | `PASS` | Verified in `tests/test_unknown.py` |
| Grid-search threshold tuner (`tune_gate_thresholds_grid`) on `ood_tune` | `OOD FAR <= 0.05` | `PASS (OOD FAR = 0.0000 on local calibration fixture)` | Full BioCLIP 2 crop-logit calibration on GCS `validation` + `ood_tune` embeddings is `NOT_RUN` locally (executes in Colab GPU after `06_bioclip2_embed_and_probe.ipynb`) |
