# Classifier v2 — Tier 0 (`T0`) BioCLIP 2 Zero-Shot Report

- **Model:** `hf-hub:imageomics/bioclip-2` (ViT-L/14, MIT License)
- **Crop Version:** `crop-v1` · **Detector Version:** `md6-apa-rtdetr-e@apache2.0`
- **Prompt Templates (3 averaged per species):**
  1. `"a photo of {scientific_name}, {common_name}"`
  2. `"a wildlife photo of a {common_name} ({scientific_name})"`
  3. `"an image of the animal {common_name}, {scientific_name}"`

## Gate & Execution Status (Ground Rule #1 Compliance)

| Gate / Metric | Target Condition | Status / Value | Notes |
|---|---|---|---|
| Prompt & zero-shot pipeline verification (`build_bioclip2_zero_shot_prompts`) | 20 species × 3 templates | `PASS (60 prompts)` | Verified in `tests/test_train_backend.py` |
| `G-macro` (Validation Macro-F1) | `>= 0.90` | `NOT_RUN` | Requires Colab GPU runtime (`torch` + `open_clip_torch` not installed on local Mac host; run `notebooks/06_bioclip2_embed_and_probe.ipynb`) |
| `G-pairs` (`M08, M09, B03, B04, B05` recall) | `>= 0.80` each | `NOT_RUN` | Requires Colab GPU runtime |
| `G-floor` (Minimum per-class recall) | `>= 0.70` | `NOT_RUN` | Requires Colab GPU runtime |
