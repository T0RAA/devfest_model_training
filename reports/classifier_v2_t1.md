# Classifier v2 — Tier 1 (`T1`) Frozen BioCLIP 2 Embeddings + Linear Head Report

- **Backbone:** `hf-hub:imageomics/bioclip-2` (frozen ViT-L/14, 768-D L2-normalized float16 embeddings)
- **Hardware Executed On:** Google Colab Pro G4 GPU (`backend = bioclip2_open_clip_gpu`)
- **Head:** 768 -> 20 multinomial logistic regression (`GroupKFold` on `observation_id`, selected **`C = 10.0`**, `class_weight="balanced"`)
- **Dataset:** `manifest_v2.parquet` / `split_v2.csv` (`train_samples = 373` crops × 4 augmented views = 1,492 embeddings; `val_samples = 59` crops × 1 center view)
- **Checkpoint SHA-256:** `57e85caf5f3e2bda63b1ce706ceacf0486c939195a26def957dadd312e1ffcef`
- **Detector & Crop Version Parity:** `detector_version = md6-apa-rtdetr-e@apache2.0`, `crop_version = crop-v1`

## Measured Results & Gate Status

| Gate / Metric | Target Condition | Measured Value (Colab G4 GPU) | Status |
|---|---|---|---|
| `G-macro` (Validation Macro-F1) | `>= 0.90` | **`0.9304`** | **`PASS`** |
| Validation Top-1 Accuracy | — | **`0.9492`** | Measured (`56 / 59`) |
| Validation Top-3 Accuracy | — | **`0.9492`** | Measured (`56 / 59`) |
| `G-pairs` (`M08, M09, B03, B04, B05` recall) | `>= 0.80` each | `M08=1.0`, `M09=0.0`, `B03=1.0`, `B04=1.0`, `B05=1.0` | **`FAIL`** (on `M09` Western gray squirrel in 59-crop pilot val split) |
| `G-floor` (Minimum per-class recall) | `>= 0.70` | **`0.0000`** (`M09`) | **`FAIL`** (pilot val split has sparse per-class support; scale up via `scripts/colab_auto_train.py` or S3 DuckDB pull to 150–800 obs/species and re-check before T2 escalation) |
