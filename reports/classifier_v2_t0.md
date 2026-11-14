# Classifier v2 — Tier 0 (`T0`) BioCLIP 2 Zero-Shot Report

- **Model:** `hf-hub:imageomics/bioclip-2` (ViT-L/14, MIT License)
- **Hardware Executed On:** Google Colab Pro G4 GPU (`open_clip_torch`, `fp16` autocast)
- **Dataset:** `manifest_v2.parquet` / `split_v2.csv` (`529` total records, `360` downloaded real CC0/CC-BY iNaturalist photographs, `val_samples = 59`)
- **Crop Version:** `crop-v1` · **Detector Version:** `md6-apa-rtdetr-e@apache2.0`
- **Prompt Templates (3 averaged per species):**
  1. `"a photo of {scientific_name}, {common_name}"`
  2. `"a wildlife photo of a {common_name} ({scientific_name})"`
  3. `"an image of the animal {common_name}, {scientific_name}"`

## Measured Results & Gate Status

| Gate / Metric | Target Condition | Measured Value (Colab G4 GPU) | Notes |
|---|---|---|---|
| `val_macro_f1` (`G-macro`) | `>= 0.90` | **`0.5885`** (`FAIL` -> escalates to `T1`) | Real BioCLIP 2 zero-shot on `59` validation crops |
| `val_top1_accuracy` | — | **`0.6102`** | `36 / 59` correct top-1 |
| `val_top3_accuracy` | — | **`0.6441`** | `38 / 59` correct in top-3 |
| `G-pairs` & `G-floor` | `>= 0.80` / `>= 0.70` | `FAIL` -> escalates to `T1` | Text embeddings saved as baseline before fitting `T1` linear head |
