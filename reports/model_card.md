# Model Card — UCSC Wildlife Vision v2 (`species-v2.0`)

## 1. Architecture & Version Summary

- **Architecture:** Two-Stage Detect-then-Classify Pipeline (`src/wildlife/pipeline.py`, mirroring Google SpeciesNet's crop-then-classify structure).
  - **Stage 1 — Detector (`src/wildlife/detect/`):** PyTorch-Wildlife **MegaDetector V6** (`MDV6-apa-rtdetr-e`, Apache-2.0 RT-DETR) with zero-shot **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache-2.0) fallback for small perched birds (`detector_version = md6-apa-rtdetr-e@apache2.0`, `tau_det = 0.25`, `max_boxes = 5`). AGPL YOLO variants are blocked by `validate_megadetector_variant_license()`.
  - **Stage 2 — Crop (`src/wildlife/detect/crops.py`):** `crop-v1` pads each animal bounding box by 15% on all sides, expands the shorter dimension to a square using real image context (clamping and shifting inward at frame edges), and resizes to 512 × 512 px.
  - **Stage 3 — Classifier (`src/wildlife/models/backbones.py`):** Follows the §0.1 efficiency ladder:
    - `T0`: **BioCLIP 2** (`hf-hub:imageomics/bioclip-2`, ViT-L/14, MIT) zero-shot across 3 prompt templates per species.
    - `T1` (Default production path): Frozen **BioCLIP 2** image embeddings (768-D float16 cached in GCS, 4 augmented views per train crop) with a group-regularized multinomial logistic regression head.
    - `T2 / T3 / Distill`: Escalation to last-4-block unfreezing or distillation into **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k_ft_in1k`, Apache-2.0) if `T1` misses a gate or Cloud Run CPU p95 exceeds `400 ms`.
  - **Stage 4 — Calibration & Unknown/OOD Gate (`src/wildlife/models/calibration.py`, `unknown.py`):** Temperature scaling (`temp-v2`) + config-driven gate (`max_prob_threshold = 0.78`, `margin_threshold = 0.15`, `energy_threshold = -1.20`), full 20-class coarse-group probability rollup (`>= 0.75` -> `"Bird (species uncertain)"`), and OWLv2 low-confidence cap (`< tau_det + 0.10` -> `uncertain`).
  - **Stage 5 — Photo Quality (`Q`) & Rating (`R`):** Computed on the detector crop using box geometry (`inside_frame_ratio`, `touches_edge`) for subject visibility.
  - **Stage 6 — Optional Multimodal Explanation (`gemini-4-argon`):** Invoked only on user request (`explain=True`) or when `status == "uncertain"`. Never overrides classifier species predictions or ground-truth labels.

> **Explicit v1 Smoke-Test Disclosure (Ground Rule #2):** The v1 checkpoint `runs/efficientnet_b0_v1/best.ckpt` (62 KB) and numbers in `reports/baseline_metrics.md` were generated on 100 synthetic images using the 132-D NumPy surrogate for pipeline smoke testing. They do **not** represent real model accuracy and must never be cited as wildlife classification performance. Real training and evaluation require `--require-torch` (default `True` in `src/wildlife/train.py`).

## 2. Data Sources, Licenses & Real Campus Counts

| Source | License | Role | Allowed in Classifier? | Allowed in Detector? |
|---|---|---|---|---|
| iNaturalist Open Data (`s3://inaturalist-open-data`) | `cc0`, `cc-by` only | Primary 20-species & hard-negative OOD images (`large` 1024 px tier) | **Yes** | **Yes** |
| LILA BC Camera Traps (`Caltech`, `ENA24`, `Channel Islands`) | `cdla-permissive-1.0` | Bounding-box evaluation, empty frames, optional detector fine-tuning (Decision D1) | **No** (Blocked by `validate_photo_license`) | **Yes** |
| UCSC Campus Natural Reserve (`place_id=94596` & `3519`) | `cc0`, `cc-by` | Held-out `campus_test` split only (Decision D5) | **Test only** | **Test only** |
| UCSC Field Holdout (`data/field_holdout/`) | Written contributor permission | Final blind evaluation in Milestone 12 (`eval_v2.md`) | **Test only** | **Test only** |

- **California Pool Verification (`data/manifests/source_snapshot.json`):** Live queries against `api.inaturalist.org` on `2026-11-14` verified that all 20 target species have `>= 572` research-grade CC0/CC-BY photographed observations in California (`M04` Gray fox has `572`; `B07` Anna's hummingbird has `9,035`). In the local 380-row API pilot pull (`reports/data_card_v2.md`), all classes have `< 150` local rows and are explicitly marked `low_data (local pilot sample; CA S3 pool >= 150)` per Decision D4 until the full S3 tarball scan is run in Colab.

## 3. Train/Serve Parity & Safety Rules

- **Version Parity:** Every v2 classifier checkpoint records `detector_version` and `crop_version`. `load_v2_checkpoint()` raises `VersionParityError` if a checkpoint's detector or crop version does not match the serving pipeline.
- **Human-Only Labels:** Ground-truth labels originate strictly from `inat_research_grade`, `lila_expert`, or `field_owner`. `validate_manifest_schema()` raises `ManifestSchemaError` if Gemini is ever listed as a `label_source`.
- **Person & Privacy Rejection:** Any frame containing only `person` or `vehicle` detections returns `status = "no_animal"` with zero species output, and EXIF/GPS metadata is stripped on ingest.
