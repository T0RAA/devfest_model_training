# UCSC Wildlife Vision v2 — Two-Stage Detect-then-Classify Pipeline

A reproducible two-stage computer-vision system for identifying 20 wild mammal and bird species documented on the UC Santa Cruz Campus Natural Reserve:
1. **Stage 1 — Class-Agnostic Animal Detector (`src/wildlife/detect/`):** PyTorch-Wildlife **MegaDetector V6** (`MDV6-apa-rtdetr-e`, Apache-2.0) with zero-shot **OWLv2** (`google/owlv2-base-patch16-ensemble`) fallback and `crop-v1` (15%-padded square context crops).
2. **Stage 2 — 20-Species Crop Classifier (`src/wildlife/models/backbones.py`):** **BioCLIP 2** (`hf-hub:imageomics/bioclip-2`, ViT-L/14, MIT) frozen embeddings + group-regularized linear head (`T1`, with `T0` zero-shot baseline and `T2`/`T3`/`distill` escalation to `EfficientNetV2-S`).
3. **Stage 3 — Calibrated Unknown/OOD Gate & Group Rollup (`src/wildlife/models/unknown.py`):** Post-hoc temperature scaling (`temp-v2`), energy + margin + max-probability gating, full 20-class coarse-group rollup (`"Bird (species uncertain)"`), and OWLv2 confidence cap.
4. **Stage 4 — Crop-Geometry Photo Quality (`Q`: 0–100) & Composite Rating (`R`: 0–100)** + optional **Gemini 4 Argon (`gemini-4-argon`)** explanation on request or uncertain predictions.

> **Honesty & Provenance Note:** Per `MILESTONES_v2_detector_classifier.md` §0, `runs/efficientnet_b0_v1/best.ckpt` (62 KB) and `reports/baseline_metrics.md` from v1 are synthetic smoke-test outputs on 100 synthetic images, not real model accuracy. See `reports/data_card_v2.md`, `reports/detector_v2.md`, `reports/classifier_v2_t0.md`, `reports/classifier_v2_t1.md`, `reports/calibration_v2.md`, `reports/eval_v2.md`, and `reports/gemini_budget.md` for verified v2 data counts and execution gates.

## Milestone Git Branches

- `milestone-0-skeleton` through `milestone-5-deployment`: v1 scaffolding, taxonomy, manifest/split schemas, quality rubric, and FastAPI foundation.
- `milestone-6-data-pull`: Live UCSC campus frequency audit (`place_id=94596`), resolved OOD taxon IDs (`configs/ood_taxa.yaml`), Decision D1 `cdla-permissive-1.0` detector-only license gate, fixed TSV DuckDB query (`s3_open_data_query.sql`), real iNaturalist CC0/CC-BY pull (`manifest_v2.parquet`, `split_v2.csv`, `reports/data_card_v2.md`), and `tests/test_license_v2.py`.
- `milestone-7-detector`: `Detection` & `AnimalDetector` contract, `MDV6-apa-rtdetr-e` (Apache-2.0) + `OWLv2` fallback, AGPL variant blocker, `gemini_box_to_pixels()` conversion, `notebooks/05_detector_eval_labeling.ipynb`, `reports/detector_v2.md`, and detector unit tests.
- `milestone-8-crops`: `crop-v1` (15% padding, square context expansion, 512 px storage, train jitter), WebDataset `.tar` shard writer, `crops_index.parquet`, and `reports/crop_contact_sheet_v1.png`.
- `milestone-9-classifier-v2`: `BioCLIP 2` (`T0`/`T1`/`T2`) and `EfficientNetV2-S` (`T3`/`distill`) backbones, float16 `.npy` embedding cache, WebDataset loader, `--require-torch` fail-loudly enforcement, `notebooks/06` & `07`, and `tests/test_train_backend.py` + `tests/test_embedding_cache.py`.
- `milestone-10-calibration-v2`: Config-driven gate thresholds (`configs/rating.yaml`), full 20-class coarse-group rollup, OWLv2 low-confidence cap, threshold grid search (`tune_gate_thresholds_grid`), and `reports/calibration_v2.md`.
- `milestone-11-pipeline`: `WildlifePipeline` (`src/wildlife/pipeline.py`), multi-animal `/v2/predict` endpoint + `/v1/predict` wrapper, detector/crop version parity enforcement (`VersionParityError`), and 5 pipeline test suites (`tests/test_pipeline_*.py`, `tests/test_version_parity.py`, `tests/test_api_v2.py`).
- `milestone-12-eval-release`: `reports/eval_v2.md`, `reports/gemini_budget.md`, updated `reports/model_card.md`, `README.md`, and end-to-end v2 smoke test.

## Quickstart Commands

```bash
# 1. Run full v1 + v2 unit and integration test suite (44 tests)
pytest

# 2. Refresh live UCSC campus observation-frequency audit (place_id=94596)
python3 scripts/audit_campus_frequency.py --online --place-id 94596

# 3. Pull real CC0/CC-BY iNaturalist candidates and build Manifest v2 + Splits v2
python3 scripts/pull_v2_data.py --per-species-sample 12 --download-sample-images 1

# 4. Build crop-v1 dataset, WebDataset shards, and 10x20 contact sheet
python3 scripts/build_crops_v1.py

# 5. Train classifier on Colab/Vertex GPU (--require-torch enforced by default)
PYTHONPATH=src python3 -m wildlife.train \
  --config configs/train_v2.yaml \
  --manifest data/manifests/manifest_v2.parquet \
  --splits data/splits/split_v2.csv \
  --tier t1 \
  --require-torch \
  --output runs/classifier_v2_t1

# 6. Run local end-to-end smoke test (explicitly sets WILDLIFE_TEST_MODE=1 for CPU surrogate)
WILDLIFE_TEST_MODE=1 python3 scripts/smoke_test.py
```
