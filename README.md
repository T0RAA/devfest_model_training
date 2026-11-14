# UCSC Wildlife Vision — 20-Species Identification and Photo Rating

A reproducible computer-vision system for identifying 20 wild mammal and bird species documented on the UC Santa Cruz Campus Natural Reserve, with calibrated top-3 probabilities, out-of-distribution (OOD) / unknown species rejection, an independent 5-component photograph quality rubric (`Q`: 0–100), a composite identification-ready rating (`R`: 0–100), and **Gemini 4 Argon (`gemini-4-argon`)** multimodal explanation & audit integration across **Google Colab GPU** and **Google Cloud (GCS, Vertex AI, Cloud Run)**.

> **Data Scope & Honesty Note:** The 20 classes represent the **UCSC Wildlife 20 Starter Species** documented on the UCSC Campus Natural Reserve mammal and bird inventories. They are a practical, documented starter set—not an unverified claim of population abundance. See `data/manifests/campus_frequency_audit.md` and `scripts/audit_campus_frequency.py` for the observation-frequency audit workflow.

## Milestone Git Branches

Each milestone is committed and pushed to its dedicated Git branch:

- `milestone-0-skeleton`: Project skeleton, 20-species canonical taxonomy (`configs/species.yaml`), schemas, campus observation-frequency audit (`scripts/audit_campus_frequency.py`), and `tests/test_taxonomy.py`.
- `milestone-1-dataset`: License-aware iNaturalist & S3 Open Data ingestion, EXIF stripping, perceptual deduplication (`phash`), 22-column `manifest.parquet`, frozen observation-grouped `split_v1.csv`, `photo_attribution.csv`, `license_audit.json`, and `tests/test_manifest.py` + `tests/test_split_leakage.py`.
- `milestone-2-model`: Two-stage `efficientnet_b0` transfer learning (`src/wildlife/models/classifier.py`, `src/wildlife/train.py`, `src/wildlife/evaluate.py`), Colab GPU notebooks (`notebooks/00`–`02`), GCS checkpointing, and Vertex AI custom training scripts.
- `milestone-3-calibration`: Held-out temperature scaling (`T`), ECE / NLL / Brier evaluation, unknown/OOD rejection gate (`max(p) >= 0.80`, `margin >= 0.15`, energy score, coarse-group fallback), Gemini 4 Argon (`gemini-4-argon`) explanation module, and `tests/test_calibration.py` + `tests/test_unknown.py`.
- `milestone-4-rating`: Independent 5-component photograph quality rubric (`Q`: sharpness 30%, visibility 25%, exposure 15%, resolution 15%, composition 15%) and identification-ready composite rating (`R = round(0.65 * Q + 0.35 * (100 * C))` when accepted, `null` when uncertain), plus `tests/test_quality.py`.
- `milestone-5-deployment`: FastAPI backend (`api/main.py`, `api/Dockerfile`), interactive crop & rating web UI (`web/`), full end-to-end inference (`src/wildlife/infer.py`), evaluation reports (`reports/`), model card (`reports/model_card.md`), and `tests/test_api.py`.

## Quickstart Commands

```bash
# 1. Run full unit and failure-case test suite
pytest

# 2. Run campus observation-frequency audit (offline verifiable mode or --online with iNaturalist API)
python3 scripts/audit_campus_frequency.py --output-csv data/manifests/campus_frequency_audit.csv --output-md data/manifests/campus_frequency_audit.md

# 3. Build synthetic smoke-test manifest and observation-grouped splits
python3 scripts/build_manifest.py --generate-synthetic --output data/manifests/manifest.parquet --splits-out data/splits/split_v1.csv

# 4. Train 20-class classifier (Stage A frozen encoder + Stage B fine-tuning)
PYTHONPATH=src python3 -m wildlife.train \
  --config configs/train.yaml \
  --manifest data/manifests/manifest.parquet \
  --splits data/splits/split_v1.csv \
  --output runs/efficientnet_b0_v1

# 5. Evaluate checkpoint, fit/apply temperature calibration, and generate reports
PYTHONPATH=src python3 -m wildlife.evaluate \
  --checkpoint runs/efficientnet_b0_v1/best.ckpt \
  --split validation \
  --out reports/

# 6. Run end-to-end smoke test (training, calibration, OOD rejection, photo scoring, and FastAPI endpoints)
python3 scripts/smoke_test.py
```
