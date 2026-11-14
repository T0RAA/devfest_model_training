# End-to-End Evaluation Report v2 (`reports/eval_v2.md`)

- **Pipeline Version:** Two-Stage Detect-then-Classify (`WildlifePipeline` v2.1)
- **Detector Version:** `md6-apa-rtdetr-e@apache2.0` (`MegaDetectorV6Apache` RT-DETR + `OWLv2` zero-shot fallback)
- **Crop Version:** `crop-v1` (15% padding, square context expansion, 512 px storage)
- **Classifier Tier:** `T1` (`hf-hub:imageomics/bioclip-2` frozen ViT-L/14 embeddings + `GroupKFold` linear head, `C = 10.0`)
- **Calibration Version:** `temp-v2` (`configs/rating.yaml`) · **Photo Quality Rubric:** `quality-v2`

> **Ground Rule #1 & #2 Disclosure:** Per `MILESTONES_v2_detector_classifier.md` §0, no numbers are fabricated. The Colab G4 GPU numbers below come directly from running `wildlife.train --tier t1 --require-torch` on `373` train / `59` validation real CC0/CC-BY iNaturalist crops (`529` total rows, `360` downloaded images). Metrics on `test`, `campus_test`, `field_holdout`, and `ood_test` remain `NOT_RUN` until the full S3 dataset pull (`150–800` obs/species) and campus `field_holdout` collection are evaluated.

## Colab G4 GPU Classifier Measurements (Pilot v2 Split)

| Tier | Split | Macro-F1 | Top-1 | Top-3 | Hard-Pair Recalls (`M08, M09, B03, B04, B05`) |
|---|---|---:|---:|---:|---|
| `T0` (BioCLIP 2 Zero-Shot) | `validation` (`n=59`) | `0.5885` | `0.6102` | `0.6441` | Escalated to `T1` |
| `T1` (BioCLIP 2 Linear Probe, `C=10.0`) | `validation` (`n=59`) | **`0.9304`** | **`0.9492`** | **`0.9492`** | `M08=1.0, M09=0.0, B03=1.0, B04=1.0, B05=1.0` |

## Milestone 12 Evaluation Table (§10.2)

| Metric | Split | Target | Measured Result / Status | Provenance & Reason |
|---|---|---|---|---|
| Detector image-level recall | `field_holdout` | `>= 0.95` | `NOT_RUN` | Requires `data/field_holdout/` (600 campus photos) and Colab/Vertex GPU with `PytorchWildlife` (`notebooks/05_detector_eval_labeling.ipynb`) |
| Species macro-F1 (all images, gate off) | `validation` / `test` / `campus_test` / `field_holdout` | Report all | `val = 0.9304` (Colab G4 GPU); `test / campus_test / field_holdout = NOT_RUN` | `test` & `campus_test` run via `scripts/colab_auto_train.py` after scaling to 150–800 obs/species; `field_holdout` pending collection |
| Selective precision at chosen operating point | `field_holdout` | `>= 0.95` | `NOT_RUN` | Requires `data/field_holdout/` and Colab GPU BioCLIP 2 logits |
| Coverage at that operating point | `field_holdout` | Report | `NOT_RUN` | Requires `data/field_holdout/` and Colab GPU BioCLIP 2 logits |
| OOD false-accept rate | `ood_test` | `<= 0.05` | `NOT_RUN` (Real BioCLIP 2) / `0.0000` (Local pipeline smoke test) | `ood_test` manifest (`18` real iNaturalist records across 9 disjoint hard-negative taxa) is built in `manifest_v2.parquet` |
| ECE after calibration | `test` | `<= 0.05` | `NOT_RUN` (Real BioCLIP 2) | Temperature scaling (`TemperatureScaler`, 15 bins) verified in `tests/test_calibration.py`; full `test` crop evaluation executes in Colab via `scripts/colab_auto_train.py` |
| Hard-pair confusions (`M08<->M09`, `B03<->B04`, `B05<->crow`) | `validation` / `field_holdout` | Report counts | `val`: `M08=1.0, M09=0.0, B03=1.0, B04=1.0, B05=1.0`; `field_holdout = NOT_RUN` | `M09` Western gray squirrel missed in 59-image pilot validation; increase per-species observations (`>= 150`) before deciding on `T2` escalation |
| End-to-end CPU latency `p50 / p95` | Local CPU (`WILDLIFE_TEST_MODE=1`) / Cloud Run | `<= 400 ms` p95 per crop | `p50 = 6.8 ms / p95 = 11.4 ms` (local CPU surrogate); `NOT_RUN` on Cloud Run 2 vCPU BioCLIP 2 | If BioCLIP 2 ViT-L/14 CPU p95 exceeds `400 ms` on Cloud Run (2 vCPU, 4 GiB), run `distill` tier into `EfficientNetV2-S` (`configs/train_v2.yaml`, Decision D2) |
| Gemini zero-shot macro-F1 (benchmark only) | `field_holdout` | Report | `NOT_RUN` | Requires `data/field_holdout/` and Vertex AI billing credentials (`reports/gemini_budget.md`) |
