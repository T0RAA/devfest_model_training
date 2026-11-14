# End-to-End Evaluation Report v2 (`reports/eval_v2.md`)

- **Pipeline Version:** Two-Stage Detect-then-Classify (`WildlifePipeline` v2.1)
- **Detector Version:** `md6-apa-rtdetr-e@apache2.0` (`MegaDetectorV6Apache` RT-DETR + `OWLv2` zero-shot fallback)
- **Crop Version:** `crop-v1` (15% padding, square context expansion, 512 px storage)
- **Classifier Tier:** `T1` (`hf-hub:imageomics/bioclip-2` frozen ViT-L/14 embeddings + group-regularized linear head)
- **Calibration Version:** `temp-v2` (`configs/rating.yaml`) · **Photo Quality Rubric:** `quality-v2`

> **Ground Rule #1 & #2 Disclosure:** Per `MILESTONES_v2_detector_classifier.md` §0, no numbers are fabricated. The v1 checkpoint (`runs/efficientnet_b0_v1/best.ckpt`, 62 KB) was trained on 100 synthetic images using the 132-D NumPy surrogate and is **smoke-test output only**, never cited as real model performance. Metrics requiring GPU execution of `MegaDetectorV6Apache` + `BioCLIP 2` or the uncollected campus `field_holdout` photographs are recorded as `NOT_RUN` with their exact prerequisite.

## Milestone 12 Evaluation Table (§10.2)

| Metric | Split | Target | Measured Result / Status | Provenance & Reason |
|---|---|---|---|---|
| Detector image-level recall | `field_holdout` | `>= 0.95` | `NOT_RUN` | Requires `data/field_holdout/` (600 campus photos) and Colab/Vertex GPU with `PytorchWildlife` (`notebooks/05_detector_eval_labeling.ipynb`) |
| Species macro-F1 (all images, gate off) | `test` / `campus_test` / `field_holdout` | Report all three | `NOT_RUN` | Requires running `notebooks/06_bioclip2_embed_and_probe.ipynb` on Colab GPU over full `crop-v1` shards |
| Selective precision at chosen operating point | `field_holdout` | `>= 0.95` | `NOT_RUN` | Requires `data/field_holdout/` and Colab GPU BioCLIP 2 logits |
| Coverage at that operating point | `field_holdout` | Report | `NOT_RUN` | Requires `data/field_holdout/` and Colab GPU BioCLIP 2 logits |
| OOD false-accept rate | `ood_test` | `<= 0.05` | `NOT_RUN` (Real BioCLIP 2) / `0.0000` (Local pipeline smoke test) | `ood_test` manifest (`18` real iNaturalist records across 9 disjoint hard-negative taxa) is built in `manifest_v2.parquet`; BioCLIP 2 GPU forward pass executes in Colab |
| ECE after calibration | `test` | `<= 0.05` | `NOT_RUN` (Real BioCLIP 2) | Temperature scaling (`TemperatureScaler`, 15 bins) verified in `tests/test_calibration.py`; full `test` crop evaluation executes in Colab |
| Hard-pair confusions (`M08<->M09`, `B03<->B04`, `B05<->crow`) | `field_holdout` | Report counts | `NOT_RUN` | Requires `data/field_holdout/` evaluation |
| End-to-end CPU latency `p50 / p95` | Local CPU (`WILDLIFE_TEST_MODE=1`) / Cloud Run | `<= 400 ms` p95 per crop | `p50 = 6.8 ms / p95 = 11.4 ms` (local CPU surrogate); `NOT_RUN` on Cloud Run 2 vCPU BioCLIP 2 | If BioCLIP 2 ViT-L/14 CPU p95 exceeds `400 ms` on Cloud Run (2 vCPU, 4 GiB), run `distill` tier into `EfficientNetV2-S` (`configs/train_v2.yaml`, Decision D2) |
| Gemini zero-shot macro-F1 (benchmark only) | `field_holdout` | Report | `NOT_RUN` | Requires `data/field_holdout/` and Vertex AI billing credentials (`reports/gemini_budget.md`) |
