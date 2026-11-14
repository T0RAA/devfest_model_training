# Detector Evaluation Report v2 (`md6-apa-rtdetr-e@apache2.0` + `owlv2`)

- **Primary Detector:** `MDV6-apa-rtdetr-e` (PyTorch-Wildlife MegaDetector V6 Apache-2.0 RT-DETR)
- **Speed Alternative (MIT):** `MDV6-mit-yolov9-c` (AGPL variants strictly blocked by `validate_megadetector_variant_license()`)
- **Zero-Shot Fallback Detector:** `google/owlv2-base-patch16-ensemble` (Apache-2.0, queries: `["a photo of a bird", "a photo of a mammal"]`)
- **Configured Threshold (`tau_det`):** `0.25` (`configs/detector.yaml`)
- **NMS IoU Threshold:** `0.50` · **Max Boxes per Image:** `5`

## Execution & Gate Status (Ground Rule #1 Compliance)

| Metric / Step | Target Gate | Measured Value / Status | Notes |
|---|---|---|---|
| Detector contract & license guard (`tests/test_detector_contract.py`) | 100% pass | `PASS` | Rejects AGPL variants; enforces `Detection` invariants (`x2 > x1`, `y2 > y1`, valid labels/sources) |
| Gemini box conversion (`tests/test_gemini_box_conversion.py`) | 100% pass | `PASS` | Converts `[ymin, xmin, ymax, xmax]` (`0..1000`) to pixel `(x1, y1, x2, y2)` with clamping |
| Local sample detection pass (`20` real downloaded iNaturalist images) | Verify pipeline & crops | `20 / 20` boxed (`mean CPU salience fallback = 1.42 ms/image`) | Executed via `scripts/build_crops_v1.py` on real downloaded CC0/CC-BY images in `data/raw/inat/` |
| Full 600-image `data/detector_eval/` GPU recall & FPR benchmark (`MDV6-apa-rtdetr-e` + `OWLv2`) | Recall >= 0.97 overall, >= 0.90 per species, non-animal FPR <= 0.05 | `NOT_RUN` | Requires Colab/Vertex GPU with `PytorchWildlife` weights and human box verification in `notebooks/05_detector_eval_labeling.ipynb` |
| Detector fine-tuning on LILA + pseudo-boxes (§5.3 Step 6) | Only if stock recall gate fails | `NOT_RUN` | Stock `MDV6-apa-rtdetr-e` is frozen by default pending 600-image GPU eval |
