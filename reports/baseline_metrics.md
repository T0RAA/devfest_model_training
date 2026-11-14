# Baseline & Calibration Evaluation Report

- **Checkpoint:** `/Users/harshranjan/DEVFEST/runs/efficientnet_b0_v1/best.ckpt`
- **Evaluated Split:** `validation` (`n=20` images across 20 species)
- **Data Provenance Note:** Metrics below are computed from the local deterministic synthetic smoke-test fixture (`synthetic_fixture`). Full-scale public iNaturalist S3 Open Data and blind UCSC Campus Field Holdout evaluations require running `notebooks/02_train_efficientnet.ipynb` and `notebooks/03_calibrate_evaluate.ipynb` on Colab GPU with GCS dataset shards.
- **Top-1 Accuracy:** `0.4500`
- **Top-3 Accuracy:** `0.5500`
- **Macro-F1:** `0.3367`
- **Fitted Temperature (`T`):** `0.3531`
- **Post-Calibration ECE:** `0.0999`
- **Post-Calibration NLL:** `2.0965`
- **Post-Calibration Brier Score:** `0.7503`
- **Selective Precision (Accepted IDs):** `1.0000` at Coverage `0.0500`
- **OOD False-Accept Rate (FAR):** `0.0000` (`0/40`)
- **Inference Latency (CPU per image):** `6.10 ms`

## Hard-Pair Confusion Audit (§11.3 #8)

- **Eastern (`M08`) vs. Western (`M09`) Gray Squirrel:** `M08->M09 = 0`, `M09->M08 = 0`
- **Steller's Jay (`B03`) vs. California Scrub-Jay (`B04`):** `B03->B04 = 0`, `B04->B03 = 0`
- **Common Raven (`B05`) Recall:** `0.0000`

## Per-Class Support & Metrics

| Class ID | Common Name | Scientific Name | Group | Support | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| `M01` | Black-tailed deer | `Odocoileus hemionus` | `large_mammal` | 1 | 0.5000 | 1.0000 | 0.6667 |
| `M02` | Coyote | `Canis latrans` | `large_mammal` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `M03` | Bobcat | `Lynx rufus` | `large_mammal` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `M04` | Gray fox | `Urocyon cinereoargenteus` | `large_mammal` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `M05` | Northern raccoon | `Procyon lotor` | `small_mammal` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `M06` | Striped skunk | `Mephitis mephitis` | `small_mammal` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `M07` | California ground squirrel | `Otospermophilus beecheyi` | `small_mammal` | 1 | 1.0000 | 1.0000 | 1.0000 |
| `M08` | Eastern gray squirrel | `Sciurus carolinensis` | `small_mammal` | 1 | 0.2500 | 1.0000 | 0.4000 |
| `M09` | Western gray squirrel | `Sciurus griseus` | `small_mammal` | 1 | 1.0000 | 1.0000 | 1.0000 |
| `M10` | Brush rabbit | `Sylvilagus bachmani` | `small_mammal` | 1 | 0.5000 | 1.0000 | 0.6667 |
| `B01` | Wild turkey | `Meleagris gallopavo` | `bird` | 1 | 0.3333 | 1.0000 | 0.5000 |
| `B02` | California quail | `Callipepla californica` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B03` | Steller's jay | `Cyanocitta stelleri` | `bird` | 1 | 0.3333 | 1.0000 | 0.5000 |
| `B04` | California scrub-jay | `Aphelocoma californica` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B05` | Common raven | `Corvus corax` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B06` | Acorn woodpecker | `Melanerpes formicivorus` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B07` | Anna's hummingbird | `Calypte anna` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B08` | American robin | `Turdus migratorius` | `bird` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `B09` | Dark-eyed junco | `Junco hyemalis` | `bird` | 1 | 1.0000 | 1.0000 | 1.0000 |
| `B10` | Spotted towhee | `Pipilo maculatus` | `bird` | 1 | 1.0000 | 1.0000 | 1.0000 |
