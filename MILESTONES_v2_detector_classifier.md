# UCSC Wildlife Vision · Milestones v2
## Two-Stage Detect-then-Classify Pipeline, Real Data Pull, and Model Training

**Audience.** Gemini (Work / Antigravity agent) operating on the `DEVFEST` repository at `/Users/harshranjan/DEVFEST`.
**Owner.** Harsh Ranjan.
**Version.** 2.0 · 2026-11-14.
**Supersedes.** Milestones 0 to 5 remain valid as scaffolding. This document adds Milestones 6 to 12 and changes the model architecture from a single whole-image classifier to a separate detector and classifier.

---

## 0. Read This First (Ground Rules for the Agent)

These rules are mandatory. Breaking any of them means the milestone is not complete.

1. **No fabricated results.** Every number written to `reports/`, `README.md`, a model card, or a commit message must come from a script that actually ran on real data in this milestone. If a step could not run, write `NOT_RUN` and the reason. Never write a plausible-looking number.
2. **The current checkpoint is a stub, not EfficientNet.** `runs/efficientnet_b0_v1/best.ckpt` is 62 KB and was trained on 100 synthetic images with the NumPy surrogate backend in `src/wildlife/models/classifier.py` (144-D hand-crafted features, `image_size: 96`). A real EfficientNet-B0 checkpoint is roughly 20 MB. Treat all numbers in `reports/baseline_metrics.md` as smoke-test output only. Do not cite them as model performance anywhere.
3. **Fail loudly on fallback.** Any training or evaluation run in Milestones 8 to 12 must use a real PyTorch backbone. Add a `--require-torch` flag (default `true` on Colab and Vertex) that raises an error if the model would fall back to the NumPy surrogate.
4. **Labels come from humans, not from Gemini.** Ground-truth species labels come from iNaturalist research-grade community identifications or from LILA expert annotations. Gemini may *flag* suspicious labels and *propose* bounding boxes for human review. Gemini output must never be written into a label column, and must never touch the validation, test, or field-holdout labels.
5. **No raw images in Git.** Images, crops, shards, and checkpoints live in Google Cloud Storage. Git holds code, configs, manifests (Parquet/CSV), and reports.
6. **License gate stays on.** Only `cc0` and `cc-by` iNaturalist photos enter commercial-eligible manifests (`configs/data_sources.yaml`). LILA datasets are `cdla-permissive-1.0` and need an explicit owner decision before use (see §11, Decision D1).
7. **One milestone per branch.** Continue the existing pattern with `milestone-6-data-pull`, `milestone-7-detector`, `milestone-8-crops`, `milestone-9-classifier-v2`, `milestone-10-calibration-v2`, `milestone-11-pipeline`, `milestone-12-eval-release`. Each branch must pass `pytest` before it is pushed.
8. **Stop and ask** before doing any of the following. Deleting data in GCS. Spending more than the Gemini budget cap in §9.4. Changing the 20-class taxonomy or class order in `configs/species.yaml`. Adding any dataset with a license not already allowed.

---

## 1. Why the Architecture Changes

The v1 design classified the whole photograph. It required the user to crop manually when several animals appeared, and it scored "visibility" against the full frame. Real user photos show a small jay in a large redwood scene, or a deer at the edge of a meadow. A whole-image classifier spends most of its resolution on background and learns background shortcuts (redwood bark means Steller's jay).

v2 separates the problem into two models with different jobs.

| Stage | Model | Question it answers | Output |
|---|---|---|---|
| Detector | Class-agnostic animal detector (MegaDetector V6, optional fine-tune) | Is there an animal, and where is it? | Boxes with `animal` / `person` / `vehicle` labels and confidences |
| Classifier | 20-species fine-grained classifier on the detector crop | Which of the 20 species is this animal? | Calibrated 20-way probabilities, top-3, acceptance decision |

This is the same structure Google uses in SpeciesNet. MegaDetector finds animals, humans, and vehicles but does not classify species, and a separate EfficientNet V2 M classifier assigns the species. SpeciesNet's default classifier variant crops to the top detection before classification, and its crop classifier expects tight crops resized to 480 × 480 px. Copy that pattern. Do not copy SpeciesNet itself as the classifier, because it targets roughly 2,000 camera-trap labels, not phone photos of these 20 species.

---

## 2. Target Architecture

### 2.1 Inference flow

```
          ┌──────────────────────────────────────────────────────────┐
 upload → │ 0. Validate & normalize                                   │
          │    EXIF orientation fix → strip EXIF/GPS → size limits    │
          └───────────────┬──────────────────────────────────────────┘
                          ▼
          ┌──────────────────────────────────────────────────────────┐
          │ 1. DETECTOR  (src/wildlife/detect/)                       │
          │    MegaDetector V6  MDV6-apa-rtdetr-e  (Apache-2.0)       │
          │    → boxes [x1,y1,x2,y2] px, label, conf                  │
          │    no animal ≥ τ_det → OWLv2 fallback ("a bird",          │
          │    "a mammal") → still none → status = no_animal          │
          │    person/vehicle only → status = no_animal               │
          └───────────────┬──────────────────────────────────────────┘
                          ▼  for each animal box ≥ τ_det (max 5)
          ┌──────────────────────────────────────────────────────────┐
          │ 2. CROP  (src/wildlife/detect/crops.py)                   │
          │    pad 15% each side → square with context → resize       │
          │    (same function used to build the training set)         │
          └───────────────┬──────────────────────────────────────────┘
                          ▼
          ┌──────────────────────────────────────────────────────────┐
          │ 3. CLASSIFIER  (src/wildlife/models/)                     │
          │    backbone chosen in M9 → 20 logits                      │
          │    ÷ temperature T → softmax → top-3                      │
          │    UnknownSpeciesGate: max p, margin, energy, group sum   │
          │    → identified | uncertain(group rollup) | unsupported   │
          └───────────────┬──────────────────────────────────────────┘
                          ▼
          ┌──────────────────────────────────────────────────────────┐
          │ 4. PHOTO QUALITY Q  (src/wildlife/quality/)               │
          │    computed on the crop, visibility uses box geometry     │
          │ 5. RATING R = round(0.65·Q + 0.35·100·C)  if accepted     │
          └───────────────┬──────────────────────────────────────────┘
                          ▼
          ┌──────────────────────────────────────────────────────────┐
          │ 6. OPTIONAL GEMINI EXPLANATION (src/wildlife/models/argon)│
          │    only on request or for uncertain results               │
          └──────────────────────────────────────────────────────────┘
```

### 2.2 Training flow

```
 iNat Open Data (S3, cc0/cc-by) ─┐
 LILA camera traps (if D1=yes) ──┼─► manifest v2 ─► grouped splits v2 ─┐
 UCSC field photos (holdout) ────┘                                     │
                                                                       ▼
                      ┌──────────────── DETECTOR TRACK ─────────────────┐
                      │ M7  run MD V6 on everything → detections.jsonl   │
                      │     hand-box detector eval set (600 images)      │
                      │     gate: recall ≥ target? ── yes ─► freeze MD   │
                      │                          └── no ─► fine-tune MD  │
                      └───────────────────────┬─────────────────────────┘
                                              ▼
                      ┌──────────────── CROP DATASET ───────────────────┐
                      │ M8  crops from the *frozen* detector + jitter    │
                      │     WebDataset shards in GCS                     │
                      └───────────────────────┬─────────────────────────┘
                                              ▼
                      ┌──────────────── CLASSIFIER TRACK ───────────────┐
                      │ M9  zero-shot → linear probes → fine-tune best   │
                      │ M10 temperature, OOD thresholds, group rollup    │
                      └───────────────────────┬─────────────────────────┘
                                              ▼
                       M11 pipeline integration → M12 field holdout eval
```

### 2.3 Why the detector and classifier are trained separately

1. **Different labels.** The detector needs boxes but no species. The classifier needs species but no boxes. iNaturalist has species labels and no boxes. LILA has boxes. Training them apart lets each use the data it can actually get.
2. **Different failure modes.** "No animal found" and "wrong species" need different fixes. Separate metrics make the fault obvious.
3. **Swappable parts.** The detector can be upgraded without retraining the classifier, as long as crops are regenerated with the same crop function.
4. **Train/serve parity.** The classifier must be trained on crops produced by the exact detector version and crop function used at inference. Record `detector_version` and `crop_version` in every classifier checkpoint, and refuse to load a checkpoint whose versions do not match the serving detector.

---

## 3. Data Sources

### 3.1 Primary training images: iNaturalist Licensed Observation Images on AWS

| Item | Value |
|---|---|
| Bucket | `s3://inaturalist-open-data` (region `us-east-1`, public, `--no-sign-request`) |
| Metadata | `s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz` contains `observations.csv.gz`, `observers.csv.gz`, `photos.csv.gz`, `taxa.csv.gz` (tab-separated) |
| Image URL | `https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}` |
| Sizes | `original` 2048 px, `large` 1024 px, `medium` 500 px, `small` 240 px |
| Use size | **`large`** for everything. Detector crops from `medium` lose too much resolution for small birds |
| Key join | `observations.observation_uuid = photos.observation_uuid`, `observations.taxon_id = taxa.taxon_id`, `observations.observer_id = observers.observer_id` |
| Filters | `quality_grade = 'research'`, `lower(license) in ('cc0','cc-by')`, taxon in the 20 IDs from `configs/species.yaml` |

The metadata is refreshed regularly. Record the archive filename and date in `data/manifests/source_snapshot.json` so the pull is reproducible.

**Geographic filter.** The current query in `data/manifests/s3_open_data_query.sql` limits observations to a California bounding box. Keep California as the default. If any species ends below the baseline floor of 150 observations, widen that species only to the Pacific states (CA, OR, WA) and record the widening per species in `source_snapshot.json`. Do not widen the 20 species to the whole continent, because regional plumage and pelage differ (for example, black-tailed deer vs. inland mule deer).

**Known bug to fix.** `s3_open_data_query.sql` reads `inaturalist-open-data/metadata/observations.csv.gz`. Those files ship inside the tarball. Extract the tarball first and point DuckDB at the extracted paths. Use `read_csv(..., delim='\t', header=true)` explicitly.

### 3.2 Bounding-box data for the detector: LILA BC camera traps (requires Decision D1)

| Dataset | Region | Relevant classes | Boxes | License |
|---|---|---|---|---|
| Caltech Camera Traps | Southwestern US | coyote, bobcat, raccoon, skunk, squirrel, rabbit, opossum, plus empties | ~66k | CDLA-Permissive |
| ENA24-detection | Eastern N. America | bobcat, gray fox, coyote, raccoon, striped skunk, eastern gray squirrel, wild turkey, plus crow, dog, cat | 9,772 | CDLA-Permissive-1.0 |
| Channel Islands Camera Traps | California | fox, bird, rodent, skunk | yes | CDLA-Permissive |
| Missouri Camera Traps | Missouri | 20 species | boxed subset | CDLA-Permissive |

Access paths: the LILA index at `https://lila.science`, the source.coop mirror `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/`, and Hugging Face community mirrors (`davanstrien/ena24-detection`, `lila-bc-community/*`). Annotations are COCO Camera Traps JSON.

Camera-trap images look different from phone photos (fixed camera, IR night shots, motion blur). Use them mainly for the **detector** and for **night/IR robustness**. Use LILA species labels in the classifier only if D1 allows it, and only for the overlapping classes.

### 3.3 Hard negatives and out-of-distribution (OOD) sets

These animals are common around Santa Cruz and are easy to confuse with one of the 20 classes. Pull them from iNaturalist with the same license filter. Resolve every taxon ID through `https://api.inaturalist.org/v1/taxa?q=<scientific name>` and write the resolved IDs to `configs/ood_taxa.yaml`. **Do not guess taxon IDs.**

| Confuses with | Hard negative (scientific name) |
|---|---|
| Eastern gray squirrel (M08), Western gray squirrel (M09) | Fox squirrel *Sciurus niger*, Douglas squirrel *Tamiasciurus douglasii* |
| Brush rabbit (M10) | Desert cottontail *Sylvilagus audubonii*, Black-tailed jackrabbit *Lepus californicus* |
| Gray fox (M04), Coyote (M02) | Red fox *Vulpes vulpes*, domestic dog *Canis familiaris* |
| Bobcat (M03) | Domestic cat *Felis catus*, Mountain lion *Puma concolor* |
| Raccoon (M05), Skunk (M06) | Virginia opossum *Didelphis virginiana* |
| Common raven (B05) | American crow *Corvus brachyrhynchos* |
| Steller's jay (B03), California scrub-jay (B04) | Woodhouse's scrub-jay *Aphelocoma woodhouseii*, Blue jay *Cyanocitta cristata* |
| Anna's hummingbird (B07) | Allen's hummingbird *Selasphorus sasin* |
| Spotted towhee (B10) | California towhee *Melozone crissalis* |
| Acorn woodpecker (B06) | Nuttall's woodpecker *Dryobates nuttallii*, Northern flicker *Colaptes auratus* |
| American robin (B08) | Hermit thrush *Catharus guttatus*, Varied thrush *Ixoreus naevius* |
| Dark-eyed junco (B09) | Song sparrow *Melospiza melodia* |

Non-animal OOD: LILA empty frames (Caltech Camera Traps is about 70% empty), plus 300 cc0/cc-by iNaturalist photos of plants and landscapes from the same region. People: keep MegaDetector's `person` class as a rejection path. Do not store or train on people photos.

Split the OOD taxa into two disjoint groups, `ood_tune` (used to pick thresholds in M10) and `ood_test` (used once in M12). A taxon never appears in both.

### 3.4 Field holdout (the number that matters)

`data/field_holdout/` holds UCSC campus photos taken by Harsh or contributors with written permission, target 30 per species (600 total) plus 100 OOD. Reserve these for M12 only. Never train, tune, or pick thresholds on them. Also reserve iNaturalist observations inside the UCSC campus place as a second "campus test" split. Find the place ID with `https://api.inaturalist.org/v1/places/autocomplete?q=UC Santa Cruz` and record it in `.env` as `INAT_CAMPUS_PLACE_ID`. Do not use the checklist ID `6494` as a place ID.

---

## 4. Milestone 6 · Real Data Pull (`milestone-6-data-pull`)

**Goal.** Replace the synthetic fixture with real, license-clean, observation-grouped data in GCS.

### Steps

1. **Environment.** In Colab, run `notebooks/00_colab_gpu_setup.ipynb`. Install `awscli duckdb pyarrow pandas google-cloud-storage imagehash pillow`. Authenticate to GCP with `google.colab.auth.authenticate_user()`. Set `GCS_BUCKET` and `GCS_PREFIX=ucsc-wildlife` from `.env`.
2. **Metadata snapshot.**
   ```bash
   aws s3 cp --no-sign-request \
     s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz /content/meta/
   tar -xzf /content/meta/inaturalist-open-data-latest.tar.gz -C /content/meta/
   ```
   Write archive name, size, SHA-256, and download date to `data/manifests/source_snapshot.json`.
3. **Query.** Fix and run `s3_open_data_query.sql` in DuckDB for the 20 target taxa, then a second query for the OOD taxa from `configs/ood_taxa.yaml`. Also select `p.extension`, `p.width`, `p.height`, `o.positional_accuracy`, and `observers.login`/`observers.name` for attribution. Output `data/manifests/inat_candidates.parquet`.
4. **Sampling policy.**
   1. Per species, sort observations randomly with seed 42.
   2. Take up to **800 observations** per species (the "improved" tier). Minimum acceptable is 150 (the "baseline" tier). Report any species below 150.
   3. Take up to **3 photos per observation** for train, and **only the first photo** for validation and test.
   4. Cap any single observer at 5% of a species' observations to stop one prolific photographer dominating.
5. **Download.** Fetch `large.{extension}` for each selected photo straight from the S3 bucket. Use 32 to 64 parallel workers with retries and exponential backoff. Write to `gs://$GCS_BUCKET/ucsc-wildlife/raw/inat/{taxon_id}/{photo_id}.jpg`. Strip EXIF on write. Do not use `api.inaturalist.org` for bulk image download. The API rate limits in `data_sources.yaml` (1 req/s, 10k/day) apply to API calls, and the API is only for taxon and place lookups.
6. **Integrity.** Compute `image_sha256` and `phash` per image. Drop exact duplicates. Drop near-duplicates with Hamming distance ≤ 4 inside the same species. Drop images under 256 px on the short side. Log every drop with a reason to `data/manifests/drop_log.csv`.
7. **Manifest v2.** Extend the 22-column schema in `src/wildlife/data/manifest.py` with these columns.

   | New column | Type | Meaning |
   |---|---|---|
   | `source_dataset` | str | `inat_open_data`, `lila_cct`, `lila_ena24`, `field_holdout`, ... |
   | `label_source` | str | `inat_research_grade`, `lila_expert`, `field_owner` |
   | `is_ood` | bool | true for hard negatives and non-animal frames |
   | `ood_group` | str | `ood_tune`, `ood_test`, or empty |
   | `gcs_uri` | str | full GCS path to the image |
   | `width`, `height` | int | pixels |
   | `det_*` | various | filled in M7 |
   | `crop_uri` | str | filled in M8 |
   | `gemini_flag` | str | filled in M7.5, review only, never a label |

8. **Splits v2.** Generate `data/splits/split_v2.csv` with 70/15/15 grouped by `observation_id`, observers separated where feasible, and no phash collision across splits. Campus-place observations go to `campus_test` regardless of the ratio. Keep `split_v1.csv` unchanged.
9. **Audit.** Rerun `scripts/audit_campus_frequency.py --online --place-id $INAT_CAMPUS_PLACE_ID` and replace the `UNVERIFIED_OFFLINE` cells with real counts.

### Deliverables

`source_snapshot.json`, `inat_candidates.parquet`, `manifest_v2.parquet`, `split_v2.csv`, `drop_log.csv`, `photo_attribution_v2.csv`, `license_audit_v2.json`, `reports/data_card_v2.md` with a per-species table of observations, photos, observers, and split sizes.

### Acceptance gate

All 20 species have ≥ 150 train observations, or the shortfall is reported and Harsh has approved it. `tests/test_split_leakage.py` and `tests/test_manifest.py` pass on v2. A new `tests/test_license_v2.py` proves that no non-allowed license reached the manifest.

---

## 5. Milestone 7 · Detector (`milestone-7-detector`)

**Goal.** A frozen, versioned, measured animal detector whose boxes feed the classifier.

### 5.1 Model choice

Use **PyTorch-Wildlife MegaDetector V6**, variant **`MDV6-apa-rtdetr-e`** (RT-DETR, Apache-2.0). It has the highest animal recall in the MegaDetector V6 table and a permissive license.

```python
from PytorchWildlife.models import detection as pw_detection
detector = pw_detection.MegaDetectorV6Apache(version="MDV6-apa-rtdetr-e")
```

**Do not use the default `MegaDetectorV6()` or any `yolov9`/`yolov10` variant without the `mit` prefix.** Those weights are AGPL-3.0, and serving them from Cloud Run would trigger AGPL network-use obligations. If speed matters more than recall, the MIT alternative is `MDV6-mit-yolov9-c`.

**Fallback detector.** OWLv2 (`google/owlv2-base-patch16-ensemble` in Hugging Face Transformers, Apache-2.0) with text queries `["a photo of a bird", "a photo of a mammal"]`. Run it only when MegaDetector returns no animal box ≥ τ_det. MegaDetector was trained on camera-trap images and may miss small perched birds in phone photos. OWLv2 covers that gap.

### 5.2 Detector contract

Add `src/wildlife/detect/` with this interface and freeze it.

```python
@dataclass(frozen=True)
class Detection:
    x1: int; y1: int; x2: int; y2: int   # absolute pixels, original image, x2 > x1
    label: Literal["animal", "person", "vehicle"]
    confidence: float                    # 0..1
    source: Literal["megadetector_v6", "owlv2", "manual"]

class AnimalDetector(Protocol):
    version: str                          # e.g. "md6-apa-rtdetr-e@<sha256[:12]>"
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...
```

Gemini returns boxes as `[ymin, xmin, ymax, xmax]` normalized to 0 to 1000. Any code that reads Gemini boxes must convert with `x = round(v / 1000 * width)` and `y = round(v / 1000 * height)`, swap the axis order, and clamp to the image. Put this in one tested helper, `gemini_box_to_pixels()`.

### 5.3 Steps

1. **Batch inference.** Run the detector over every image in `manifest_v2` on a Colab GPU. Write `gs://.../detections/md6-apa-rtdetr-e/{source_dataset}.jsonl` with all boxes ≥ 0.05 confidence, so thresholds can be retuned later without rerunning.
2. **Detector evaluation set.** Build `data/detector_eval/` with 600 images. That is 20 iNaturalist validation photos per species (400), 100 non-animal frames, and 100 hard-negative animals. Hand-check one box per animal using a simple labeling notebook (`notebooks/05_detector_eval_labeling.ipynb`) that shows MegaDetector's box and lets Harsh accept, adjust, or redraw it. Gemini may pre-draw boxes here to save clicks, but a human approves every box.
3. **Metrics.** Report all of these in `reports/detector_v2.md`.
   1. Image-level animal recall (at least one box with IoU ≥ 0.5 against the true box) per species, at τ_det ∈ {0.1, 0.2, 0.3, 0.5}.
   2. False-positive rate on the 100 non-animal frames.
   3. Mean IoU of the top box against the true box.
   4. Recall split by `bird` vs `mammal` and by box area (< 5%, 5 to 20%, > 20% of the frame).
   5. OWLv2 fallback gain, meaning how many misses it recovers and how many false positives it adds.
   6. GPU and CPU latency per image.
4. **Choose τ_det.** Pick the lowest threshold with a non-animal false-positive rate ≤ 5%. Write it to `configs/detector.yaml`.
5. **Gate.**
   1. If image-level recall ≥ 0.97 overall and ≥ 0.90 for every species, **freeze the stock detector**. Go to M8.
   2. Otherwise, go to step 6.
6. **Fine-tune (only if the gate fails).** Use the PyTorch-Wildlife detection fine-tuning module, starting from the pretrained MegaDetector V6 weights. Training data is the LILA boxes (if D1 allows) plus iNaturalist *pseudo-boxes*. Pseudo-boxes are MegaDetector boxes ≥ 0.6 confidence that Gemini also confirms (see §5.4), plus a random 10% human-checked sample. Keep the label set as `animal`, `person`, `vehicle`. Hold out the 600-image eval set completely. Re-measure. If the fine-tuned model beats stock by ≥ 2 recall points without raising false positives above 5%, adopt it.
7. **Freeze.** Write `configs/detector.yaml` with the model name, weights SHA-256, τ_det, NMS IoU, max boxes = 5, and fallback settings. Bump `detector_version`.

### 5.4 Optional Gemini box verification (M7.5)

For pseudo-box quality control and label auditing, send each image with its MegaDetector box to Gemini with a structured-output schema. Ask exactly these questions. Is there one clearly visible wild animal inside the box (yes/no)? Is the box tight (yes/no)? If not, give the corrected `box_2d`. Does the animal look like `{common_name}` (yes / no / cannot tell)? Store the answers in `gemini_flag` as JSON. Rows where Gemini says "no" go to a human review queue `data/manifests/review_queue.csv`. Nothing is relabeled or deleted automatically. Use the cheaper Flash-tier model for bulk calls and reserve `gemini-4-argon` for the review queue. Check the current model IDs on the Gemini API models page before you start.

### Deliverables

`src/wildlife/detect/{__init__,megadetector,owl_fallback,crops,gemini_boxes}.py`, `configs/detector.yaml`, `detections/*.jsonl` in GCS, `data/detector_eval/` labels (boxes only, no images in Git), `reports/detector_v2.md`, `tests/test_detector_contract.py`, `tests/test_gemini_box_conversion.py`.

---

## 6. Milestone 8 · Crop Dataset (`milestone-8-crops`)

**Goal.** A crop dataset made by the frozen detector, identical to what the classifier will see in production.

### Steps

1. **Crop function** (`src/wildlife/detect/crops.py`, version `crop-v1`).
   1. Take the top-confidence `animal` box ≥ τ_det. For training images with several boxes, take the largest box whose confidence is within 0.1 of the top box.
   2. Pad 15% of box width and height on each side.
   3. Expand the shorter side to make the region square, using real image context, not black bars. Clamp to the image and shift the square inward if it hits an edge.
   4. Resize to the classifier input size chosen in M9 (224 for BioCLIP 2, 384 or 480 for EfficientNetV2). Store crops at **512 px** so any later input size can be produced without re-cropping.
2. **No-detection images.** If neither detector finds an animal in a *training* image of a target species, keep the whole image as a fallback sample (`crop_mode = "full"`), up to 10% of that species. Log the rest to `drop_log.csv`. These cases teach the classifier to cope when the detector fails at inference.
3. **Train-time box jitter.** For the train split only, also save the raw box so the data loader can jitter it on the fly (scale 0.9 to 1.25, center shift ± 5% of box size). Validation and test use the exact crop function with no jitter.
4. **Shards.** Write WebDataset `.tar` shards of about 1,000 samples each to `gs://.../crops/crop-v1/{split}/shard-{000000}.tar`. Each sample holds `jpg`, `cls` (class index from the frozen `classes.json`), and `json` (observation_id, photo_id, box, crop_mode, source_dataset, license). Write `crops_index.parquet` to Git.
5. **Visual check.** Save a 10 × 20 contact sheet (10 random crops per species) to `reports/crop_contact_sheet_v1.png`. Harsh reviews it before M9 starts.

### Acceptance gate

Every crop record links back to an observation in `manifest_v2`. Crop version and detector version are recorded in the index. The contact sheet shows the animal centered in at least 95% of tiles, judged by eye.

---

## 7. Milestone 9 · Classifier v2 (`milestone-9-classifier-v2`)

**Goal.** Choose the best backbone by measurement, then fine-tune it on crops.

### 7.1 Candidate backbones

| ID | Backbone | Why it is a candidate | License | Notes |
|---|---|---|---|---|
| C0 | Current EfficientNet-B0 (real torchvision weights) | Existing baseline, fastest | BSD-3 | Must actually run with torch this time |
| C1 | EfficientNetV2-S / V2-M (`timm`) | Same family SpeciesNet uses for wildlife | Apache-2.0 | Train at 384 or 480 |
| C2 | **BioCLIP 2** (`hf-hub:imageomics/bioclip-2`, ViT-L/14, OpenCLIP) | Trained on TreeOfLife-200M (214M organism images), built for species ID, supports zero-shot from species names | MIT | 224 input. Strongest prior for this task |
| C3 | SigLIP 2 or DINOv3 (optional) | Strong general-purpose encoders for linear probes | Check each model's license before use | Only if C2 underperforms |

Gemma and PaliGemma are **not** candidates for the classifier. They produce text, not a 20-way class distribution, which breaks temperature scaling and the energy-score OOD gate. They are too slow for a per-request classifier. Gemini can be evaluated as a zero-shot comparison in M12 (§9.3) but is not the production classifier.

### 7.2 Step 1 · Zero-shot baseline (no training, about 30 minutes)

Run BioCLIP 2 zero-shot on the validation crops. Text prompts use the scientific name and common name (for example, `"a photo of Cyanocitta stelleri, Steller's jay"`). Report macro-F1 and the confusion matrix. This is the floor every trained model must beat.

### 7.3 Step 2 · Linear probes (about 1 to 2 hours on one GPU)

For C0, C1, C2 (and C3 if used), freeze the backbone, extract one embedding per validation and train crop (center crop only, no augmentation), cache the embeddings to GCS as `.npy`, and fit a multinomial logistic regression (`sklearn`, `C` tuned on 5-fold grouped CV inside train). Report in `reports/backbone_probe_v2.md`.

1. Macro-F1 and top-1 / top-3 accuracy on validation.
2. Per-class recall, with the three hard pairs called out (M08/M09, B03/B04, B05 vs American crow from `ood_tune`).
3. ECE before calibration.
4. Embedding plus head latency on CPU (Cloud Run target) and on GPU.

**Selection rule.** Pick the backbone with the highest validation macro-F1. If two are within 1.5 points, pick the one with better hard-pair recall. If still tied, pick the faster one.

### 7.4 Step 3 · Fine-tune the winner

Use the two-stage recipe already in `src/wildlife/train.py`, rewritten as a real PyTorch training loop with a WebDataset loader.

| Setting | ConvNet (C0/C1) | ViT-L (C2 BioCLIP 2) |
|---|---|---|
| Input size | 384 (V2-S) or 480 (V2-M) | 224 |
| Stage A (head only) | 4 epochs, lr 1e-3 | 3 epochs, lr 1e-3 |
| Stage B (unfreeze) | full network, backbone lr 3e-5, head lr 1e-3 | last 6 transformer blocks + final norm, layer-wise lr decay 0.75, top lr 2e-5, head lr 1e-3 |
| Epochs B | up to 24, patience 5 | up to 15, patience 4 |
| Optimizer | AdamW, wd 1e-4 | AdamW, wd 0.05 |
| Schedule | cosine with 1 epoch warmup | cosine with 1 epoch warmup |
| Precision | bf16 on A100/L4, fp16 + GradScaler on T4 | bf16, gradient checkpointing on |
| Batch | 64 | 32 to 64 with gradient accumulation to effective 128 |
| Loss | cross-entropy, label smoothing 0.05 | same |
| Sampling | class-balanced sampler, cap 800 observations per class | same |
| Selection metric | validation macro-F1 | same |
| Seeds | 42, 1337, 2026 (report mean ± std) | same |

**Augmentation (species-safe).** Random resized crop scale 0.7 to 1.0 over the jittered box, horizontal flip 0.5, rotation ± 12°, brightness/contrast ± 0.15, saturation ± 0.10, **hue jitter 0** (color is diagnostic, for example the Steller's jay blue, robin breast, and towhee flank), Gaussian blur p 0.1, JPEG quality 60 to 95 p 0.3. For LILA night images (if used), add grayscale p 0.1 to all training images so IR frames are not a shortcut. No vertical flip. No mixup on hard-pair classes.

**Compute plan.** Run on Colab with whatever GPU is assigned. Prefer A100 or L4 for BioCLIP 2. Checkpoint to `gs://.../checkpoints/{run_id}/` every epoch so a disconnected Colab session can resume with `--resume`. If Colab is unavailable for more than a day, use `scripts/vertex_custom_job.sh` with an L4 or A100 machine instead of the T4 in `train.yaml`.

**Checkpoint payload** must include `backbone`, `input_size`, `classes.json` SHA-256, `detector_version`, `crop_version`, `train_manifest_sha256`, `split_version`, git commit, and seed.

### 7.5 Optional distillation for Cloud Run

If the winner is BioCLIP 2 and CPU latency on Cloud Run is above 400 ms per crop, distill it into EfficientNetV2-S. The teacher's temperature-2 soft labels plus ground-truth cross-entropy (weight 0.5 each), trained on the same crops. Ship the student only if its macro-F1 is within 2 points of the teacher's.

### Deliverables

`src/wildlife/models/backbones.py` (factory for C0 to C3), `src/wildlife/data/webdataset_loader.py`, rewritten `src/wildlife/train.py`, `notebooks/06_backbone_probe.ipynb`, `notebooks/07_finetune_v2.ipynb`, `reports/backbone_probe_v2.md`, `reports/classifier_v2_training.md`, checkpoints in GCS, and `tests/test_train_backend.py` (fails if the NumPy surrogate is used with `--require-torch`).

---

## 8. Milestone 10 · Calibration, Unknown Rejection, Group Rollup (`milestone-10-calibration-v2`)

**Goal.** Probabilities that mean what they say, and a gate that says "uncertain" instead of guessing.

### Steps

1. **Temperature scaling.** Fit a single T on validation-crop logits by minimizing NLL (`src/wildlife/models/calibration.py` already implements this). Report ECE (15 bins), NLL, Brier, and reliability diagrams before and after.
2. **Gate thresholds.** Using validation crops (in-distribution) and `ood_tune` crops (out-of-distribution), grid-search `max_prob_threshold`, `margin_threshold`, and `energy_threshold`. Choose the setting that maximizes in-distribution coverage subject to an OOD false-accept rate ≤ 0.05 (the `false_accept_rate_target` in `configs/rating.yaml`). Write the chosen values to `configs/rating.yaml` and bump `calibration_version`. Remove the hard-coded thresholds in `WildlifePredictor.__init__` and read them from config.
3. **Group rollup.** When the species decision fails the gate, sum calibrated probabilities within `large_mammal`, `small_mammal`, and `bird`. If one group's sum is ≥ `coarse_group_agreement_min_sum` (0.75), return `uncertain` with `"Bird (species uncertain)"` or the matching group label. This is the same idea as SpeciesNet's label rollup, where low-confidence species predictions fall back to a higher taxon.
4. **Detector confidence in the decision.** If the selected box confidence is below τ_det + 0.1 and came from the OWLv2 fallback, cap the result at `uncertain` regardless of classifier confidence.
5. **Report.** `reports/calibration_v2.md` with risk-coverage curves, selective precision at the chosen operating point, OOD false-accept rate on `ood_tune`, and the hard-pair confusion counts after gating.

### Acceptance gate

Post-calibration ECE ≤ 0.05 on validation. OOD false-accept rate ≤ 0.05 on `ood_tune`. Thresholds live in config and no longer in code.

---

## 9. Milestone 11 · Pipeline Integration (`milestone-11-pipeline`)

**Goal.** One call from photo to result, using detector then classifier, behind the existing FastAPI service.

### 9.1 Code changes

1. **`src/wildlife/pipeline.py`** (new). `WildlifePipeline(detector, classifier, gate, quality, argon)` with `run(raw_bytes, manual_box=None) -> PredictionResponse`. A manual box from the UI skips the detector and is marked `source="manual"`.
2. **`src/wildlife/infer.py`.** `WildlifePredictor.predict_bytes` delegates to `WildlifePipeline`. Remove `image_size = 96` and the default `use_torch_if_available=False` path for production. Keep the NumPy surrogate only behind an explicit `WILDLIFE_TEST_MODE=1`.
3. **Multiple animals.** Classify up to 5 animal boxes. The response returns a `detections` array, one result per box, and a `primary` result for the box with the highest `confidence × box_area`. Replace the v1 behavior of "uncertain, please crop" with per-animal results. Keep `uncertain` when two boxes overlap with IoU > 0.5 but disagree on species.
4. **Quality on the crop.** `score_photograph` receives the crop and the box geometry. Visibility uses how much of the padded box lies inside the frame and whether the box touches an edge.
5. **Response schema** (`src/wildlife/schemas.py`) adds `detections[]`, `primary_index`, `detector_version`, `crop_version`, and per-detection `box`, `box_source`, `detector_confidence`, `status`, `species`, `top_3`, `quality`, `rating`. Bump the API to `/v2/predict` and keep `/v1/predict` as a wrapper that returns `primary` only.
6. **Docker.** `api/Dockerfile` installs `PytorchWildlife` (pinned), `open_clip_torch` if BioCLIP 2 wins, and `timm`. Weights are downloaded at build time from GCS, not at request time. Verify the image runs with 2 vCPU and 4 GiB on Cloud Run.

### 9.2 Tests to add

`tests/test_pipeline_no_animal.py` (landscape → `no_animal`), `tests/test_pipeline_person.py` (person box only → `no_animal`, no species output), `tests/test_pipeline_multi.py` (two boxes → two results), `tests/test_version_parity.py` (checkpoint with mismatched `detector_version` refuses to load), and `tests/test_api_v2.py`.

### 9.3 Gemini roles in production

Gemini stays outside the per-request critical path.

1. **Explanation** (existing `argon.py`). Called only when the user taps "Explain" or when the status is `uncertain`. Send the crop, the top-3 calibrated probabilities, and the quality components. The prompt must forbid Gemini from overriding the classifier's species and must ask it to describe visible features only.
2. **Zero-shot comparison** (M12 only). Run Gemini on the test crops with a fixed prompt that lists the 20 species plus "none of these". This is a benchmark, not a production path.

### 9.4 Gemini budget

Before any bulk Gemini job (M7.5 or M12), run 50 calls, measure the actual cost from billing, extrapolate, and write the estimate to `reports/gemini_budget.md`. Stop and ask Harsh if the estimate is above **$25** for the job. `gemini-4-argon` pricing changes on 2026-11-30, so recompute after that date.

---

## 10. Milestone 12 · Evaluation and Release (`milestone-12-eval-release`)

**Goal.** Honest, final numbers on data the model has never influenced.

### Steps

1. Run the full pipeline (detector, crop, classifier, gate) on `test`, `campus_test`, `field_holdout`, and `ood_test`. Never retune anything after seeing these results. If a retune is needed, it becomes v2.1 with a new holdout.
2. Report end-to-end metrics in `reports/eval_v2.md`.

   | Metric | Split | Target |
   |---|---|---|
   | Detector image-level recall | field_holdout | ≥ 0.95 |
   | Species macro-F1 (all images, gate off) | test / campus_test / field_holdout | report all three |
   | Selective precision at chosen operating point | field_holdout | ≥ 0.95 |
   | Coverage at that operating point | field_holdout | report |
   | OOD false-accept rate | ood_test | ≤ 0.05 |
   | ECE after calibration | test | ≤ 0.05 |
   | Hard-pair confusions (M08↔M09, B03↔B04, B05↔crow) | field_holdout | report counts |
   | End-to-end CPU latency p50 / p95 | Cloud Run | report |
   | Gemini zero-shot macro-F1 (benchmark only) | field_holdout | report |

3. Update `reports/model_card.md` to v2. Architecture, data sources with licenses and counts, detector and classifier versions, known failure modes from the field holdout, and the explicit statement that numbers from v1 were synthetic.
4. Update `README.md` milestone list and quickstart. Remove any claim that is not backed by a v2 report.
5. Deploy to Cloud Run with `gcloud run deploy`, keeping v1 as a separate revision for rollback.

---

## 11. Open Decisions for Harsh (agent must ask, not assume)

| ID | Decision | Default if no answer |
|---|---|---|
| D1 | Allow `cdla-permissive-1.0` (LILA camera traps) for detector training? For classifier training? | Detector yes, classifier no |
| D2 | Final classifier size limit for Cloud Run (allow ViT-L or require distillation)? | Distill if p95 > 400 ms |
| D3 | Gemini budget cap per bulk job | $25 |
| D4 | Accept species below 150 observations, or drop them from v2? | Keep and flag |
| D5 | Use iNaturalist observations inside the campus place for training at all? | No, test only |

---

## 12. Updated Repository Layout (new or changed paths only)

```
configs/
  detector.yaml            # NEW  detector name, weights sha, τ_det, NMS, fallback
  ood_taxa.yaml            # NEW  resolved hard-negative taxon IDs, ood_tune/ood_test
  train_v2.yaml            # NEW  backbone, input size, stage A/B settings
  rating.yaml              # CHANGED  thresholds from M10
  data_sources.yaml        # CHANGED  LILA entries, cdla decision
src/wildlife/
  detect/                  # NEW  megadetector.py, owl_fallback.py, crops.py, gemini_boxes.py
  data/webdataset_loader.py# NEW
  models/backbones.py      # NEW
  pipeline.py              # NEW
  train.py                 # REWRITTEN  real torch loop, --require-torch, --resume
  infer.py                 # CHANGED  delegates to pipeline
notebooks/
  05_detector_eval_labeling.ipynb   # NEW
  06_backbone_probe.ipynb           # NEW
  07_finetune_v2.ipynb              # NEW
reports/
  data_card_v2.md, detector_v2.md, backbone_probe_v2.md,
  classifier_v2_training.md, calibration_v2.md, eval_v2.md, gemini_budget.md
```

GCS layout under `gs://$GCS_BUCKET/ucsc-wildlife/`.

```
raw/inat/{taxon_id}/{photo_id}.jpg
raw/lila/{dataset}/...
detections/{detector_version}/{source_dataset}.jsonl
crops/{crop_version}/{split}/shard-*.tar
embeddings/{backbone}/{split}.npy
checkpoints/{run_id}/{epoch}.ckpt
```

---

## 13. Definition of Done for v2

1. The detector and classifier are separate, versioned, and independently measured.
2. Every reported number traces to a script, a commit, and a data snapshot.
3. The field holdout was used once, at the end.
4. `pytest` passes on `main` after merging milestones 6 to 12.
5. `/v2/predict` on Cloud Run returns boxes, per-animal species with calibrated top-3, quality, rating, and versions for a real campus photo.

---

## 14. Sources

**Data**
1. iNaturalist Licensed Observation Images, AWS Open Data Registry. https://registry.opendata.aws/inaturalist-open-data/
2. iNaturalist Open Data documentation (bucket layout, metadata files, image sizes, attribution). https://github.com/inaturalist/inaturalist-open-data
3. Getting the iNaturalist AWS Open Data metadata files and working with them in a database (iNaturalist Forum). https://forum.inaturalist.org/t/getting-the-inaturalist-aws-open-data-metadata-files-and-working-with-them-in-a-database/22135
4. iNaturalist API v1 (taxa and places lookups only). https://api.inaturalist.org/v1/docs/
5. LILA BC, Labeled Information Library of Alexandria. Biology and Conservation. https://lila.science/datasets
6. LILA mirror on source.coop. https://source.coop/agentmorris/lila-wildlife
7. Caltech Camera Traps. https://beerys.github.io/CaltechCameraTraps
8. ENA24-detection (Hugging Face mirror, CDLA-Permissive-1.0). https://huggingface.co/datasets/davanstrien/ena24-detection
9. LILA BC community detection mirrors. https://huggingface.co/lila-bc-community
10. UCSC Campus Natural Reserve mammal and bird inventories (class list validation only). https://ucsccampusreserve.ucsc.edu/species-habitats/

**Detector**
11. MegaDetector model zoo (V6 variants, recall, licenses). https://microsoft.github.io/MegaDetector/model_zoo/
12. PyTorch-Wildlife model zoo and loaders. https://microsoft.github.io/Pytorch-Wildlife/model_zoo/
13. PyTorch-Wildlife repository (detection fine-tuning module). https://github.com/microsoft/CameraTraps
14. OWLv2, Hugging Face Transformers documentation. https://huggingface.co/docs/transformers/model_doc/owlv2

**Classifier and reference architecture**
15. SpeciesNet (Google), detector + classifier ensemble, crop vs whole-image variants, label rollup. https://github.com/google/cameratrapai
16. Gadot et al., *To crop or not to crop. Comparing whole-image and cropped classification on a large dataset of camera trap images*, IET Computer Vision, 2024. https://doi.org/10.1049/cvi2.12318
17. BioCLIP 2 model card. https://huggingface.co/imageomics/bioclip-2
18. Gu et al., *BioCLIP 2. Emergent Properties from Scaling Hierarchical Contrastive Learning*, NeurIPS 2025. https://arxiv.org/abs/2505.23883
19. timm (EfficientNetV2 weights). https://github.com/huggingface/pytorch-image-models
20. WebDataset. https://github.com/webdataset/webdataset

**Calibration and OOD**
21. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (temperature scaling). https://arxiv.org/abs/1706.04599
22. Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020. https://arxiv.org/abs/2010.03759

**Google tools**
23. Gemini API image understanding and object detection (`box_2d`, 0 to 1000 normalization). https://ai.google.dev/gemini-api/docs/image-understanding
24. Vertex AI bounding box detection. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection
25. Gemini 4 Argon model card. https://deepmind.google/models/model-cards/gemini-4-argon/
26. Gemma vision documentation (why Gemma is not used as the classifier). https://ai.google.dev/gemma/docs/capabilities/vision
27. Vertex AI custom training. https://cloud.google.com/vertex-ai/docs/training/overview
28. Cloud Run. https://cloud.google.com/run/docs
