# Model Card — UCSC Wildlife Vision (`species-v1.0` + `gemini-4-argon`)

## 1. Model Overview

- **Model Version:** `species-v1.0`
- **Calibration Version:** `temp-v1` (Post-hoc single-parameter temperature scaling `T > 0`)
- **Photo Quality Rubric Version:** `quality-v1` (5-component independent rubric `Q`: 0–100)
- **Multimodal Explanation & Audit Model:** `gemini-4-argon` (via Google Cloud Vertex AI / Google Gen AI SDK, with deterministic offline rule synthesis)
- **Architecture:** `torchvision.models.efficientnet_b0` pretrained on ImageNet-1K, 20-class linear head (`M01`–`M10` mammals, `B01`–`B10` birds documented on the UC Santa Cruz Campus Natural Reserve).

## 2. Intended Use & Non-Goals

- **Intended Use:** Assist students, naturalists, and campus visitors in identifying 20 documented UCSC Campus Natural Reserve mammal and bird species from single-animal photographs, while providing an independent photograph quality score (`Q`: 0–100) and an identification-ready composite rating (`R`: 0–100).
- **Explicit Non-Goals:**
  - Does **not** infer population abundance, animal health, sex, age, or behavioral state from a photograph.
  - Does **not** grade the intrinsic worth or attractiveness of an animal—only how clear and identification-ready the photograph is.
  - Does **not** publish exact coordinates of sensitive wildlife sightings or retain unconsented EXIF metadata.
  - Does **not** automatically retrain on user feedback (`/v1/feedback` writes to a human-reviewed JSONL queue only).

## 3. Training & Licensing Governance

- **Permitted Licenses:** Commercial-eligible training manifests (`data/manifests/manifest.parquet`) enforce `cc0` and `cc-by` photo licenses exclusively, tracked in `data/manifests/photo_attribution.csv` and `data/manifests/license_audit.json`.
- **Research-Only Separation:** CUB-200-2011 (`cub_200_2011`) and non-commercial (`cc-by-nc`, `cc-by-nd`, `cc-by-sa`) images are rejected by `validate_photo_license()` unless `allow_research_only=True` is explicitly set for non-commercial ablation audits.
- **Leakage Prevention:** Splits (`data/splits/split_v1.csv`) are grouped by `observation_id` and verified against exact (`image_sha256`) and perceptual (`phash` Hamming distance <= 4) duplicate leakage across `train`, `validation`, and `test`.

## 4. Known Confusion Pairs & Failure Modes

1. **Eastern (`M08`) vs. Western (`M09`) Gray Squirrel:** Similar dorsal pelage in shaded redwood understory; ear-tuft and warm rufous wash on `M08` can be obscured in low light.
2. **Steller's Jay (`B03`) vs. California Scrub-Jay (`B04`):** Back-lit silhouettes where the crest of `B03` is flattened or cropped out.
3. **Common Raven (`B05`) vs. American Crow (OOD hard negative):** Requires bill shape, throat hackles, or wedge-shaped tail visibility; distant silhouettes should trigger `uncertain` or `Bird (species uncertain)`.
4. **Multiple Animals in Frame:** V1 is a whole-image/single-crop classifier. When multiple animals are present without a manual crop, the system returns `uncertain` and prompts the user to crop around a single individual.
