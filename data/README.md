# Data Directory & Governance

## Privacy, Licensing, and Git Policy

1. **No Raw Images in Git:** `data/raw/`, `data/processed/`, `data/field_holdout/`, and `data/ood/` are excluded by `.gitignore`. Store image shards and checkpoints in Google Cloud Storage (`gs://<YOUR_BUCKET>/ucsc-wildlife/`).
2. **License Enforcement:** Only photographs licensed under `cc0` or `cc-by` are permitted into commercial-eligible training manifests. Every ingested photograph is logged in `data/manifests/photo_attribution.csv` and audited in `data/manifests/license_audit.json`. Research-only datasets (such as CUB-200-2011) are blocked from entering production manifests unless explicitly flagged for non-commercial research comparison.
3. **Animal & Photographer Privacy:** Exact GPS coordinates of sensitive wildlife sightings and unconsented camera EXIF metadata are stripped before storage or public display.
4. **Observation-Grouped Splits:** `data/splits/split_v1.csv` assigns `train`, `validation`, and `test` partitions strictly grouped by `observation_id` (and separated by `observer_id` where feasible). No observation or perceptually identical image (`phash` Hamming distance <= 4) may cross split boundaries.
