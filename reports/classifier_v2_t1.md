# Classifier v2 — Tier 1 (`T1`) Frozen BioCLIP 2 Embeddings + Linear Head Report

- **Backbone:** `hf-hub:imageomics/bioclip-2` (frozen ViT-L/14, 768-D L2-normalized float16 embeddings)
- **Head:** 768 -> 20 multinomial logistic regression (`GroupKFold` on `observation_id`, `C in {0.1, 0.3, 1.0, 3.0, 10.0}`, `class_weight="balanced"`)
- **Embedding Cache Layout:** `embeddings/bioclip2/crop-v1/{split}.npy` (4 views per train crop, 1 center view per eval crop)
- **Detector & Crop Version Parity:** `detector_version = md6-apa-rtdetr-e@apache2.0`, `crop_version = crop-v1`

## Gate & Execution Status (Ground Rule #1 Compliance)

| Gate / Metric | Target Condition | Status / Value | Notes |
|---|---|---|---|
| `--require-torch` fail-loudly guard (`tests/test_train_backend.py`) | Raises `TorchBackendRequiredError` when `torch` is absent | `PASS` | Prevents silent fallback to NumPy surrogate in production training |
| Embedding cache 1-to-1 row & split alignment (`tests/test_embedding_cache.py`) | 100% match against `crops_index.parquet` | `PASS` | Verified for 1-view eval and 4-view train caches |
| `G-macro` (Full BioCLIP 2 Validation Macro-F1) | `>= 0.90` | `NOT_RUN` | Requires Colab GPU (`notebooks/06_bioclip2_embed_and_probe.ipynb`) to extract ViT-L/14 embeddings |
| `G-pairs` (`M08, M09, B03, B04, B05` recall) | `>= 0.80` each | `NOT_RUN` | Requires Colab GPU (`notebooks/06_bioclip2_embed_and_probe.ipynb`) |
| `G-floor` (Minimum per-class recall) | `>= 0.70` | `NOT_RUN` | Requires Colab GPU (`notebooks/06_bioclip2_embed_and_probe.ipynb`) |
