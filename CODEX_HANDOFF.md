# Codex handoff: UCSC Wildlife Vision

Written 2026-11-23 so Codex can work in parallel with Claude Code while Claude
usage is limited. Read [`AGENTS.md`](AGENTS.md) first.

## Where things stand

- `main` holds milestones 0–12 (the v2 two-stage pipeline): a MegaDetector V6
  plus OWLv2 detector, a BioCLIP 2 crop classifier, a calibrated unknown/OOD
  gate, a photo-quality and composite rating, and `/v2/predict`.
- Verified Colab G4 GPU metrics: BioCLIP 2 T0 0.5885 and T1 0.9304 (see
  `reports/`). Per the README's provenance note, the v1 checkpoint and
  baseline metrics are **synthetic**. Never present them as real.
- **Three pushed branches are ahead of `main` with no PR:**
  - `bioclip2-serving`: serves the trained BioCLIP 2 head from `/v2/predict`,
    loads MegaDetector at startup, and fixes the EXIF orientation parser that
    crashed on every phone photo. The iPhone app needs this to upload.
  - `feature-app-ui`: a React + TS + Tailwind Campus Wildlife web app in `web/`.
  - `fix-taxon-ids`: fixes 9 wrong iNaturalist taxon IDs in the 20-species
    taxonomy, plus `scripts/verify_taxon_ids.py`.
- The iOS app's camera path is untested.

## Claude's work in flight: don't touch

Nothing active. The three branches above are Claude's finished work waiting on
review. Don't rewrite them; open PRs from them instead (C-1).

## Codex queue

### C-1. Open PRs for the three branches, in order
`fix-taxon-ids` → `bioclip2-serving` → `feature-app-ui`. For each one: run the
gate on the branch, open a PR against `main` with what it changes and the test
output, and flag conflicts between them. Don't merge.

### C-2. Check whether the taxon fix invalidates anything
After `fix-taxon-ids`, list every manifest, split, cached embedding and report
that was built with the old IDs and needs rebuilding. Write the list to
`reports/taxon-fix-impact.md`. Rebuilding needs a GPU, so don't claim you did it.

### C-3. Contract tests for `/v2/predict`
FastAPI `TestClient` tests with stubbed detector and classifier: multi-animal
responses, the unknown-gate path, version-parity refusal, and EXIF-rotated JPEG
input (all eight orientations).

## Can't be done in the cloud

GPU training, large iNaturalist pulls (`scripts/pull_150_per_species.py`) and
Colab runs. Prepare commands and configs for Tora instead.
