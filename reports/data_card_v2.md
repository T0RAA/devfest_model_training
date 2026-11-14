# Data Card v2 — Real iNaturalist CC0/CC-BY Pull & Observation-Grouped Splits

- **Snapshot Date:** `2026-11-14T23:10:29Z`
- **Campus Place ID Verified:** `94596` (UC Santa Cruz Campus) & `3519` (UC Santa Cruz Campus Natural Reserve)
- **Total Verified Candidate Photos Pulled via Live API:** `380` (`{'train': 262, 'validation': 40, 'test': 40, 'ood_tune': 20, 'ood_test': 18}`)
- **Real Image Files Downloaded Locally (`data/raw/inat/`):** `20` (gitignored; full 800-obs/species S3 pull staged to GCS via Colab)
- **Leakage Check (`verify_no_split_leakage`):** `PASS (leakage_free=True)`
- **License Gate (`cc0`, `cc-by` only for classifier; `cdla-permissive-1.0` detector-only):** `PASS (rejected=0)`

## Per-Species California Availability & Local v2 Manifest Breakdown

| Class ID | Common Name | Scientific Name | CA Available CC0/CC-BY Obs (Live API) | Local Pull Obs | Photos | Unique Observers | Train | Val | Test | Campus Test | Data Flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M01` | Black-tailed deer | `Odocoileus hemionus` | 6234 | 12 | 20 | 8 | 16 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M02` | Coyote | `Canis latrans` | 2930 | 12 | 16 | 8 | 12 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M03` | Bobcat | `Lynx rufus` | 897 | 12 | 18 | 8 | 14 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M04` | Gray fox | `Urocyon cinereoargenteus` | 572 | 12 | 19 | 10 | 15 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M05` | Northern raccoon | `Procyon lotor` | 1380 | 12 | 14 | 4 | 10 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M06` | Striped skunk | `Mephitis mephitis` | 781 | 11 | 17 | 11 | 13 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M07` | California ground squirrel | `Otospermophilus beecheyi` | 5824 | 10 | 13 | 7 | 9 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M08` | Eastern gray squirrel | `Sciurus carolinensis` | 1146 | 12 | 16 | 3 | 12 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M09` | Western gray squirrel | `Sciurus griseus` | 1656 | 12 | 22 | 9 | 18 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `M10` | Brush rabbit | `Sylvilagus bachmani` | 1020 | 12 | 16 | 9 | 12 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B01` | Wild turkey | `Meleagris gallopavo` | 3608 | 12 | 17 | 10 | 13 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B02` | California quail | `Callipepla californica` | 3058 | 12 | 13 | 7 | 9 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B03` | Steller's jay | `Cyanocitta stelleri` | 2608 | 12 | 19 | 6 | 15 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B04` | California scrub-jay | `Aphelocoma californica` | 6601 | 12 | 20 | 8 | 16 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B05` | Common raven | `Corvus corax` | 4793 | 12 | 18 | 11 | 14 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B06` | Acorn woodpecker | `Melanerpes formicivorus` | 4950 | 12 | 16 | 9 | 12 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B07` | Anna's hummingbird | `Calypte anna` | 9035 | 12 | 20 | 9 | 16 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B08` | American robin | `Turdus migratorius` | 3045 | 12 | 13 | 11 | 9 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B09` | Dark-eyed junco | `Junco hyemalis` | 5509 | 12 | 15 | 12 | 11 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
| `B10` | Spotted towhee | `Pipilo maculatus` | 2798 | 12 | 20 | 6 | 16 | 2 | 2 | 0 | `low_data (local pilot sample; CA S3 pool >= 150)` |
