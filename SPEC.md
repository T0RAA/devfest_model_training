# UCSC Wildlife Vision — 20-Species Identification and Photo Rating

**Implementation specification for Google Antigravity**  
**Version:** 1.0 · **Date:** 2026-11-14  
**Training:** Google Colab GPU (primary); Google Cloud Vertex AI custom training (optional scale-up)  
**Multimodal Assist & Audit:** Gemini 4 Argon (`gemini-4-argon`) via Google Cloud Vertex AI / Google Gen AI SDK  
**Scope:** Wild mammals and birds documented in the UC Santa Cruz campus reserve. No reptiles, amphibians, insects, pets, or people are species classes.

> **Important data limitation.** The 20 species below are a **campus-documented, practical starter set**, chosen for recognizability and likely opportunities for photography. They have **not** been validated as the exact 20 most abundant or most photographed animals on campus. The UCSC inventories are presence/habitat lists, not abundance rankings. Before claiming “20 most common,” perform the observation-frequency audit in §3 and record its results. iNaturalist reporting frequency measures observer activity and detectability, not population abundance.

## 1. Product requirements

Create a reproducible computer-vision system that accepts a photograph and returns:

1. Whether a supported wild animal is identifiable; otherwise `no_animal`, `unsupported_animal`, or `uncertain`.
2. A predicted **species**, common name, scientific name, and one of `large_mammal`, `small_mammal`, `bird`.
3. **Top-3 species probabilities** calibrated on held-out validation images (not raw softmax presented as fact).
4. A **photograph quality score (0–100)**, independent of species identity: sharpness, exposure, subject visibility, useful image resolution, and composition.
5. An **identification-ready photo rating (0–100)** that combines calibrated identification confidence and image quality; always display both components separately.
6. An interpretable explanation, e.g., “well-focused bird, partially hidden tail; likely Steller's jay.” Avoid invented biological certainty.
7. Model version and the right to correct a prediction. Store corrections for *review*, not automatic retraining.

## 2. Initial 20 target species

### 2.1 Mammals (10 species)

| ID | Group | Common name | Scientific name / dataset search key | Likely campus habitat | Primary image source |
|---|---|---|---|---|---|
| M01 | Large | Black-tailed deer | `Odocoileus hemionus` (coastal black-tailed form) | Meadows, woodland edges | iNaturalist licensed observations, local photos |
| M02 | Large | Coyote | `Canis latrans` | Open meadows, woodland | iNaturalist, licensed camera-trap data |
| M03 | Large | Bobcat | `Lynx rufus` | Forest edges, meadows | iNaturalist, licensed camera-trap data |
| M04 | Large | Gray fox | `Urocyon cinereoargenteus` | Woodlands, mixed forest | iNaturalist, licensed camera-trap data |
| M05 | Small/medium | Northern raccoon | `Procyon lotor` | Riparian, disturbed and wooded areas | iNaturalist, licensed camera-trap data |
| M06 | Small/medium | Striped skunk | `Mephitis mephitis` | Multiple habitats | iNaturalist, licensed camera-trap data |
| M07 | Small | California ground squirrel | `Otospermophilus beecheyi` | Grasslands, agricultural edges | iNaturalist, campus photos |
| M08 | Small | Eastern gray squirrel | `Sciurus carolinensis` | Developed and forested zones | iNaturalist, campus photos |
| M09 | Small | Western gray squirrel | `Sciurus griseus` | Forested areas | iNaturalist, campus photos |
| M10 | Small | Brush rabbit | `Sylvilagus bachmani` | Chaparral, meadow edges | iNaturalist, campus photos |

### 2.2 Birds (10 species)

| ID | Group | Common name | Scientific name / dataset search key | Likely campus habitat | Primary image source |
|---|---|---|---|---|---|
| B01 | Bird | Wild turkey | `Meleagris gallopavo` | Mixed woodland, open edges | iNaturalist, optional CUB audit |
| B02 | Bird | California quail | `Callipepla californica` | Arboretum, open edges | iNaturalist, optional CUB audit |
| B03 | Bird | Steller's jay | `Cyanocitta stelleri` | Redwood and mixed forest | iNaturalist, optional CUB audit |
| B04 | Bird | California scrub-jay | `Aphelocoma californica` | Chaparral and woodland edge | iNaturalist; inspect older “Western scrub-jay” labels |
| B05 | Bird | Common raven | `Corvus corax` | Forest and grassland | iNaturalist, optional CUB audit |
| B06 | Bird | Acorn woodpecker | `Melanerpes formicivorus` | Oak and mixed woodland | iNaturalist, optional CUB audit |
| B07 | Bird | Anna's hummingbird | `Calypte anna` | Flowering shrubs and edges | iNaturalist, optional CUB audit |
| B08 | Bird | American robin | `Turdus migratorius` | Forest edges and lawns | iNaturalist, optional CUB audit |
| B09 | Bird | Dark-eyed junco | `Junco hyemalis` | Redwood and mixed forest | iNaturalist, optional CUB audit |
| B10 | Bird | Spotted towhee | `Pipilo maculatus` | Chaparral and understory | iNaturalist, optional CUB audit |

See `configs/species.yaml`, `configs/train.yaml`, `configs/rating.yaml`, and `configs/data_sources.yaml` for machine-readable parameters.
