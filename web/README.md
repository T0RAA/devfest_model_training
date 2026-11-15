# Campus Wildlife web app

React 19 + TypeScript + Vite + Tailwind v4 front end for the UCSC Wildlife Vision API.
It follows the team's Figma wireframes (Home feed, View post, Species, Rankings) and
adds the Identify flow the wireframes were missing: photo → detector boxes →
species + top‑3 + photo quality → confirm or correct → post.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Requests to `/v1/*`, `/v2/*` and `/healthz` are proxied to the FastAPI service on
`http://localhost:8000` (`api/main.py`). Start it to get real predictions:

```bash
PYTHONPATH=src uvicorn api.main:app --port 8000
```

When the API isn't reachable, Identify falls back to an in-browser **demo predictor**
that picks a random species. Every demo result is labelled as such in the UI.

## Data and honesty rules

- Seed posts are real CC0 / CC‑BY research‑grade iNaturalist observations near campus
  (`scripts/fetch_seed_photos.py` → `src/data/seed.json`), credited to each observer.
  Their species comes from iNaturalist; their boxes, confidence and rating are sample
  values and the UI says so.
- Locations are rounded to ~1 km and drawn as a circle, never a pin.
- Uploaded photos are re‑encoded through a canvas, which drops EXIF/GPS.
- Corrections go to `POST /v1/feedback` (review queue only, `consent: true`), never
  straight into training data.

## Structure

```
src/types.ts          v2 response types, mirroring schemas.py to_v2_dict()
src/api.ts            /v2/predict and /v1/feedback, demo fallback
src/mock.ts           demo predictor and sample detections
src/store.tsx         posts, likes, comments (localStorage for your own posts)
src/components.tsx    nav, tab bar, DetectionPhoto (corner brackets), result widgets
src/pages/            Feed, PostView, Species, Rankings, Identify
```
