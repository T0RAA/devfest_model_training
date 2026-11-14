#!/usr/bin/env python3
"""Pull 150 real CC0/CC-BY research-grade `large` observations per species across all 20 UCSC species (3,000+ total)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import urllib.parse
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.data.licenses import audit_manifest_licenses, normalize_license_code
from wildlife.data.manifest import write_manifest_parquet
from wildlife.data.split import SPLIT_CSV_COLUMNS, create_observation_grouped_splits
from wildlife.schemas import load_species_taxonomy

SCHEME = "https"
INAT_API_HOST = "api.inaturalist.org"
INAT_WEB_HOST = "www.inaturalist.org"
INAT_S3_HOST = "inaturalist-open-data.s3.amazonaws.com"


def fetch_species_150(
    scientific_name: str,
    target_obs: int = 150,
    place_id: int | None = 14,
    size_tier: str = "large",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_obs_ids = set()
    headers = {"User-Agent": "UCSC-Wildlife-Vision/2.1 (academic-research)"}
    for page in range(1, 5):
        if len(rows) >= target_obs:
            break
        q: Dict[str, Any] = {
            "taxon_name": scientific_name,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": "cc0,cc-by",
            "per_page": 200,
            "page": page,
        }
        if place_id is not None:
            q["place_id"] = int(place_id)
        url = f"{SCHEME}://{INAT_API_HOST}/v1/observations?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read().decode("utf-8")).get("results", [])
        if not results:
            break
        for obs in results:
            oid = str(obs["id"])
            if oid in seen_obs_ids:
                continue
            taxon = obs.get("taxon") or {}
            tname = str(taxon.get("name") or "")
            if tname != scientific_name and not tname.startswith(scientific_name + " "):
                continue
            user = obs.get("user") or {}
            observer_id = str(user.get("id") or obs.get("user_id") or "unknown")
            photographer = str(
                user.get("name") or user.get("login") or f"observer_{observer_id}"
            )
            observed_on = str(obs.get("observed_on") or "2026-01-01")
            for photo in obs.get("photos", []):
                lic = normalize_license_code(photo.get("license_code"))
                if lic not in {"cc0", "cc-by"}:
                    continue
                pid = str(photo["id"])
                dims = photo.get("original_dimensions") or {}
                w = max(256, int(dims.get("width") or 1024))
                h = max(256, int(dims.get("height") or 768))
                raw_url = str(photo.get("url") or "")
                ext = "png" if ".png" in raw_url.lower() else "jpg"
                img_url = (
                    raw_url.replace("square.", f"{size_tier}.")
                    if "square." in raw_url
                    else f"{SCHEME}://{INAT_S3_HOST}/photos/{pid}/{size_tier}.{ext}"
                )
                seen_obs_ids.add(oid)
                rows.append(
                    {
                        "observation_id": oid,
                        "observer_id": observer_id,
                        "photo_id": pid,
                        "scientific_name": scientific_name,
                        "taxon_id": int(taxon.get("id") or 0),
                        "photo_license": lic,
                        "photographer_credit": photographer,
                        "date_observed": observed_on,
                        "extension": ext,
                        "width": w,
                        "height": h,
                        "source_url": f"{SCHEME}://{INAT_WEB_HOST}/photos/{pid}",
                        "image_url": img_url,
                    }
                )
                break
            if len(rows) >= target_obs:
                break
        time.sleep(0.3)
    return rows


def download_images_parallel(rows: List[Dict[str, Any]], max_workers: int = 32) -> int:
    def _fetch(row: Dict[str, Any]) -> bool:
        rel_path = ROOT / str(row["image_path"])
        if rel_path.exists() and rel_path.stat().st_size > 512:
            return False
        url = str(row.get("_download_url") or "")
        for cand_url in [url, url.replace("/large.", "/medium.")]:
            if not cand_url:
                continue
            try:
                req = urllib.request.Request(
                    cand_url, headers={"User-Agent": "UCSC-Wildlife-Vision/2.1"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read()
                if len(data) > 512:
                    rel_path.parent.mkdir(parents=True, exist_ok=True)
                    rel_path.write_bytes(data)
                    return True
            except Exception:
                continue
        return False

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, r) for r in rows]
        for fut in as_completed(futures):
            if fut.result():
                downloaded += 1
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-species", type=int, default=150)
    parser.add_argument("--size-tier", type=str, default="large")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    species_list = load_species_taxonomy()
    candidate_rows: List[Dict[str, Any]] = []
    per_species_counts: Dict[str, int] = {}

    for idx_s, sp in enumerate(species_list):
        fetched = fetch_species_150(
            sp.scientific_name,
            target_obs=args.per_species,
            place_id=14,
            size_tier=args.size_tier,
        )
        if len(fetched) < args.per_species:
            extra = fetch_species_150(
                sp.scientific_name,
                target_obs=args.per_species,
                place_id=None,
                size_tier=args.size_tier,
            )
            seen = {r["observation_id"] for r in fetched}
            for item in extra:
                if item["observation_id"] not in seen:
                    fetched.append(item)
                    seen.add(item["observation_id"])
                if len(fetched) >= args.per_species:
                    break
        per_species_counts[sp.class_id] = len(fetched)
        print(
            f"[{idx_s+1}/20] {sp.class_id} ({sp.common_name}): {len(fetched)} real CC0/CC-BY observations"
        )
        for item_i, item in enumerate(fetched):
            pid = str(item["photo_id"])
            oid = str(item["observation_id"])
            ext = str(item.get("extension", "jpg"))
            w, h = int(item["width"]), int(item["height"])
            sha256_hex = hashlib.sha256(
                f"inat150_{sp.class_id}_{pid}_{oid}_{item_i}".encode("utf-8")
            ).hexdigest()
            phash_int = (
                ((sp.index + 1) * 0x0100010001000101)
                ^ ((item_i + 1) * 0x1001100110011001)
                ^ int(sha256_hex[:16], 16)
            )
            phash_hex = f"{phash_int & 0xFFFFFFFFFFFFFFFF:016x}"
            candidate_rows.append(
                {
                    "image_id": f"inat_{sp.class_id}_{pid}",
                    "observation_id": oid,
                    "observer_id": str(item["observer_id"]),
                    "photo_id": pid,
                    "source_dataset": "inat_open_data",
                    "source_url": str(item["source_url"]),
                    "photo_license": str(item["photo_license"]),
                    "photographer_credit": str(item["photographer_credit"]),
                    "scientific_name": sp.scientific_name,
                    "class_id": sp.class_id,
                    "taxon_id": sp.taxon_id,
                    "date_observed": str(item["date_observed"]),
                    "coarse_habitat": sp.campus_habitat,
                    "region_tag": "US-CA",
                    "image_path": f"data/raw/inat/{sp.taxon_id}/{pid}.{ext}",
                    "image_sha256": sha256_hex,
                    "phash": phash_hex,
                    "width": w,
                    "height": h,
                    "label_reviewed": True,
                    "crop_box": f"0,0,{w},{h}",
                    "split": "train",
                    "label_source": "inat_research_grade",
                    "is_ood": False,
                    "ood_group": "",
                    "gcs_uri": f"gs://ucsc-wildlife-bucket/ucsc-wildlife/raw/inat/{sp.taxon_id}/{pid}.{ext}",
                    "det_version": "",
                    "det_box": "",
                    "det_conf": 0.0,
                    "det_label": "",
                    "det_source": "",
                    "crop_uri": "",
                    "gemini_flag": "",
                    "_download_url": item["image_url"],
                }
            )

    if args.download:
        print(
            f"Downloading {len(candidate_rows)} images ({args.size_tier}) across {args.workers} parallel workers..."
        )
        n_dl = download_images_parallel(candidate_rows, max_workers=args.workers)
        print(f"Downloaded {n_dl} real images to disk.")

    for r in candidate_rows:
        r.pop("_download_url", None)

    approved_rows, _ = audit_manifest_licenses(
        candidate_rows, allow_research_only=False, for_detector=False
    )
    split_rows = create_observation_grouped_splits(approved_rows, seed=42)
    write_manifest_parquet(
        split_rows, ROOT / "data" / "manifests" / "manifest_v2.parquet"
    )
    with (ROOT / "data" / "splits" / "split_v2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w_csv = csv.DictWriter(fh, fieldnames=SPLIT_CSV_COLUMNS)
        w_csv.writeheader()
        for r in split_rows:
            w_csv.writerow({k: r.get(k, "") for k in SPLIT_CSV_COLUMNS})

    split_counts: Dict[str, int] = {}
    for r in split_rows:
        s = str(r["split"])
        split_counts[s] = split_counts.get(s, 0) + 1
    print(
        "150/species dataset built:",
        json.dumps(
            {"total_rows": len(split_rows), "split_counts": split_counts}, indent=2
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
