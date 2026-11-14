"""iNaturalist API collector and AWS S3 Open Data TSV DuckDB query builder (v2 §3.1 & §4)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.parse
import urllib.request
import yaml

from wildlife.data.licenses import ALLOWED_PHOTO_LICENSES, normalize_license_code
from wildlife.schemas import SpeciesRecord, load_species_taxonomy

SCHEME = "https"
INAT_HOST = "api.inaturalist.org"
INAT_WEB_HOST = "www.inaturalist.org"
INAT_S3_HOST = "inaturalist-open-data.s3.amazonaws.com"
BASE = f"{SCHEME}://{INAT_HOST}/v1/observations"


def get_pilot_observations(
    scientific_name: str,
    pages: int = 1,
    per_page: int = 200,
    place_id: Optional[int] = None,
    allowed_licenses: Set[str] = ALLOWED_PHOTO_LICENSES,
    user_agent: str = "UCSC-Wildlife-Vision/2.1 (academic project)",
    rate_limit_sec: float = 1.05,
    max_retries: int = 3,
    checkpoint_path: Optional[Path] = None,
    http_get_json: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Discover research-grade, license-approved observations from the iNaturalist API."""
    rows: List[Dict[str, Any]] = []
    seen_photo_ids: Set[str] = set()

    if checkpoint_path and checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows.extend(loaded)
        seen_photo_ids.update(str(r["photo_id"]) for r in loaded)

    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    def _default_fetch(url: str, req_headers: Dict[str, str]) -> Dict[str, Any]:
        for attempt in range(max_retries):
            req = urllib.request.Request(url, headers=req_headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt + 1 < max_retries:
                    time.sleep(rate_limit_sec * (2 ** (attempt + 1)))
                    continue
                raise
        raise RuntimeError(f"Exhausted retries fetching {url}")

    fetch_fn = http_get_json or _default_fetch

    for page in range(1, pages + 1):
        query_dict: Dict[str, Any] = {
            "taxon_name": scientific_name,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": "cc0,cc-by",
            "per_page": per_page,
            "page": page,
        }
        if place_id is not None:
            query_dict["place_id"] = int(place_id)

        params = urllib.parse.urlencode(query_dict)
        url = f"{BASE}?{params}"
        payload = fetch_fn(url, headers)

        for obs in payload.get("results", []):
            taxon = obs.get("taxon") or {}
            obs_taxon_name = str(taxon.get("name") or "")
            if obs_taxon_name != scientific_name and not obs_taxon_name.startswith(
                scientific_name + " "
            ):
                continue

            user = obs.get("user") or {}
            observer_id = str(user.get("id") or obs.get("user_id") or "unknown")
            observer_login = str(user.get("login") or f"observer_{observer_id}")
            photographer = str(user.get("name") or observer_login)
            observed_on = str(obs.get("observed_on") or "2026-01-01")
            pos_acc = obs.get("positional_accuracy")
            obs_place_ids = obs.get("place_ids") or []

            for photo_idx, photo in enumerate(obs.get("photos", [])):
                lic = normalize_license_code(photo.get("license_code"))
                if lic not in allowed_licenses:
                    continue
                pid = str(photo["id"])
                if pid in seen_photo_ids:
                    continue
                seen_photo_ids.add(pid)
                orig_dims = photo.get("original_dimensions") or {}
                w = int(orig_dims.get("width") or 1024)
                h = int(orig_dims.get("height") or 768)
                raw_url = str(photo.get("url") or "")
                ext = "jpg"
                if ".png" in raw_url.lower():
                    ext = "png"
                large_url = (
                    raw_url.replace("square.", "large.")
                    if "square." in raw_url
                    else f"{SCHEME}://{INAT_S3_HOST}/photos/{pid}/large.{ext}"
                )
                rows.append(
                    {
                        "observation_id": str(obs["id"]),
                        "observer_id": observer_id,
                        "observer_login": observer_login,
                        "photo_id": pid,
                        "photo_rank_in_obs": photo_idx,
                        "scientific_name": scientific_name,
                        "taxon_id": int(taxon.get("id") or 0),
                        "photo_license": lic,
                        "photographer_credit": photographer,
                        "date_observed": observed_on,
                        "positional_accuracy": pos_acc,
                        "extension": ext,
                        "width": w,
                        "height": h,
                        "is_campus_place": bool(
                            3519 in obs_place_ids or 94596 in obs_place_ids
                        ),
                        "source_dataset": "inat_open_data",
                        "label_source": "inat_research_grade",
                        "source_url": f"{SCHEME}://{INAT_WEB_HOST}/photos/{pid}",
                        "image_url": large_url,
                    }
                )

        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        if page < pages and rate_limit_sec > 0 and http_get_json is None:
            time.sleep(rate_limit_sec)

    return rows


def build_s3_open_data_duckdb_query(
    species_records: Optional[List[SpeciesRecord]] = None,
    extracted_metadata_dir: str = "/content/meta",
    california_bbox: Tuple[float, float, float, float] = (32.5, -124.5, 42.0, -114.1),
    ood_taxa_path: Optional[Path] = None,
) -> str:
    """Generate the fixed v2 DuckDB SQL query over extracted iNaturalist Open Data TSVs (§3.1 & §4.3).

    Fixes v1 bug:
    - Points to extracted tarball TSV files (`observations.csv.gz`, `photos.csv.gz`, `taxa.csv.gz`, `observers.csv.gz`).
    - Explicitly uses `read_csv(..., delim='\\t', header=true)`.
    - Selects `p.extension`, `p.width`, `p.height`, `o.positional_accuracy`, and `obs.login`/`obs.name` for attribution.
    - Generates `large.{extension}` S3 URLs.
    """
    records = species_records or load_species_taxonomy()
    target_taxon_ids = ", ".join(str(r.taxon_id) for r in records)
    lat_min, lon_min, lat_max, lon_max = california_bbox

    ood_ids_list: List[str] = []
    if ood_taxa_path and Path(ood_taxa_path).exists():
        raw_ood = yaml.safe_load(Path(ood_taxa_path).read_text(encoding="utf-8"))
        ood_ids_list = [str(item["taxon_id"]) for item in raw_ood.get("ood_taxa", [])]
    ood_taxon_ids = ", ".join(ood_ids_list) if ood_ids_list else "47144, 118552, 8021"

    return f"""-- UCSC Wildlife Vision v2 — Reproducible iNaturalist Open Data TSV Scan (CC0 / CC-BY only)
-- Prerequisite: extract s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz into {extracted_metadata_dir}/

-- Query 1: 20 Target Species Candidates (California default bounding box)
COPY (
    SELECT
        o.observation_uuid AS observation_id,
        o.observer_id,
        obs.login AS observer_login,
        COALESCE(NULLIF(obs.name, ''), obs.login) AS photographer_credit,
        p.photo_id,
        p.photo_uuid,
        p.extension,
        p.width,
        p.height,
        LOWER(p.license) AS photo_license,
        t.name AS scientific_name,
        t.taxon_id,
        o.observed_on AS date_observed,
        o.positional_accuracy,
        o.latitude,
        o.longitude,
        FALSE AS is_ood,
        '' AS ood_group,
        CONCAT('{SCHEME}://{INAT_S3_HOST}/photos/', p.photo_id, '/large.', p.extension) AS large_image_url
    FROM read_csv('{extracted_metadata_dir}/observations.csv.gz', delim='\\t', header=true) AS o
    JOIN read_csv('{extracted_metadata_dir}/photos.csv.gz', delim='\\t', header=true) AS p
      ON o.observation_uuid = p.observation_uuid
    JOIN read_csv('{extracted_metadata_dir}/taxa.csv.gz', delim='\\t', header=true) AS t
      ON o.taxon_id = t.taxon_id
    LEFT JOIN read_csv('{extracted_metadata_dir}/observers.csv.gz', delim='\\t', header=true) AS obs
      ON o.observer_id = obs.observer_id
    WHERE o.quality_grade = 'research'
      AND LOWER(p.license) IN ('cc0', 'cc-by')
      AND t.taxon_id IN ({target_taxon_ids})
      AND p.width >= 256
      AND p.height >= 256
      AND o.latitude BETWEEN {lat_min} AND {lat_max}
      AND o.longitude BETWEEN {lon_min} AND {lon_max}
) TO 'data/manifests/inat_candidates.parquet' (FORMAT PARQUET);

-- Query 2: Resolved Hard-Negative OOD Taxa Candidates (configs/ood_taxa.yaml)
COPY (
    SELECT
        o.observation_uuid AS observation_id,
        o.observer_id,
        obs.login AS observer_login,
        COALESCE(NULLIF(obs.name, ''), obs.login) AS photographer_credit,
        p.photo_id,
        p.photo_uuid,
        p.extension,
        p.width,
        p.height,
        LOWER(p.license) AS photo_license,
        t.name AS scientific_name,
        t.taxon_id,
        o.observed_on AS date_observed,
        o.positional_accuracy,
        o.latitude,
        o.longitude,
        TRUE AS is_ood,
        CONCAT('{SCHEME}://{INAT_S3_HOST}/photos/', p.photo_id, '/large.', p.extension) AS large_image_url
    FROM read_csv('{extracted_metadata_dir}/observations.csv.gz', delim='\\t', header=true) AS o
    JOIN read_csv('{extracted_metadata_dir}/photos.csv.gz', delim='\\t', header=true) AS p
      ON o.observation_uuid = p.observation_uuid
    JOIN read_csv('{extracted_metadata_dir}/taxa.csv.gz', delim='\\t', header=true) AS t
      ON o.taxon_id = t.taxon_id
    LEFT JOIN read_csv('{extracted_metadata_dir}/observers.csv.gz', delim='\\t', header=true) AS obs
      ON o.observer_id = obs.observer_id
    WHERE o.quality_grade = 'research'
      AND LOWER(p.license) IN ('cc0', 'cc-by')
      AND t.taxon_id IN ({ood_taxon_ids})
      AND p.width >= 256
      AND p.height >= 256
      AND o.latitude BETWEEN {lat_min} AND {lat_max}
      AND o.longitude BETWEEN {lon_min} AND {lon_max}
) TO 'data/manifests/inat_ood_candidates.parquet' (FORMAT PARQUET);
"""
