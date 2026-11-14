"""iNaturalist pilot API collector and Open Data metadata query builder."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.parse
import urllib.request

from wildlife.data.licenses import ALLOWED_PHOTO_LICENSES, normalize_license_code
from wildlife.schemas import SpeciesRecord, load_species_taxonomy

SCHEME = "https"
INAT_HOST = "api.inaturalist.org"
INAT_WEB_HOST = "www.inaturalist.org"
BASE = f"{SCHEME}://{INAT_HOST}/v1/observations"


def get_pilot_observations(
    scientific_name: str,
    pages: int = 1,
    per_page: int = 200,
    allowed_licenses: Set[str] = ALLOWED_PHOTO_LICENSES,
    user_agent: str = "UCSC-Wildlife-Vision/1.0 (academic project)",
    rate_limit_sec: float = 1.1,
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
        params = urllib.parse.urlencode(
            {
                "taxon_name": scientific_name,
                "quality_grade": "research",
                "photos": "true",
                "per_page": per_page,
                "page": page,
            }
        )
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
            photographer = str(
                user.get("name") or user.get("login") or f"observer_{observer_id}"
            )
            observed_on = str(obs.get("observed_on") or "2026-01-01")

            for photo in obs.get("photos", []):
                lic = normalize_license_code(photo.get("license_code"))
                if lic not in allowed_licenses:
                    continue
                pid = str(photo["id"])
                if pid in seen_photo_ids:
                    continue
                seen_photo_ids.add(pid)
                rows.append(
                    {
                        "observation_id": str(obs["id"]),
                        "observer_id": observer_id,
                        "photo_id": pid,
                        "scientific_name": scientific_name,
                        "taxon_id": int(taxon.get("id") or 0),
                        "photo_license": lic,
                        "photographer_credit": photographer,
                        "date_observed": observed_on,
                        "source_dataset": "inaturalist_pilot_api",
                        "source_url": f"{SCHEME}://{INAT_WEB_HOST}/photos/{pid}",
                        "image_url": photo.get("url"),
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
    metadata_dir: str = "inaturalist-open-data/metadata",
    california_bbox: Tuple[float, float, float, float] = (32.5, -124.5, 42.0, -114.1),
) -> str:
    """Generate a reproducible DuckDB SQL query over iNaturalist Open Data metadata tables."""
    records = species_records or load_species_taxonomy()
    taxon_ids = ", ".join(str(r.taxon_id) for r in records)
    lat_min, lon_min, lat_max, lon_max = california_bbox
    return f"""-- Reproducible iNaturalist Open Data metadata scan (CC0 / CC-BY only)
SELECT
    o.observation_uuid AS observation_id,
    o.observer_id,
    p.photo_id,
    p.photo_uuid,
    LOWER(p.license) AS photo_license,
    t.name AS scientific_name,
    t.taxon_id,
    o.observed_on AS date_observed,
    o.latitude,
    o.longitude
FROM read_csv_auto('{metadata_dir}/observations.csv.gz') AS o
JOIN read_csv_auto('{metadata_dir}/photos.csv.gz') AS p
  ON o.observation_uuid = p.observation_uuid
JOIN read_csv_auto('{metadata_dir}/taxa.csv.gz') AS t
  ON o.taxon_id = t.taxon_id
WHERE o.quality_grade = 'research'
  AND LOWER(p.license) IN ('cc0', 'cc-by')
  AND t.taxon_id IN ({taxon_ids})
  AND o.latitude BETWEEN {lat_min} AND {lat_max}
  AND o.longitude BETWEEN {lon_min} AND {lon_max};
"""
