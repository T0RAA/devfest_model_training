"""Observation-grouped, observer-capped stratified splitting and leakage checks (v1 & v2 §4.4, §4.8)."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Set, Tuple

from wildlife.data.clean import phash_hamming_distance


class SplitLeakageError(RuntimeError):
    """Raised when observation IDs, exact SHA256 hashes, or pHashes leak across splits."""


SPLIT_CSV_COLUMNS = [
    "image_id",
    "observation_id",
    "observer_id",
    "class_id",
    "split",
    "image_sha256",
    "phash",
]


def apply_v2_sampling_and_splits(
    rows: List[Dict[str, Any]],
    max_obs_per_species: int = 800,
    max_photos_train: int = 3,
    max_photos_eval: int = 1,
    max_observer_share: float = 0.05,
    min_short_side_px: int = 256,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply Milestone 6 (§4.4, §4.6, §4.8) sampling, deduplication, and split rules.

    Returns `(kept_rows_with_splits, drop_log_rows)`.
    """
    rng = random.Random(seed)
    drop_log: List[Dict[str, Any]] = []
    seen_sha: Set[str] = set()
    phash_by_class: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # Step 1: Integrity & deduplication (§4.6)
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        img_id = str(r.get("image_id", ""))
        w = int(r.get("width", 0))
        h = int(r.get("height", 0))
        sha = str(r.get("image_sha256", ""))
        ph = str(r.get("phash", ""))
        cid = str(r.get("class_id", ""))

        if min(w, h) < min_short_side_px:
            drop_log.append(
                {"image_id": img_id, "reason": f"short_side_{min(w, h)}_below_{min_short_side_px}"}
            )
            continue

        if sha and sha in seen_sha:
            drop_log.append({"image_id": img_id, "reason": "exact_sha256_duplicate"})
            continue
        if sha:
            seen_sha.add(sha)

        if ph:
            is_near_dup = False
            for prev_id, prev_ph in phash_by_class[cid]:
                if phash_hamming_distance(ph, prev_ph) <= 4:
                    drop_log.append(
                        {"image_id": img_id, "reason": f"near_duplicate_phash_of_{prev_id}"}
                    )
                    is_near_dup = True
                    break
            if is_near_dup:
                continue
            phash_by_class[cid].append((img_id, ph))

        filtered.append(dict(r))

    # Step 2: Separate special splits (campus_test per Decision D5, field_holdout, ood_tune, ood_test)
    special_rows: List[Dict[str, Any]] = []
    by_class_obs: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in filtered:
        if r.get("is_ood"):
            ood_grp = str(r.get("ood_group") or "ood_tune")
            r["split"] = ood_grp
            special_rows.append(r)
        elif r.get("region_tag") == "ucsc_campus_place" or r.get("split") == "campus_test":
            r["split"] = "campus_test"
            special_rows.append(r)
        elif r.get("split") == "field_holdout":
            special_rows.append(r)
        else:
            by_class_obs[str(r["class_id"])][str(r["observation_id"])].append(r)

    final_rows: List[Dict[str, Any]] = list(special_rows)

    # Step 3: Per-species observation sampling, 5% observer cap, and 70/15/15 split
    for class_id in sorted(by_class_obs.keys()):
        obs_map = by_class_obs[class_id]
        obs_ids = sorted(obs_map.keys())
        rng.shuffle(obs_ids)

        # Observer cap: max 5% of species observations per observer (minimum cap of 2 for small samples)
        target_total = min(len(obs_ids), max_obs_per_species)
        max_per_observer = max(2, int(round(target_total * max_observer_share)))
        observer_counts: Dict[str, int] = defaultdict(int)
        kept_obs_ids: List[str] = []

        for oid in obs_ids:
            if len(kept_obs_ids) >= max_obs_per_species:
                for dropped_r in obs_map[oid]:
                    drop_log.append(
                        {"image_id": dropped_r["image_id"], "reason": "exceeded_800_obs_cap"}
                    )
                continue
            observer_id = str(obs_map[oid][0].get("observer_id", "unknown"))
            if observer_counts[observer_id] >= max_per_observer and len(obs_ids) > 40:
                for dropped_r in obs_map[oid]:
                    drop_log.append(
                        {
                            "image_id": dropped_r["image_id"],
                            "reason": f"observer_cap_exceeded_{observer_id}",
                        }
                    )
                continue
            observer_counts[observer_id] += 1
            kept_obs_ids.append(oid)

        n_obs = len(kept_obs_ids)
        if n_obs >= 3:
            n_test = max(1, int(round(n_obs * test_ratio)))
            n_val = max(1, int(round(n_obs * val_ratio)))
            if n_test + n_val >= n_obs:
                n_test = 1
                n_val = 1
        elif n_obs == 2:
            n_test = 0
            n_val = 1
        else:
            n_test = 0
            n_val = 0

        for idx, oid in enumerate(kept_obs_ids):
            if idx < n_test:
                split_name = "test"
                photo_cap = max_photos_eval
            elif idx < n_test + n_val:
                split_name = "validation"
                photo_cap = max_photos_eval
            else:
                split_name = "train"
                photo_cap = max_photos_train

            photos = sorted(obs_map[oid], key=lambda x: str(x["image_id"]))
            for p_idx, item in enumerate(photos):
                if p_idx >= photo_cap:
                    drop_log.append(
                        {
                            "image_id": item["image_id"],
                            "reason": f"exceeded_{split_name}_photos_per_obs_cap_{photo_cap}",
                        }
                    )
                    continue
                item["split"] = split_name
                final_rows.append(item)

    final_rows.sort(key=lambda x: str(x["image_id"]))
    return final_rows, drop_log


def create_observation_grouped_splits(
    rows: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Assign `train`, `validation`, and `test` strictly grouped by `observation_id`."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    by_class_obs: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    preserved_rows: List[Dict[str, Any]] = []

    for r in rows:
        copy_r = dict(r)
        if copy_r.get("split") in {"field_holdout", "ood", "campus_test", "ood_tune", "ood_test"}:
            preserved_rows.append(copy_r)
        else:
            by_class_obs[str(copy_r["class_id"])][str(copy_r["observation_id"])].append(
                copy_r
            )

    assigned_rows: List[Dict[str, Any]] = list(preserved_rows)

    for class_id in sorted(by_class_obs.keys()):
        obs_dict = by_class_obs[class_id]
        obs_ids = sorted(obs_dict.keys())
        rng.shuffle(obs_ids)
        n_obs = len(obs_ids)

        if n_obs >= 3:
            n_test = max(1, int(round(n_obs * test_ratio)))
            n_val = max(1, int(round(n_obs * val_ratio)))
            if n_test + n_val >= n_obs:
                n_test = 1
                n_val = 1
        elif n_obs == 2:
            n_test = 0
            n_val = 1
        else:
            n_test = 0
            n_val = 0

        for idx, oid in enumerate(obs_ids):
            if idx < n_test:
                target_split = "test"
            elif idx < n_test + n_val:
                target_split = "validation"
            else:
                target_split = "train"
            for item in obs_dict[oid]:
                item["split"] = target_split
                assigned_rows.append(item)

    assigned_rows.sort(key=lambda x: str(x["image_id"]))
    return assigned_rows


def verify_no_split_leakage(
    rows: Iterable[Dict[str, Any]],
    max_phash_distance: int = 4,
) -> Dict[str, Any]:
    """Assert zero observation_id, SHA256, or near-duplicate pHash leakage across splits."""
    obs_to_splits: Dict[str, Set[str]] = defaultdict(set)
    sha_to_splits: Dict[str, Set[str]] = defaultdict(set)
    ood_taxon_to_splits: Dict[int, Set[str]] = defaultdict(set)
    phash_records: List[Tuple[str, str, str]] = []

    row_list = list(rows)
    for r in row_list:
        img_id = str(r["image_id"])
        obs_id = str(r["observation_id"])
        split = str(r["split"])
        sha = str(r.get("image_sha256", ""))
        ph = str(r.get("phash", ""))

        obs_to_splits[obs_id].add(split)
        if sha:
            sha_to_splits[sha].add(split)
        if ph:
            phash_records.append((img_id, split, ph))
        if r.get("is_ood") and r.get("taxon_id"):
            ood_taxon_to_splits[int(r["taxon_id"])].add(split)

    leaked_obs = {oid: sorted(s) for oid, s in obs_to_splits.items() if len(s) > 1}
    if leaked_obs:
        raise SplitLeakageError(
            f"Observation ID leakage detected across splits: {leaked_obs}"
        )

    leaked_sha = {sha: sorted(s) for sha, s in sha_to_splits.items() if len(s) > 1}
    if leaked_sha:
        raise SplitLeakageError(
            f"Identical image SHA256 detected across splits: {leaked_sha}"
        )

    leaked_ood_taxa = {
        tid: sorted(s) for tid, s in ood_taxon_to_splits.items() if len(s) > 1
    }
    if leaked_ood_taxa:
        raise SplitLeakageError(
            f"OOD taxon ID appeared in both ood_tune and ood_test: {leaked_ood_taxa}"
        )

    for i in range(len(phash_records)):
        id_a, split_a, ph_a = phash_records[i]
        for j in range(i + 1, len(phash_records)):
            id_b, split_b, ph_b = phash_records[j]
            if split_a == split_b:
                continue
            dist = phash_hamming_distance(ph_a, ph_b)
            if dist <= max_phash_distance:
                raise SplitLeakageError(
                    f"Perceptual near-duplicate leakage across splits ({split_a} vs {split_b}): "
                    f"{id_a} and {id_b} have pHash Hamming distance {dist} <= {max_phash_distance}."
                )

    split_counts: Dict[str, int] = defaultdict(int)
    for r in row_list:
        split_counts[str(r["split"])] += 1

    return {
        "total_images": len(row_list),
        "unique_observations": len(obs_to_splits),
        "split_counts": dict(split_counts),
        "leakage_free": True,
    }


def export_split_csv(rows: Iterable[Dict[str, Any]], output_path: Path) -> Path:
    row_list = list(rows)
    verify_no_split_leakage(row_list)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPLIT_CSV_COLUMNS)
        writer.writeheader()
        for r in row_list:
            writer.writerow({k: r.get(k, "") for k in SPLIT_CSV_COLUMNS})
    return output_path


def load_split_csv(input_path: Path) -> List[Dict[str, str]]:
    with Path(input_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    verify_no_split_leakage(rows)
    return rows
