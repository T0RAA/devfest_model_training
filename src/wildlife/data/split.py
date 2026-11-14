"""Observation-grouped, observer-aware stratified splitting and leakage checks."""

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
        if copy_r.get("split") in {"field_holdout", "ood"}:
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
