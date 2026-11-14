"""Cached BioCLIP 2 L2-normalized float16 embedding store and index verifier (§7.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from wildlife.data.manifest import read_manifest_parquet, write_manifest_parquet


class EmbeddingCacheMismatchError(RuntimeError):
    """Raised when cached embedding matrix rows do not match the crop index rows or splits."""


def compute_l2_normalized_float16_embeddings(raw_features: np.ndarray, target_dim: int = 768) -> np.ndarray:
    """Project or pad features to `target_dim` (768 for BioCLIP 2 ViT-L/14) and L2-normalize to float16."""
    arr = np.asarray(raw_features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    n, d = arr.shape
    if d < target_dim:
        reps = (target_dim + d - 1) // d
        tiled = np.tile(arr, (1, reps))[:, :target_dim]
        arr = tiled
    elif d > target_dim:
        arr = arr[:, :target_dim]

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = (arr / norms).astype(np.float16)
    return normed


def save_embedding_cache(
    embeddings: np.ndarray,
    index_rows: List[Dict[str, Any]],
    npy_path: Path,
    index_parquet_path: Path,
) -> Tuple[Path, Path]:
    """Save float16 `.npy` embeddings and matching index `.parquet` (§7.4.1)."""
    if len(embeddings) != len(index_rows):
        raise EmbeddingCacheMismatchError(
            f"Embedding row count {len(embeddings)} != index row count {len(index_rows)}"
        )
    npy_path = Path(npy_path)
    index_parquet_path = Path(index_parquet_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    index_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(npy_path, np.asarray(embeddings, dtype=np.float16))
    write_manifest_parquet(index_rows, index_parquet_path)
    return npy_path, index_parquet_path


def load_and_verify_embedding_cache(
    npy_path: Path,
    index_parquet_path: Path,
    expected_crop_rows: List[Dict[str, Any]] | None = None,
    expected_split: str | None = None,
    views_per_crop: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load cached `.npy` embeddings and verify 1-to-1 row & split alignment against crop index rows."""
    emb = np.load(Path(npy_path))
    idx_rows = read_manifest_parquet(Path(index_parquet_path))

    if emb.shape[0] != len(idx_rows):
        raise EmbeddingCacheMismatchError(
            f"Cached embedding shape {emb.shape} does not match index rows ({len(idx_rows)})."
        )

    if expected_split is not None:
        mismatched_splits = {str(r["split"]) for r in idx_rows if str(r["split"]) != expected_split}
        if mismatched_splits:
            raise EmbeddingCacheMismatchError(
                f"Expected split '{expected_split}' in embedding cache, found {mismatched_splits}"
            )

    if expected_crop_rows is not None:
        filtered_crops = [
            r for r in expected_crop_rows
            if (expected_split is None or str(r["split"]) == expected_split)
        ]
        if len(idx_rows) != len(filtered_crops) * views_per_crop:
            raise EmbeddingCacheMismatchError(
                f"Cache index rows ({len(idx_rows)}) != expected crop rows ({len(filtered_crops)}) * views ({views_per_crop})."
            )
        crop_obs_ids = {str(r["observation_id"]) for r in filtered_crops}
        cache_obs_ids = {str(r["observation_id"]) for r in idx_rows}
        if crop_obs_ids != cache_obs_ids:
            raise EmbeddingCacheMismatchError(
                "Observation IDs in embedding cache do not match crop index!"
            )

    return emb, idx_rows
