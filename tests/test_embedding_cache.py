"""Unit tests for BioCLIP 2 float16 embedding cache and WebDataset shard loader (Milestone 9)."""

from pathlib import Path
import numpy as np
import pytest

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import generate_synthetic_species_rgb
from wildlife.data.embedding_cache import (
    EmbeddingCacheMismatchError,
    compute_l2_normalized_float16_embeddings,
    load_and_verify_embedding_cache,
    save_embedding_cache,
)
from wildlife.data.webdataset_loader import iter_webdataset_shard
from wildlife.detect.crops import write_webdataset_shard


def _make_crop_row(image_id: str, obs_id: str, split: str = "validation") -> dict:
    return {
        "image_id": image_id,
        "observation_id": obs_id,
        "observer_id": "u_1",
        "photo_id": "p_1",
        "source_dataset": "inat_open_data",
        "source_url": "inaturalist://photos/1",
        "photo_license": "cc0",
        "photographer_credit": "Tester",
        "scientific_name": "Cyanocitta stelleri",
        "class_id": "B03",
        "taxon_id": 8229,
        "date_observed": "2026-10-01",
        "coarse_habitat": "Redwood forest",
        "region_tag": "US-CA",
        "image_path": "data/raw/inat/8229/1.jpg",
        "image_sha256": "a" * 64,
        "phash": "0123456789abcdef",
        "width": 512,
        "height": 512,
        "label_reviewed": True,
        "crop_box": "0,0,512,512",
        "split": split,
    }


def test_embedding_cache_roundtrip_and_alignment_check(tmp_path: Path):
    crop_rows = [
        _make_crop_row("img_1", "obs_1", "validation"),
        _make_crop_row("img_2", "obs_2", "validation"),
    ]
    raw_feats = np.random.RandomState(42).randn(2, 132).astype(np.float32)
    emb = compute_l2_normalized_float16_embeddings(raw_feats, target_dim=768)
    assert emb.shape == (2, 768)
    assert emb.dtype == np.float16
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)

    npy_path = tmp_path / "validation.npy"
    idx_path = tmp_path / "validation_index.parquet"
    save_embedding_cache(emb, crop_rows, npy_path, idx_path)

    loaded_emb, loaded_rows = load_and_verify_embedding_cache(
        npy_path,
        idx_path,
        expected_crop_rows=crop_rows,
        expected_split="validation",
        views_per_crop=1,
    )
    assert loaded_emb.shape == (2, 768)
    assert len(loaded_rows) == 2

    # Mismatched expected split must fail
    with pytest.raises(EmbeddingCacheMismatchError, match="Expected split"):
        load_and_verify_embedding_cache(npy_path, idx_path, expected_split="train")


def test_webdataset_shard_roundtrip(tmp_path: Path):
    rgb = generate_synthetic_species_rgb(12, sample_idx=1, size=64)
    png_bytes = encode_rgb_to_png_bytes(rgb)
    shard_path = tmp_path / "shard-000000.tar"
    n = write_webdataset_shard(
        [
            {
                "image_id": "crop_001",
                "image_bytes": png_bytes,
                "class_idx": 12,
                "metadata": {"class_id": "B03", "crop_version": "crop-v1"},
            }
        ],
        shard_path,
    )
    assert n == 1
    records = list(iter_webdataset_shard(shard_path))
    assert len(records) == 1
    dec_rgb, cls_idx, meta = records[0]
    assert dec_rgb.shape == (64, 64, 3)
    assert cls_idx == 12
    assert meta["class_id"] == "B03"
