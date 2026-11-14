"""Unit tests for observation-grouped splits and duplicate leakage detection (Milestone 1)."""

import pytest

from wildlife.data.split import (
    SplitLeakageError,
    create_observation_grouped_splits,
    verify_no_split_leakage,
)


def _sample_rows():
    rows = []
    for obs_idx in range(10):
        # Multiple photos per observation to test group integrity
        for photo_sub in range(2):
            idx = obs_idx * 2 + photo_sub
            rows.append(
                {
                    "image_id": f"img_{idx:03d}",
                    "observation_id": f"obs_{obs_idx:03d}",
                    "observer_id": f"user_{obs_idx % 4}",
                    "class_id": "B03",
                    "split": "train",
                    "image_sha256": f"sha_{idx:04d}",
                    "phash": f"{(obs_idx + 1) * 0x1111111111111111 + photo_sub * 0x0101010101010101:016x}",
                }
            )
    return rows


def test_observation_grouped_splits_keep_observations_together():
    rows = _sample_rows()
    split_rows = create_observation_grouped_splits(rows, seed=42)
    report = verify_no_split_leakage(split_rows, max_phash_distance=2)
    assert report["leakage_free"] is True
    assert report["total_images"] == 20
    assert report["unique_observations"] == 10
    assert set(report["split_counts"].keys()) == {"train", "validation", "test"}


def test_same_observation_across_splits_fails():
    # Failure case #1: Same iNaturalist observation appears in train and test -> pipeline must fail
    rows = [
        {
            "image_id": "img_1",
            "observation_id": "obs_shared_999",
            "observer_id": "u1",
            "class_id": "M01",
            "split": "train",
            "image_sha256": "sha_1",
            "phash": "0000000000000000",
        },
        {
            "image_id": "img_2",
            "observation_id": "obs_shared_999",
            "observer_id": "u1",
            "class_id": "M01",
            "split": "test",
            "image_sha256": "sha_2",
            "phash": "ffffffffffffffff",
        },
    ]
    with pytest.raises(SplitLeakageError, match="Observation ID leakage"):
        verify_no_split_leakage(rows)


def test_perceptually_identical_photos_across_splits_fails():
    # Failure case #2: Two perceptually identical photos across splits -> pipeline must flag/fail
    rows = [
        {
            "image_id": "img_a",
            "observation_id": "obs_1",
            "observer_id": "u1",
            "class_id": "B03",
            "split": "train",
            "image_sha256": "sha_unique_a",
            "phash": "a5a5a5a5a5a5a5a5",
        },
        {
            "image_id": "img_b",
            "observation_id": "obs_2",
            "observer_id": "u2",
            "class_id": "B03",
            "split": "test",
            "image_sha256": "sha_unique_b",
            # Hamming distance = 1 bit difference (<= 4 threshold)
            "phash": "a5a5a5a5a5a5a5a4",
        },
    ]
    with pytest.raises(SplitLeakageError, match="Perceptual near-duplicate leakage"):
        verify_no_split_leakage(rows, max_phash_distance=4)
