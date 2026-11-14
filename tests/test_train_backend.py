"""Unit tests for Milestone 9 `--require-torch` enforcement, tier gates, and T0 prompts."""

from pathlib import Path
import pytest

from wildlife.models.backbones import (
    TorchBackendRequiredError,
    build_bioclip2_zero_shot_prompts,
    evaluate_tier_gates,
    verify_torch_available,
)
from wildlife.train import run_training


def test_require_torch_fails_loudly_when_torch_missing(tmp_path: Path):
    # Ground Rule #3: Any training run with require_torch=True must raise TorchBackendRequiredError
    # when PyTorch is not installed, never silently falling back to the NumPy surrogate.
    try:
        import torch  # type: ignore

        has_torch = True
    except Exception:
        has_torch = False

    if not has_torch:
        with pytest.raises(TorchBackendRequiredError, match="Real PyTorch backbone is required"):
            verify_torch_available(require_torch=True)

        root = Path(__file__).resolve().parents[1]
        with pytest.raises(TorchBackendRequiredError):
            run_training(
                config_path=root / "configs" / "train_v2.yaml",
                manifest_path=root / "data" / "manifests" / "manifest_v2.parquet",
                splits_path=root / "data" / "splits" / "split_v2.csv",
                output_dir=tmp_path / "run_fail",
                require_torch=True,
            )


def test_bioclip2_zero_shot_prompts_and_tier_gates():
    prompts = build_bioclip2_zero_shot_prompts()
    assert len(prompts) == 20
    assert len(prompts["B03"]) == 3
    assert "Cyanocitta stelleri" in prompts["B03"][0]

    passing_metrics = {
        "macro_f1": 0.92,
        "per_class": {i: {"recall": 0.85} for i in range(20)},
    }
    res_pass = evaluate_tier_gates(passing_metrics)
    assert res_pass["all_gates_pass"] is True

    failing_pair_metrics = {
        "macro_f1": 0.91,
        "per_class": {i: {"recall": 0.75 if i == 7 else 0.88} for i in range(20)},  # idx 7 = M08
    }
    res_fail = evaluate_tier_gates(failing_pair_metrics)
    assert res_fail["G_macro_pass"] is True
    assert res_fail["G_pairs_pass"] is False
    assert res_fail["all_gates_pass"] is False
