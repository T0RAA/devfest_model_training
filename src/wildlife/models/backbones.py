"""v2 Classifier backbones (`BioCLIP 2` T0/T1/T2 and `EfficientNetV2-S` T3/distill), gate checks, and version parity."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wildlife.detect import CROP_VERSION_DEFAULT, DETECTOR_VERSION_DEFAULT
from wildlife.models.classifier import (
    CheckpointClassOrderError,
    WildlifeClassifier,
    load_checkpoint,
)
from wildlife.schemas import build_classes_Index_map, load_species_taxonomy

HARD_PAIR_CLASS_IDS = ["M08", "M09", "B03", "B04", "B05"]


class TorchBackendRequiredError(RuntimeError):
    """Raised when `--require-torch` is active and PyTorch / open_clip / timm is unavailable (Ground Rule #3)."""


class VersionParityError(RuntimeError):
    """Raised when checkpoint `detector_version` or `crop_version` mismatches the serving pipeline (§2.3 #4)."""


def verify_torch_available(require_torch: bool = True) -> bool:
    """Verify real PyTorch backbone availability; fail loudly if `require_torch=True` and unavailable."""
    try:
        import torch  # type: ignore

        return True
    except Exception as exc:
        if require_torch:
            raise TorchBackendRequiredError(
                "Real PyTorch backbone is required (`--require-torch` is True), but `torch` is not installed "
                "in this environment. Run on Google Colab GPU / Vertex AI or explicitly pass "
                "`--no-require-torch` / `WILDLIFE_TEST_MODE=1` for local smoke testing."
            ) from exc
        return False


def build_bioclip2_zero_shot_prompts() -> Dict[str, List[str]]:
    """Construct the 3 canonical text prompts per species for BioCLIP 2 T0 zero-shot (§7.3)."""
    species = load_species_taxonomy()
    prompts: Dict[str, List[str]] = {}
    for sp in species:
        prompts[sp.class_id] = [
            f"a photo of {sp.scientific_name}, {sp.common_name}",
            f"a wildlife photo of a {sp.common_name} ({sp.scientific_name})",
            f"an image of the animal {sp.common_name}, {sp.scientific_name}",
        ]
    return prompts


def evaluate_tier_gates(
    metrics: Dict[str, Any],
    macro_f1_min: float = 0.90,
    hard_pair_recall_min: float = 0.80,
    per_class_recall_floor: float = 0.70,
) -> Dict[str, Any]:
    """Evaluate Milestone 9 tier gates (`G-macro`, `G-pairs`, `G-floor`) (§7.2)."""
    species_map = {r.class_id: r.index for r in load_species_taxonomy()}
    per_class = metrics.get("per_class", {})

    macro_f1 = float(metrics.get("macro_f1", 0.0))
    g_macro = macro_f1 >= macro_f1_min

    hard_pair_recalls: Dict[str, float] = {}
    for cid in HARD_PAIR_CLASS_IDS:
        idx = species_map[cid]
        pc = per_class.get(idx) or per_class.get(str(idx)) or {}
        hard_pair_recalls[cid] = float(pc.get("recall", 0.0))
    g_pairs = all(r >= hard_pair_recall_min for r in hard_pair_recalls.values())

    all_recalls: List[float] = []
    for idx in range(20):
        pc = per_class.get(idx) or per_class.get(str(idx)) or {}
        all_recalls.append(float(pc.get("recall", 0.0)))
    min_recall = min(all_recalls) if all_recalls else 0.0
    g_floor = min_recall >= per_class_recall_floor

    return {
        "G_macro_pass": g_macro,
        "G_pairs_pass": g_pairs,
        "G_floor_pass": g_floor,
        "all_gates_pass": bool(g_macro and g_pairs and g_floor),
        "macro_f1": round(macro_f1, 4),
        "hard_pair_recalls": hard_pair_recalls,
        "min_class_recall": round(min_recall, 4),
    }


def save_v2_checkpoint(
    model: WildlifeClassifier,
    checkpoint_path: Path,
    tier: str,
    backbone: str,
    input_size: int,
    detector_version: str,
    crop_version: str,
    train_manifest_sha256: str,
    split_version: str,
    git_commit: str,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    temperature: float = 1.0,
) -> Tuple[Path, str]:
    """Save a v2 classifier checkpoint containing full provenance and detector/crop parity metadata (§7.5)."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    classes_bytes = str(model.class_map).encode("utf-8")
    classes_sha = hashlib.sha256(classes_bytes).hexdigest()
    backbone_sha = hashlib.sha256(backbone.encode("utf-8")).hexdigest()[:12]

    payload: Dict[str, Any] = {
        "model_version": f"species-v2.0-{tier}",
        "tier": tier,
        "backbone": backbone,
        "backbone_sha": backbone_sha,
        "backend": model.backend,
        "input_size": int(input_size),
        "num_classes": model.num_classes,
        "class_ids": list(model.class_ids),
        "class_to_idx": dict(model.class_map["class_to_idx"]),
        "classes_json_sha256": classes_sha,
        "detector_version": detector_version,
        "crop_version": crop_version,
        "train_manifest_sha256": train_manifest_sha256,
        "split_version": split_version,
        "git_commit": git_commit,
        "epoch": int(epoch),
        "seed": int(model.seed),
        "config": config,
        "metrics": metrics,
        "temperature": float(temperature),
        "numpy_state": {
            "W_enc": model.W_enc,
            "b_enc": model.b_enc,
            "W_head": model.W_head,
            "b_head": model.b_head,
        },
    }
    raw_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint_path.write_bytes(raw_bytes)
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    sha_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    sha_path.write_text(f"{sha256_hex}  {checkpoint_path.name}\n", encoding="utf-8")
    return checkpoint_path, sha256_hex


def load_v2_checkpoint(
    checkpoint_path: Path,
    expected_detector_version: str = DETECTOR_VERSION_DEFAULT,
    expected_crop_version: str = CROP_VERSION_DEFAULT,
    enforce_version_parity: bool = True,
) -> Tuple[WildlifeClassifier, Dict[str, Any]]:
    """Load a checkpoint and enforce both 20-species class order and detector/crop version parity (§2.3 #4)."""
    model, payload = load_checkpoint(checkpoint_path)
    if enforce_version_parity:
        ckpt_det = payload.get("detector_version")
        ckpt_crop = payload.get("crop_version")
        if ckpt_det is not None and ckpt_det != expected_detector_version:
            raise VersionParityError(
                f"Checkpoint detector_version '{ckpt_det}' does not match serving detector "
                f"'{expected_detector_version}'! Refusing to load mismatched crop classifier."
            )
        if ckpt_crop is not None and ckpt_crop != expected_crop_version:
            raise VersionParityError(
                f"Checkpoint crop_version '{ckpt_crop}' does not match serving crop function "
                f"'{expected_crop_version}'! Refusing to load mismatched crop classifier."
            )
    return model, payload
