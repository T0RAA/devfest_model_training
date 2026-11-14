"""Two-stage 20-species training CLI (`python -m wildlife.train`) with v2 efficiency ladder (`--tier`, `--require-torch`, `--resume`).

Implements SPEC.md §7.4 & MILESTONES_v2 §0.3, §7.1-7.6:
- Enforces `--require-torch` (default True on Colab/Vertex; raises `TorchBackendRequiredError` if PyTorch is absent
  unless `--no-require-torch` or `WILDLIFE_TEST_MODE=1` is explicitly passed for local smoke tests).
- Supports `--tier {t0,t1,t2,t3,distill}` and `--resume`.
- Records `detector_version`, `crop_version`, `train_manifest_sha256`, `split_version`, and `git_commit` in every checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

import numpy as np

from wildlife.data.dataset import WildlifeDataset
from wildlife.detect import CROP_VERSION_DEFAULT, DETECTOR_VERSION_DEFAULT
from wildlife.models.backbones import (
    TorchBackendRequiredError,
    evaluate_tier_gates,
    load_v2_checkpoint,
    save_v2_checkpoint,
    verify_torch_available,
)
from wildlife.models.classifier import (
    WildlifeClassifier,
    compute_classification_metrics,
    save_checkpoint,
)


def _collect_split_arrays(dataset: WildlifeDataset) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(len(dataset)):
        item_x, label_idx, _ = dataset[i]
        arr = np.asarray(item_x, dtype=np.float32)
        xs.append(arr)
        ys.append(int(label_idx))
    if not xs:
        raise ValueError(f"Split '{dataset.split}' has 0 samples.")
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def run_training(
    config_path: Path,
    manifest_path: Path,
    splits_path: Path,
    output_dir: Path,
    max_epochs_override: Optional[int] = None,
    tier: str = "t1",
    require_torch: bool = False,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    # Ground Rule #3: Fail loudly on fallback when require_torch=True
    verify_torch_available(require_torch=require_torch)

    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    image_size = int(cfg.get("image_size", 96))
    batch_size = int(cfg.get("batch_size", 32))
    head_epochs = int(cfg.get("head_epochs", 4))
    finetune_epochs = int(cfg.get("finetune_epochs", 24))
    if max_epochs_override is not None:
        head_epochs = min(head_epochs, max(1, max_epochs_override // 3))
        finetune_epochs = max(1, max_epochs_override - head_epochs)

    head_lr = float(cfg.get("head_lr", 0.001))
    backbone_lr = float(cfg.get("backbone_lr", 0.00003))
    weight_decay = float(cfg.get("weight_decay", 0.0001))
    label_smoothing = float(cfg.get("label_smoothing", 0.05))
    patience = int(cfg.get("patience", 5))
    detector_version = str(cfg.get("detector_version", DETECTOR_VERSION_DEFAULT))
    crop_version = str(cfg.get("crop_version", CROP_VERSION_DEFAULT))
    split_version = str(cfg.get("split_version", Path(splits_path).stem))

    manifest_sha = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    git_commit = _get_git_commit()

    train_ds = WildlifeDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="train",
        image_size=image_size,
        is_train=True,
        seed=seed,
    )
    val_ds = WildlifeDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="validation",
        image_size=image_size,
        is_train=False,
        seed=seed,
    )

    x_train, y_train = _collect_split_arrays(train_ds)
    x_val, y_val = _collect_split_arrays(val_ds)

    start_epoch = 1
    if resume_checkpoint and Path(resume_checkpoint).exists():
        model, resumed_payload = load_v2_checkpoint(
            Path(resume_checkpoint),
            expected_detector_version=detector_version,
            expected_crop_version=crop_version,
        )
        start_epoch = int(resumed_payload.get("epoch", 0)) + 1
    else:
        model = WildlifeClassifier(
            num_classes=int(cfg.get("num_classes", 20)),
            seed=seed,
            pretrained=False,
            use_torch_if_available=require_torch,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "classes.json").write_text(
        json.dumps(model.class_map, indent=2), encoding="utf-8"
    )

    metrics_jsonl_path = output_dir / "metrics.jsonl"
    if start_epoch == 1 and metrics_jsonl_path.exists():
        metrics_jsonl_path.unlink()

    total_epochs = head_epochs + finetune_epochs
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_ckpt_path = output_dir / "best.ckpt"
    last_ckpt_path = output_dir / "last.ckpt"
    best_sha256 = ""
    last_val_metrics: Dict[str, Any] = {}

    rng = np.random.RandomState(seed)
    effective_head_lr = max(head_lr * 25.0, 0.02)
    effective_backbone_lr = max(backbone_lr * 150.0, 0.005)
    backbone_name = str(cfg.get("model", "hf-hub:imageomics/bioclip-2"))

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.perf_counter()
        stage = "stage_a_head" if epoch <= head_epochs else "stage_b_finetune"
        if epoch <= head_epochs:
            model.freeze_encoder()
            cur_head_lr = effective_head_lr
            cur_bb_lr = 0.0
        else:
            model.unfreeze_encoder()
            progress = (epoch - head_epochs - 1) / float(max(1, finetune_epochs))
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            cur_head_lr = effective_head_lr * (0.1 + 0.9 * cosine_mult)
            cur_bb_lr = effective_backbone_lr * (0.1 + 0.9 * cosine_mult)

        perm = rng.permutation(len(x_train))
        batch_losses: List[float] = []
        for start in range(0, len(x_train), batch_size):
            idx = perm[start : start + batch_size]
            loss = model.train_step_numpy(
                x_train[idx],
                y_train[idx],
                head_lr=cur_head_lr,
                backbone_lr=cur_bb_lr,
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
            )
            batch_losses.append(loss)

        val_logits = model.forward_logits(x_val)
        val_metrics = compute_classification_metrics(val_logits, y_val, num_classes=20)
        last_val_metrics = val_metrics
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        epoch_record = {
            "epoch": epoch,
            "tier": tier,
            "stage": stage,
            "train_loss": round(float(np.mean(batch_losses)), 4),
            "val_loss": val_metrics["val_loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top1_acc": val_metrics["top1_accuracy"],
            "val_top3_acc": val_metrics["top3_accuracy"],
            "epoch_time_ms": round(elapsed_ms, 2),
        }
        with metrics_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_record) + "\n")

        save_v2_checkpoint(
            model=model,
            checkpoint_path=last_ckpt_path,
            tier=tier,
            backbone=backbone_name,
            input_size=image_size,
            detector_version=detector_version,
            crop_version=crop_version,
            train_manifest_sha256=manifest_sha,
            split_version=split_version,
            git_commit=git_commit,
            epoch=epoch,
            config=cfg,
            metrics=val_metrics,
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _, best_sha256 = save_v2_checkpoint(
                model=model,
                checkpoint_path=best_ckpt_path,
                tier=tier,
                backbone=backbone_name,
                input_size=image_size,
                detector_version=detector_version,
                crop_version=crop_version,
                train_manifest_sha256=manifest_sha,
                split_version=split_version,
                git_commit=git_commit,
                epoch=epoch,
                config=cfg,
                metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1
            if epoch > head_epochs and epochs_without_improvement >= patience:
                break

    gate_eval = evaluate_tier_gates(last_val_metrics)
    summary = {
        "tier": tier,
        "backend": model.backend,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_checkpoint": str(best_ckpt_path),
        "best_checkpoint_sha256": best_sha256,
        "detector_version": detector_version,
        "crop_version": crop_version,
        "gates": gate_eval,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_v2.yaml"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/manifest_v2.parquet")
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits/split_v2.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/classifier_v2_t1")
    )
    parser.add_argument(
        "--tier",
        choices=["t0", "t1", "t2", "t3", "distill"],
        default="t1",
    )
    parser.add_argument(
        "--require-torch",
        dest="require_torch",
        action="store_true",
        default=(os.environ.get("WILDLIFE_TEST_MODE", "0") != "1"),
        help="Fail loudly if PyTorch is unavailable (default True unless WILDLIFE_TEST_MODE=1)",
    )
    parser.add_argument(
        "--no-require-torch",
        dest="require_torch",
        action="store_false",
        help="Allow deterministic NumPy surrogate for local smoke tests",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    summary = run_training(
        config_path=args.config,
        manifest_path=args.manifest,
        splits_path=args.splits,
        output_dir=args.output,
        max_epochs_override=args.max_epochs,
        tier=args.tier,
        require_torch=args.require_torch,
        resume_checkpoint=args.resume,
    )
    print(
        f"Training complete (tier={summary['tier']}): best_epoch={summary['best_epoch']} "
        f"val_macro_f1={summary['best_val_macro_f1']:.4f} "
        f"checkpoint={summary['best_checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
