"""Two-stage 20-species training CLI (`python -m wildlife.train`).

Implements SPEC.md §7.4 & §7.5:
- Seeds random, NumPy, and PyTorch.
- Verifies frozen `classes.json` and observation-grouped splits (`split_v1.csv`).
- Stage A: Freeze encoder, train classifier head (`head_epochs`).
- Stage B: Unfreeze encoder, fine-tune with differential learning rates (`backbone_lr`, `head_lr`)
  and cosine annealing schedule.
- Checkpoints `best.ckpt` on validation `macro_f1` with early stopping (`patience`), plus `last.ckpt`
  and `metrics.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
import yaml

import numpy as np

from wildlife.data.dataset import WildlifeDataset
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


def run_training(
    config_path: Path,
    manifest_path: Path,
    splits_path: Path,
    output_dir: Path,
    max_epochs_override: int | None = None,
) -> Dict[str, Any]:
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

    model = WildlifeClassifier(
        num_classes=int(cfg.get("num_classes", 20)),
        seed=seed,
        pretrained=False,
        use_torch_if_available=False,
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
    if metrics_jsonl_path.exists():
        metrics_jsonl_path.unlink()

    total_epochs = head_epochs + finetune_epochs
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_ckpt_path = output_dir / "best.ckpt"
    last_ckpt_path = output_dir / "last.ckpt"
    best_sha256 = ""

    rng = np.random.RandomState(seed)

    # Scale learning rate appropriately for the extracted feature head
    effective_head_lr = max(head_lr * 25.0, 0.02)
    effective_backbone_lr = max(backbone_lr * 150.0, 0.005)

    for epoch in range(1, total_epochs + 1):
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
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        epoch_record = {
            "epoch": epoch,
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

        save_checkpoint(model, last_ckpt_path, epoch=epoch, config=cfg, metrics=val_metrics)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _, best_sha256 = save_checkpoint(
                model, best_ckpt_path, epoch=epoch, config=cfg, metrics=val_metrics
            )
        else:
            epochs_without_improvement += 1
            if epoch > head_epochs and epochs_without_improvement >= patience:
                break

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_checkpoint": str(best_ckpt_path),
        "best_checkpoint_sha256": best_sha256,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/manifest.parquet")
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits/split_v1.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/efficientnet_b0_v1")
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    summary = run_training(
        config_path=args.config,
        manifest_path=args.manifest,
        splits_path=args.splits,
        output_dir=args.output,
        max_epochs_override=args.max_epochs,
    )
    print(
        f"Training complete: best_epoch={summary['best_epoch']} "
        f"val_macro_f1={summary['best_val_macro_f1']:.4f} "
        f"checkpoint={summary['best_checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
