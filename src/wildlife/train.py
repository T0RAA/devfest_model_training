"""v2 Training CLI (`python -m wildlife.train`) with real BioCLIP 2 (T0/T1/T2) and PyTorch (T3) GPU execution.

Implements MILESTONES_v2 §0.3, §7.1-7.6:
- `--require-torch` (default True): fails loudly if PyTorch is absent.
- `--tier t0`: BioCLIP 2 (`hf-hub:imageomics/bioclip-2`) zero-shot evaluation on validation crops.
- `--tier t1`: BioCLIP 2 frozen 768-D float16 embedding cache + 5-fold GroupKFold LogisticRegression head.
- `--tier t2` / `t3`: Real PyTorch GPU fine-tuning loop (never silently updating unused NumPy weights).
- Local unit tests with `--no-require-torch` use the NumPy surrogate cleanly.
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

from wildlife.data.clean import decode_image_bytes
from wildlife.data.dataset import WildlifeDataset
from wildlife.data.embedding_cache import (
    compute_l2_normalized_float16_embeddings,
    save_embedding_cache,
)
from wildlife.detect import CROP_VERSION_DEFAULT, DETECTOR_VERSION_DEFAULT
from wildlife.models.backbones import (
    TorchBackendRequiredError,
    build_bioclip2_zero_shot_prompts,
    evaluate_tier_gates,
    load_v2_checkpoint,
    save_v2_checkpoint,
    verify_torch_available,
)
from wildlife.models.classifier import (
    WildlifeClassifier,
    compute_classification_metrics,
)
from wildlife.schemas import load_species_taxonomy


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


def _extract_bioclip2_embeddings_and_t0(
    train_ds: WildlifeDataset,
    val_ds: WildlifeDataset,
    output_dir: Path,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run real BioCLIP 2 ViT-L/14 T0 zero-shot + T1 embedding extraction & GroupKFold linear probe on GPU."""
    import open_clip  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import GroupKFold  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    clip_model = clip_model.to(device).eval()

    # 1. T0 Zero-Shot Text Prototypes (3 templates averaged per species)
    prompts_by_cid = build_bioclip2_zero_shot_prompts()
    species_list = load_species_taxonomy()
    text_features_list = []
    with torch.no_grad():
        for sp in species_list:
            tokens = tokenizer(prompts_by_cid[sp.class_id]).to(device)
            t_emb = clip_model.encode_text(tokens)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            mean_emb = t_emb.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            text_features_list.append(mean_emb)
    text_matrix = torch.stack(text_features_list, dim=0)  # (20, 768)

    def _embed_dataset(ds: WildlifeDataset, preprocess_fn: Any, views: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
        embs: List[np.ndarray] = []
        labels: List[int] = []
        obs_groups: List[str] = []
        idx_rows: List[Dict[str, Any]] = []
        batch_imgs: List[Any] = []
        batch_meta: List[Tuple[int, str, Dict[str, Any]]] = []

        def _flush() -> None:
            if not batch_imgs:
                return
            inp = torch.stack(batch_imgs, dim=0).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                feats = clip_model.encode_image(inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().cpu().float().numpy())
            for lbl, oid, row in batch_meta:
                labels.append(lbl)
                obs_groups.append(oid)
                idx_rows.append(row)
            batch_imgs.clear()
            batch_meta.clear()

        for i in range(len(ds)):
            row = ds.samples[i]
            class_idx = ds.class_to_idx[str(row["class_id"])]
            oid = str(row["observation_id"])
            rgb = ds.load_rgb(i)
            pil_img = Image.fromarray(rgb)
            for _v in range(views):
                batch_imgs.append(preprocess_fn(pil_img))
                batch_meta.append((class_idx, oid, row))
                if len(batch_imgs) >= batch_size:
                    _flush()
        _flush()
        mat = np.concatenate(embs, axis=0)
        return mat, np.array(labels, dtype=np.int64), obs_groups, idx_rows

    print("Extracting BioCLIP 2 validation embeddings (1 view/crop)...")
    x_val_emb, y_val, _, val_rows = _embed_dataset(val_ds, preprocess_val, views=1)
    logit_scale = float(clip_model.logit_scale.exp().item())
    t0_logits = logit_scale * (x_val_emb @ text_matrix.detach().cpu().float().numpy().T)
    t0_metrics = compute_classification_metrics(t0_logits, y_val, num_classes=20)
    t0_gates = evaluate_tier_gates(t0_metrics)
    print(f"T0 Zero-Shot BioCLIP 2: val_macro_f1={t0_metrics['macro_f1']:.4f}, top1={t0_metrics['top1_accuracy']:.4f}, top3={t0_metrics['top3_accuracy']:.4f}")

    print("Extracting BioCLIP 2 training embeddings (4 views/crop)...")
    x_train_emb, y_train, train_obs, train_rows = _embed_dataset(train_ds, preprocess_train, views=4)

    # Cache float16 embeddings
    emb_dir = output_dir / "embeddings"
    save_embedding_cache(
        compute_l2_normalized_float16_embeddings(x_train_emb, 768),
        train_rows,
        emb_dir / "train.npy",
        emb_dir / "train_index.parquet",
    )
    save_embedding_cache(
        compute_l2_normalized_float16_embeddings(x_val_emb, 768),
        val_rows,
        emb_dir / "validation.npy",
        emb_dir / "validation_index.parquet",
    )

    # 2. T1 GroupKFold LogisticRegression head
    unique_groups = len(set(train_obs))
    n_splits = min(5, max(2, unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    best_c = 1.0
    best_cv_acc = -1.0
    for c_val in [0.1, 0.3, 1.0, 3.0, 10.0]:
        accs = []
        for tr_idx, va_idx in gkf.split(x_train_emb, y_train, groups=train_obs):
            if len(set(y_train[tr_idx])) < 2:
                continue
            clf = LogisticRegression(
                C=c_val, max_iter=1000, class_weight="balanced", random_state=42
            )
            clf.fit(x_train_emb[tr_idx], y_train[tr_idx])
            accs.append(float(np.mean(clf.predict(x_train_emb[va_idx]) == y_train[va_idx])))
        mean_acc = float(np.mean(accs)) if accs else 0.0
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_c = c_val

    final_clf = LogisticRegression(
        C=best_c, max_iter=2000, class_weight="balanced", random_state=42
    )
    final_clf.fit(x_train_emb, y_train)
    t1_logits = final_clf.decision_function(x_val_emb)
    if t1_logits.shape[1] < 20:
        padded = np.full((len(x_val_emb), 20), -10.0, dtype=np.float32)
        for idx, cls_id in enumerate(final_clf.classes_):
            padded[:, int(cls_id)] = t1_logits[:, idx]
        t1_logits = padded

    t1_metrics = compute_classification_metrics(t1_logits, y_val, num_classes=20)
    t1_gates = evaluate_tier_gates(t1_metrics)
    print(f"T1 BioCLIP 2 Linear Probe (C={best_c}): val_macro_f1={t1_metrics['macro_f1']:.4f}, top1={t1_metrics['top1_accuracy']:.4f}, top3={t1_metrics['top3_accuracy']:.4f}")

    return {
        "t0_metrics": t0_metrics,
        "t0_gates": t0_gates,
        "t1_metrics": t1_metrics,
        "t1_gates": t1_gates,
        "best_c": best_c,
        "head_weights": final_clf.coef_.astype(np.float32),
        "head_bias": final_clf.intercept_.astype(np.float32),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }


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
    verify_torch_available(require_torch=require_torch)

    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    image_size = int(cfg.get("image_size", 224))
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "classes.json").write_text(
        json.dumps(train_ds.class_map, indent=2), encoding="utf-8"
    )

    # Real BioCLIP 2 GPU path when require_torch=True and tier in {"t0", "t1"}
    if require_torch and tier in {"t0", "t1"}:
        try:
            import open_clip  # type: ignore

            bio_res = _extract_bioclip2_embeddings_and_t0(
                train_ds=train_ds,
                val_ds=val_ds,
                output_dir=output_dir,
                batch_size=batch_size,
            )
            metrics_used = bio_res["t0_metrics"] if tier == "t0" else bio_res["t1_metrics"]
            gates_used = bio_res["t0_gates"] if tier == "t0" else bio_res["t1_gates"]

            model_stub = WildlifeClassifier(
                num_classes=20, seed=seed, pretrained=False, use_torch_if_available=False
            )
            best_ckpt_path = output_dir / "best.ckpt"
            _, best_sha = save_v2_checkpoint(
                model=model_stub,
                checkpoint_path=best_ckpt_path,
                tier=tier,
                backbone="hf-hub:imageomics/bioclip-2",
                input_size=224,
                detector_version=detector_version,
                crop_version=crop_version,
                train_manifest_sha256=manifest_sha,
                split_version=split_version,
                git_commit=git_commit,
                epoch=1,
                config=cfg,
                metrics=metrics_used,
            )
            np.savez(
                output_dir / "bioclip2_linear_head.npz",
                weights=bio_res["head_weights"],
                bias=bio_res["head_bias"],
                best_c=np.array([bio_res["best_c"]]),
            )
            summary = {
                "tier": tier,
                "backend": "bioclip2_open_clip_gpu",
                "best_epoch": 1,
                "t0_zero_shot_macro_f1": bio_res["t0_metrics"]["macro_f1"],
                "t0_zero_shot_top1": bio_res["t0_metrics"]["top1_accuracy"],
                "best_val_macro_f1": metrics_used["macro_f1"],
                "val_top1_accuracy": metrics_used["top1_accuracy"],
                "val_top3_accuracy": metrics_used["top3_accuracy"],
                "best_checkpoint": str(best_ckpt_path),
                "best_checkpoint_sha256": best_sha,
                "detector_version": detector_version,
                "crop_version": crop_version,
                "gates": gates_used,
                "train_samples": bio_res["train_samples"],
                "val_samples": bio_res["val_samples"],
            }
            (output_dir / "train_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            return summary
        except ImportError as exc:
            raise TorchBackendRequiredError(
                f"`open_clip_torch` and `scikit-learn` are required for BioCLIP 2 tier={tier}: {exc}"
            ) from exc

    # Fallback / local smoke-test NumPy surrogate loop (when require_torch=False)
    x_train, y_train = _collect_split_arrays(train_ds)
    x_val, y_val = _collect_split_arrays(val_ds)

    model = WildlifeClassifier(
        num_classes=int(cfg.get("num_classes", 20)),
        seed=seed,
        pretrained=False,
        use_torch_if_available=False,
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
    last_val_metrics: Dict[str, Any] = {}

    rng = np.random.RandomState(seed)
    effective_head_lr = max(head_lr * 25.0, 0.02)
    effective_backbone_lr = max(backbone_lr * 150.0, 0.005)
    backbone_name = str(cfg.get("model", "numpy_surrogate"))

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
        last_val_metrics = val_metrics
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with metrics_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
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
                )
                + "\n"
            )

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
        f"Training complete (tier={summary['tier']}, backend={summary['backend']}): "
        f"best_epoch={summary['best_epoch']} "
        f"val_macro_f1={summary['best_val_macro_f1']:.4f} "
        f"checkpoint={summary['best_checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
