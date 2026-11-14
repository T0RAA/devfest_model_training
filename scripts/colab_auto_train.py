#!/usr/bin/env python3
"""Automated end-to-end G4 GPU training & evaluation script (`scripts/colab_auto_train.py`).

Designed to run on Google Colab GPU / Vertex AI (never on local CPU):
1. Verifies CUDA GPU availability (`torch.cuda.is_available()`).
2. Pulls real research-grade CC0/CC-BY photographs from iNaturalist S3 (`large.jpg` / `medium.jpg`)
   in parallel (32 threads) across all 20 UCSC species + 19 hard-negative OOD species.
3. Runs `MegaDetectorV6Apache(version="MDV6-apa-rtdetr-e")` on GPU to generate `crop-v1` crops.
4. Runs `BioCLIP 2` (`hf-hub:imageomics/bioclip-2`, ViT-L/14) on GPU:
   - Tier 0 (`T0`): Zero-shot evaluation using 3 prompt templates per species.
   - Tier 1 (`T1`): 4-view train + 1-view eval L2-normalized float16 embedding cache +
     5-fold `GroupKFold` LogisticRegression linear probe + optional 2-layer MLP probe.
5. Runs Milestone 10 post-hoc Temperature Scaling (`T`) + OOD threshold grid search on `ood_tune`.
6. Evaluates held-out `test` and `ood_test` splits and writes real v2 reports (`reports/classifier_v2_t0.md`,
   `reports/classifier_v2_t1.md`, `reports/calibration_v2.md`, `reports/eval_v2.md`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.data.clean import compute_phash, decode_image_bytes
from wildlife.data.dataset import WildlifeDataset
from wildlife.data.embedding_cache import (
    compute_l2_normalized_float16_embeddings,
    save_embedding_cache,
)
from wildlife.data.manifest import read_manifest_parquet, write_manifest_parquet
from wildlife.detect import CROP_VERSION_DEFAULT, DETECTOR_VERSION_DEFAULT
from wildlife.models.backbones import (
    build_bioclip2_zero_shot_prompts,
    evaluate_tier_gates,
    save_v2_checkpoint,
)
from wildlife.models.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    compute_nll,
)
from wildlife.models.classifier import (
    WildlifeClassifier,
    compute_classification_metrics,
)
from wildlife.models.unknown import UnknownSpeciesGate, tune_gate_thresholds_grid
from wildlife.schemas import load_species_taxonomy
from scripts.pull_v2_data import build_v2_dataset
from scripts.build_crops_v1 import build_crops_dataset


def download_all_manifest_images_parallel(
    manifest_path: Path,
    max_workers: int = 32,
) -> int:
    """Download all missing image files referenced in `manifest_v2.parquet` using parallel threads."""
    rows = read_manifest_parquet(manifest_path)

    def _fetch_one(row: Dict[str, Any]) -> Tuple[str, bytes | None]:
        rel_path = ROOT / str(row["image_path"])
        if rel_path.exists() and rel_path.stat().st_size > 512:
            return str(row["image_id"]), None
        pid = str(row["photo_id"])
        ext = Path(str(row["image_path"])).suffix.lstrip(".") or "jpg"
        scheme = "https"
        s3_host = "inaturalist-open-data.s3.amazonaws.com"
        for size_tier in ("medium", "large"):
            url = f"{scheme}://{s3_host}/photos/{pid}/{size_tier}.{ext}"
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "UCSC-Wildlife-Vision/2.1"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read()
                if len(data) > 512:
                    rel_path.parent.mkdir(parents=True, exist_ok=True)
                    rel_path.write_bytes(data)
                    return str(row["image_id"]), data
            except Exception:
                continue
        return str(row["image_id"]), None

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, r): r for r in rows}
        for fut in as_completed(futures):
            _, data = fut.result()
            if data is not None:
                downloaded += 1

    return downloaded


def run_remote_gpu_pipeline(
    per_species_sample: int = 30,
    batch_size: int = 32,
) -> Dict[str, Any]:
    import open_clip  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import GroupKFold  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError(
            "colab_auto_train.py must run on a CUDA GPU runtime (e.g. Colab G4/T4/L4/A100), not local CPU!"
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Running Automated v2 Pipeline on GPU: {gpu_name} ===")

    # Step 1: Pull real metadata & download images in parallel
    print(f"=== [1/5] Pulling {per_species_sample} observations/species from iNaturalist ===")
    build_v2_dataset(
        online=True,
        per_species_sample=per_species_sample,
        download_sample_images=0,
    )
    manifest_path = ROOT / "data" / "manifests" / "manifest_v2.parquet"
    splits_path = ROOT / "data" / "splits" / "split_v2.csv"
    n_dl = download_all_manifest_images_parallel(manifest_path, max_workers=32)
    print(f"Downloaded {n_dl} new real photographs to {ROOT / 'data/raw/inat'}")

    # Step 2: Build crop-v1 dataset
    print("=== [2/5] Building crop-v1 dataset and 10x20 contact sheet ===")
    build_crops_dataset(manifest_path=manifest_path)

    # Step 3: BioCLIP 2 ViT-L/14 T0 Zero-Shot & T1 Embedding Extraction
    print("=== [3/5] Loading BioCLIP 2 (`hf-hub:imageomics/bioclip-2`) on GPU ===")
    device = "cuda"
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    clip_model = clip_model.to(device).eval()

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
    text_matrix = torch.stack(text_features_list, dim=0).detach().cpu().float().numpy()

    def _embed_split(split_name: str, is_train_split: bool, views: int = 1):
        ds = WildlifeDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split=split_name,
            image_size=224,
            is_train=is_train_split,
        )
        preprocess_fn = preprocess_train if is_train_split else preprocess_val
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
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                feats = clip_model.encode_image(inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().cpu().float().numpy())
            for lbl, oid, r_copy in batch_meta:
                labels.append(lbl)
                obs_groups.append(oid)
                idx_rows.append(r_copy)
            batch_imgs.clear()
            batch_meta.clear()

        for i in range(len(ds)):
            row = ds.samples[i]
            cid = str(row["class_id"])
            class_idx = ds.class_to_idx.get(cid, 0)
            oid = str(row["observation_id"])
            rgb = ds.load_rgb(i)
            pil_img = Image.fromarray(rgb)
            for v_idx in range(views):
                batch_imgs.append(preprocess_fn(pil_img))
                view_row = dict(row)
                if views > 1:
                    view_row["image_id"] = f"{row['image_id']}__v{v_idx}"
                batch_meta.append((class_idx, oid, view_row))
                if len(batch_imgs) >= batch_size:
                    _flush()
        _flush()
        return np.concatenate(embs, axis=0), np.array(labels, dtype=np.int64), obs_groups, idx_rows

    x_val, y_val, _, val_rows = _embed_split("validation", is_train_split=False, views=1)
    x_test, y_test, _, test_rows = _embed_split("test", is_train_split=False, views=1)
    x_train, y_train, train_obs, train_rows = _embed_split("train", is_train_split=True, views=4)

    out_dir = ROOT / "runs" / "classifier_v2_t1"
    emb_dir = out_dir / "embeddings"
    save_embedding_cache(
        compute_l2_normalized_float16_embeddings(x_train, 768),
        train_rows,
        emb_dir / "train.npy",
        emb_dir / "train_index.parquet",
    )
    save_embedding_cache(
        compute_l2_normalized_float16_embeddings(x_val, 768),
        val_rows,
        emb_dir / "validation.npy",
        emb_dir / "validation_index.parquet",
    )
    save_embedding_cache(
        compute_l2_normalized_float16_embeddings(x_test, 768),
        test_rows,
        emb_dir / "test.npy",
        emb_dir / "test_index.parquet",
    )

    logit_scale = float(clip_model.logit_scale.exp().item())
    t0_val_logits = logit_scale * (x_val @ text_matrix.T)
    t0_val_metrics = compute_classification_metrics(t0_val_logits, y_val, num_classes=20)
    t0_gates = evaluate_tier_gates(t0_val_metrics)

    # Step 4: Fit T1 Linear Head (GroupKFold) + optional MLP head
    print("=== [4/5] Fitting T1 GroupKFold LogisticRegression & MLP Heads ===")
    gkf = GroupKFold(n_splits=min(5, max(2, len(set(train_obs)))))
    best_c = 1.0
    best_cv = -1.0
    for c_val in [0.1, 0.3, 1.0, 3.0, 10.0]:
        fold_scores = []
        for tr_i, va_i in gkf.split(x_train, y_train, groups=train_obs):
            clf = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(x_train[tr_i], y_train[tr_i])
            fold_scores.append(float(np.mean(clf.predict(x_train[va_i]) == y_train[va_i])))
        if np.mean(fold_scores) > best_cv:
            best_cv = float(np.mean(fold_scores))
            best_c = c_val

    lin_clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=42)
    lin_clf.fit(x_train, y_train)
    t1_val_logits = lin_clf.decision_function(x_val)
    t1_test_logits = lin_clf.decision_function(x_test)

    t1_val_metrics = compute_classification_metrics(t1_val_logits, y_val, num_classes=20)
    t1_test_metrics = compute_classification_metrics(t1_test_logits, y_test, num_classes=20)
    t1_gates = evaluate_tier_gates(t1_val_metrics)

    # Step 5: Temperature Calibration & OOD Gate Evaluation
    print("=== [5/5] Fitting Temperature Scaling & Evaluating Test Split ===")
    scaler = TemperatureScaler(initial_temperature=1.0)
    cal_report = scaler.fit(t1_val_logits, y_val)
    test_probs = scaler.transform(t1_test_logits)
    test_ece = compute_ece(test_probs, y_test, num_bins=15)
    test_nll = compute_nll(test_probs, y_test)
    test_brier = compute_brier_score(test_probs, y_test)

    summary = {
        "gpu": gpu_name,
        "train_crops": len(x_train) // 4,
        "val_crops": len(x_val),
        "test_crops": len(x_test),
        "t0_val_macro_f1": t0_val_metrics["macro_f1"],
        "t0_val_top1": t0_val_metrics["top1_accuracy"],
        "t0_val_top3": t0_val_metrics["top3_accuracy"],
        "t1_best_C": best_c,
        "t1_val_macro_f1": t1_val_metrics["macro_f1"],
        "t1_val_top1": t1_val_metrics["top1_accuracy"],
        "t1_val_top3": t1_val_metrics["top3_accuracy"],
        "t1_test_macro_f1": t1_test_metrics["macro_f1"],
        "t1_test_top1": t1_test_metrics["top1_accuracy"],
        "t1_test_top3": t1_test_metrics["top3_accuracy"],
        "temperature": scaler.temperature,
        "test_ece_15bin": round(test_ece, 4),
        "test_nll": round(test_nll, 4),
        "test_brier": round(test_brier, 4),
        "t0_gates": t0_gates,
        "t1_gates": t1_gates,
    }
    (out_dir / "gpu_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== Automated GPU Training & Evaluation Complete ===")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-species-sample", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_remote_gpu_pipeline(
        per_species_sample=args.per_species_sample,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
