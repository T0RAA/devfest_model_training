#!/usr/bin/env python3
"""Colab G4 GPU script: Train BioCLIP 2 (T0 + T1) on 150 images/species (3,000+ images) & test every species."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wildlife.data.dataset import WildlifeDataset
from wildlife.models.backbones import build_bioclip2_zero_shot_prompts, evaluate_tier_gates
from wildlife.models.calibration import TemperatureScaler, compute_ece
from wildlife.models.classifier import compute_classification_metrics
from wildlife.models.unknown import UnknownSpeciesGate, tune_gate_thresholds_grid
from wildlife.schemas import load_species_taxonomy


def main() -> int:
    import open_clip  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import GroupKFold  # type: ignore

    assert torch.cuda.is_available(), "Run this script on a Colab GPU runtime!"
    device = "cuda"
    print(f"=== Running 150/species BioCLIP 2 Train & 20-Species Test on {torch.cuda.get_device_name(0)} ===")

    manifest_path = ROOT / "data" / "manifests" / "manifest_v2.parquet"
    splits_path = ROOT / "data" / "splits" / "split_v2.csv"

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    clip_model = clip_model.to(device).eval()

    def _embed(split_name: str, is_train: bool, views: int = 1):
        ds = WildlifeDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split=split_name,
            image_size=224,
            is_train=is_train,
        )
        prep = preprocess_train if is_train else preprocess_val
        embs, labels, obs = [], [], []
        batch_imgs, batch_lbls, batch_obs = [], [], []

        def _flush():
            if not batch_imgs:
                return
            inp = torch.stack(batch_imgs, dim=0).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                f = clip_model.encode_image(inp)
                f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.detach().cpu().float().numpy())
            labels.extend(batch_lbls)
            obs.extend(batch_obs)
            batch_imgs.clear()
            batch_lbls.clear()
            batch_obs.clear()

        for i in range(len(ds)):
            row = ds.samples[i]
            cid = str(row["class_id"])
            c_idx = ds.class_to_idx.get(cid, 0)
            oid = str(row["observation_id"])
            pil_img = Image.fromarray(ds.load_rgb(i))
            for _ in range(views):
                batch_imgs.append(prep(pil_img))
                batch_lbls.append(c_idx)
                batch_obs.append(oid)
                if len(batch_imgs) >= 64:
                    _flush()
        _flush()
        return np.concatenate(embs, axis=0), np.array(labels, dtype=np.int64), obs

    print("1/4 Embedding validation, test, and train (4 views/crop) splits...")
    x_val, y_val, _ = _embed("validation", is_train=False, views=1)
    x_test, y_test, _ = _embed("test", is_train=False, views=1)
    x_train, y_train, train_obs = _embed("train", is_train=True, views=4)
    print(f"    Embedded: train={len(x_train)} views, val={len(x_val)}, test={len(x_test)}")

    print("2/4 Fitting 5-fold GroupKFold LogisticRegression head (T1)...")
    gkf = GroupKFold(n_splits=5)
    best_c, best_cv = 1.0, -1.0
    for c_val in [0.3, 1.0, 3.0, 10.0, 30.0]:
        scores = []
        for tr_i, va_i in gkf.split(x_train, y_train, groups=train_obs):
            clf = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(x_train[tr_i], y_train[tr_i])
            scores.append(float(np.mean(clf.predict(x_train[va_i]) == y_train[va_i])))
        if np.mean(scores) > best_cv:
            best_cv = float(np.mean(scores))
            best_c = c_val

    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(x_train, y_train)

    val_logits = clf.decision_function(x_val)
    test_logits = clf.decision_function(x_test)
    val_metrics = compute_classification_metrics(val_logits, y_val, num_classes=20)
    test_metrics = compute_classification_metrics(test_logits, y_test, num_classes=20)
    gates = evaluate_tier_gates(test_metrics)

    print("3/4 Fitting Temperature Scaling on validation logits...")
    scaler = TemperatureScaler(initial_temperature=1.0)
    cal_rep = scaler.fit(val_logits, y_val)
    test_probs = scaler.transform(test_logits)
    test_ece = compute_ece(test_probs, y_test, num_bins=15)

    print("4/4 Per-Species Held-Out Test Results (All 20 Species):")
    species_list = load_species_taxonomy()
    print(f"{'ID':<4} {'Common Name':<26} {'Support':>7} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 68)
    for sp in species_list:
        pc = test_metrics["per_class"][sp.index]
        print(
            f"{sp.class_id:<4} {sp.common_name:<26} {pc['support']:>7d} "
            f"{pc['precision']:>9.4f} {pc['recall']:>8.4f} {pc['f1']:>8.4f}"
        )

    summary = {
        "best_c": best_c,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_top1": test_metrics["top1_accuracy"],
        "test_top3": test_metrics["top3_accuracy"],
        "temperature": scaler.temperature,
        "test_ece_15bin": round(test_ece, 4),
        "test_gates": gates,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
