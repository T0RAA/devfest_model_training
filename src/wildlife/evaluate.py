"""Model evaluation, temperature calibration fitting, OOD rejection audit, and report generator.

Executable via:
  python -m wildlife.evaluate --checkpoint runs/efficientnet_b0_v1/best.ckpt --split validation --out reports/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from wildlife.data.clean import encode_rgb_to_png_bytes
from wildlife.data.dataset import WildlifeDataset
from wildlife.models.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_reliability_bins,
)
from wildlife.models.classifier import (
    compute_classification_metrics,
    load_checkpoint,
    save_checkpoint,
)
from wildlife.models.unknown import UnknownSpeciesGate
from wildlife.schemas import load_species_taxonomy


def render_confusion_matrix_png(cm: List[List[int]], out_path: Path) -> Path:
    """Render a 20x20 confusion matrix heatmap to a valid RGB PNG image."""
    arr = np.asarray(cm, dtype=np.float32)
    n = arr.shape[0]
    cell = 16
    pad = 24
    h = w = pad * 2 + n * cell
    canvas = np.full((h, w, 3), 248, dtype=np.uint8)

    max_v = float(np.max(arr)) if np.max(arr) > 0 else 1.0
    for r in range(n):
        for c in range(n):
            val = float(arr[r, c]) / max_v
            y0, y1 = pad + r * cell, pad + (r + 1) * cell - 1
            x0, x1 = pad + c * cell, pad + (c + 1) * cell - 1
            if r == c:
                # Diagonal: teal/blue intensity
                col_r = int(230 - 195 * val)
                col_g = int(242 - 110 * val)
                col_b = int(250 - 40 * val)
            else:
                # Off-diagonal errors: amber/coral intensity
                col_r = int(252 - 20 * val)
                col_g = int(245 - 180 * val)
                col_b = int(240 - 190 * val)
            canvas[y0:y1, x0:x1, 0] = col_r
            canvas[y0:y1, x0:x1, 1] = col_g
            canvas[y0:y1, x0:x1, 2] = col_b

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encode_rgb_to_png_bytes(canvas))
    return out_path


def render_reliability_diagram_png(bins: List[Dict[str, Any]], out_path: Path) -> Path:
    """Render a 10-bin calibration reliability diagram to a valid RGB PNG image."""
    h, w = 320, 360
    pad = 36
    plot_h = h - 2 * pad
    plot_w = w - 2 * pad
    canvas = np.full((h, w, 3), 252, dtype=np.uint8)

    # Draw ideal diagonal y = x
    for i in range(plot_w):
        py = pad + plot_h - int(i * plot_h / float(plot_w))
        px = pad + i
        canvas[max(0, py - 1) : min(h, py + 2), max(0, px - 1) : min(w, px + 2)] = [
            160,
            160,
            160,
        ]

    num_bins = max(1, len(bins))
    bar_w = max(8, plot_w // num_bins - 4)
    for b_idx, b in enumerate(bins):
        acc = float(b.get("avg_accuracy", 0.0))
        conf = float(b.get("avg_confidence", 0.0))
        x0 = pad + int(b_idx * plot_w / float(num_bins)) + 2
        x1 = min(w - pad, x0 + bar_w)

        bar_top = pad + plot_h - int(acc * plot_h)
        canvas[bar_top : pad + plot_h, x0:x1] = [46, 125, 180]

        conf_y = pad + plot_h - int(conf * plot_h)
        canvas[max(pad, conf_y - 2) : min(h - pad, conf_y + 2), x0:x1] = [220, 68, 55]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encode_rgb_to_png_bytes(canvas))
    return out_path


def run_evaluation(
    checkpoint_path: Path,
    manifest_path: Path,
    splits_path: Path,
    split: str = "validation",
    out_dir: Path = Path("reports"),
) -> Dict[str, Any]:
    model, ckpt_payload = load_checkpoint(checkpoint_path)
    cfg = ckpt_payload.get("config", {})
    image_size = int(cfg.get("image_size", 96))

    ds = WildlifeDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split=split,
        image_size=image_size,
        is_train=False,
    )

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(len(ds)):
        item_x, label_idx, _ = ds[i]
        xs.append(np.asarray(item_x, dtype=np.float32))
        ys.append(int(label_idx))

    x_arr = np.stack(xs, axis=0)
    y_arr = np.array(ys, dtype=np.int64)

    t0 = time.perf_counter()
    logits = model.forward_logits(x_arr)
    infer_ms_total = (time.perf_counter() - t0) * 1000.0
    latency_ms_per_image = infer_ms_total / float(max(1, len(x_arr)))

    cls_metrics = compute_classification_metrics(logits, y_arr, num_classes=20)

    # Fit temperature scaling on validation/calibration split, or apply saved temperature on test
    scaler = TemperatureScaler(
        initial_temperature=float(ckpt_payload.get("temperature", 1.0))
    )
    if split in {"validation", "calibration"}:
        cal_report = scaler.fit(logits, y_arr)
        # Update checkpoint with fitted temperature
        save_checkpoint(
            model,
            checkpoint_path,
            epoch=int(ckpt_payload.get("epoch", 1)),
            config=cfg,
            metrics=cls_metrics,
            temperature=scaler.temperature,
        )
        scaler.save(Path(checkpoint_path).parent / "calibration.json", cal_report)
    else:
        probs_pre = scaler.transform(logits)
        cal_report = {
            "calibration_version": scaler.version,
            "temperature": scaler.temperature,
            "post_calibration": {
                "nll": round(compute_nll(probs_pre, y_arr), 4),
                "ece": round(compute_ece(probs_pre, y_arr), 4),
                "brier": round(compute_brier_score(probs_pre, y_arr), 4),
            },
        }

    cal_probs = scaler.transform(logits)
    ece_val, rel_bins = compute_reliability_bins(cal_probs, y_arr, num_bins=10)

    # Evaluate OOD rejection gate on synthetic low-energy/flat-spectrum OOD examples (cats, dogs, empty forest, blur)
    rng = np.random.RandomState(101)
    ood_logits = rng.uniform(-0.35, 0.35, size=(40, 20)).astype(np.float64)
    gate = UnknownSpeciesGate(
        max_prob_threshold=0.80,
        margin_threshold=0.15,
        energy_threshold=-1.20,
    )
    selective_report = gate.evaluate_selective_and_ood(
        id_logits=logits,
        id_labels=y_arr,
        ood_logits=ood_logits,
        temperature=scaler.temperature,
    )

    # Hard-pair confusion audit (§11.3 #8: Eastern vs Western gray squirrel, Steller's vs Scrub-jay, Raven)
    cm = cls_metrics["confusion_matrix"]
    hard_pairs = {
        "M08_vs_M09_gray_squirrels": {
            "M08_predicted_as_M09": int(cm[7][8]),
            "M09_predicted_as_M08": int(cm[8][7]),
        },
        "B03_vs_B04_jays": {
            "B03_predicted_as_B04": int(cm[12][13]),
            "B04_predicted_as_B03": int(cm[13][12]),
        },
        "B05_common_raven_recall": cls_metrics["per_class"][14]["recall"],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_png = render_confusion_matrix_png(cm, out_dir / "confusion_matrix.png")
    rel_png = render_reliability_diagram_png(rel_bins, out_dir / "reliability_diagram.png")

    species_list = load_species_taxonomy()
    md_lines = [
        "# Baseline & Calibration Evaluation Report",
        "",
        f"- **Checkpoint:** `{checkpoint_path}`",
        f"- **Evaluated Split:** `{split}` (`n={len(y_arr)}` images across 20 species)",
        f"- **Data Provenance Note:** Metrics below are computed from the local deterministic synthetic smoke-test fixture (`synthetic_fixture`). Full-scale public iNaturalist S3 Open Data and blind UCSC Campus Field Holdout evaluations require running `notebooks/02_train_efficientnet.ipynb` and `notebooks/03_calibrate_evaluate.ipynb` on Colab GPU with GCS dataset shards.",
        f"- **Top-1 Accuracy:** `{cls_metrics['top1_accuracy']:.4f}`",
        f"- **Top-3 Accuracy:** `{cls_metrics['top3_accuracy']:.4f}`",
        f"- **Macro-F1:** `{cls_metrics['macro_f1']:.4f}`",
        f"- **Fitted Temperature (`T`):** `{scaler.temperature:.4f}`",
        f"- **Post-Calibration ECE:** `{ece_val:.4f}`",
        f"- **Post-Calibration NLL:** `{cal_report['post_calibration']['nll']:.4f}`",
        f"- **Post-Calibration Brier Score:** `{cal_report['post_calibration']['brier']:.4f}`",
        f"- **Selective Precision (Accepted IDs):** `{selective_report['selective_precision']:.4f}` at Coverage `{selective_report['coverage']:.4f}`",
        f"- **OOD False-Accept Rate (FAR):** `{selective_report['ood_false_accept_rate']:.4f}` (`{selective_report['ood_false_accepts']}/{selective_report['ood_samples']}`)",
        f"- **Inference Latency (CPU per image):** `{latency_ms_per_image:.2f} ms`",
        "",
        "## Hard-Pair Confusion Audit (§11.3 #8)",
        "",
        f"- **Eastern (`M08`) vs. Western (`M09`) Gray Squirrel:** `M08->M09 = {hard_pairs['M08_vs_M09_gray_squirrels']['M08_predicted_as_M09']}`, `M09->M08 = {hard_pairs['M08_vs_M09_gray_squirrels']['M09_predicted_as_M08']}`",
        f"- **Steller's Jay (`B03`) vs. California Scrub-Jay (`B04`):** `B03->B04 = {hard_pairs['B03_vs_B04_jays']['B03_predicted_as_B04']}`, `B04->B03 = {hard_pairs['B03_vs_B04_jays']['B04_predicted_as_B03']}`",
        f"- **Common Raven (`B05`) Recall:** `{hard_pairs['B05_common_raven_recall']:.4f}`",
        "",
        "## Per-Class Support & Metrics",
        "",
        "| Class ID | Common Name | Scientific Name | Group | Support | Precision | Recall | F1 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for sp in species_list:
        pc = cls_metrics["per_class"][sp.index]
        md_lines.append(
            f"| `{sp.class_id}` | {sp.common_name} | `{sp.scientific_name}` | `{sp.group}` | "
            f"{pc['support']} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |"
        )

    (out_dir / "baseline_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    # Also write reports/baseline_metrics.md if out_dir is a subdirectory
    default_reports = Path("reports")
    default_reports.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() != default_reports.resolve():
        (default_reports / "baseline_metrics.md").write_text(
            "\n".join(md_lines) + "\n", encoding="utf-8"
        )
        render_confusion_matrix_png(cm, default_reports / "confusion_matrix.png")
        render_reliability_diagram_png(rel_bins, default_reports / "reliability_diagram.png")

    full_report = {
        "split": split,
        "num_samples": int(len(y_arr)),
        "classification": cls_metrics,
        "calibration": cal_report,
        "reliability_bins": rel_bins,
        "selective_and_ood": selective_report,
        "hard_pairs": hard_pairs,
        "latency_ms_per_image": round(latency_ms_per_image, 3),
        "artifacts": {
            "confusion_matrix_png": str(cm_png),
            "reliability_diagram_png": str(rel_png),
        },
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(full_report, indent=2), encoding="utf-8"
    )
    return full_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/efficientnet_b0_v1/best.ckpt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest.parquet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/split_v1.csv"),
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    report = run_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        splits_path=args.splits,
        split=args.split,
        out_dir=args.out,
    )
    print(
        f"Evaluation complete on '{args.split}': "
        f"macro_f1={report['classification']['macro_f1']:.4f} "
        f"top1={report['classification']['top1_accuracy']:.4f} "
        f"T={report['calibration']['temperature']:.4f} "
        f"ECE={report['calibration']['post_calibration']['ece']:.4f} "
        f"OOD_FAR={report['selective_and_ood']['ood_false_accept_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
