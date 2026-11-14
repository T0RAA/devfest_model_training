"""20-species classifier with PyTorch EfficientNet-B0 and deterministic NumPy fallback.

Implements SPEC.md §4 & §7.4:
- Primary architecture: `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`
  with 20-way linear head, Stage A encoder freeze, and Stage B differential-LR fine-tuning.
- Local CPU/offline fallback: deterministic spatial-pyramid + gradient/color feature encoder
  with two-stage trainable projection and 20-way classifier head so `python -m wildlife.train`,
  `python -m wildlife.evaluate`, and `pytest` execute on machines without PyTorch wheels installed.
- Strict class-order validation on checkpoint load (SPEC.md §11.3 #3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wildlife.schemas import build_classes_Index_map


class CheckpointClassOrderError(RuntimeError):
    """Raised when a checkpoint's class list or order mismatches canonical taxonomy (§11.3 #3)."""


def extract_spatial_features_batch(chw_batch: np.ndarray) -> np.ndarray:
    """Extract a 144-D spatial color, contrast, and gradient descriptor from N x 3 x H x W float32."""
    if chw_batch.ndim == 3:
        chw_batch = chw_batch[None, ...]
    n, c, h, w = chw_batch.shape
    feats: List[np.ndarray] = []

    # Global channel mean, std, min/max percentiles (12 dims)
    feats.append(chw_batch.mean(axis=(2, 3)))
    feats.append(chw_batch.std(axis=(2, 3)))
    feats.append(np.percentile(chw_batch, 25, axis=(2, 3)))
    feats.append(np.percentile(chw_batch, 75, axis=(2, 3)))

    # 4x4 spatial grid mean and std per channel (4*4*3*2 = 96 dims)
    gh, gw = 4, 4
    sh, sw = max(1, h // gh), max(1, w // gw)
    for r in range(gh):
        for col in range(gw):
            patch = chw_batch[:, :, r * sh : (r + 1) * sh, col * sw : (col + 1) * sw]
            feats.append(patch.mean(axis=(2, 3)))
            feats.append(patch.std(axis=(2, 3)))

    # Horizontal & vertical spatial gradient magnitudes and frequencies (24 dims)
    gx = np.diff(chw_batch, axis=3)
    gy = np.diff(chw_batch, axis=2)
    feats.append(np.abs(gx).mean(axis=(2, 3)))
    feats.append(np.abs(gy).mean(axis=(2, 3)))
    feats.append(gx.std(axis=(2, 3)))
    feats.append(gy.std(axis=(2, 3)))

    # Center vs border contrast (12 dims)
    ch0, ch1 = h // 4, (3 * h) // 4
    cw0, cw1 = w // 4, (3 * w) // 4
    center = chw_batch[:, :, ch0:ch1, cw0:cw1]
    feats.append(center.mean(axis=(2, 3)))
    feats.append(center.std(axis=(2, 3)))
    feats.append(center.mean(axis=(2, 3)) - chw_batch.mean(axis=(2, 3)))
    feats.append(np.abs(np.diff(center, axis=3)).mean(axis=(2, 3)))

    out = np.concatenate(feats, axis=1).astype(np.float32)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)


class WildlifeClassifier:
    """Unified 20-species classifier supporting PyTorch EfficientNet-B0 and NumPy surrogate."""

    def __init__(
        self,
        num_classes: int = 20,
        seed: int = 42,
        pretrained: bool = False,
        use_torch_if_available: bool = True,
        class_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.class_map = class_map or build_classes_Index_map()
        self.class_ids: List[str] = list(self.class_map["class_ids"])
        if len(self.class_ids) != self.num_classes:
            raise CheckpointClassOrderError(
                f"Expected {self.num_classes} class_ids, got {len(self.class_ids)}"
            )

        self.backend = "numpy"
        self.torch_model = None
        self.device = "cpu"
        self.encoder_frozen = False

        if use_torch_if_available:
            try:
                import torch
                from torch import nn
                from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

                torch.manual_seed(self.seed)
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                net = efficientnet_b0(weights=weights)
                in_features = net.classifier[1].in_features
                net.classifier[1] = nn.Linear(in_features, self.num_classes)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.torch_model = net.to(self.device)
                self.backend = "torch_efficientnet_b0"
            except Exception:
                self.backend = "numpy"

        # Deterministic two-layer parameters for the NumPy backend (and fast fallback)
        rng = np.random.RandomState(self.seed)
        self.in_dim = 132
        self.hidden_dim = 96
        self.W_enc = (rng.randn(self.in_dim, self.hidden_dim) * np.sqrt(2.0 / self.in_dim)).astype(
            np.float32
        )
        self.b_enc = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.W_head = (rng.randn(self.hidden_dim, self.num_classes) * 0.08).astype(np.float32)
        self.b_head = np.zeros((self.num_classes,), dtype=np.float32)

    def freeze_encoder(self) -> None:
        """Stage A: Freeze encoder backbone, train only classification head."""
        self.encoder_frozen = True
        if self.torch_model is not None:
            for param in self.torch_model.features.parameters():
                param.requires_grad = False
            for param in self.torch_model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_encoder(self) -> None:
        """Stage B: Unfreeze encoder backbone for differential-LR fine-tuning."""
        self.encoder_frozen = False
        if self.torch_model is not None:
            for param in self.torch_model.parameters():
                param.requires_grad = True

    def forward_logits(self, chw_batch: Any) -> np.ndarray:
        """Compute raw 20-class logits (N x 20) from a batch of 3xHxW images."""
        if self.torch_model is not None:
            import torch

            self.torch_model.eval()
            with torch.no_grad():
                if not isinstance(chw_batch, torch.Tensor):
                    t = torch.from_numpy(np.asarray(chw_batch, dtype=np.float32))
                else:
                    t = chw_batch.float()
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                t = t.to(self.device)
                logits = self.torch_model(t)
                return logits.detach().cpu().numpy().astype(np.float32)

        arr = np.asarray(chw_batch, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        feats = extract_spatial_features_batch(arr)
        hidden = np.maximum(0.0, feats @ self.W_enc + self.b_enc)
        logits = hidden @ self.W_head + self.b_head
        return logits.astype(np.float32)

    def train_step_numpy(
        self,
        chw_batch: np.ndarray,
        labels: np.ndarray,
        head_lr: float = 1e-3,
        backbone_lr: float = 3e-5,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
    ) -> float:
        """Perform one mini-batch gradient update with label smoothing cross-entropy."""
        feats = extract_spatial_features_batch(chw_batch)
        n = feats.shape[0]
        pre_hidden = feats @ self.W_enc + self.b_enc
        hidden = np.maximum(0.0, pre_hidden)
        logits = hidden @ self.W_head + self.b_head
        probs = _softmax(logits)

        # Label-smoothed target distribution
        smooth_val = label_smoothing / float(self.num_classes)
        targets = np.full((n, self.num_classes), smooth_val, dtype=np.float32)
        targets[np.arange(n), labels] += 1.0 - label_smoothing

        eps = 1e-8
        loss = float(-np.sum(targets * np.log(np.clip(probs, eps, 1.0))) / n)

        dlogits = (probs - targets) / float(n)
        dW_head = hidden.T @ dlogits + weight_decay * self.W_head
        db_head = dlogits.sum(axis=0)

        if not self.encoder_frozen and backbone_lr > 0.0:
            dhidden = dlogits @ self.W_head.T
            dpre = dhidden * (pre_hidden > 0.0).astype(np.float32)
            dW_enc = feats.T @ dpre + weight_decay * self.W_enc
            db_enc = dpre.sum(axis=0)
            self.W_enc -= float(backbone_lr) * dW_enc
            self.b_enc -= float(backbone_lr) * db_enc

        self.W_head -= float(head_lr) * dW_head
        self.b_head -= float(head_lr) * db_head
        return loss


def compute_classification_metrics(
    logits: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int = 20,
) -> Dict[str, Any]:
    """Compute top-1, top-3, macro-F1, per-class precision/recall/F1/support, and 20x20 confusion matrix."""
    n = len(true_labels)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty label array.")

    preds_top1 = np.argmax(logits, axis=1)
    top1_acc = float(np.mean(preds_top1 == true_labels))

    top3_indices = np.argsort(-logits, axis=1)[:, :3]
    top3_hits = [int(true_labels[i] in top3_indices[i]) for i in range(n)]
    top3_acc = float(np.mean(top3_hits))

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, preds_top1):
        cm[int(t), int(p)] += 1

    per_class: Dict[int, Dict[str, float]] = {}
    f1_list: List[float] = []
    active_f1_list: List[float] = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)
        if support > 0:
            active_f1_list.append(f1)

    macro_f1 = float(np.mean(active_f1_list)) if active_f1_list else 0.0

    probs = _softmax(logits)
    nll = float(
        -np.mean(np.log(np.clip(probs[np.arange(n), true_labels], 1e-8, 1.0)))
    )

    return {
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "val_loss": round(nll, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def save_checkpoint(
    model: WildlifeClassifier,
    checkpoint_path: Path,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    temperature: float = 1.0,
) -> Tuple[Path, str]:
    """Serialize model checkpoint and write a SHA256 digest sidecar."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_version": "species-v1.0",
        "backend": model.backend,
        "num_classes": model.num_classes,
        "class_ids": list(model.class_ids),
        "class_to_idx": dict(model.class_map["class_to_idx"]),
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
    if model.torch_model is not None:
        try:
            import torch

            payload["torch_state_dict"] = {
                k: v.detach().cpu().numpy()
                for k, v in model.torch_model.state_dict().items()
            }
        except Exception:
            pass

    raw_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint_path.write_bytes(raw_bytes)
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    sha_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    sha_path.write_text(f"{sha256_hex}  {checkpoint_path.name}\n", encoding="utf-8")
    return checkpoint_path, sha256_hex


def load_checkpoint(
    checkpoint_path: Path,
    expected_classes_json: Optional[Path] = None,
) -> Tuple[WildlifeClassifier, Dict[str, Any]]:
    """Load checkpoint and enforce strict 20-species class order verification (§11.3 #3)."""
    checkpoint_path = Path(checkpoint_path)
    raw_bytes = checkpoint_path.read_bytes()

    sha_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if sha_path.exists():
        recorded_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if recorded_sha != actual_sha:
            raise CheckpointClassOrderError(
                f"Checkpoint SHA256 mismatch for {checkpoint_path}: expected {recorded_sha}, got {actual_sha}"
            )

    payload = pickle.loads(raw_bytes)
    if expected_classes_json and Path(expected_classes_json).exists():
        expected_map = json.loads(Path(expected_classes_json).read_text(encoding="utf-8"))
    else:
        expected_map = build_classes_Index_map()

    expected_ids = expected_map.get("class_ids", [])
    ckpt_ids = payload.get("class_ids", [])
    ckpt_num = payload.get("num_classes")

    if ckpt_num != 20 or ckpt_ids != expected_ids:
        raise CheckpointClassOrderError(
            f"Checkpoint class list/order mismatch! Expected {expected_ids}, got {ckpt_ids} (num={ckpt_num})."
        )
    if payload.get("class_to_idx") != expected_map.get("class_to_idx"):
        raise CheckpointClassOrderError(
            "Checkpoint class_to_idx mapping does not match canonical taxonomy."
        )

    model = WildlifeClassifier(
        num_classes=20,
        seed=int(payload.get("seed", 42)),
        pretrained=False,
        use_torch_if_available=(payload.get("backend") == "torch_efficientnet_b0"),
        class_map=expected_map,
    )
    np_state = payload.get("numpy_state")
    if np_state:
        model.W_enc = np.asarray(np_state["W_enc"], dtype=np.float32)
        model.b_enc = np.asarray(np_state["b_enc"], dtype=np.float32)
        model.W_head = np.asarray(np_state["W_head"], dtype=np.float32)
        model.b_head = np.asarray(np_state["b_head"], dtype=np.float32)

    if model.torch_model is not None and "torch_state_dict" in payload:
        import torch

        state = {
            k: torch.from_numpy(np.asarray(v))
            for k, v in payload["torch_state_dict"].items()
        }
        model.torch_model.load_state_dict(state)

    return model, payload
