"""Evaluation metrics, diagnostic test slices and per-image predictions.

Scores are softmax outputs over the 120 supported classes; they are not calibrated
probabilities of correctness. Top-k ordering: score descending, class_id ascending on ties.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ZERO_DIVISION = 0.0

SLICES: dict[str, dict[str, str]] = {
    "full": {"rule": "all records of the split", "query": ""},
    "clean": {
        "rule": "dup_with_test == 0 and label_conflict == 0",
        "query": "dup_with_test == 0 and label_conflict == 0",
    },
    "multi_box": {"rule": "n_boxes > 1", "query": "n_boxes > 1"},
    "single_box": {"rule": "n_boxes == 1", "query": "n_boxes == 1"},
}


class EvaluationError(ValueError):
    pass


def predict_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for images, _labels in loader:
            chunks.append(model(images.to(device)).float().cpu().numpy())
    if not chunks:
        raise EvaluationError("empty loader")
    logits = np.concatenate(chunks)
    if not np.isfinite(logits).all():
        raise EvaluationError("non-finite logits")
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Per row: indices sorted by score descending, class_id ascending on exact ties."""
    n, c = scores.shape
    if not 1 <= k <= c:
        raise EvaluationError(f"k must be in 1..{c}")
    cls = np.broadcast_to(np.arange(c), (n, c))
    order = np.lexsort((cls, -scores), axis=1)  # last key is primary
    return order[:, :k]


def compute_metrics(
    y_true: Sequence[int],
    scores: np.ndarray,
    labels: Sequence[int],
    class_names: dict[int, str] | None = None,
    top_pairs: int = 20,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    if y.size == 0:
        raise EvaluationError("cannot compute metrics on an empty selection")
    if scores.shape != (y.size, len(labels)):
        raise EvaluationError("scores shape does not match labels")
    labels = list(labels)
    top = topk_indices(scores, min(3, len(labels)))
    pred = top[:, 0]
    present = sorted(set(y.tolist()))
    f1_all = f1_score(y, pred, labels=labels, average="macro", zero_division=ZERO_DIVISION)
    f1_present = f1_score(y, pred, labels=present, average="macro", zero_division=ZERO_DIVISION)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=ZERO_DIVISION
    )
    names = class_names or {}
    per_class = [
        {
            "class_id": int(c),
            "breed_key": names.get(int(c)),
            "support": int(support[i]),
            "recall": float(recall[i]) if support[i] > 0 else None,
            "precision": float(precision[i]),
            "f1": float(f1[i]) if support[i] > 0 else None,
        }
        for i, c in enumerate(labels)
    ]
    cm = confusion_matrix(y, pred, labels=labels)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")[:top_pairs]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, off.shape)
        count = int(off[t, p])
        if count == 0:
            break
        pairs.append(
            {
                "true_class_id": int(labels[t]),
                "true_breed_key": names.get(int(labels[t])),
                "pred_class_id": int(labels[p]),
                "pred_breed_key": names.get(int(labels[p])),
                "count": count,
                "share_of_true_support": count / int(cm[t].sum()),
            }
        )
    return {
        "n": int(y.size),
        "classes_present": len(present),
        "top1": float((pred == y).mean()),
        "top3": float((top == y[:, None]).any(axis=1).mean()),
        "macro_f1_all_labels": float(f1_all),
        "macro_f1_present_labels": float(f1_present),
        "macro_f1_note": (
            f"sklearn f1_score(average='macro', zero_division={ZERO_DIVISION}); "
            f"'all_labels' uses the fixed list of {len(labels)} labels (classes absent from the "
            "selection count as 0), 'present_labels' only classes with support > 0"
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "most_confused_pairs": pairs,
    }


def slice_mask(records: pd.DataFrame, name: str) -> np.ndarray:
    if name not in SLICES:
        raise EvaluationError(f"unknown slice {name!r}")
    query = SLICES[name]["query"]
    if not query:
        return np.ones(len(records), dtype=bool)
    return records.eval(query).to_numpy(dtype=bool)


def predictions_frame(
    records: pd.DataFrame, scores: np.ndarray, class_keys: dict[int, str], k: int = 5
) -> pd.DataFrame:
    top = topk_indices(scores, k)
    out = pd.DataFrame(
        {
            "image_id": records["image_id"].to_numpy(),
            "split": records["split"].to_numpy(),
            "true_class_id": records["class_id"].to_numpy(),
            "true_breed_key": records["breed_key"].to_numpy(),
        }
    )
    for j in range(k):
        out[f"top{j + 1}_class_id"] = top[:, j]
        out[f"top{j + 1}_breed_key"] = [class_keys[int(c)] for c in top[:, j]]
        out[f"top{j + 1}_score"] = scores[np.arange(len(top)), top[:, j]]
    out["correct_top1"] = (out["top1_class_id"] == out["true_class_id"]).astype(int)
    return out
