"""Evaluation metrics, tie ordering, slices and prediction tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from doglens.evaluation import (
    SLICES,
    EvaluationError,
    compute_metrics,
    predictions_frame,
    slice_mask,
    softmax,
    topk_indices,
)


def test_topk_orders_by_score_then_class_id() -> None:
    scores = np.array([[0.1, 0.4, 0.4, 0.1], [0.25, 0.25, 0.25, 0.25]])
    assert topk_indices(scores, 3).tolist() == [[1, 2, 0], [0, 1, 2]]
    assert topk_indices(scores, 4)[1].tolist() == [0, 1, 2, 3]
    with pytest.raises(EvaluationError):
        topk_indices(scores, 0)
    with pytest.raises(EvaluationError):
        topk_indices(scores, 5)


def test_softmax_rows_sum_to_one_and_are_finite() -> None:
    logits = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    p = softmax(logits)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] == pytest.approx(1.0)


def _scores(preds: list[list[int]], n_labels: int) -> np.ndarray:
    """Build score rows whose top-k order is exactly the given class lists."""
    s = np.zeros((len(preds), n_labels))
    for i, order in enumerate(preds):
        for rank, c in enumerate(order):
            s[i, c] = 1.0 - 0.1 * rank
    return s


def test_compute_metrics_known_values() -> None:
    labels = [0, 1, 2, 3]
    y = [0, 0, 1, 1, 2]
    preds = [[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 3, 1], [2, 0, 1]]
    m = compute_metrics(y, _scores(preds, 4), labels, {0: "a", 1: "b", 2: "c", 3: "d"})
    top1 = [p[0] for p in preds]
    assert m["n"] == 5 and m["classes_present"] == 3
    assert m["top1"] == pytest.approx(3 / 5)
    assert m["top3"] == pytest.approx(5 / 5)
    assert m["macro_f1_all_labels"] == pytest.approx(
        f1_score(y, top1, labels=labels, average="macro", zero_division=0.0)
    )
    assert m["macro_f1_present_labels"] == pytest.approx(
        f1_score(y, top1, labels=[0, 1, 2], average="macro", zero_division=0.0)
    )
    assert m["macro_f1_all_labels"] < m["macro_f1_present_labels"]  # absent class counts as 0
    per = {c["class_id"]: c for c in m["per_class"]}
    assert per[0]["support"] == 2 and per[0]["recall"] == pytest.approx(0.5)
    assert per[3]["support"] == 0 and per[3]["recall"] is None
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert m["confusion_matrix"][1] == [0, 1, 1, 0]
    pairs = {(p["true_class_id"], p["pred_class_id"]): p for p in m["most_confused_pairs"]}
    assert set(pairs) == {(0, 1), (1, 2)}
    assert pairs[(0, 1)]["share_of_true_support"] == pytest.approx(0.5)
    assert pairs[(0, 1)]["true_breed_key"] == "a"


def test_compute_metrics_rejects_empty_and_bad_shapes() -> None:
    with pytest.raises(EvaluationError):
        compute_metrics([], np.zeros((0, 4)), [0, 1, 2, 3])
    with pytest.raises(EvaluationError):
        compute_metrics([0], np.zeros((1, 3)), [0, 1, 2, 3])


def test_slices_rules() -> None:
    df = pd.DataFrame(
        {
            "dup_with_test": [0, 1, 0, 0, 1],
            "label_conflict": [0, 0, 1, 0, 1],
            "n_boxes": [1, 2, 1, 3, 1],
        }
    )
    assert slice_mask(df, "full").tolist() == [True] * 5
    assert slice_mask(df, "clean").tolist() == [True, False, False, True, False]
    assert slice_mask(df, "multi_box").tolist() == [False, True, False, True, False]
    assert slice_mask(df, "single_box").tolist() == [True, False, True, False, True]
    assert set(SLICES) == {"full", "clean", "multi_box", "single_box"}
    with pytest.raises(EvaluationError):
        slice_mask(df, "other")


def test_predictions_frame() -> None:
    records = pd.DataFrame(
        {"image_id": ["a", "b"], "split": ["test"] * 2, "class_id": [2, 0], "breed_key": ["c", "a"]}
    )
    scores = softmax(np.array([[0.0, 1.0, 3.0, 2.0], [0.5, 0.5, 0.0, 0.0]]))
    df = predictions_frame(records, scores, {0: "a", 1: "b", 2: "c", 3: "d"}, k=3)
    assert df.loc[0, ["top1_class_id", "top2_class_id", "top3_class_id"]].tolist() == [2, 3, 1]
    assert df.loc[1, "top1_class_id"] == 0 and df.loc[1, "top2_class_id"] == 1  # tie -> id asc
    assert df["correct_top1"].tolist() == [1, 1]
    assert df.loc[0, "top1_score"] == pytest.approx(scores[0, 2])
