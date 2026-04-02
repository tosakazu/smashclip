"""Evaluation metrics for SmashClip benchmark tasks."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)


# ── Task A: Aesthetic Score Classification (5-class) ──


def compute_task_a_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute Task A classification metrics.

    Parameters
    ----------
    logits : ndarray of shape (N, 5)
        Raw logits for 5 classes (scores 1-5 mapped to 0-4).
    labels : ndarray of shape (N,)
        Ground-truth labels in [0, 4].

    Returns
    -------
    dict with accuracy, weighted_f1, srcc, plcc.
    """
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    wf1 = f1_score(labels, preds, average="weighted")
    srcc, _ = spearmanr(labels, preds)
    plcc, _ = pearsonr(labels.astype(float), preds.astype(float))
    bacc = balanced_accuracy_score(labels, preds)

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "weighted_f1": round(float(wf1), 4),
        "srcc": round(float(srcc), 4),
        "plcc": round(float(plcc), 4),
    }


# ── Task A: Regression metrics ──


def compute_regression_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """For Task A regression: SRCC, PLCC, MAE.

    Parameters
    ----------
    predictions : ndarray of shape (N,)
        Predicted scores (continuous).
    labels : ndarray of shape (N,)
        Ground-truth scores (continuous).
    """
    srcc, _ = spearmanr(labels, predictions)
    plcc, _ = pearsonr(labels.astype(float), predictions.astype(float))
    mae = np.mean(np.abs(predictions - labels))
    return {
        "srcc": round(float(srcc), 4),
        "plcc": round(float(plcc), 4),
        "mae": round(float(mae), 4),
    }


# ── Task B: Metadata prediction metrics ──


def compute_metadata_metrics(
    preds: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """For Task B metadata prediction: Accuracy, Balanced Accuracy."""
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bacc), 4),
    }


# ── Task C: Scene Tag Prediction (20-class multi-label) ──


def compute_task_c_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute Task C metrics.

    Parameters
    ----------
    logits : ndarray of shape (N, 25)
        Raw logits (pre-sigmoid).
    labels : ndarray of shape (N, 25)
        Binary ground-truth labels.

    Returns
    -------
    dict with mAP, micro_f1, macro_f1.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    map_score = average_precision_score(labels, probs, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "mAP": round(float(map_score), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


# ── Task C: Per-layer mAP ──


def compute_task_c_metrics_by_layer(
    logits: np.ndarray, labels: np.ndarray, tag_layers: dict[str, list[int]]
) -> dict[str, float]:
    """Compute per-layer mAP for Task C scene tags.

    Parameters
    ----------
    logits : ndarray of shape (N, 25)
    labels : ndarray of shape (N, 25)
    tag_layers : dict mapping layer name to list of column indices,
        e.g. {"technique": [0,1,2,3,4], "context": [5,...,17], "meta": [18,...,24]}
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    result = {}
    for layer_name, indices in tag_layers.items():
        layer_probs = probs[:, indices]
        layer_labels = labels[:, indices]
        ap = average_precision_score(layer_labels, layer_probs, average="macro")
        result[f"mAP_{layer_name}"] = round(float(ap), 4)
    return result


# ── Statistical baselines ──


def compute_random_baseline(
    labels: np.ndarray, task: str, n_trials: int = 100, seed: int = 0
) -> dict[str, float]:
    """Expected metrics for a random predictor."""
    rng = np.random.RandomState(seed)
    metrics_list: list[dict[str, float]] = []

    for _ in range(n_trials):
        if task == "A":
            logits = rng.randn(len(labels), 5)
            metrics_list.append(compute_task_a_metrics(logits, labels))
        else:
            logits = rng.randn(len(labels), 25)
            metrics_list.append(compute_task_c_metrics(logits, labels))

    keys = metrics_list[0].keys()
    return {k: round(float(np.mean([m[k] for m in metrics_list])), 4) for k in keys}


def compute_mean_predictor_baseline(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> dict[str, float]:
    """Always predict training set mean."""
    mean_pred = np.full_like(test_labels, train_labels.mean(), dtype=float)
    return compute_regression_metrics(mean_pred, test_labels)


def compute_random_regression_baseline(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_trials: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Random predictions sampled from training distribution."""
    rng = np.random.RandomState(seed)
    metrics_list: list[dict[str, float]] = []
    for _ in range(n_trials):
        preds = rng.choice(train_labels, size=len(test_labels))
        metrics_list.append(compute_regression_metrics(preds, test_labels))
    keys = metrics_list[0].keys()
    return {k: round(float(np.mean([m[k] for m in metrics_list])), 4) for k in keys}


def compute_majority_baseline(
    train_labels: np.ndarray, test_labels: np.ndarray, task: str
) -> dict[str, float]:
    """Metrics for a majority-class predictor."""
    if task == "A":
        majority_class = int(np.bincount(train_labels).argmax())
        n = len(test_labels)
        logits = np.full((n, 5), -10.0)
        logits[:, majority_class] = 10.0
        return compute_task_a_metrics(logits, test_labels)
    else:
        tag_freq = train_labels.mean(axis=0)
        majority_vec = (tag_freq >= 0.5).astype(float)
        n = len(test_labels)
        logits = np.where(majority_vec[None, :] > 0.5, 10.0, -10.0)
        logits = np.broadcast_to(logits, (n, 25)).copy()
        return compute_task_c_metrics(logits, test_labels)
