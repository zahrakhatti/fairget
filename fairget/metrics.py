"""Empirical metrics for binary predictions.

The functions in this module evaluate thresholded predictions. They do not
make claims about the population from which an evaluation sample was drawn.
Undefined quantities are represented by ``numpy.nan`` instead of being
silently replaced with a favorable value.
"""

from typing import Sequence

import numpy as np


def _as_1d(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _validate_same_length(**arrays):
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Inputs must have the same length; received {details}.")


def _validate_binary(values, name, allow_empty=False):
    if len(values) == 0 and allow_empty:
        return
    unique = np.unique(values)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _finite_difference(first, second):
    if not (np.isfinite(first) and np.isfinite(second)):
        return float("nan")
    return float(abs(first - second))


def _rate_ratio(rate0, rate1):
    """Return the symmetric smaller-to-larger rate ratio."""
    if not (np.isfinite(rate0) and np.isfinite(rate1)):
        return float("nan")
    if rate0 == 0.0 and rate1 == 0.0:
        return float("nan")
    if rate0 == 0.0 or rate1 == 0.0:
        return 0.0
    return float(min(rate0, rate1) / max(rate0, rate1))


def positive_rates(pred, group):
    """Return positive-prediction rates for encoded groups 0 and 1."""
    pred = _as_1d(pred, "pred")
    group = _as_1d(group, "group")
    _validate_same_length(pred=pred, group=group)
    _validate_binary(pred, "pred", allow_empty=True)

    def _rate(group_value):
        mask = group == group_value
        return float(pred[mask].mean()) if np.any(mask) else float("nan")

    return _rate(0), _rate(1)


def demographic_parity_gap(pred, group):
    """Return the absolute difference between group selection rates."""
    rate0, rate1 = positive_rates(pred, group)
    return _finite_difference(rate0, rate1)


def disparate_impact_ratio(pred, group):
    """Return ``min(rate0, rate1) / max(rate0, rate1)``.

    Two zero selection rates produce ``nan`` rather than a misleading ratio of
    1.0. If exactly one rate is zero, the ratio is zero.
    """
    rate0, rate1 = positive_rates(pred, group)
    return _rate_ratio(rate0, rate1)


def true_positive_rates(label, pred, group):
    """Return true-positive rates for encoded groups 0 and 1."""
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    group = _as_1d(group, "group")
    _validate_same_length(label=label, pred=pred, group=group)
    _validate_binary(label, "label", allow_empty=True)
    _validate_binary(pred, "pred", allow_empty=True)

    def _tpr(group_value):
        positives = (label == 1) & (group == group_value)
        if not np.any(positives):
            return float("nan")
        return float(pred[positives].mean())

    return _tpr(0), _tpr(1)


def equal_opportunity_gap(label, pred, group):
    """Return the absolute difference between group true-positive rates."""
    rate0, rate1 = true_positive_rates(label, pred, group)
    return _finite_difference(rate0, rate1)


def equal_opportunity_ratio(label, pred, group):
    """Return the symmetric ratio between group true-positive rates."""
    rate0, rate1 = true_positive_rates(label, pred, group)
    return _rate_ratio(rate0, rate1)


def accuracy(label, pred):
    """Return empirical accuracy, or ``nan`` for an empty sample."""
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    _validate_same_length(label=label, pred=pred)
    if len(label) == 0:
        return float("nan")
    return float((label == pred).mean())


def accuracy_by_group(label, pred, group):
    """Return empirical accuracy for encoded groups 0 and 1."""
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    group = _as_1d(group, "group")
    _validate_same_length(label=label, pred=pred, group=group)

    def _accuracy(group_value):
        mask = group == group_value
        return accuracy(label[mask], pred[mask])

    return _accuracy(0), _accuracy(1)


def confusion_counts(label, pred):
    """Return binary confusion-matrix counts."""
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    _validate_same_length(label=label, pred=pred)
    _validate_binary(label, "label", allow_empty=True)
    _validate_binary(pred, "pred", allow_empty=True)

    return {
        "tp": int(np.sum((label == 1) & (pred == 1))),
        "tn": int(np.sum((label == 0) & (pred == 0))),
        "fp": int(np.sum((label == 0) & (pred == 1))),
        "fn": int(np.sum((label == 1) & (pred == 0))),
    }


def _roc_auc(label, score):
    """Compute ROC AUC using average ranks for tied scores."""
    positives = int(np.sum(label == 1))
    negatives = int(np.sum(label == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        # Ranks are one-based for the Mann--Whitney identity.
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop

    positive_rank_sum = float(ranks[label == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _average_precision(label, score):
    """Compute non-interpolated average precision at distinct score values."""
    positives = int(np.sum(label == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        block = sorted_label[start:stop]
        true_positives += int(np.sum(block == 1))
        false_positives += int(np.sum(block == 0))
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(area)


def binary_classification_metrics(label, pred, proba=None):
    """Return utility and behavior metrics for one evaluation sample.

    Rate metrics whose denominator is zero are returned as ``nan``. When
    probabilities are provided, probability-based utility metrics are included.
    """
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    _validate_same_length(label=label, pred=pred)
    _validate_binary(label, "label", allow_empty=True)
    _validate_binary(pred, "pred", allow_empty=True)

    counts = confusion_counts(label, pred)
    n = int(len(label))
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    predicted_positives = counts["tp"] + counts["fp"]
    predicted_negatives = counts["tn"] + counts["fn"]

    prevalence = _safe_divide(positives, n)
    selection_rate = _safe_divide(predicted_positives, n)
    tpr = _safe_divide(counts["tp"], positives)
    tnr = _safe_divide(counts["tn"], negatives)
    fpr = _safe_divide(counts["fp"], negatives)
    fnr = _safe_divide(counts["fn"], positives)
    precision = _safe_divide(counts["tp"], predicted_positives)
    negative_predictive_value = _safe_divide(counts["tn"], predicted_negatives)
    balanced_accuracy = (
        float((tpr + tnr) / 2.0)
        if np.isfinite(tpr) and np.isfinite(tnr)
        else float("nan")
    )
    majority_baseline = (
        float(max(prevalence, 1.0 - prevalence))
        if np.isfinite(prevalence)
        else float("nan")
    )

    result = {
        "n": n,
        "positives": int(positives),
        "negatives": int(negatives),
        "predicted_positives": int(predicted_positives),
        "predicted_negatives": int(predicted_negatives),
        "prevalence": prevalence,
        "selection_rate": selection_rate,
        "majority_baseline_accuracy": majority_baseline,
        "accuracy": accuracy(label, pred),
        "balanced_accuracy": balanced_accuracy,
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "negative_predictive_value": negative_predictive_value,
    }

    if proba is not None:
        proba = _as_1d(proba, "proba").astype(float)
        _validate_same_length(label=label, proba=proba)
        if np.any(~np.isfinite(proba)) or np.any((proba < 0.0) | (proba > 1.0)):
            raise ValueError("proba must contain finite values in [0, 1].")
        if n == 0:
            result.update({
                "mean_predicted_probability": float("nan"),
                "brier_score": float("nan"),
                "log_loss": float("nan"),
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
            })
        else:
            epsilon = np.finfo(float).eps
            clipped = np.clip(proba, epsilon, 1.0 - epsilon)
            result.update({
                "mean_predicted_probability": float(proba.mean()),
                "brier_score": float(np.mean((proba - label) ** 2)),
                "log_loss": float(
                    -np.mean(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))
                ),
                "roc_auc": _roc_auc(label, proba),
                "average_precision": _average_precision(label, proba),
            })

    return result


def evaluate_binary_predictions(
    label,
    pred,
    group=None,
    proba=None,
    group_values: Sequence = (0, 1),
):
    """Evaluate a binary classifier overall and by sensitive group."""
    label = _as_1d(label, "label")
    pred = _as_1d(pred, "pred")
    _validate_same_length(label=label, pred=pred)
    proba_array = None if proba is None else _as_1d(proba, "proba")
    if proba_array is not None:
        _validate_same_length(label=label, proba=proba_array)

    overall = binary_classification_metrics(label, pred, proba_array)
    result = {
        "overall": overall,
        "groups": {},
        "accuracy_gap": float("nan"),
        "balanced_accuracy_gap": float("nan"),
        "selection_rate_gap": float("nan"),
        "selection_rate_ratio": float("nan"),
        "worst_group_accuracy": float("nan"),
        "worst_group_balanced_accuracy": float("nan"),
    }
    if group is None:
        return result

    group = _as_1d(group, "group")
    _validate_same_length(label=label, group=group)
    if len(set(group_values)) != len(group_values):
        raise ValueError("group_values must not contain duplicates.")

    for group_value in group_values:
        mask = group == group_value
        group_proba = None if proba_array is None else proba_array[mask]
        result["groups"][str(group_value)] = binary_classification_metrics(
            label[mask], pred[mask], group_proba
        )

    group_metrics = list(result["groups"].values())
    if len(group_metrics) >= 2:
        first, second = group_metrics[0], group_metrics[1]
        result["accuracy_gap"] = _finite_difference(
            first["accuracy"], second["accuracy"]
        )
        result["balanced_accuracy_gap"] = _finite_difference(
            first["balanced_accuracy"], second["balanced_accuracy"]
        )
        result["selection_rate_gap"] = _finite_difference(
            first["selection_rate"], second["selection_rate"]
        )
        result["selection_rate_ratio"] = _rate_ratio(
            first["selection_rate"], second["selection_rate"]
        )

    finite_accuracy = [
        item["accuracy"] for item in group_metrics if np.isfinite(item["accuracy"])
    ]
    finite_balanced = [
        item["balanced_accuracy"]
        for item in group_metrics
        if np.isfinite(item["balanced_accuracy"])
    ]
    if finite_accuracy:
        result["worst_group_accuracy"] = float(min(finite_accuracy))
    if finite_balanced:
        result["worst_group_balanced_accuracy"] = float(min(finite_balanced))
    return result


# Concise aliases for callers evaluating one sample or a grouped sample.
classification_metrics = binary_classification_metrics
grouped_classification_metrics = evaluate_binary_predictions
