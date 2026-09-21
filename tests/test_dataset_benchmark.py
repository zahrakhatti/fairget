"""Tests for the validation-first dataset benchmark helpers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from examples.dataset_benchmark import (
    Candidate,
    DatasetSpec,
    joint_stratified_split,
    parse_architecture,
    rejection_reasons,
    ratio_passes,
)


def test_architecture_is_supplied_explicitly():
    assert parse_architecture("logistic") == ()
    assert parse_architecture("8") == (8,)
    assert parse_architecture("16,8") == (16, 8)


def test_joint_split_is_deterministic_disjoint_and_stratified():
    rows = []
    for group in ("A", "B"):
        for outcome in ("no", "yes"):
            for number in range(10):
                rows.append({
                    "row_id": f"{group}-{outcome}-{number}",
                    "feature": number,
                    "group": group,
                    "outcome": outcome,
                })
    frame = pd.DataFrame(rows)
    spec = DatasetSpec("test", "outcome", "group", "yes", "B")

    first = joint_stratified_split(frame, spec, seed=19)
    second = joint_stratified_split(frame, spec, seed=19)
    first_ids = [set(part["row_id"]) for part in first]
    second_ids = [set(part["row_id"]) for part in second]

    assert first_ids == second_ids
    assert not (first_ids[0] & first_ids[1])
    assert not (first_ids[0] & first_ids[2])
    assert not (first_ids[1] & first_ids[2])
    assert set.union(*first_ids) == set(frame["row_id"])
    for part in first:
        assert len(part.groupby(["group", "outcome"])) == 4


def test_validation_gate_uses_accuracy_without_equalizing_group_accuracy():
    constraint = SimpleNamespace(
        satisfied=True,
        imposed_delta=0.8,
        achieved_ratio=0.8,
    )
    report = SimpleNamespace(
        constraints=[constraint],
        baseline_metrics={"overall": {"accuracy": 0.80}},
        metrics={
            "overall": {
                "accuracy": 0.79,
                "majority_baseline_accuracy": 0.60,
            }
        },
    )
    candidate = Candidate((), 5.0, None, report)
    settings = SimpleNamespace(
        minimum_accuracy=0.0,
        minimum_majority_margin=0.0,
        max_accuracy_drop=0.05,
        ratio_goal="minimum",
        ratio_tolerance=0.0,
    )

    assert rejection_reasons(candidate, settings) == []

    report.metrics["overall"]["accuracy"] = 0.60
    reasons = rejection_reasons(candidate, settings)
    assert any("majority baseline" in reason for reason in reasons)
    assert any("accuracy drop" in reason for reason in reasons)


def test_target_ratio_band_is_inclusive_and_distinct_from_minimum_rule():
    constraint = SimpleNamespace(
        satisfied=False,
        imposed_delta=0.8,
        achieved_ratio=0.8,
    )
    report = SimpleNamespace(
        constraints=[constraint],
        baseline_metrics={"overall": {"accuracy": 0.80}},
        metrics={
            "overall": {
                "accuracy": 0.80,
                "majority_baseline_accuracy": 0.50,
            }
        },
    )
    candidate = Candidate((), 50.0, None, report)
    target = SimpleNamespace(ratio_goal="target", ratio_tolerance=0.01)

    for ratio in (0.79, 0.791, 0.80, 0.809, 0.81):
        constraint.achieved_ratio = ratio
        assert ratio_passes(candidate, target)
    for ratio in (0.789, 0.815, 0.977, np.nan):
        constraint.achieved_ratio = ratio
        assert not ratio_passes(candidate, target)

    minimum = SimpleNamespace(ratio_goal="minimum", ratio_tolerance=0.0)
    constraint.achieved_ratio = 0.977
    assert ratio_passes(candidate, minimum)
    constraint.achieved_ratio = 0.791
    assert not ratio_passes(candidate, minimum)
