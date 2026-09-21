"""End-to-end and unit tests for fairget."""

import numpy as np
import pandas as pd
import pytest

from fairget import (
    DemographicParity,
    DisparateImpact,
    EqualOpportunity,
    FairClassifier,
)
from fairget.preprocessing import Preprocessor


def make_biased_dataset(n=2000, seed=0):
    """Synthetic dataset with a deliberate correlation between the sensitive
    attribute and the label, so fairness constraints have something to fix."""
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Label correlated with sex (the bias) plus signal from features.
    logit = 1.2 * sex + 0.8 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    y = (logit > 0.5).astype(int)
    return pd.DataFrame({
        "x1": x1, "x2": x2,
        "sex": np.where(sex == 1, "M", "F"),
        "income": np.where(y == 1, ">50K", "<=50K"),
    })


# --- preprocessing tests -----------------------------------------------------

def test_preprocessor_binarizes_and_scales():
    df = make_biased_dataset(n=200)
    pre = Preprocessor(target="income", sensitive="sex")
    X, y, s = pre.fit_transform(df)
    assert set(np.unique(y)) <= {0, 1}
    assert set(np.unique(s)) <= {0, 1}
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert X.shape[0] == len(df)


def test_preprocessor_rejects_multiclass_sensitive():
    df = make_biased_dataset(n=100)
    df.loc[df.index[:10], "sex"] = "X"  # third category
    pre = Preprocessor(target="income", sensitive="sex")
    with pytest.raises(ValueError, match="binary"):
        pre.fit_transform(df)


def test_preprocessor_rejects_multiclass_target():
    df = make_biased_dataset(n=100)
    df.loc[df.index[:10], "income"] = "MID"
    pre = Preprocessor(target="income", sensitive="sex")
    with pytest.raises(ValueError, match="exactly 2 classes"):
        pre.fit_transform(df)


def test_preprocessor_rejects_multiple_sensitive_columns():
    df = make_biased_dataset(n=100)
    with pytest.raises(ValueError, match="one binary sensitive"):
        Preprocessor(target="income", sensitive=["sex", "x1"])


# --- criteria tests ----------------------------------------------------------

def test_delta_validation():
    with pytest.raises(ValueError):
        DemographicParity(delta=1.5)
    with pytest.raises(ValueError):
        DemographicParity(delta=0.0)
    DemographicParity(delta=0.8)  # ok


# --- end-to-end tests --------------------------------------------------------

def test_fit_predict_demographic_parity():
    df = make_biased_dataset(n=1500)
    clf = FairClassifier(
        constraints=[DemographicParity(delta=0.8)],
        hidden_sizes=(), epochs=150, alpha=50.0,
    )
    clf.fit(df, target="income", sensitive="sex")
    preds = clf.predict(df)
    assert set(np.unique(preds)) <= {0, 1}
    assert len(preds) == len(df)

    report = clf.fairness_report()
    assert len(report.constraints) == 1
    c = report.constraints[0]
    assert 0.0 <= c.achieved_ratio <= 1.0
    # The report exposes the requested and empirically observed values.
    assert c.imposed_delta == 0.8


def test_report_is_honest_about_achieved_fairness():
    """The reported ratio matches an independent empirical calculation."""
    from fairget import metrics

    df = make_biased_dataset(n=1500)
    clf = FairClassifier(constraints=[DemographicParity(delta=0.8)],
                         epochs=200, alpha=50.0)
    clf.fit(df, target="income", sensitive="sex")

    reported = clf.fairness_report().constraints[0].achieved_ratio

    # Independently recompute from raw predictions.
    preds = clf.predict(df)
    # Reconstruct the binarized sensitive column the same way the preprocessor did.
    s = (df["sex"] == clf._pre.sensitive_privileged_).astype(int).to_numpy()
    independent = metrics.disparate_impact_ratio(preds, s)

    assert abs(reported - independent) < 1e-9, (
        "Reported achieved fairness does not match the model's actual predictions."
    )


def test_satisfied_flag_matches_target():
    """The satisfaction flag applies the documented evaluation tolerance."""
    df = make_biased_dataset(n=1500)
    clf = FairClassifier(constraints=[DemographicParity(delta=0.8)],
                         epochs=200, alpha=50.0)
    clf.fit(df, target="income", sensitive="sex")
    c = clf.fairness_report().constraints[0]
    expected = c.achieved_ratio >= c.imposed_delta - clf.satisfied_tol
    assert c.satisfied == expected


def test_multiple_constraints():
    df = make_biased_dataset(n=1500)
    clf = FairClassifier(
        constraints=[DemographicParity(delta=0.8), EqualOpportunity(delta=0.8)],
        epochs=150,
    )
    clf.fit(df, target="income", sensitive="sex")
    report = clf.fairness_report()
    assert len(report.constraints) == 2
    names = {c.criterion for c in report.constraints}
    assert names == {"DemographicParity", "EqualOpportunity"}


def test_sigmoid_surrogate_runs():
    df = make_biased_dataset(n=800)
    clf = FairClassifier(constraints=[DisparateImpact(delta=0.8)],
                         surrogate="sigmoid", epochs=100)
    clf.fit(df, target="income", sensitive="sex")
    assert clf.fairness_report() is not None


def test_report_dict_serializable():
    df = make_biased_dataset(n=600)
    clf = FairClassifier(constraints=[DemographicParity(delta=0.8)], epochs=80)
    clf.fit(df, target="income", sensitive="sex")
    d = clf.fairness_report().to_dict()
    assert "constraints" in d and "overall_accuracy" in d
    import json
    json.dumps(d)  # must be JSON-serializable


def test_predict_before_fit_raises():
    clf = FairClassifier(constraints=[DemographicParity(delta=0.8)])
    with pytest.raises(RuntimeError):
        clf.predict(make_biased_dataset(n=10))
