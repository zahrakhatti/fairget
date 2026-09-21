"""Tests for matched reporting, held-out evaluation, and custom models."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from fairget import DemographicParity, FairClassifier
from fairget import metrics
from fairget.surrogates import _smoothed_step, constraint_values


def make_proxy_dataset(n=900, seed=13):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)
    skill = rng.normal(size=n)
    proxy = group + rng.normal(scale=0.30, size=n)
    latent = 2.5 * skill + 3.0 * group - 1.5 + rng.normal(scale=0.60, size=n)
    outcome = (latent > 0).astype(int)
    return pd.DataFrame({
        "skill": skill,
        "proxy": proxy,
        "group": np.where(group == 1, "B", "A"),
        "outcome": np.where(outcome == 1, "yes", "no"),
    })


@pytest.fixture(scope="module")
def fitted_proxy_classifier():
    data = make_proxy_dataset()
    train = data.iloc[:650].reset_index(drop=True)
    test = data.iloc[650:].reset_index(drop=True)
    classifier = FairClassifier(
        constraints=[DemographicParity(delta=0.8)],
        alpha=5.0,
        epochs=180,
        seed=3,
    )
    classifier.fit(
        train,
        target="outcome",
        sensitive="group",
        positive_label="yes",
        privileged_group="B",
    )
    return classifier, test


def test_report_compares_baseline_and_constrained(fitted_proxy_classifier):
    classifier, _ = fitted_proxy_classifier
    report = classifier.fairness_report()

    assert report.baseline_metrics is not None
    assert "roc_auc" in report.metrics["overall"]
    assert report.surrogate == "smoothed_step"
    assert report.alpha == 5.0
    assert report.architecture["source"] == "hidden_sizes"

    expected_change = (
        report.metrics["overall"]["accuracy"]
        - report.baseline_metrics["overall"]["accuracy"]
    )
    assert report.metric_changes["overall"]["accuracy"] == pytest.approx(
        expected_change
    )

    constraint = report.constraints[0]
    assert np.isfinite(constraint.baseline_ratio)
    assert constraint.ratio_change == pytest.approx(
        constraint.achieved_ratio - constraint.baseline_ratio
    )
    parsed = json.loads(report.to_json())
    assert parsed["surrogate"] == "smoothed_step"
    assert parsed["constraints"][0]["baseline_observed_ratio"] is not None


def test_held_out_and_single_group_evaluation(fitted_proxy_classifier):
    classifier, test = fitted_proxy_classifier
    held_out = classifier.evaluate(test, split_name="validation")
    assert held_out.split_name == "validation"
    assert held_out.metrics["overall"]["n"] == len(test)
    assert classifier.evaluation_reports_["validation"] is held_out

    alternate = classifier.evaluate(
        test,
        split_name="validation-threshold-0.4",
        threshold=0.4,
    )
    assert alternate.threshold == 0.4
    alternate_codes = {warning.code for warning in alternate.warnings}
    assert "evaluation_threshold_differs_from_surrogate" in alternate_codes

    only_group_a = test[test["group"] == "A"].reset_index(drop=True)
    single_group = classifier.evaluate(only_group_a, split_name="group-a-only")
    assert np.isnan(single_group.constraints[0].achieved_ratio)
    warning_codes = {warning.code for warning in single_group.warnings}
    assert "empty_evaluation_group" in warning_codes
    assert "undefined_fairness_ratio" in warning_codes
    assert single_group.to_dict()["constraints"][0]["observed_ratio"] is None


def test_custom_model_factory_is_called_for_each_model():
    calls = []

    def model_factory(input_size):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid(),
        )
        calls.append(model)
        return model

    classifier = FairClassifier(
        constraints=[DemographicParity(delta=0.8)],
        model_factory=model_factory,
        epochs=5,
    )
    classifier.fit(
        make_proxy_dataset(n=250),
        target="outcome",
        sensitive="group",
        positive_label="yes",
        privileged_group="B",
    )

    assert len(calls) == 2
    assert calls[0] is not calls[1]
    assert classifier.architecture_["source"] == "model_factory"
    assert classifier.architecture_["hidden_sizes"] is None


def test_invalid_architecture_and_utility_configuration_are_rejected():
    def model_factory(input_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    with pytest.raises(ValueError, match="not both"):
        FairClassifier(
            constraints=[DemographicParity()],
            model_factory=model_factory,
            hidden_sizes=(4,),
        )
    with pytest.raises(ValueError, match="Unknown utility threshold"):
        FairClassifier(
            constraints=[DemographicParity()],
            utility_thresholds={"accuracy_typo": 0.5},
        )
    with pytest.raises(ValueError, match="between 0 and 0.5"):
        FairClassifier(
            constraints=[DemographicParity()],
            utility_thresholds={"near_constant_selection_rate": 0.8},
        )

    def wrong_output_shape(input_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, 2),
            torch.nn.Sigmoid(),
        )

    invalid_model = FairClassifier(
        constraints=[DemographicParity()],
        model_factory=wrong_output_shape,
        epochs=1,
    )
    with pytest.raises(ValueError, match="one probability per input row"):
        invalid_model.fit(
            make_proxy_dataset(n=100),
            target="outcome",
            sensitive="group",
            positive_label="yes",
            privileged_group="B",
        )


def test_undefined_zero_over_zero_ratio_is_not_reported_as_parity():
    prediction = np.zeros(6, dtype=int)
    group = np.array([0, 0, 0, 1, 1, 1])
    assert np.isnan(metrics.disparate_impact_ratio(prediction, group))


def test_smoothed_step_is_the_default_surrogate():
    classifier = FairClassifier(constraints=[DemographicParity()])
    assert classifier.surrogate == "smoothed_step"


def test_demographic_parity_rows_use_the_smoothed_step_rates():
    class IdentityProbability(torch.nn.Module):
        def forward(self, values):
            return values[:, 0]

    probabilities = torch.tensor([[0.2], [0.6], [0.4], [0.8]])
    groups = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 1, 0, 1])
    alpha = 5.0
    mu = 1e-4

    first, second = constraint_values(
        IdentityProbability(),
        probabilities,
        groups,
        labels,
        criterion_delta=0.8,
        conditions_on_label=False,
        surrogate="smoothed_step",
        alpha=alpha,
        mu=mu,
    )
    smooth = _smoothed_step(alpha * (probabilities[:, 0] - 0.5), mu)
    rate0 = smooth[groups == 0].mean()
    rate1 = smooth[groups == 1].mean()

    torch.testing.assert_close(first, 0.8 * rate0 - rate1)
    torch.testing.assert_close(second, 0.8 * rate1 - rate0)


def test_balanced_accuracy_setting_changes_warnings_not_training():
    data = make_proxy_dataset(n=250)
    common = {
        "constraints": [DemographicParity(delta=0.8)],
        "epochs": 5,
        "seed": 7,
    }
    with_warning = FairClassifier(
        **common,
        utility_thresholds={"minimum_balanced_accuracy": 0.95},
    )
    without_warning = FairClassifier(
        **common,
        utility_thresholds={"minimum_balanced_accuracy": None},
    )
    for classifier in (with_warning, without_warning):
        classifier.fit(
            data,
            target="outcome",
            sensitive="group",
            positive_label="yes",
            privileged_group="B",
        )

    for first, second in zip(
        with_warning.model_.parameters(), without_warning.model_.parameters()
    ):
        assert torch.equal(first, second)
