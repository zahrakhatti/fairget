# fairget

[![tests](https://github.com/zahrakhatti/fairget/actions/workflows/tests.yml/badge.svg)](https://github.com/zahrakhatti/fairget/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`fairget` trains binary PyTorch classifiers with differentiable fairness
constraints and evaluates the result on thresholded predictions. Each fit also
trains a matched unconstrained baseline from the same initialization. The report
therefore shows both sides of the decision: whether the requested statistical
constraint was met and how much predictive utility changed.

The package currently supports demographic parity, disparate impact, and equal
opportunity for one binary sensitive attribute. Meeting one of these criteria
does not establish that a model is fair in every sense, suitable for deployment,
or legally compliant.

## Requirements

- Python 3.9 or newer
- NumPy 1.23 or newer
- pandas 1.5 or newer
- PyTorch 2.0 or newer

These dependencies are declared in `pyproject.toml` and installed automatically
by `pip`. A separate `requirements.txt` is intentionally not needed for normal
package installation.

## Installation

Install the tagged release directly from GitHub:

```bash
python -m pip install "fairget @ https://github.com/zahrakhatti/fairget/archive/refs/tags/v0.1.0.zip"
```

`fairget` is not on PyPI yet, so `pip install fairget` is not the supported
installation command. For editable development:

```bash
git clone https://github.com/zahrakhatti/fairget.git
cd fairget
python -m pip install -e '.[test]'
python -m pytest
```

## Basic use

Split data before fitting. The training report is useful for optimization
diagnostics; the held-out report is the appropriate report for model comparison.

```python
import pandas as pd

from fairget import DemographicParity, FairClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

classifier = FairClassifier(
    constraints=[DemographicParity(delta=0.8)],
    hidden_sizes=(),       # logistic regression
    alpha=5.0,
    epochs=500,
    satisfied_tol=0.0,    # strict comparison with the requested minimum
    seed=1,
)
classifier.fit(
    train_df,
    target="hired",
    sensitive="group",
    positive_label="yes",
    privileged_group="B",
)

print(classifier.fairness_report())
test_report = classifier.evaluate(test_df, split_name="test")
print(test_report)

predictions = classifier.predict(test_df.drop(columns="hired"))
probabilities = classifier.predict_proba(test_df.drop(columns="hired"))
```

The main public interface is deliberately small:

| Object | Purpose |
|---|---|
| `FairClassifier` | Fit, predict, and compare constrained and matched-baseline models |
| `DemographicParity` | Constrain the ratio of positive-prediction rates |
| `DisparateImpact` | Alias for the demographic-parity ratio constraint |
| `EqualOpportunity` | Constrain the ratio of true-positive rates |
| `FairnessReport` | Inspect or serialize fairness and utility results |

`fit` trains the constrained model and a matched unconstrained baseline. The
baseline starts from an exact copy of the constrained model's initialization, so
the presence of the fairness constraints is the intended difference between the
two runs. `fairness_report()` evaluates the training rows; `evaluate(...)`
applies the same preprocessing and decision threshold to labeled held-out rows.

From a cloned repository, run the complete synthetic example with:

```bash
python -m examples.quickstart
```

Its data includes a model feature correlated with the sensitive group. That is
important: a label may differ by group while an unconstrained classifier still
has similar selection rates if none of its input features carries group-related
information.

`examples/dataset_benchmark.py` provides validation-first runs for Law School,
COMPAS, and Dutch Census CSV files. The datasets are not redistributed with the
package. Supply the architecture candidates yourself; `logistic` means no
hidden layer, `8` means one eight-unit layer, and `16,8` means two layers:

```bash
python -m examples.dataset_benchmark law \
  --data /path/to/law.csv --architectures logistic 8 --alphas 2 10 50 \
  --epochs 1000 --lr 2 --ratio-goal target --ratio-tolerance 0.01

python -m examples.dataset_benchmark compas \
  --data /path/to/compas-scores-two-years.csv \
  --architectures logistic 8 --alphas 2 5 10 20 50 \
  --ratio-goal target --ratio-tolerance 0.01

python -m examples.dataset_benchmark dutch \
  --data /path/to/dutch_census_2001.csv --sample-size 6000 \
  --architectures logistic --alphas 2 5 10 20 50 --epochs 600 --lr 0.5 \
  --ratio-goal target --ratio-tolerance 0.01
```

Each run creates group-and-target-stratified training, validation, and test
splits. The commands above use an experimental ratio-selection target of
`0.800 +/- 0.010`, so ratios from `0.790` through `0.810` pass that selection
gate. This does not redefine demographic parity: constrained training still
enforces the standard minimum-ratio inequalities. A ratio above `0.810` is a
miss for this experimental target, not a demographic-parity failure.
The reported ratio is symmetric—smaller group rate divided by larger group
rate—so this band does not specify which group has the lower selection rate. A
directional protected-to-reference ratio would be a different metric.

Candidates must also beat the majority-class rule by more than the configured
margin (one percentage point by default) and stay within the configured
ordinary accuracy loss on validation. These are model-selection gates, not
training constraints, and they do not make group accuracies equal. The passing
candidate with the highest ordinary validation accuracy is selected. If none
passes, the script reports the closest rejected candidate and leaves the test
set untouched. `--evaluate-rejected` permits an explicitly labelled diagnostic
test, but it does not convert a validation failure into an accepted model.

With the commands above and model seed 1/split seed 19, one reference run gave:

| Dataset | Validation decision | Test accuracy, baseline → constrained | Test DP ratio, baseline → constrained | Final decision |
|---|---|---:|---:|---|
| Law School | no acceptable candidate | not evaluated by this protocol | not evaluated by this protocol | — |
| Dutch Census, 6,000 rows | logistic, alpha 50 selected at 0.791 | 78.05% → 73.62% | 0.748 → 0.763 | test target failed |
| COMPAS | no acceptable candidate | not evaluated by this protocol | not evaluated by this protocol | — |

On Dutch Census, alpha 50 correctly passed validation because `0.791` is only
`0.009` from the requested target. After that configuration was frozen, however,
its one allowed test evaluation produced `0.763`, outside the target band, and
the command returned a failure status. It must not be retuned against that test
set. On COMPAS no candidate met both the target band and utility gates. On Law
School, the hidden `(8,)` model reached 89.76% validation accuracy and a 0.875 DP
ratio, but the majority-class rule already reached 88.99%; it missed both the
target band and the default one-percentage-point utility margin. These are
finite-sample integration results, not claims of population fairness or model
suitability; repeat model selection across documented seeds and use a new final
test set after any revision.

## Reading the report

The report keeps fairness and utility conclusions separate. It compares the
baseline and constrained model overall and by group, including counts, label
prevalence, predicted-positive rates, ordinary accuracy, majority-class
baselines, and confusion-rate diagnostics. Balanced accuracy and ROC AUC are
also reported as diagnostics; neither is an optimization objective or fairness
constraint. The report separately gives the requested minimum ratio,
matched-baseline ratio, constrained ratio, their change, margin, violation, and
any tolerance used in the decision.

For example:

```text
requested minimum ratio: 0.800
observed ratio:          0.977
margin:                 +0.177
```

The requested value is a lower bound, not a value the optimizer tries to match.
An observed ratio of `0.977` therefore satisfies a `0.800` requirement; it does
not indicate an error or imply that `alpha` should be increased.

For an experiment that deliberately selects ratios close to `0.800`, use
`--ratio-goal target --ratio-tolerance 0.01` in the dataset benchmark. Under
that separate selection rule, `0.791` is inside the band and `0.977` is outside
it. The latter remains a standard demographic-parity pass; it is only a miss for
the experimental target-calibration goal. Alpha changes the smooth surrogate's
steepness; it does not directly pull the empirical ratio toward `0.800`.

Accuracy near 50% is not, by itself, proof of random guessing. Its meaning
depends on label prevalence. The report compares accuracy with a majority-class
baseline and balanced accuracy, and warns about behavior such as predicting one
class almost everywhere. A model can have useful probability rankings while a
fixed threshold produces poor hard classifications, so the underlying rates
should be inspected rather than relying on one headline number.

A fairness target can pass while utility becomes unacceptable. Applications
should define utility requirements before model selection, such as a maximum
accuracy loss or a minimum worst-group balanced accuracy. If no candidate meets
both the fairness and utility requirements on validation data, the correct
conclusion is that no acceptable candidate was found—not that a collapsed model
is an adequate solution.

Warning thresholds can be configured on the classifier, for example:

```python
classifier = FairClassifier(
    constraints=[DemographicParity(delta=0.8)],
    utility_thresholds={
        "max_accuracy_drop": 0.03,
        "minimum_balanced_accuracy": 0.60,
        "minimum_worst_group_accuracy": 0.65,
    },
)
```

Reports are also available as structured data:

```python
result = classifier.evaluate(test_df, split_name="test")
result_dict = result.to_dict()
result_json = result.to_json()
```

## Choosing the model architecture

Architecture is a user-controlled modeling decision. `hidden_sizes=()` is the
default logistic model; tuples such as `(16,)` or `(32, 16)` select the built-in
feed-forward network. A custom probability-producing PyTorch model can be
supplied with `model_factory`:

```python
import torch


def make_model(input_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 16),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid(),
    )


classifier = FairClassifier(
    constraints=[DemographicParity(delta=0.8)],
    model_factory=make_model,
)
```

The factory must accept the engineered input size and return a fresh
`torch.nn.Module`. Its output must contain one probability in `[0, 1]` per row,
with shape `(n,)` or `(n, 1)`. Do not combine `model_factory` with nonempty
`hidden_sizes`.

More hidden layers are not automatically better. Extra capacity can capture
nonlinear signal, but it can also learn sensitive-group proxies more accurately,
overfit small groups, or find a nearly constant solution to a fairness
constraint. Compare architectures without constraints first, then select the
architecture and fairness hyperparameters together on validation data. Keep the
test set untouched until one configuration has been selected, and repeat the
comparison across several random seeds when results are variable.

## Criteria and surrogate

The criterion specifies the rate relationship to constrain:

- `DemographicParity(delta=...)` constrains the ratio of positive-prediction
  rates between the two groups.
- `DisparateImpact(delta=...)` is an alias with the same pair of constraints.
- `EqualOpportunity(delta=...)` constrains the ratio of true-positive rates.

These criteria do not constrain overall accuracy or require the two groups to
have equal accuracy. Accuracy is measured before and after constrained training
to expose the utility change; it is not made equal across groups.

For the two-sided ratio criteria, the package constrains both directions:

```text
delta * rate_group0 <= rate_group1
delta * rate_group1 <= rate_group0
```

`"smoothed_step"` is the default surrogate. It replaces the hard decision at
probability `0.5` with the package's smooth-step approximation during
optimization. The report still evaluates the empirical criterion on hard
predictions.

`alpha` controls the steepness of the surrogate. A larger value is a sharper
approximation, but its effect on optimization and held-out fairness is not
monotonic. Choose it on validation data over a documented grid rather than
increasing it automatically whenever a run misses its target. Training duration,
learning rate, architecture, seed, and threshold can also affect the observed
result.

The training surrogate is centered at probability `0.5`. You may inspect a
different empirical decision threshold on validation data with
`classifier.evaluate(validation_df, threshold=...)`. The resulting report is
labelled when its threshold differs from the training surrogate. Freeze the
chosen threshold before the final test evaluation; do not select it by
maximizing test results.

`satisfied_tol=0.0` requires the observed ratio to meet the requested minimum
strictly. A positive tolerance relaxes that decision to
`observed >= requested - tolerance`; the report displays the tolerance and
effective minimum so that a below-target result is not silently presented as a
strict pass.

## Data and evaluation requirements

- The target must have exactly two observed classes.
- Exactly one binary sensitive column is supported.
- Pass `positive_label` and `privileged_group` explicitly. If omitted, the
  current preprocessor uses a lexical default, which may not match the intended
  semantics.
- Numeric features are scaled using training-set ranges; categorical features
  are one-hot encoded using training-set categories.
- Evaluation rows containing categorical levels not observed during fitting are
  rejected instead of being silently assigned an ambiguous encoding.
- The sensitive column is used to construct and evaluate constraints but is
  excluded from the prediction features. Other features can still act as
  proxies for it.
- `evaluate` requires target and sensitive columns. `predict` and
  `predict_proba` require only the feature columns used during fitting.
- Fairness and utility should be assessed on held-out data, with group sample
  sizes and uncertainty considered before deployment.

## Current scope

- Binary classification and one binary sensitive attribute only.
- Full-batch training with an active-set SQP subproblem solved in constraint
  space. Large networks are still expensive, and active-set enumeration grows
  rapidly as simultaneous constraint rows are added.
- A `0.5` training-surrogate and default prediction threshold. Different hard
  thresholds can be passed to `predict` and `evaluate`; threshold selection is
  not automated.
- The package evaluates statistical group criteria. It does not provide causal,
  individual-fairness, multiclass, regression, calibration, or legal analyses.
- The built-in SQP solver is the supported solver backend in this release.

## License

MIT. See `LICENSE`.
