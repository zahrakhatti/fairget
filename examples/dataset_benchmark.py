"""Validation-first benchmarks for Law School, COMPAS, and Dutch Census data.

The datasets are not distributed with fairget. Run this module from the
package root and pass a local CSV, for example:

    python -m examples.dataset_benchmark compas --data /path/to/compas.csv \
        --architectures logistic 8 --alphas 2 5 10 20 50 \
        --ratio-goal target --ratio-tolerance 0.01

Architecture and alpha candidates are selected using validation data. The
benchmark can apply either the standard minimum-ratio rule or an experimental
target band around ``delta``. The test split is evaluated once only after a
candidate meets the chosen ratio rule and the configured ordinary-accuracy
gates on validation. Use ``--evaluate-rejected`` only when an explicit
diagnostic test of the closest rejected candidate is intended.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fairget import (
    DemographicParity,
    DisparateImpact,
    EqualOpportunity,
    FairClassifier,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target: str
    sensitive: str
    positive_label: object
    privileged_group: object


@dataclass
class Candidate:
    architecture: tuple
    alpha: float
    classifier: FairClassifier
    report: object

    @property
    def result(self):
        return self.report.constraints[0]

    @property
    def accuracy(self):
        return self.report.metrics["overall"]["accuracy"]


def _require_columns(frame, columns, dataset_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} CSV is missing required columns: {missing}"
        )


def load_law(path):
    frame = pd.read_csv(path)
    columns = [
        "decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa",
        "fulltime", "fam_inc", "male", "tier", "race", "pass_bar",
    ]
    _require_columns(frame, columns, "Law School")
    frame = frame[columns].dropna().reset_index(drop=True)
    if set(frame["race"].unique()) != {"White", "Non-White"}:
        raise ValueError(
            "Law School race must contain exactly 'White' and 'Non-White'."
        )
    frame["pass_bar"] = frame["pass_bar"].astype(int)
    return frame, DatasetSpec(
        name="Law School",
        target="pass_bar",
        sensitive="race",
        positive_label=1,
        privileged_group="White",
    )


def load_compas(path):
    frame = pd.read_csv(path)
    filter_columns = [
        "days_b_screening_arrest", "is_recid", "c_charge_degree",
        "score_text", "race",
    ]
    model_columns = [
        "priors_count", "age_cat", "c_charge_degree", "sex", "race",
        "two_year_recid",
    ]
    _require_columns(frame, filter_columns + model_columns, "COMPAS")
    frame = frame[
        frame["days_b_screening_arrest"].between(-30, 30)
        & (frame["is_recid"] != -1)
        & (frame["c_charge_degree"] != "O")
        & (frame["score_text"] != "NA")
        & frame["race"].isin(["African-American", "Caucasian"])
    ]
    frame = frame[model_columns].dropna().reset_index(drop=True)
    frame["two_year_recid"] = frame["two_year_recid"].astype(int)
    return frame, DatasetSpec(
        name="COMPAS",
        target="two_year_recid",
        sensitive="race",
        positive_label=1,
        privileged_group="Caucasian",
    )


def load_dutch(path):
    frame = pd.read_csv(path)
    columns = [
        "sex", "age", "household_position", "household_size",
        "prev_residence_place", "citizenship", "country_birth", "edu_level",
        "economic_status", "cur_eco_activity", "Marital_status", "occupation",
    ]
    _require_columns(frame, columns, "Dutch Census")
    frame = frame[columns].dropna().reset_index(drop=True)
    if set(frame["sex"].unique()) != {1, 2}:
        raise ValueError("Dutch Census sex must contain exactly codes 1 and 2.")
    if set(frame["occupation"].unique()) != {"2_1", "5_4_9"}:
        raise ValueError(
            "Dutch Census occupation must contain exactly '2_1' and '5_4_9'."
        )

    frame["sex_group"] = frame["sex"].map({1: "Male", 2: "Female"})
    frame = frame.drop(columns="sex")
    categorical_codes = [
        "household_position", "household_size", "prev_residence_place",
        "citizenship", "country_birth", "economic_status",
        "cur_eco_activity", "Marital_status",
    ]
    for column in categorical_codes:
        frame[column] = frame[column].astype(str)
    return frame, DatasetSpec(
        name="Dutch Census",
        target="occupation",
        sensitive="sex_group",
        positive_label="5_4_9",
        privileged_group="Male",
    )


LOADERS = {
    "law": load_law,
    "compas": load_compas,
    "dutch": load_dutch,
}


def joint_stratified_sample(frame, spec, sample_size, seed):
    """Take an approximately proportional sample of every group-label stratum."""
    if not sample_size or sample_size >= len(frame):
        return frame.reset_index(drop=True)
    strata = list(frame.groupby([spec.sensitive, spec.target], sort=True).indices.items())
    if any(len(indices) < 5 for _, indices in strata):
        raise ValueError(
            "Every group-label stratum must contain at least five rows."
        )
    if sample_size < 5 * len(strata):
        raise ValueError(
            "sample_size must leave at least five rows per group-label stratum."
        )
    raw_sizes = np.array([len(indices) for _, indices in strata], dtype=float)
    quotas = np.floor(sample_size * raw_sizes / raw_sizes.sum()).astype(int)
    quotas = np.maximum(quotas, 5)
    while quotas.sum() > sample_size:
        eligible = np.where(quotas > 5)[0]
        quotas[eligible[np.argmax(quotas[eligible])]] -= 1
    remainders = sample_size * raw_sizes / raw_sizes.sum() - np.floor(
        sample_size * raw_sizes / raw_sizes.sum()
    )
    while quotas.sum() < sample_size:
        eligible = np.where(quotas < raw_sizes)[0]
        index = eligible[np.argmax(remainders[eligible])]
        quotas[index] += 1
        remainders[index] = -1.0

    rng = np.random.default_rng(seed)
    selected = []
    for (_, indices), quota in zip(strata, quotas):
        indices = np.asarray(indices, dtype=int).copy()
        rng.shuffle(indices)
        selected.extend(indices[:quota])
    rng.shuffle(selected)
    return frame.iloc[selected].reset_index(drop=True)


def joint_stratified_split(frame, spec, seed):
    """Create deterministic 60/20/20 splits within each group-label stratum."""
    rng = np.random.default_rng(seed)
    train_indices, validation_indices, test_indices = [], [], []
    grouped = frame.groupby([spec.sensitive, spec.target], sort=True).indices
    for key, values in grouped.items():
        indices = np.asarray(values, dtype=int).copy()
        if len(indices) < 5:
            raise ValueError(
                f"Stratum {key!r} has only {len(indices)} rows; at least five are required."
            )
        rng.shuffle(indices)
        n_train = max(1, int(round(0.60 * len(indices))))
        n_validation = max(1, int(round(0.20 * len(indices))))
        if n_train + n_validation >= len(indices):
            n_train = len(indices) - n_validation - 1
        train_indices.extend(indices[:n_train])
        validation_indices.extend(indices[n_train:n_train + n_validation])
        test_indices.extend(indices[n_train + n_validation:])
    for indices in (train_indices, validation_indices, test_indices):
        rng.shuffle(indices)
    return tuple(
        frame.iloc[indices].reset_index(drop=True)
        for indices in (train_indices, validation_indices, test_indices)
    )


def parse_architecture(value):
    text = value.strip().lower()
    if text in {"logistic", "none", "()"}:
        return ()
    try:
        sizes = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use 'logistic' or comma-separated positive sizes such as '16,8'"
        ) from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("hidden-layer sizes must be positive integers")
    return sizes


def architecture_name(sizes):
    return "logistic" if not sizes else "hidden=" + ",".join(map(str, sizes))


def make_criterion(name, delta):
    criteria = {
        "demographic_parity": DemographicParity,
        "disparate_impact": DisparateImpact,
        "equal_opportunity": EqualOpportunity,
    }
    return criteria[name](delta=delta)


def finite_number(value, digits=3):
    return "undefined" if not np.isfinite(value) else f"{value:.{digits}f}"


def percent(value):
    return "undefined" if not np.isfinite(value) else f"{100 * value:.2f}%"


def percentage_points(value):
    return "undefined" if not np.isfinite(value) else f"{100 * value:+.2f} pp"


def point_magnitude(value):
    return "undefined" if not np.isfinite(value) else f"{100 * value:.2f} pp"


def fit_candidate(args, spec, train, validation, architecture, alpha):
    classifier = FairClassifier(
        constraints=[make_criterion(args.criterion, args.delta)],
        hidden_sizes=architecture,
        surrogate="smoothed_step",
        alpha=alpha,
        epochs=args.epochs,
        lr=args.lr,
        satisfied_tol=args.ratio_tolerance,
        seed=args.model_seed,
    )
    classifier.fit(
        train,
        target=spec.target,
        sensitive=spec.sensitive,
        positive_label=spec.positive_label,
        privileged_group=spec.privileged_group,
    )
    report = classifier.evaluate(validation, split_name="validation")
    return Candidate(architecture, alpha, classifier, report)


def ratio_bounds(candidate, args):
    delta = candidate.result.imposed_delta
    lower = max(delta - args.ratio_tolerance, 0.0)
    upper = (
        min(delta + args.ratio_tolerance, 1.0)
        if args.ratio_goal == "target" else 1.0
    )
    return lower, upper


def ratio_passes(candidate, args):
    ratio = candidate.result.achieved_ratio
    if not np.isfinite(ratio):
        return False
    lower, upper = ratio_bounds(candidate, args)
    return lower - 1e-12 <= ratio <= upper + 1e-12


def ratio_violation(candidate, args):
    ratio = candidate.result.achieved_ratio
    if not np.isfinite(ratio):
        return float("inf")
    lower, upper = ratio_bounds(candidate, args)
    return max(lower - ratio, ratio - upper, 0.0)


def candidate_sort_key(candidate, args):
    violation = ratio_violation(candidate, args)
    accuracy = candidate.accuracy
    return (violation, -accuracy if np.isfinite(accuracy) else float("inf"))


def rejection_reasons(candidate, args):
    """Return failed validation gates; none of these gates changes training."""
    report = candidate.report
    baseline_accuracy = report.baseline_metrics["overall"]["accuracy"]
    majority = report.metrics["overall"]["majority_baseline_accuracy"]
    reasons = []
    if not ratio_passes(candidate, args):
        lower, upper = ratio_bounds(candidate, args)
        if args.ratio_goal == "target":
            reasons.append(
                "ratio is outside the experimental target band "
                f"[{lower:.3f}, {upper:.3f}]"
            )
        else:
            reasons.append(f"ratio is below the minimum {lower:.3f}")
    if not np.isfinite(candidate.accuracy):
        reasons.append("ordinary accuracy is undefined")
        return reasons
    if candidate.accuracy < args.minimum_accuracy:
        reasons.append(
            f"accuracy is below {percent(args.minimum_accuracy)}"
        )
    if (
        np.isfinite(majority)
        and candidate.accuracy
        <= majority + args.minimum_majority_margin
    ):
        reasons.append(
            "accuracy does not exceed the majority baseline by more than "
            f"{point_magnitude(args.minimum_majority_margin)}"
        )
    if (
        np.isfinite(baseline_accuracy)
        and baseline_accuracy - candidate.accuracy > args.max_accuracy_drop
    ):
        reasons.append(
            "accuracy drop exceeds "
            f"{point_magnitude(args.max_accuracy_drop)}"
        )
    return reasons


def print_validation_table(candidates, args):
    print("\nValidation candidates")
    print(
        "  architecture       alpha  base acc  fair acc       change  "
        "base ratio  fair ratio  validation"
    )
    for candidate in candidates:
        report = candidate.report
        baseline_accuracy = report.baseline_metrics["overall"]["accuracy"]
        change = report.metric_changes["overall"]["accuracy"]
        result = candidate.result
        print(
            f"  {architecture_name(candidate.architecture):<18} "
            f"{candidate.alpha:>5g}  {percent(baseline_accuracy):>8}  "
            f"{percent(candidate.accuracy):>8}  {percentage_points(change):>11}  "
            f"{finite_number(result.baseline_ratio):>10}  "
            f"{finite_number(result.achieved_ratio):>10}  "
            f"{'PASS' if not rejection_reasons(candidate, args) else 'REJECT'}"
        )


def print_final_report(candidate, report, accepted, args):
    evaluated_candidate = Candidate(
        candidate.architecture,
        candidate.alpha,
        candidate.classifier,
        report,
    )
    result = report.constraints[0]
    baseline = report.baseline_metrics["overall"]
    constrained = report.metrics["overall"]
    changes = report.metric_changes["overall"]
    label = (
        "validation-selected candidate"
        if accepted else "rejected diagnostic fallback"
    )
    print(f"\nTest result ({label})")
    print(
        f"  architecture={architecture_name(candidate.architecture)}, "
        f"alpha={candidate.alpha:g}"
    )
    print(
        f"  matched baseline: accuracy={percent(baseline['accuracy'])}, "
        f"fairness ratio={finite_number(result.baseline_ratio)}"
    )
    print(
        f"  constrained:      accuracy={percent(constrained['accuracy'])}, "
        f"fairness ratio={finite_number(result.achieved_ratio)}, "
        f"standard minimum-rule status={result.status}"
    )
    lower, upper = ratio_bounds(evaluated_candidate, args)
    if args.ratio_goal == "target":
        goal = f"target band=[{lower:.3f}, {upper:.3f}]"
    else:
        goal = f"minimum ratio={lower:.3f}"
    print(
        f"  benchmark ratio goal: {goal}; "
        f"status={'PASS' if ratio_passes(evaluated_candidate, args) else 'FAIL'}"
    )
    print(
        f"  change: accuracy={percentage_points(changes['accuracy'])}, "
        f"fairness ratio={finite_number(result.ratio_change)}"
    )
    print(
        "  diagnostics only: "
        f"majority baseline={percent(constrained['majority_baseline_accuracy'])}, "
        f"balanced accuracy={percent(constrained['balanced_accuracy'])}, "
        f"ROC AUC={finite_number(constrained['roc_auc'])}"
    )
    print(
        f"  constraint activity: {result.active_epochs}/{result.total_epochs} epochs; "
        f"final surrogate violation={finite_number(result.surrogate_violation, 6)}"
    )
    if report.warnings:
        print("  warnings:")
        for warning in report.warnings:
            print(f"    [{warning.code}] {warning.message}")


def run(args):
    frame, spec = LOADERS[args.dataset](args.data)
    loaded_rows = len(frame)
    frame = joint_stratified_sample(
        frame, spec, args.sample_size, args.split_seed
    )
    train, validation, test = joint_stratified_split(
        frame, spec, args.split_seed
    )
    print(
        f"{spec.name}: usable={loaded_rows:,}, sampled={len(frame):,}, "
        f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}"
    )
    print(
        f"target={spec.target!r}, positive={spec.positive_label!r}; "
        f"sensitive={spec.sensitive!r}, privileged={spec.privileged_group!r}"
    )
    print(
        f"criterion={args.criterion}, training constraint=minimum ratio "
        f">= {args.delta:.3f}, surrogate=smoothed_step"
    )
    if args.ratio_goal == "target":
        print(
            "benchmark selection goal=experimental target ratio "
            f"{args.delta:.3f} +/- {args.ratio_tolerance:.3f}; this does not "
            "change the one-sided training constraint"
        )
    else:
        print(
            "benchmark selection goal=minimum ratio "
            f">= {max(args.delta - args.ratio_tolerance, 0.0):.3f}"
        )

    candidates = []
    for architecture in args.architectures:
        for alpha in args.alphas:
            candidates.append(
                fit_candidate(
                    args, spec, train, validation, architecture, alpha
                )
            )
    print_validation_table(candidates, args)

    accepted = [
        candidate for candidate in candidates
        if not rejection_reasons(candidate, args)
    ]
    if accepted:
        selected = max(
            accepted,
            key=lambda candidate: (
                candidate.accuracy,
                -abs(candidate.result.achieved_ratio - candidate.result.imposed_delta),
                -candidate.alpha,
            ),
        )
        print(
            "\nSelected on validation: "
            f"{architecture_name(selected.architecture)}, alpha={selected.alpha:g}; "
            "it passed the configured ratio-selection rule and ordinary-accuracy "
            "gates, and had the highest validation accuracy among passing candidates."
        )
        test_report = selected.classifier.evaluate(test, split_name="test")
        print_final_report(selected, test_report, accepted=True, args=args)
        test_candidate = Candidate(
            selected.architecture,
            selected.alpha,
            selected.classifier,
            test_report,
        )
        test_rejections = rejection_reasons(test_candidate, args)
        if test_rejections:
            print(
                "  final test assessment: FAILED. Do not retune on this test set; "
                "report the failure and collect a new test set after revising the model."
            )
            for reason in test_rejections:
                print(f"    test failure: {reason}")
            return 3
        print("  final test assessment: PASSED the configured empirical gates.")
        return 0

    selected = min(candidates, key=lambda candidate: candidate_sort_key(candidate, args))
    print(
        "\nNo candidate met every validation ratio-selection and utility requirement. "
        f"Closest rejected candidate: {architecture_name(selected.architecture)}, "
        f"alpha={selected.alpha:g}, ratio={finite_number(selected.result.achieved_ratio)}."
    )
    for reason in rejection_reasons(selected, args):
        print(f"  rejection reason: {reason}")
    if not args.evaluate_rejected:
        print(
            "The test split was not evaluated. Expand or revise the validation "
            "grid before using the test set."
        )
        return 2

    print(
        "--evaluate-rejected was supplied. The following test result is "
        "diagnostic and does not turn the rejected candidate into an accepted one."
    )
    test_report = selected.classifier.evaluate(test, split_name="test")
    print_final_report(selected, test_report, accepted=False, args=args)
    return 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(LOADERS))
    parser.add_argument("--data", type=Path, required=True, help="path to the dataset CSV")
    parser.add_argument(
        "--criterion",
        choices=("demographic_parity", "disparate_impact", "equal_opportunity"),
        default="demographic_parity",
    )
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument(
        "--ratio-goal",
        choices=("minimum", "target"),
        default="minimum",
        help="minimum is standard DP; target is an experimental selection band",
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.0,
        help="one-sided shortfall for minimum, or target-band half-width",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        type=parse_architecture,
        default=[()],
        metavar="LAYERS",
        help="user-supplied candidates: logistic, 8, 16,8, and so on",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[2, 5, 10, 20, 50])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.05,
        help="largest allowed baseline-minus-constrained accuracy loss on validation",
    )
    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=0.0,
        help="minimum ordinary validation accuracy",
    )
    parser.add_argument(
        "--minimum-majority-margin",
        type=float,
        default=0.01,
        help="required validation accuracy margin above the majority-class baseline",
    )
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=19)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="jointly stratified row limit; 0 uses every usable row",
    )
    parser.add_argument(
        "--evaluate-rejected",
        action="store_true",
        help="evaluate the closest validation failure on test for diagnosis only",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"dataset CSV not found: {args.data}")
    if not 0 < args.delta <= 1:
        parser.error("--delta must be in (0, 1]")
    if args.sample_size < 0:
        parser.error("--sample-size cannot be negative")
    if not args.alphas or any(alpha <= 0 for alpha in args.alphas):
        parser.error("--alphas must contain positive values")
    for option in (
        "ratio_tolerance", "max_accuracy_drop", "minimum_accuracy",
        "minimum_majority_margin",
    ):
        value = getattr(args, option)
        if not 0 <= value <= 1:
            parser.error(f"--{option.replace('_', '-')} must be between 0 and 1")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
