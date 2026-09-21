"""Structured evaluation reports for constrained binary classifiers."""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, List, Optional


DEFAULT_UTILITY_THRESHOLDS = {
    "max_accuracy_drop": 0.05,
    "minimum_balanced_accuracy": 0.50,
    "near_constant_selection_rate": 0.01,
    "minimum_worst_group_accuracy": None,
}


def _is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value):
    """Convert nested values to strict JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metric(metrics, section, name, default=float("nan")):
    if not metrics:
        return default
    values = metrics.get(section, {})
    return values.get(name, default) if isinstance(values, dict) else default


def _format_number(value, digits=3):
    return f"{float(value):.{digits}f}" if _is_finite(value) else "undefined"


def _format_percent(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}%" if _is_finite(value) else "undefined"


def _format_points(value, digits=2):
    return (
        f"{100.0 * float(value):+.{digits}f} pp"
        if _is_finite(value) else "undefined"
    )


@dataclass
class ReportWarning:
    """One actionable qualification attached to an evaluation report."""

    code: str
    message: str
    section: str = "utility"
    severity: str = "warning"
    group: Optional[str] = None
    criterion: Optional[str] = None

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "section": self.section,
            "group": self.group,
            "criterion": self.criterion,
            "message": self.message,
        }


@dataclass
class ConstraintResult:
    criterion: str
    imposed_delta: float
    achieved_ratio: float
    achieved_gap: float
    was_active: Optional[bool]
    satisfied: bool
    tolerance: float = 0.0
    split_name: Optional[str] = None
    metric_name: Optional[str] = None
    active_epochs: Optional[int] = None
    total_epochs: Optional[int] = None
    active_rows: Optional[Any] = None
    surrogate_violation: Optional[float] = None
    baseline_ratio: Optional[float] = None
    baseline_gap: Optional[float] = None

    def __post_init__(self):
        # An undefined empirical ratio can never establish satisfaction.
        if not _is_finite(self.achieved_ratio):
            self.satisfied = False

    @property
    def requested_minimum(self):
        return self.imposed_delta

    @property
    def observed_ratio(self):
        return self.achieved_ratio

    @property
    def margin(self):
        if not _is_finite(self.achieved_ratio):
            return float("nan")
        return float(self.achieved_ratio - self.imposed_delta)

    @property
    def violation(self):
        if not _is_finite(self.achieved_ratio):
            return float("nan")
        return float(max(self.imposed_delta - self.achieved_ratio, 0.0))

    @property
    def ratio_change(self):
        """Return constrained minus matched-baseline empirical ratio."""
        if not (
            _is_finite(self.achieved_ratio)
            and _is_finite(self.baseline_ratio)
        ):
            return float("nan")
        return float(self.achieved_ratio - self.baseline_ratio)

    @property
    def effective_minimum(self):
        return float(max(self.imposed_delta - self.tolerance, 0.0))

    @property
    def status(self):
        if not _is_finite(self.achieved_ratio):
            return "NOT EVALUABLE"
        if self.achieved_ratio >= self.imposed_delta:
            return "SATISFIED"
        if self.satisfied:
            return "SATISFIED WITHIN TOLERANCE"
        return "NOT SATISFIED"

    def to_dict(self):
        return _json_safe({
            # Existing names are retained for callers using the 0.1 API.
            "criterion": self.criterion,
            "imposed_delta": self.imposed_delta,
            "achieved_ratio": self.achieved_ratio,
            "achieved_gap": self.achieved_gap,
            "was_active": self.was_active,
            "satisfied": self.satisfied,
            # Explicit evaluation terminology.
            "requested_minimum": self.requested_minimum,
            "observed_ratio": self.observed_ratio,
            "baseline_observed_ratio": self.baseline_ratio,
            "observed_ratio_change": self.ratio_change,
            "baseline_difference": self.baseline_gap,
            "margin": self.margin,
            "violation": self.violation,
            "tolerance": self.tolerance,
            "effective_minimum": self.effective_minimum,
            "status": self.status,
            "split_name": self.split_name,
            "difference_metric": self.metric_name,
            "active_epochs": self.active_epochs,
            "total_epochs": self.total_epochs,
            "active_rows": self.active_rows,
            "surrogate_violation": self.surrogate_violation,
        })

    def __str__(self):
        difference_name = self.metric_name or "rate difference"
        activity = "not available"
        if self.was_active is True:
            activity = "observed during optimization"
        elif self.was_active is False:
            activity = "not observed in the monitored epochs"
        lines = [
            self.criterion,
            f"  requested minimum ratio: >= {self.imposed_delta:.3f}",
            f"  matched-baseline ratio: {_format_number(self.baseline_ratio)}",
            f"  constrained ratio: {_format_number(self.achieved_ratio)}",
            f"  ratio change: {_format_number(self.ratio_change)}",
            (
                f"  matched-baseline {difference_name}: "
                f"{_format_number(self.baseline_gap, 4)}"
            ),
            f"  constrained {difference_name}: {_format_number(self.achieved_gap, 4)}",
            f"  margin relative to requested minimum: {_format_number(self.margin)}",
            f"  violation relative to requested minimum: {_format_number(self.violation)}",
            (
                f"  evaluation tolerance: {self.tolerance:.3f} "
                f"(effective pass threshold >= {self.effective_minimum:.3f})"
            ),
            f"  status: {self.status}",
            f"  solver activity: {activity}",
        ]
        if self.active_epochs is not None:
            denominator = "" if self.total_epochs is None else f"/{self.total_epochs}"
            lines.append(f"  active epochs: {self.active_epochs}{denominator}")
        if self.surrogate_violation is not None:
            lines.append(
                "  final surrogate violation: "
                f"{_format_number(self.surrogate_violation, 6)}"
            )
        return "\n".join(lines)


@dataclass
class FairnessReport:
    # The first five fields retain the original constructor and attributes.
    overall_accuracy: float = float("nan")
    accuracy_group0: float = float("nan")
    accuracy_group1: float = float("nan")
    constraints: list = field(default_factory=list)
    preprocessing: Optional[dict] = None
    split_name: str = "training"
    metrics: Optional[dict] = None
    baseline_metrics: Optional[dict] = None
    tolerance: float = 0.0
    threshold: float = 0.5
    architecture: Optional[Any] = None
    diagnostics: Optional[dict] = None
    warnings: List[Any] = field(default_factory=list)
    utility_thresholds: Optional[dict] = None
    surrogate: Optional[str] = None
    alpha: Optional[float] = None
    mu: Optional[float] = None
    solver: Optional[str] = None
    surrogate_threshold: Optional[float] = None

    def __post_init__(self):
        if self.metrics:
            self.overall_accuracy = _metric(self.metrics, "overall", "accuracy")
            groups = self.metrics.get("groups", {})
            self.accuracy_group0 = groups.get("0", {}).get("accuracy", float("nan"))
            self.accuracy_group1 = groups.get("1", {}).get("accuracy", float("nan"))
        self._normalize_warnings()
        self._add_utility_warnings()
        for result in self.constraints:
            self._add_constraint_warnings(result)

    @property
    def split(self):
        """Alias retained for callers that use ``report.split``."""
        return self.split_name

    @property
    def model_metrics(self):
        """Explicit alias for the constrained model's metrics."""
        return self.metrics

    def add(self, result: ConstraintResult):
        self.constraints.append(result)
        self._add_constraint_warnings(result)

    @property
    def all_satisfied(self):
        return all(result.satisfied for result in self.constraints)

    @property
    def fairness_status(self):
        if not self.constraints:
            return "NOT REQUESTED"
        if any(not _is_finite(result.achieved_ratio) for result in self.constraints):
            return "NOT EVALUABLE"
        return "PASSED" if self.all_satisfied else "FAILED"

    @property
    def utility_status(self):
        warning_sections = {"utility", "model_behavior", "data_quality"}
        return (
            "WARNING"
            if any(self._warning_dict(item).get("section") in warning_sections
                   for item in self.warnings)
            else "NO BUILT-IN WARNING"
        )

    @property
    def metric_changes(self):
        """Return constrained-minus-baseline changes on the same split."""
        if not self.metrics or not self.baseline_metrics:
            return None
        names = (
            "accuracy", "balanced_accuracy", "selection_rate", "precision",
            "tpr", "tnr", "fpr", "fnr", "brier_score", "log_loss",
            "roc_auc", "average_precision",
        )

        def changes(current, baseline):
            result = {}
            for name in names:
                current_value = current.get(name, float("nan"))
                baseline_value = baseline.get(name, float("nan"))
                result[name] = (
                    float(current_value - baseline_value)
                    if _is_finite(current_value) and _is_finite(baseline_value)
                    else float("nan")
                )
            return result

        output = {
            "overall": changes(
                self.metrics.get("overall", {}),
                self.baseline_metrics.get("overall", {}),
            ),
            "groups": {},
        }
        group_names = set(self.metrics.get("groups", {})) | set(
            self.baseline_metrics.get("groups", {})
        )
        for group_name in sorted(group_names):
            output["groups"][group_name] = changes(
                self.metrics.get("groups", {}).get(group_name, {}),
                self.baseline_metrics.get("groups", {}).get(group_name, {}),
            )
        return output

    @property
    def effective_utility_thresholds(self):
        """Return default warning thresholds with user overrides applied."""
        settings = dict(DEFAULT_UTILITY_THRESHOLDS)
        if self.utility_thresholds:
            settings.update(self.utility_thresholds)
        return settings

    def _warning_dict(self, warning):
        if isinstance(warning, ReportWarning):
            return warning.to_dict()
        if isinstance(warning, dict):
            result = dict(warning)
            result.setdefault("severity", "warning")
            result.setdefault("section", "utility")
            return result
        return {
            "code": "user_warning",
            "severity": "warning",
            "section": "utility",
            "group": None,
            "criterion": None,
            "message": str(warning),
        }

    def _normalize_warnings(self):
        self.warnings = [
            warning if isinstance(warning, ReportWarning)
            else ReportWarning(**self._warning_dict(warning))
            for warning in self.warnings
        ]

    def _add_warning(self, warning):
        candidate = self._warning_dict(warning)
        identity = (
            candidate.get("code"), candidate.get("group"),
            candidate.get("criterion"), candidate.get("message"),
        )
        existing = {
            (item.get("code"), item.get("group"), item.get("criterion"), item.get("message"))
            for item in (self._warning_dict(value) for value in self.warnings)
        }
        if identity not in existing:
            self.warnings.append(warning)

    def _add_utility_warnings(self):
        if not self.metrics:
            return
        settings = self.effective_utility_thresholds

        scopes = [("overall", self.metrics.get("overall", {}), None)]
        scopes.extend(
            (f"group {name}", values, str(name))
            for name, values in self.metrics.get("groups", {}).items()
        )
        for label, values, group_name in scopes:
            n = values.get("n", 0)
            if n == 0:
                self._add_warning(ReportWarning(
                    code="empty_evaluation_group",
                    section="data_quality",
                    group=group_name,
                    message=f"No observations are available for {label} on {self.split_name}.",
                ))
                continue
            predicted_positives = values.get("predicted_positives")
            selection_rate = values.get("selection_rate")
            if predicted_positives in (0, n):
                predicted_class = 0 if predicted_positives == 0 else 1
                self._add_warning(ReportWarning(
                    code="constant_predictions",
                    section="model_behavior",
                    group=group_name,
                    message=(
                        f"The constrained model predicts only class {predicted_class} for "
                        f"{label} on {self.split_name}; ratio-based fairness may be undefined "
                        "or uninformative."
                    ),
                ))
            else:
                boundary = settings["near_constant_selection_rate"]
                if (boundary is not None and _is_finite(selection_rate)
                        and (selection_rate <= boundary or selection_rate >= 1.0 - boundary)):
                    self._add_warning(ReportWarning(
                        code="near_constant_predictions",
                        section="model_behavior",
                        group=group_name,
                        message=(
                            f"The constrained selection rate for {label} is "
                            f"{_format_percent(selection_rate)}, close to a constant prediction."
                        ),
                    ))

            observed_accuracy = values.get("accuracy")
            majority = values.get("majority_baseline_accuracy")
            if (_is_finite(observed_accuracy) and _is_finite(majority)
                    and observed_accuracy + 1e-12 < majority):
                self._add_warning(ReportWarning(
                    code="accuracy_below_majority_baseline",
                    group=group_name,
                    message=(
                        f"Constrained accuracy for {label} ({_format_percent(observed_accuracy)}) "
                        f"is below its majority-class baseline ({_format_percent(majority)})."
                    ),
                ))

            minimum_balanced = settings["minimum_balanced_accuracy"]
            balanced = values.get("balanced_accuracy")
            if (minimum_balanced is not None and _is_finite(balanced)
                    and balanced <= minimum_balanced):
                self._add_warning(ReportWarning(
                    code="low_balanced_accuracy",
                    group=group_name,
                    message=(
                        f"Balanced accuracy for {label} is {_format_percent(balanced)}, "
                        f"at or below the configured {_format_percent(minimum_balanced)} threshold."
                    ),
                ))

        if self.baseline_metrics:
            maximum_drop = settings["max_accuracy_drop"]
            if maximum_drop is not None:
                comparison_scopes = [("overall", None)] + [
                    (f"group {name}", str(name))
                    for name in self.metrics.get("groups", {})
                ]
                for label, group_name in comparison_scopes:
                    if group_name is None:
                        current = self.metrics.get("overall", {}).get("accuracy")
                        baseline = self.baseline_metrics.get("overall", {}).get("accuracy")
                    else:
                        current = self.metrics.get("groups", {}).get(group_name, {}).get("accuracy")
                        baseline = self.baseline_metrics.get("groups", {}).get(group_name, {}).get("accuracy")
                    drop = baseline - current if _is_finite(baseline) and _is_finite(current) else None
                    if drop is not None and drop > maximum_drop:
                        self._add_warning(ReportWarning(
                            code="accuracy_drop_exceeds_threshold",
                            group=group_name,
                            message=(
                                f"Accuracy for {label} decreased by "
                                f"{_format_number(100.0 * drop, 2)} percentage points "
                                f"relative to the matched baseline; the configured warning "
                                f"threshold is {_format_number(100.0 * maximum_drop, 2)} "
                                "percentage points."
                            ),
                        ))

        minimum_worst = settings["minimum_worst_group_accuracy"]
        worst = self.metrics.get("worst_group_accuracy")
        if minimum_worst is not None and _is_finite(worst) and worst < minimum_worst:
            self._add_warning(ReportWarning(
                code="worst_group_accuracy_below_threshold",
                message=(
                    f"Worst-group accuracy is {_format_percent(worst)}, below the configured "
                    f"minimum of {_format_percent(minimum_worst)}."
                ),
            ))

    def _add_constraint_warnings(self, result):
        if not _is_finite(result.achieved_ratio):
            self._add_warning(ReportWarning(
                code="undefined_fairness_ratio",
                section="fairness",
                criterion=result.criterion,
                message=(
                    f"{result.criterion} cannot be evaluated on {self.split_name}: its "
                    "empirical rate ratio is undefined."
                ),
            ))
        elif not result.satisfied:
            self._add_warning(ReportWarning(
                code="fairness_target_not_met",
                section="fairness",
                criterion=result.criterion,
                message=(
                    f"{result.criterion} did not meet the requested minimum on "
                    f"{self.split_name}; observed {_format_number(result.achieved_ratio)} "
                    f"versus requested {result.imposed_delta:.3f}."
                ),
            ))

    def _utility_lines(self):
        if not self.metrics:
            return [
                f"Overall accuracy: {_format_percent(self.overall_accuracy)}",
                (
                    f"Group 0 accuracy: {_format_percent(self.accuracy_group0)}; "
                    f"group 1 accuracy: {_format_percent(self.accuracy_group1)}"
                ),
            ]

        lines = [
            "UTILITY AND MODEL BEHAVIOR",
            "  Ordinary accuracy is the primary utility comparison below.",
            "  Balanced accuracy and ROC AUC are diagnostics only; they are not",
            "  optimized, constrained, or compared between sensitive groups.",
        ]
        if self.baseline_metrics:
            lines.extend([
                "  scope     metric                 baseline  constrained  change (pp)",
                "  --------  ---------------------  --------  -----------  -----------",
            ])
            changes = self.metric_changes
            scopes = [("overall", "overall")]
            scopes.extend((f"group {name}", name) for name in self.metrics.get("groups", {}))
            for scope_label, group_name in scopes:
                current = (
                    self.metrics.get("overall", {}) if group_name == "overall"
                    else self.metrics.get("groups", {}).get(group_name, {})
                )
                baseline = (
                    self.baseline_metrics.get("overall", {}) if group_name == "overall"
                    else self.baseline_metrics.get("groups", {}).get(group_name, {})
                )
                scope_changes = (
                    changes.get("overall", {}) if group_name == "overall"
                    else changes.get("groups", {}).get(group_name, {})
                )
                for metric_name, display_name in (
                    ("accuracy", "accuracy"),
                    ("balanced_accuracy", "balanced acc. (diag.)"),
                    ("selection_rate", "selection rate"),
                    ("roc_auc", "ROC AUC"),
                ):
                    if metric_name not in current and metric_name not in baseline:
                        continue
                    lines.append(
                        f"  {scope_label:<8}  {display_name:<21}  "
                        f"{_format_percent(baseline.get(metric_name)):<8}  "
                        f"{_format_percent(current.get(metric_name)):<11}  "
                        f"{_format_points(scope_changes.get(metric_name))}"
                    )
        else:
            overall = self.metrics.get("overall", {})
            lines.append(
                f"  overall: n={overall.get('n', 0)}, "
                f"accuracy={_format_percent(overall.get('accuracy'))}, "
                "balanced accuracy (diagnostic only)="
                f"{_format_percent(overall.get('balanced_accuracy'))}, "
                f"selection rate={_format_percent(overall.get('selection_rate'))}"
            )

        lines.append("  detailed constrained metrics by scope:")
        scopes = [("overall", self.metrics.get("overall", {}))]
        scopes.extend(
            (f"group {name}", values)
            for name, values in self.metrics.get("groups", {}).items()
        )
        for label, values in scopes:
            lines.append(
                f"  {label}: n={values.get('n', 0)}, "
                f"prevalence={_format_percent(values.get('prevalence'))}, "
                f"selection={_format_percent(values.get('selection_rate'))}, "
                f"majority baseline={_format_percent(values.get('majority_baseline_accuracy'))}, "
                f"accuracy={_format_percent(values.get('accuracy'))}, "
                "balanced accuracy (diagnostic only)="
                f"{_format_percent(values.get('balanced_accuracy'))}"
            )
            lines.append(
                f"    TP={values.get('tp', 0)} TN={values.get('tn', 0)} "
                f"FP={values.get('fp', 0)} FN={values.get('fn', 0)}; "
                f"TPR={_format_percent(values.get('tpr'))}, "
                f"TNR={_format_percent(values.get('tnr'))}, "
                f"FPR={_format_percent(values.get('fpr'))}, "
                f"FNR={_format_percent(values.get('fnr'))}, "
                f"precision={_format_percent(values.get('precision'))}"
            )
        return lines

    def __str__(self):
        lines = [
            "=" * 78,
            "fairget evaluation report",
            "=" * 78,
            f"Evaluation split: {self.split_name}",
            f"Decision threshold: {self.threshold:.3f}",
        ]
        group_encoding = (
            self.preprocessing.get("sensitive_group_encoding", {})
            if self.preprocessing else {}
        )
        if group_encoding:
            lines.append(
                "Sensitive-group encoding: "
                f"0={group_encoding.get('0')!r}; 1={group_encoding.get('1')!r}"
            )
        if self.surrogate is not None:
            settings = f"Surrogate: {self.surrogate}"
            if self.alpha is not None:
                settings += f"; alpha={self.alpha:g}"
            if self.mu is not None:
                settings += f"; mu={self.mu:g}"
            if self.solver is not None:
                settings += f"; solver={self.solver}"
            if self.surrogate_threshold is not None:
                settings += f"; training threshold={self.surrogate_threshold:g}"
            lines.append(settings)
        if self.architecture:
            if self.architecture.get("source") == "model_factory":
                architecture_name = (
                    f"model_factory={self.architecture.get('model_factory')}"
                )
            else:
                architecture_name = (
                    f"hidden_sizes={self.architecture.get('hidden_sizes')}"
                )
            lines.append(
                f"Architecture: {architecture_name}; "
                f"input features={self.architecture.get('input_features')}; "
                f"trainable parameters={self.architecture.get('trainable_parameters')}"
            )
        lines.extend([
            f"Fairness status: {self.fairness_status}",
            f"Utility status: {self.utility_status}",
            "-" * 78,
        ])
        lines.extend(self._utility_lines())
        lines.extend(["-" * 78, "FAIRNESS CONSTRAINTS"])
        if self.constraints:
            for result in self.constraints:
                lines.append(str(result))
        else:
            lines.append("  No fairness constraints were evaluated.")
        lines.extend(["-" * 78, "WARNINGS"])
        if self.warnings:
            for warning in self.warnings:
                item = self._warning_dict(warning)
                lines.append(f"  [{item.get('code')}] {item.get('message')}")
        else:
            lines.append("  No built-in warning threshold was crossed.")
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_dict(self):
        return _json_safe({
            "schema_version": "0.2",
            "split_name": self.split_name,
            "decision_threshold": self.threshold,
            "fairness_status": self.fairness_status,
            "utility_status": self.utility_status,
            # Existing summary fields.
            "overall_accuracy": self.overall_accuracy,
            "accuracy_group0": self.accuracy_group0,
            "accuracy_group1": self.accuracy_group1,
            "all_satisfied": self.all_satisfied,
            # Full matched evaluation.
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "metric_changes": self.metric_changes,
            "constraints": [result.to_dict() for result in self.constraints],
            "warnings": [self._warning_dict(item) for item in self.warnings],
            "tolerance": self.tolerance,
            "surrogate": self.surrogate,
            "alpha": self.alpha,
            "mu": self.mu,
            "solver": self.solver,
            "surrogate_threshold": self.surrogate_threshold,
            "architecture": self.architecture,
            "utility_thresholds": self.effective_utility_thresholds,
            "diagnostics": self.diagnostics,
            "preprocessing": self.preprocessing,
        })

    def to_json(self, *, indent=2):
        """Serialize the report as standards-compliant JSON."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)
