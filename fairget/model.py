"""Estimator interface for constrained binary classification."""

import copy
from numbers import Integral, Real

import numpy as np
import torch

from fairget import metrics
from fairget.criteria import _Criterion, EqualOpportunity
from fairget.net import Feedforward
from fairget.optimizer import make_solver
from fairget.preprocessing import Preprocessor
from fairget.report import ConstraintResult, FairnessReport
from fairget.surrogates import THRESHOLD as SURROGATE_THRESHOLD
from fairget.training import train


class FairClassifier:
    """Train matched unconstrained and fairness-constrained probability models."""

    def __init__(
        self,
        constraints,
        *,
        model_factory=None,
        surrogate="smoothed_step",
        hidden_sizes=(),
        alpha=50.0,
        mu=1e-4,
        epochs=500,
        lr=0.5,
        device="cpu",
        satisfied_tol=0.0,
        utility_thresholds=None,
        seed=1,
        verbose=False,
    ):
        if not constraints:
            raise ValueError("Provide at least one fairness constraint.")
        for criterion in constraints:
            if not isinstance(criterion, _Criterion):
                raise TypeError(f"{criterion!r} is not a fairness criterion object.")
        if surrogate not in ("smoothed_step", "sigmoid", "covariance"):
            raise ValueError(f"Unknown surrogate {surrogate!r}.")
        if model_factory is not None and not callable(model_factory):
            raise TypeError("model_factory must be callable.")

        hidden_sizes = tuple(hidden_sizes)
        if any(
            isinstance(size, bool)
            or not isinstance(size, Integral)
            or size < 1
            for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers.")
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if model_factory is not None and hidden_sizes:
            raise ValueError(
                "Choose model_factory or nonempty hidden_sizes, not both."
            )
        for name, value in (("alpha", alpha), ("mu", mu), ("lr", lr)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite numeric value.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if (
            isinstance(satisfied_tol, bool)
            or not isinstance(satisfied_tol, Real)
        ):
            raise TypeError("satisfied_tol must be a finite numeric value.")
        if not np.isfinite(satisfied_tol) or not 0.0 <= satisfied_tol <= 1.0:
            raise ValueError("satisfied_tol must be between 0 and 1.")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer.")
        if utility_thresholds is not None and not isinstance(
            utility_thresholds, dict
        ):
            raise TypeError("utility_thresholds must be a dictionary or None.")
        allowed_utility_thresholds = {
            "max_accuracy_drop": 1.0,
            "minimum_balanced_accuracy": 1.0,
            "near_constant_selection_rate": 0.5,
            "minimum_worst_group_accuracy": 1.0,
        }
        unknown_thresholds = set(utility_thresholds or {}) - set(
            allowed_utility_thresholds
        )
        if unknown_thresholds:
            raise ValueError(
                "Unknown utility threshold(s): "
                + ", ".join(sorted(unknown_thresholds))
            )
        for name, value in (utility_thresholds or {}).items():
            if value is None:
                continue
            upper = allowed_utility_thresholds[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"utility_thresholds[{name!r}] must be numeric or None.")
            if not np.isfinite(value) or not 0.0 <= value <= upper:
                raise ValueError(
                    f"utility_thresholds[{name!r}] must be between 0 and {upper}."
                )

        self.constraints = list(constraints)
        self.model_factory = model_factory
        self.surrogate = surrogate
        self.hidden_sizes = hidden_sizes
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.satisfied_tol = float(satisfied_tol)
        self.utility_thresholds = dict(utility_thresholds or {})
        self.seed = int(seed)
        self.verbose = verbose

        self._model = None
        self._baseline_model = None
        self._pre = None
        self._report = None
        self._solver_name = "sqp"
        self._training_diagnostics = None
        self._baseline_diagnostics = None
        self._architecture = None
        self._evaluation_reports = {}

        # Public fitted attributes follow the conventional trailing-underscore
        # naming used by estimator libraries.
        self.model_ = None
        self.baseline_model_ = None
        self.training_diagnostics_ = None
        self.baseline_diagnostics_ = None
        self.architecture_ = None
        self.evaluation_reports_ = self._evaluation_reports

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _new_model(self, input_size):
        if self.model_factory is None:
            model = Feedforward(input_size, self.hidden_sizes)
            source = "hidden_sizes"
            factory_name = None
        else:
            model = self.model_factory(input_size)
            source = "model_factory"
            factory_name = getattr(
                self.model_factory, "__qualname__",
                type(self.model_factory).__qualname__,
            )
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "model_factory(input_size) must return a torch.nn.Module."
            )
        return model.to(self.device), source, factory_name

    def _validate_model_output(self, model, X):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(X[: min(len(X), 8)].to(self.device))
        model.train(was_training)
        valid_shape = (
            output.ndim == 1
            or (output.ndim == 2 and output.shape[1] == 1)
        )
        if not valid_shape or output.shape[0] != min(len(X), 8):
            raise ValueError(
                "The model must return one probability per input row with "
                "shape (n,) or (n, 1)."
            )
        if not torch.is_floating_point(output):
            raise ValueError("The model output must be floating point.")
        if not torch.isfinite(output).all():
            raise ValueError("The model output contains non-finite values.")
        if torch.any(output < 0) or torch.any(output > 1):
            raise ValueError(
                "The model must return probabilities in [0, 1]; include a "
                "sigmoid output layer."
            )

    @staticmethod
    def _model_parameter_counts(model):
        return {
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameters": int(sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )),
        }

    def _architecture_summary(self, model, source, factory_name, input_size):
        counts = self._model_parameter_counts(model)
        return {
            "source": source,
            "model_class": type(model).__qualname__,
            "model_factory": factory_name,
            "input_features": int(input_size),
            "hidden_sizes": (
                list(self.hidden_sizes) if source == "hidden_sizes" else None
            ),
            **counts,
        }

    def fit(
        self,
        df,
        *,
        target,
        sensitive,
        positive_label=None,
        privileged_group=None,
        drop_columns=None,
        solver="sqp",
    ):
        """Fit constrained and matched unconstrained models on a DataFrame."""
        if (solver or "sqp").lower() != "sqp":
            raise ValueError("The available solver is 'sqp'.")
        self._solver_name = "sqp"
        self._set_seed()

        self._pre = Preprocessor(
            target, sensitive, positive_label, privileged_group, drop_columns
        )
        X_np, y_np, s_np = self._pre.fit_transform(df)
        self._check_groups(y_np, s_np)
        X = torch.from_numpy(np.array(X_np, copy=True)).float()
        y = torch.from_numpy(np.array(y_np, copy=True)).long()
        s = torch.from_numpy(np.array(s_np, copy=True)).long()

        initial_model, source, factory_name = self._new_model(self._pre.n_features)
        self._validate_model_output(initial_model, X)
        self._architecture = self._architecture_summary(
            initial_model, source, factory_name, self._pre.n_features
        )

        # Both runs start from the same parameters. A custom factory is called
        # once per fitted model so its documented fresh-model contract is
        # checked instead of being hidden behind a deep copy.
        if self.model_factory is None:
            self._baseline_model = copy.deepcopy(initial_model)
            self._model = copy.deepcopy(initial_model)
        else:
            second_model, _, _ = self._new_model(self._pre.n_features)
            self._validate_model_output(second_model, X)
            if second_model is initial_model:
                raise ValueError(
                    "model_factory must return a fresh model on every call."
                )
            first_parameter_ids = {id(value) for value in initial_model.parameters()}
            if any(id(value) in first_parameter_ids for value in second_model.parameters()):
                raise ValueError(
                    "Models returned by model_factory must not share parameters."
                )
            try:
                second_model.load_state_dict(copy.deepcopy(initial_model.state_dict()))
            except RuntimeError as error:
                raise ValueError(
                    "model_factory returned inconsistent model architectures."
                ) from error
            self._model = initial_model
            self._baseline_model = second_model

        def solver_factory(**kwargs):
            return make_solver(self._solver_name, **kwargs)

        self._set_seed()
        self._baseline_model, self._baseline_diagnostics = train(
            self._baseline_model, X, y, s, [], solver_factory,
            surrogate=self.surrogate,
            alpha=self.alpha,
            mu=self.mu,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            verbose=self.verbose,
        )
        self._set_seed()
        self._model, self._training_diagnostics = train(
            self._model, X, y, s, self.constraints, solver_factory,
            surrogate=self.surrogate,
            alpha=self.alpha,
            mu=self.mu,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            verbose=self.verbose,
        )

        self.model_ = self._model
        self.baseline_model_ = self._baseline_model
        self.training_diagnostics_ = self._training_diagnostics
        self.baseline_diagnostics_ = self._baseline_diagnostics
        self.architecture_ = self._architecture
        self._evaluation_reports.clear()
        self._report = self._build_report(X, y, s, split_name="training")
        return self

    def _check_groups(self, y, s):
        for group in (0, 1):
            count = int((s == group).sum())
            if count < 2:
                raise ValueError(
                    f"Sensitive group {group} has only {count} samples; "
                    "at least 2 are required."
                )
        if any(c.conditions_on_label for c in self.constraints):
            for group in (0, 1):
                positives = int(((s == group) & (y == 1)).sum())
                if positives < 2:
                    raise ValueError(
                        "EqualOpportunity requires at least 2 positive-label "
                        f"samples in group {group}; found {positives}."
                    )

    def _probabilities_from_tensor(self, model, X):
        model.eval()
        with torch.no_grad():
            return model(X.to(self.device)).reshape(-1).cpu().numpy()

    def _predict_proba_tensor(self, df):
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_np, _, _ = self._pre.transform(df)
        X = torch.from_numpy(np.array(X_np, copy=True)).float()
        return self._probabilities_from_tensor(self._model, X)

    @staticmethod
    def _validated_threshold(threshold):
        if isinstance(threshold, bool):
            raise TypeError("threshold must be a numeric value in [0, 1].")
        try:
            value = float(threshold)
        except (TypeError, ValueError) as error:
            raise TypeError("threshold must be a numeric value in [0, 1].") from error
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return value

    def predict_proba(self, df):
        """Return the fitted constrained model's probability for class 1."""
        return self._predict_proba_tensor(df)

    def predict(self, df, threshold=SURROGATE_THRESHOLD):
        """Return thresholded predictions from the constrained model."""
        threshold = self._validated_threshold(threshold)
        return (self._predict_proba_tensor(df) > threshold).astype(int)

    @staticmethod
    def _compact_training_diagnostics(diagnostics):
        keys = (
            "epochs_requested",
            "epochs_completed",
            "termination_reason",
            "final_loss",
            "final_max_surrogate_violation",
            "final_step_size",
            "stationarity_detected",
            "surrogate_feasible",
            "convergence_status",
        )
        return {key: diagnostics.get(key) for key in keys}

    def _criterion_activity(self, criterion_index):
        diagnostics = self._training_diagnostics
        meta = diagnostics["constraint_meta"]
        row_indices = [
            index
            for index, row_meta in enumerate(meta)
            if row_meta["criterion_index"] == criterion_index
        ]
        active_history = diagnostics["active_history"]
        tail_start = len(active_history) * 2 // 3
        tail = active_history[tail_start:] or active_history
        was_active = any(
            any(row_index in active for row_index in row_indices)
            for active in tail
        )
        active_epochs = sum(
            any(row_index in active for row_index in row_indices)
            for active in active_history
        )
        counts = diagnostics["active_epoch_counts_by_row"]
        active_rows = [
            {
                "row_index": row_index,
                "row": meta[row_index]["row"],
                "active_epochs": int(counts.get(str(row_index), 0)),
            }
            for row_index in row_indices
        ]
        final_violations = diagnostics["final_constraint_violations"]
        surrogate_violation = max(
            (final_violations[index] for index in row_indices),
            default=0.0,
        )
        return was_active, active_epochs, active_rows, surrogate_violation

    def _build_report(
        self, X, y, s, split_name, threshold=SURROGATE_THRESHOLD
    ):
        constrained_proba = self._probabilities_from_tensor(self._model, X)
        baseline_proba = self._probabilities_from_tensor(self._baseline_model, X)
        constrained_pred = (constrained_proba > threshold).astype(int)
        baseline_pred = (baseline_proba > threshold).astype(int)
        y_np = y.cpu().numpy()
        s_np = s.cpu().numpy()

        constrained_metrics = metrics.evaluate_binary_predictions(
            y_np, constrained_pred, s_np, constrained_proba
        )
        baseline_metrics = metrics.evaluate_binary_predictions(
            y_np, baseline_pred, s_np, baseline_proba
        )
        diagnostics = {
            "constrained_training": self._compact_training_diagnostics(
                self._training_diagnostics
            ),
            "baseline_training": self._compact_training_diagnostics(
                self._baseline_diagnostics
            ),
        }
        report_warnings = []
        if not np.isclose(threshold, SURROGATE_THRESHOLD):
            report_warnings.append({
                "code": "evaluation_threshold_differs_from_surrogate",
                "section": "evaluation",
                "message": (
                    f"This report uses decision threshold {threshold:.6g}; the "
                    f"training surrogate was centered at {SURROGATE_THRESHOLD:.6g}. "
                    "Select thresholds on validation data and recheck empirical "
                    "fairness on untouched test data."
                ),
            })
        report = FairnessReport(
            metrics=constrained_metrics,
            baseline_metrics=baseline_metrics,
            preprocessing=self._pre.summary(),
            split_name=split_name,
            tolerance=self.satisfied_tol,
            threshold=threshold,
            architecture=self._architecture,
            diagnostics=diagnostics,
            utility_thresholds=self.utility_thresholds,
            surrogate=self.surrogate,
            alpha=self.alpha,
            mu=self.mu,
            solver=self._solver_name,
            surrogate_threshold=SURROGATE_THRESHOLD,
            warnings=report_warnings,
        )

        for criterion_index, spec in enumerate(self.constraints):
            if isinstance(spec, EqualOpportunity):
                achieved_ratio = metrics.equal_opportunity_ratio(
                    y_np, constrained_pred, s_np
                )
                achieved_gap = metrics.equal_opportunity_gap(
                    y_np, constrained_pred, s_np
                )
                baseline_ratio = metrics.equal_opportunity_ratio(
                    y_np, baseline_pred, s_np
                )
                baseline_gap = metrics.equal_opportunity_gap(
                    y_np, baseline_pred, s_np
                )
                metric_name = "true-positive-rate gap"
            else:
                achieved_ratio = metrics.disparate_impact_ratio(
                    constrained_pred, s_np
                )
                achieved_gap = metrics.demographic_parity_gap(
                    constrained_pred, s_np
                )
                baseline_ratio = metrics.disparate_impact_ratio(
                    baseline_pred, s_np
                )
                baseline_gap = metrics.demographic_parity_gap(
                    baseline_pred, s_np
                )
                metric_name = "selection-rate gap"

            was_active, active_epochs, active_rows, surrogate_violation = (
                self._criterion_activity(criterion_index)
            )
            satisfied = bool(
                np.isfinite(achieved_ratio)
                and achieved_ratio >= spec.delta - self.satisfied_tol
            )
            report.add(ConstraintResult(
                criterion=spec.name,
                imposed_delta=spec.delta,
                achieved_ratio=achieved_ratio,
                achieved_gap=achieved_gap,
                was_active=was_active,
                satisfied=satisfied,
                tolerance=self.satisfied_tol,
                split_name=split_name,
                metric_name=metric_name,
                active_epochs=active_epochs,
                total_epochs=self._training_diagnostics["epochs_completed"],
                active_rows=active_rows,
                surrogate_violation=surrogate_violation,
                baseline_ratio=baseline_ratio,
                baseline_gap=baseline_gap,
            ))
        return report

    def fairness_report(self):
        """Return the report calculated on the training data during fit."""
        if self._report is None:
            raise RuntimeError("Call fit() before fairness_report().")
        return self._report

    def evaluate(
        self,
        df,
        split_name="evaluation",
        threshold=SURROGATE_THRESHOLD,
    ):
        """Evaluate constrained and baseline models on labeled held-out data."""
        if self._model is None:
            raise RuntimeError("Call fit() before evaluate().")
        X_np, y_np, s_np = self._pre.transform(df)
        missing = []
        if y_np is None:
            missing.append(self._pre.target)
        if s_np is None:
            missing.append(self._pre.sensitive)
        if missing:
            raise ValueError(
                "evaluate() requires these fitted columns: " + ", ".join(missing)
            )
        if len(y_np) == 0:
            raise ValueError("evaluate() requires at least one row.")
        threshold = self._validated_threshold(threshold)
        X = torch.from_numpy(np.array(X_np, copy=True)).float()
        y = torch.from_numpy(np.array(y_np, copy=True)).long()
        s = torch.from_numpy(np.array(s_np, copy=True)).long()
        report = self._build_report(
            X,
            y,
            s,
            split_name=str(split_name),
            threshold=threshold,
        )
        self._evaluation_reports[str(split_name)] = report
        return report
