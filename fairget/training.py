"""Full-batch constrained training and training diagnostics."""

from collections import Counter

import torch

from fairget.surrogates import constraint_values


NO_BOUND = 1e18
BOUND_THRESHOLD = 1e12


def _trainable_parameters(model):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("The model has no trainable parameters.")
    return params


def _flat_grad(loss, parameters, device):
    """Return a flat gradient, with zeros for parameters unused by the loss."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    pieces = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces).to(device)


def _jacobian(c_list, parameters, n_params, device):
    """Return row-stacked gradients of the constraint values."""
    rows = []
    for value in c_list:
        gradients = torch.autograd.grad(
            value, parameters, retain_graph=True, allow_unused=True
        )
        pieces = [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
        rows.append(torch.cat(pieces))
    if not rows:
        return torch.zeros(0, n_params, device=device)
    return torch.stack(rows).to(device)


def _apply_step(parameters, direction, step_size):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            width = parameter.numel()
            parameter.add_(
                direction[offset:offset + width].reshape(parameter.shape),
                alpha=step_size,
            )
            offset += width


def _bound_violations(values, lower, upper):
    """Return nonnegative violations under the solver's bound convention."""
    violations = []
    for index, value in enumerate(values):
        terms = [torch.zeros((), device=value.device, dtype=value.dtype)]
        if upper[index] < BOUND_THRESHOLD:
            terms.append(value - upper[index])
        if lower[index] < BOUND_THRESHOLD:
            terms.append(-lower[index] - value)
        violations.append(torch.stack(terms).max())
    if not violations:
        return torch.empty(0, device=values.device)
    return torch.stack(violations)


def build_constraints(model, X, y, s, specs, surrogate, alpha, mu):
    """Evaluate constraint rows and return bounds and source metadata."""
    c_values = []
    lower, upper = [], []
    meta = []

    for criterion_index, spec in enumerate(specs):
        c1, c2 = constraint_values(
            model, X, s, y,
            criterion_delta=spec.delta,
            conditions_on_label=spec.conditions_on_label,
            surrogate=surrogate,
            alpha=alpha,
            mu=mu,
        )
        if c2 is None:
            epsilon = 1.0 - spec.delta
            c_values.append(c1)
            lower.append(epsilon)
            upper.append(epsilon)
            meta.append({
                "criterion_index": criterion_index,
                "criterion": spec.name,
                "row": "covariance",
            })
        else:
            c_values.extend((c1, c2))
            lower.extend((NO_BOUND, NO_BOUND))
            upper.extend((0.0, 0.0))
            meta.extend((
                {
                    "criterion_index": criterion_index,
                    "criterion": spec.name,
                    "row": "delta_rate_group0_minus_rate_group1",
                },
                {
                    "criterion_index": criterion_index,
                    "criterion": spec.name,
                    "row": "delta_rate_group1_minus_rate_group0",
                },
            ))

    values = (
        torch.stack(c_values)
        if c_values
        else torch.empty(0, device=X.device, dtype=X.dtype)
    )
    return values, lower, upper, meta


def _finish_diagnostics(
    model, X, y, s, specs, surrogate, alpha, mu,
    lower, upper, meta, history, epochs_requested, step_size,
):
    model.eval()
    with torch.no_grad():
        probabilities = model(X).reshape(-1)
        final_loss = torch.nn.functional.binary_cross_entropy(
            probabilities, y.float()
        ).item()
        final_values, _, _, _ = build_constraints(
            model, X, y, s, specs, surrogate, alpha, mu
        )
        final_violations = _bound_violations(final_values, lower, upper)

    active_epoch_counts = Counter(
        row for active_rows in history["active_history"] for row in active_rows
    )
    recent_updates = history["update_norm_history"][-10:]
    stationary = bool(recent_updates) and max(recent_updates) <= 1e-6
    feasible = (
        not len(final_violations)
        or float(final_violations.max().item()) <= 1e-6
    )
    diagnostics = dict(history)
    diagnostics.update({
        "epochs_requested": int(epochs_requested),
        "epochs_completed": len(history["loss_history"]),
        "termination_reason": "completed_requested_epochs",
        "constraint_meta": meta,
        "active_epoch_counts_by_row": {
            str(index): int(active_epoch_counts.get(index, 0))
            for index in range(len(meta))
        },
        "final_loss": float(final_loss),
        "final_constraint_values": [float(value) for value in final_values.cpu()],
        "final_constraint_violations": [
            float(value) for value in final_violations.cpu()
        ],
        "final_max_surrogate_violation": (
            float(final_violations.max().item()) if len(final_violations) else 0.0
        ),
        "final_step_size": float(step_size),
        "stationarity_tolerance": 1e-6,
        "surrogate_feasibility_tolerance": 1e-6,
        "stationarity_detected": stationary,
        "surrogate_feasible": feasible,
        "convergence_status": (
            "stationary_and_surrogate_feasible"
            if stationary and feasible
            else "not_established"
        ),
    })
    return diagnostics


def train(
    model, X, y, s, specs, solver_factory, *,
    surrogate, alpha, mu, epochs, lr, device, verbose=False,
):
    """Train a model and return the model with JSON-compatible diagnostics."""
    X = X.to(device)
    y = y.to(device)
    s = s.to(device)
    model.train()
    parameters = _trainable_parameters(model)
    n_params = sum(parameter.numel() for parameter in parameters)

    c0, lower, upper, meta = build_constraints(
        model, X, y, s, specs, surrogate, alpha, mu
    )
    n_constraints = c0.shape[0]
    solver = solver_factory(
        n_parameters=n_params,
        n_constraints=n_constraints,
        lower_bound=lower,
        upper_bound=upper,
        device=device,
    )

    step_size = float(lr)
    average_merit = 0.0
    since_adjustment = 0
    beta2 = 0.999
    squared_gradient = None
    history = {
        "loss_history": [],
        "constraint_value_history": [],
        "constraint_violation_history": [],
        "active_history": [],
        "step_size_history": [],
        "update_norm_history": [],
        "solver_status_history": [],
    }

    for epoch in range(epochs):
        outputs = model(X).reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
        constraints, _, _, _ = build_constraints(
            model, X, y, s, specs, surrogate, alpha, mu
        )
        gradient = _flat_grad(loss, parameters, device)
        jacobian = _jacobian(
            [constraints[index] for index in range(n_constraints)],
            parameters, n_params, device,
        )

        if squared_gradient is None:
            squared_gradient = (1 - beta2) * gradient.square()
        else:
            squared_gradient = (
                beta2 * squared_gradient + (1 - beta2) * gradient.square()
            )
        curvature = torch.sqrt(squared_gradient + 1e2)

        direction, info = solver.solve_step(
            gradient, jacobian, constraints.detach(), curvature
        )
        if not torch.isfinite(direction).all():
            raise RuntimeError(
                f"The solver returned a non-finite step at epoch {epoch + 1}."
            )

        violations = _bound_violations(constraints.detach(), lower, upper)
        penalty = float(violations.sum().item()) if len(violations) else 0.0
        merit = 1e-2 * loss.item() + penalty
        average_merit = (
            0.85 * merit + 0.15 * average_merit if epoch > 0 else merit
        )
        shrink = getattr(solver, "last_step_shrink", 1.0)
        if shrink < 1.0:
            step_size *= shrink
        if (
            epoch >= 200
            and step_size >= 1e-7
            and average_merit <= merit
            and since_adjustment >= 5
        ):
            step_size /= 10
            since_adjustment = 0
        since_adjustment += 1

        active_rows = [int(index) for index in info["active_constraints"]]
        update_norm = float(torch.linalg.vector_norm(direction * step_size).item())
        history["loss_history"].append(float(loss.item()))
        history["constraint_value_history"].append(
            [float(value) for value in constraints.detach().cpu()]
        )
        history["constraint_violation_history"].append(
            [float(value) for value in violations.cpu()]
        )
        history["active_history"].append(active_rows)
        history["step_size_history"].append(float(step_size))
        history["update_norm_history"].append(update_norm)
        history["solver_status_history"].append(info.get("status", "unknown"))

        _apply_step(parameters, direction, step_size)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch + 1}/{epochs} loss={loss.item():.4f} "
                f"active_rows={active_rows} step_size={step_size:.2e}"
            )

    diagnostics = _finish_diagnostics(
        model, X, y, s, specs, surrogate, alpha, mu,
        lower, upper, meta, history, epochs, step_size,
    )
    return model, diagnostics
