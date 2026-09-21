"""Differentiable approximations used by the fairness constraints.

``smoothed_step`` is the package default and implements the project's
smoothed-step formulation.  ``sigmoid`` provides a second smooth
approximation, and ``covariance`` constrains a raw covariance statistic.

For the bounded approximations, ``alpha`` changes the sharpness around the
classification threshold.  Its practical effect also depends on the fitted
score distribution and the optimizer, so larger values do not imply monotonic
improvements in held-out fairness.
"""

import torch

THRESHOLD = 0.5


def _smoothed_step(t: torch.Tensor, mu: float) -> torch.Tensor:
    """Smooth approximation of ``1{t > 0}``.

    The construction composes smooth approximations of ``max`` and ``min``:

    .. code-block:: text

        a = 0.5 * (t + 0.5 + sqrt((t+0.5)^2 + mu))   # smooth max(0, t+0.5)
        b = 1 - a
        return 1 - 0.5 * (b + sqrt(b^2 + mu))        # smooth min(a, 1)
    """
    a = 0.5 * (t + 0.5 + torch.sqrt((t + 0.5) ** 2 + mu))
    b = 1 - a
    return 1 - 0.5 * (b + torch.sqrt(b ** 2 + mu))


def _group_rates(model, inputs, groups, surrogate, alpha, mu):
    """Return (r0, r1): the surrogate positive-prediction rate for group 0 and
    group 1 respectively."""
    x0 = inputs[groups == 0]
    x1 = inputs[groups == 1]
    y0 = model(x0).view(-1)
    y1 = model(x1).view(-1)

    if surrogate == "smoothed_step":
        r0 = torch.mean(_smoothed_step(alpha * (y0 - THRESHOLD), mu))
        r1 = torch.mean(_smoothed_step(alpha * (y1 - THRESHOLD), mu))
    elif surrogate == "sigmoid":
        r0 = torch.mean(torch.sigmoid(alpha * (y0 - THRESHOLD)))
        r1 = torch.mean(torch.sigmoid(alpha * (y1 - THRESHOLD)))
    else:
        raise ValueError(f"_group_rates does not handle surrogate={surrogate!r}")
    return r0, r1


def constraint_values(model, inputs, groups, labels, *, criterion_delta,
                      conditions_on_label, surrogate, alpha, mu):
    """Compute the pair of constraint values (c1, c2) for an independence-style
    criterion, using the chosen surrogate.

    The constraint pair encodes the two-sided fairness requirement:
        c1 = delta * r0 - r1 <= 0
        c2 = delta * r1 - r0 <= 0
    where r0, r1 are the group positive-prediction rates (optionally conditioned
    on Y == 1 for equal opportunity).

    For the covariance surrogate a single value is returned as (cov, None) and
    the caller enforces |cov| <= epsilon instead.
    """
    if conditions_on_label:
        # Restrict to the positively-labeled population.
        mask = labels == 1
        inputs = inputs[mask]
        groups = groups[mask]

    if surrogate == "covariance":
        s_mean = groups.float().mean()
        y_hat = model(inputs).view(-1)
        cov = torch.dot(groups.float() - s_mean, y_hat) / len(groups)
        return cov, None

    r0, r1 = _group_rates(model, inputs, groups, surrogate, alpha, mu)
    c1 = criterion_delta * r0 - r1
    c2 = criterion_delta * r1 - r0
    return c1, c2
