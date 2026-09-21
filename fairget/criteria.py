"""Fairness-constraint specifications for binary classification.

Criteria define the statistic being constrained.  The differentiable
approximation used during optimization is configured separately through the
classifier's ``surrogate`` argument.
"""

from dataclasses import dataclass


@dataclass
class _Criterion:
    """Base criterion.

    ``delta`` is the minimum allowed ratio between the two group rates.  A
    value of 1 requests equal rates; smaller values allow a wider difference.
    """
    delta: float = 0.8

    def __post_init__(self):
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(
                f"{type(self).__name__}: delta must be in (0, 1], got {self.delta}"
            )

    @property
    def name(self) -> str:
        return type(self).__name__

    # Whether this criterion conditions on the positive label (Y == 1).
    conditions_on_label: bool = False


@dataclass
class DemographicParity(_Criterion):
    """Independence of prediction from the sensitive attribute.

    Enforces ``delta * r0 <= r1`` and ``delta * r1 <= r0``, where ``r0`` and
    ``r1`` are group positive-prediction rates.  ``delta=1`` requests equal
    rates.  ``delta=0.8`` corresponds to the ratio used in the US federal
    selection-rate screening guideline; it is not a universal definition of
    fairness or a legal conclusion.
    """
    conditions_on_label: bool = False


@dataclass
class DisparateImpact(DemographicParity):
    """Alias for DemographicParity phrased in disparate-impact language.

    This alias uses the same two-sided rate-ratio constraint as
    :class:`DemographicParity`.
    """
    pass


@dataclass
class EqualOpportunity(_Criterion):
    """Independence of prediction from the sensitive attribute *among* the
    positively-labelled population (``Y == 1``).  It constrains the ratio of
    group true-positive rates to be at least ``delta``.
    """
    conditions_on_label: bool = True


# Registry so the trainer can look criteria up by class.
_ALL_CRITERIA = (DemographicParity, DisparateImpact, EqualOpportunity)
