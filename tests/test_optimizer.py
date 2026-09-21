"""Reference cases for the active-set quadratic subproblem solver."""

import torch

from fairget.optimizer import SQPSolver


def test_solver_rejects_active_rows_with_invalid_multiplier_signs():
    """A feasible projection is not optimal when its multiplier has wrong sign."""
    solver = SQPSolver(
        n_parameters=2,
        n_constraints=3,
        lower_bound=[1e18, 1e18, 1e18],
        upper_bound=[0.0, 0.0, 2.0],
    )
    gradient = torch.tensor([-1.0, -1.0])
    jacobian = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ])
    values = torch.zeros(3)

    direction, info = solver.solve_step(
        gradient,
        jacobian,
        values,
        torch.ones(2),
    )

    assert torch.allclose(direction, torch.zeros(2), atol=1e-7)
    assert info["active_constraints"] == [0, 1]


def test_solver_applies_the_lower_bound_multiplier_sign_convention():
    solver = SQPSolver(
        n_parameters=2,
        n_constraints=3,
        lower_bound=[0.0, 0.0, 2.0],
        upper_bound=[1e18, 1e18, 1e18],
    )
    gradient = torch.tensor([1.0, 1.0])
    jacobian = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ])

    direction, info = solver.solve_step(
        gradient,
        jacobian,
        torch.zeros(3),
        torch.ones(2),
    )

    assert torch.allclose(direction, torch.zeros(2), atol=1e-7)
    assert info["active_constraints"] == [0, 1]
