"""Active-set solver for the quadratic subproblem used during training."""

from abc import ABC, abstractmethod

import torch


class QPSolver(ABC):
    """Contract for a QP-subproblem solver.

    A solver is constructed once per training run with the fixed problem size
    and box bounds, then `solve_step` is called every iteration with the current
    gradients/Jacobian/constraint values.
    """

    def __init__(self, n_parameters, n_constraints, lower_bound, upper_bound,
                 device="cpu"):
        self.n_parameters = n_parameters
        self.n_constraints = n_constraints
        self.lower_bound = list(lower_bound)
        self.upper_bound = list(upper_bound)
        self.device = device

    @abstractmethod
    def solve_step(self, g, J, c, H_diag):
        """Solve the QP subproblem for one iteration.

        Parameters
        ----------
        g : (n_parameters,) objective gradient
        J : (n_constraints, n_parameters) constraint Jacobian
        c : (n_constraints,) current constraint values
        H_diag : (n_parameters,) diagonal Hessian approximation (positive)

        Returns
        -------
        d : (n_parameters,) step direction
        info : dict with at least {"active_constraints": list[int]}
        """
        raise NotImplementedError


class SQPSolver(QPSolver):
    """Active-set sequential quadratic programming with diagonal curvature.

    Candidate active sets are enumerated.  The equality-constrained system is
    reduced to constraint space, so memory does not grow quadratically with the
    number of model parameters.
    """

    THRESH = 1e12
    FEASIBILITY_TOL = 1e-7
    MULTIPLIER_TOL = 1e-7

    def __init__(self, *args, step_size_decay=0.5, mu=1e2, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size_decay = step_size_decay
        self.mu = mu
        self.last_step_shrink = 1.0  # signals the loop to shrink LR on fallback

    def _solve_kkt(self, H_diag, g, J_sub, c_sub):
        """Solve the equality-constrained system in constraint space."""
        n_sub = J_sub.shape[0]
        H_inv = torch.reciprocal(H_diag)
        if n_sub == 0:
            d = -H_inv * g
            return d, torch.empty(0, device=g.device)

        weighted_jt = H_inv[:, None] * J_sub.T
        schur = J_sub @ weighted_jt
        rhs = c_sub - J_sub @ (H_inv * g)
        multipliers = torch.linalg.solve(schur, rhs)
        d = -H_inv * (g + J_sub.T @ multipliers)
        return d, multipliers

    def _feasible(self, d, c, J, skip):
        """Check box feasibility of the linearized constraints not in `skip`."""
        for i in range(self.n_constraints):
            if i in skip:
                continue
            val = c[i] + torch.matmul(J[i], d)
            if not torch.isfinite(val):
                return False
            lo, up = self.lower_bound[i], self.upper_bound[i]
            if lo < self.THRESH and val < -lo - self.FEASIBILITY_TOL:
                return False
            if up < self.THRESH and val > up + self.FEASIBILITY_TOL:
                return False
        return True

    def _valid_multipliers(self, multipliers, sides):
        """Check inequality-multiplier signs for an active-set candidate."""
        if not torch.isfinite(multipliers).all():
            return False
        for multiplier, side in zip(multipliers, sides):
            value = float(multiplier.item())
            if side == "u" and value < -self.MULTIPLIER_TOL:
                return False
            if side == "l" and value > self.MULTIPLIER_TOL:
                return False
        return True

    def _qp_obj(self, d, g, H_diag):
        return torch.dot(g, d) + 0.5 * torch.sum(H_diag * d.square())

    def _adjusted_c(self, c, active, side):
        """Build the RHS constraint offsets for an active set, choosing the
        upper (side='u') or lower (side='l') bound for each active index."""
        vals = []
        for i, s in zip(active, side):
            if s == "u":
                vals.append(c[i] - self.upper_bound[i])
            else:
                vals.append(c[i] + self.lower_bound[i])
        return torch.stack(vals) if vals else torch.empty(0, device=c.device)

    def solve_step(self, g, J, c, H_diag):
        self.last_step_shrink = 1.0
        n = self.n_constraints

        if not torch.isfinite(H_diag).all() or torch.any(H_diag <= 0):
            raise ValueError("H_diag must contain finite positive values.")

        if n == 0:
            d = -g / H_diag
            return d, {
                "active_constraints": [],
                "status": "unconstrained",
                "fallback": False,
            }

        d_free = -g / H_diag
        if self._feasible(d_free, c, J, skip=set()):
            return d_free, {
                "active_constraints": [],
                "status": "unconstrained_step_feasible",
                "fallback": False,
            }

        # Enumerate active-set candidates of increasing size. For each active
        # index we may hit its upper or lower bound; try both sides.
        import itertools

        best = None  # (qp_obj, d, active_list)
        indices = list(range(n))

        for size in range(1, n + 1):
            for active in itertools.combinations(indices, size):
                # side choices: for each active constraint, upper or lower if
                # that bound is finite.
                side_options = []
                for i in active:
                    opts = []
                    if self.upper_bound[i] < self.THRESH:
                        opts.append("u")
                    if self.lower_bound[i] < self.THRESH:
                        opts.append("l")
                    if not opts:
                        opts = None
                        break
                    side_options.append(opts)
                if side_options is None:
                    continue

                for sides in itertools.product(*side_options):
                    c_adj = self._adjusted_c(c, active, sides)
                    J_sub = J[list(active)]
                    try:
                        d_cand, multipliers = self._solve_kkt(
                            H_diag, g, J_sub, c_adj
                        )
                    except RuntimeError:
                        continue
                    if not torch.isfinite(d_cand).all():
                        continue
                    if not self._valid_multipliers(multipliers, sides):
                        continue
                    # Check every row, including the rows expected to be active.
                    # This catches numerical or rank-deficiency errors in the
                    # equality solve.
                    if not self._feasible(d_cand, c, J, skip=set()):
                        continue
                    obj = self._qp_obj(d_cand, g, H_diag).item()
                    if best is None or obj < best[0]:
                        best = (obj, d_cand, list(active))

            if best is not None:
                # Found feasible candidate(s) at this active-set size; the
                # smallest active set that works is preferred, so stop.
                return best[1], {
                    "active_constraints": best[2],
                    "status": "active_set_step",
                    "fallback": False,
                }

        # Nothing feasible: fall back to the unconstrained step but signal the
        # training loop to shrink the learning rate.
        self.last_step_shrink = self.step_size_decay
        return d_free, {
            "active_constraints": [],
            "status": "fallback_unconstrained_step",
            "fallback": True,
        }


def make_solver(name, **kwargs):
    name = (name or "sqp").lower()
    if name == "sqp":
        return SQPSolver(**kwargs)
    raise ValueError(f"Unknown solver {name!r}. The available solver is 'sqp'.")
