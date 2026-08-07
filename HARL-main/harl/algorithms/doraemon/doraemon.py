"""DORAEMON: Domain Randomization via Entropy Maximization (Tiboni et al., 2023).

This module contains the *environment-agnostic* algorithmic core of DORAEMON,
ported faithfully from the authors' reference implementation
(``doraemon/doraemon.py``) but stripped of the SB3/wandb training-subroutine glue
(in HARL the RL learner is HAPPO, driven by ``OnPolicyDoraemonRunner``).

Two classes are provided:

* ``DomainRandDistribution`` -- a product of independent univariate Beta
  distributions ``Be(a_i, b_i)`` over bounded ranges ``[m_i, M_i]``, with
  ``sample`` / ``pdf`` / ``kl_divergence`` / ``entropy`` and gradient-tracking
  helpers used by the constrained optimizer.

* ``DoraemonUpdater`` -- given the dynamics parameters ``xi`` sampled while
  training under the *current* distribution and the corresponding episode
  returns, solves the DORAEMON optimization problem (Eq. 4 of the paper)

      min_phi   KL( nu_phi || nu_target )           # = maximize entropy
      s.t.      G_hat(theta, phi_old, phi) >= alpha # success-rate constraint (IS)
                KL( nu_phi_old || nu_phi ) <= eps   # trust region

  via SciPy ``trust-constr`` with analytic (autograd) Jacobians, updating the
  current distribution in place.  A backup feasibility problem (Eq. 6) recovers a
  feasible starting point when the current distribution already violates the
  success constraint.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal


class DomainRandDistribution:
    """Product of independent univariate Beta distributions over bounded ranges."""

    def __init__(self, distr: List[Dict]):
        self.set(distr)

    def set(self, distr):
        """distr: list of dicts, one per dimension, with keys m, M, a, b.

        Y ~ Beta(a, b) on [0, 1] mapped to [m, M] via y = x*(M-m) + m.
        """
        self.distr = deepcopy(distr)
        self.ndims = len(self.distr)
        self.to_distr = []
        # float64 throughout: dynamics samples / sigmoid params are float64, and
        # torch.distributions requires matching dtypes between params and values.
        self.parameters = torch.zeros((self.ndims * 2), dtype=torch.float64)
        for i in range(self.ndims):
            self.parameters[i * 2] = float(distr[i]["a"])
            self.parameters[i * 2 + 1] = float(distr[i]["b"])
            self.to_distr.append(
                Beta(self.parameters[i * 2], self.parameters[i * 2 + 1])
            )

    # ------------------------------------------------------------------ sampling
    def sample(self, n_samples=1):
        values = []
        for i in range(self.ndims):
            m, M = self.distr[i]["m"], self.distr[i]["M"]
            values.append(
                self.to_distr[i].sample(sample_shape=(n_samples,)).numpy() * (M - m) + m
            )
        return np.array(values).T

    # ----------------------------------------------------------------------- pdf
    def _univariate_pdf(self, x, i, log=False, to_distr=None, standardize=False):
        to_distr = self.to_distr if to_distr is None else to_distr
        m, M = self.distr[i]["m"], self.distr[i]["M"]
        if np.isclose(M - m, 0):
            return np.isclose(x, m).astype(int)
        dtype = to_distr[i].concentration1.dtype
        if log:
            if standardize:
                return to_distr[i].log_prob(torch.as_tensor(x, dtype=dtype))
            return to_distr[i].log_prob(
                torch.as_tensor((x - m) / (M - m), dtype=dtype)
            ) - torch.log(torch.tensor(M - m, dtype=dtype))
        else:
            if standardize:
                return torch.exp(to_distr[i].log_prob(torch.as_tensor(x, dtype=dtype)))
            return torch.exp(
                to_distr[i].log_prob(torch.as_tensor((x - m) / (M - m), dtype=dtype))
            ) / (M - m)

    def pdf(self, x, log=False, requires_grad=False, standardize=False, to_params=None):
        """Compute pdf(x). x: (batch, ndims) numpy array or torch tensor."""
        assert len(x.shape) == 2, "Input tensor expected with dims (batch, ndims)"
        n = x.shape[0]
        density = (
            torch.zeros(n, dtype=torch.float64)
            if log
            else torch.ones(n, dtype=torch.float64)
        )
        custom_to_distr = None
        if requires_grad:
            custom_to_distr, to_params = self._to_distr_with_grad(to_params=to_params)
        if standardize:
            x = self._standardize_value(x)
        for i in range(self.ndims):
            if log:
                density = density + self._univariate_pdf(
                    x[:, i], i, log=True, to_distr=custom_to_distr, standardize=standardize
                )
            else:
                density = density * self._univariate_pdf(
                    x[:, i], i, log=False, to_distr=custom_to_distr, standardize=standardize
                )
        if requires_grad:
            return density, to_params
        return density

    def _standardize_value(self, x):
        """Linearly scale values from [m, M] to [0, 1]."""
        if torch.is_tensor(x):
            norm_x = x.clone().to(torch.float64)
        else:
            norm_x = np.array(x, dtype=np.float64).copy()
        for i in range(self.ndims):
            m, M = self.distr[i]["m"], self.distr[i]["M"]
            norm_x[:, i] = (x[:, i] - m) / (M - m)
        return norm_x

    # ------------------------------------------------------------- kl / entropy
    def kl_divergence(self, q, requires_grad=False, p_params=None, q_params=None):
        """KL( self || q ). q: DomainRandDistribution."""
        assert isinstance(q, DomainRandDistribution)
        if requires_grad:
            p_distr, p_params = self._to_distr_with_grad(to_params=p_params)
            q_distr, q_params = q._to_distr_with_grad(to_params=q_params)
        else:
            p_distr = self.to_distr
            q_distr = q.to_distr

        kl_div = 0
        for i in range(self.ndims):
            # KL does not depend on the [m, M] location parameters
            kl_div = kl_div + torch.distributions.kl_divergence(p_distr[i], q_distr[i])

        if requires_grad:
            return kl_div, p_params, q_params
        return kl_div

    def entropy(self, standardize=False):
        entropy = 0
        for i in range(self.ndims):
            if standardize:
                entropy = entropy + self.to_distr[i].entropy()
            else:
                # Y = aX + b  =>  H(Y) = H(X) + log(a)
                m, M = self.distr[i]["m"], self.distr[i]["M"]
                entropy = entropy + self.to_distr[i].entropy() + torch.log(
                    torch.tensor(M - m, dtype=torch.float64)
                )
        return entropy

    def _to_distr_with_grad(self, to_params=None):
        if to_params is None:
            params = self.get_stacked_params()
            to_params = torch.tensor(params, requires_grad=True)
        to_distr = []
        for i in range(self.ndims):
            to_distr.append(Beta(to_params[i * 2], to_params[i * 2 + 1]))
        return to_distr, to_params

    # ----------------------------------------------------------------- mutators
    def update_parameters(self, params):
        distr = deepcopy(self.distr)
        for i in range(self.ndims):
            distr[i]["a"] = float(params[i * 2])
            distr[i]["b"] = float(params[i * 2 + 1])
        self.set(distr)

    # ------------------------------------------------------------------ getters
    def get(self):
        return deepcopy(self.distr)

    def get_stacked_bounds(self):
        return np.array([[item["m"], item["M"]] for item in self.distr]).reshape(-1)

    def get_stacked_params(self):
        return self.parameters.detach().numpy()

    def to_string(self):
        return " | ".join(
            f"dim{i}: a={self.distr[i]['a']:.3f} b={self.distr[i]['b']:.3f} "
            f"[{self.distr[i]['m']:.3f},{self.distr[i]['M']:.3f}]"
            for i in range(self.ndims)
        )

    # ------------------------------------------------------------ static helpers
    @staticmethod
    def beta_from_stacked(stacked_bounds, stacked_params):
        distr = []
        ndim = stacked_bounds.shape[0] // 2
        for i in range(ndim):
            distr.append(
                {
                    "m": float(stacked_bounds[i * 2]),
                    "M": float(stacked_bounds[i * 2 + 1]),
                    "a": stacked_params[i * 2],
                    "b": stacked_params[i * 2 + 1],
                }
            )
        return DomainRandDistribution(distr)

    @staticmethod
    def sigmoid(x, lb=0, up=1):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        return (up - lb) / (1 + torch.exp(-x)) + lb

    @staticmethod
    def inv_sigmoid(x, lb=0, up=1):
        x = x if torch.is_tensor(x) else torch.tensor(x)
        x = torch.clamp(x, lb + 1e-6, up - 1e-6)
        return -torch.log((up - lb) / (x - lb) - 1)


class DoraemonUpdater:
    """Solves the DORAEMON constrained optimization to update the DR distribution."""

    def __init__(
        self,
        init_distr: DomainRandDistribution,
        target_distr: DomainRandDistribution,
        kl_upper_bound: float,
        alpha: float,
        return_threshold: float,
        success_mode: str = "success_rate",
        robust_estimate: bool = False,
        alpha_ci: float = 0.9,
        train_until_performance_lb: bool = True,
        min_dynamics_samples: int = 50,
        init_beta_param: float = 100.0,
        beta_param_bounds=None,
        verbose: int = 1,
    ):
        """
        Args:
            init_distr: starting (narrow) distribution.
            target_distr: max-entropy (widest) distribution to converge to.
            kl_upper_bound: epsilon, the per-step KL trust-region bound.
            alpha: desired in-distribution success rate (success_rate mode) or
                   expected-return lower bound (return mode).
            return_threshold: episode return above which a trajectory counts as a
                              success (success_rate mode only).
            success_mode: "success_rate" (constraint on Prob[return >= thr]) or
                          "return" (constraint on expected return).
            robust_estimate: constrain the alpha_ci lower-confidence bound of the
                             IS estimate instead of its sample mean.
            train_until_performance_lb: do not expand the distribution until the
                                        success constraint is first satisfied.
            min_dynamics_samples: minimum number of collected episodes required to
                                  attempt an update (otherwise skip).
        """
        self.init_distr = deepcopy(init_distr)
        self.current_distr = deepcopy(init_distr)
        self.target_distr = deepcopy(target_distr)

        self.kl_upper_bound = float(kl_upper_bound)
        self.alpha = float(alpha)
        self.return_threshold = float(return_threshold)
        self.success_mode = success_mode
        self.robust_estimate = bool(robust_estimate)
        self.alpha_ci = float(alpha_ci)
        self.train_until_performance_lb = bool(train_until_performance_lb)
        self.min_dynamics_samples = int(min_dynamics_samples)
        self.verbose = verbose

        # sigmoid bounds for the beta parameters during optimization
        if beta_param_bounds is None:
            self.min_bound = 1.0 if self.train_until_performance_lb else 0.8
            self.max_bound = init_beta_param + 10
        else:
            margin = 0.1 * (beta_param_bounds[1] - beta_param_bounds[0])
            self.min_bound = min(1, max(0, beta_param_bounds[0] - margin))
            self.max_bound = max(init_beta_param + 10, beta_param_bounds[1] + margin)

        self.train_until_done = False
        self.current_iter = 0

    # ------------------------------------------------------------- success utils
    def _perf_values(self, returns):
        if self.success_mode == "success_rate":
            return torch.tensor(
                np.asarray(returns) >= self.return_threshold, dtype=torch.float64
            )
        return torch.tensor(np.asarray(returns), dtype=torch.float64)

    def _get_ci(self, mean, stdev, N, alpha):
        import scipy.stats as st

        t_score = float(st.t.ppf((1 + alpha) / 2.0, N))
        ci = t_score * stdev / (N ** 0.5)
        return mean - ci, mean + ci

    def variance_IS_estimator(self, x, f_x, p, q, requires_grad=False, to_params=None):
        """Variance of the importance-sampling estimator IS = E_q[f(X) p(X)/q(X)]."""
        N = x.shape[0]
        if requires_grad:
            p_log_prob, _ = p.pdf(
                x, log=True, standardize=True, requires_grad=True, to_params=to_params
            )
        else:
            p_log_prob = p.pdf(x, log=True, standardize=True)
        q_log_prob = q.pdf(x, log=True, standardize=True)
        sq_is = torch.exp(2 * p_log_prob - 2 * q_log_prob)
        second_moment = torch.mean(torch.square(f_x) * sq_is)
        is_ratio = torch.exp(p_log_prob - q_log_prob)
        sq_first_moment = torch.mean(f_x * is_ratio) ** 2
        return 1 / N * (second_moment - sq_first_moment)

    def _solve_inverted(
        self, x0_opt, perf_fn, perf_prime, kl_fn, kl_prime, minimize, NonlinearConstraint
    ):
        """Backup problem (Eq. 6): max success s.t. KL(current || proposed) <= eps.

        Returns (x_opt, ok). Used to recover a feasible / max-success starting
        distribution within the trust region when the current one is infeasible.
        """

        def negative_perf(x_opt):
            return (
                -float(perf_fn(x_opt)),
                -np.asarray(perf_prime(x_opt), dtype=np.float64),
            )

        constraints = [
            NonlinearConstraint(
                fun=kl_fn,
                lb=-np.inf,
                # stay strictly inside the trust region for later numerical stability
                ub=self.kl_upper_bound - 1e-5,
                jac=kl_prime,
                keep_feasible=True,
            )
        ]
        try:
            result = minimize(
                negative_perf,
                x0_opt,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6, "maxiter": 300},
            )
        except Exception:
            return None, False
        return result.x, True

    # --------------------------------------------------------------- main update
    def update(self, dynamics: np.ndarray, returns: np.ndarray):
        """Run one DORAEMON distribution update.

        Args:
            dynamics: (N, ndims) dynamics parameters sampled under current_distr.
            returns:  (N,) episode returns matching ``dynamics``.
        Returns:
            info: (dict) diagnostics; ``updated`` is False if the distribution
                  was left unchanged (skipped/infeasible).
        """
        try:
            from scipy.optimize import NonlinearConstraint, minimize
        except ImportError:
            print(
                "[DORAEMON] scipy is required for the distribution update "
                "(pip install scipy). Keeping the current distribution."
            )
            self.current_iter += 1
            return {
                "updated": False,
                "n_samples": int(len(dynamics)),
                "entropy": float(self.current_distr.entropy().item()),
                "kl_from_target": float(
                    self.current_distr.kl_divergence(self.target_distr).item()
                ),
                "kl_step": 0.0,
                "train_success_rate": 0.0,
                "est_success": 0.0,
            }

        info = {
            "updated": False,
            "n_samples": int(len(dynamics)),
            "entropy": float(self.current_distr.entropy().item()),
            "kl_from_target": float(
                self.current_distr.kl_divergence(self.target_distr).item()
            ),
            "kl_step": 0.0,
            "train_success_rate": 0.0,
            "est_success": 0.0,
        }

        if len(dynamics) < self.min_dynamics_samples:
            if self.verbose:
                print(
                    f"[DORAEMON] iter {self.current_iter}: only {len(dynamics)} "
                    f"samples (< {self.min_dynamics_samples}); skipping update."
                )
            self.current_iter += 1
            return info

        dynamics = np.asarray(dynamics, dtype=np.float64)
        perf_values = self._perf_values(returns)
        info["train_success_rate"] = float(
            (np.asarray(returns) >= self.return_threshold).mean()
        )

        bounds = self.current_distr.get_stacked_bounds()

        def proposed_from_xopt(x_opt):
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            return (
                DomainRandDistribution.beta_from_stacked(bounds, x),
                x,
            )

        # ---------------- KL trust-region constraint: KL(current || proposed) ----
        def kl_constraint_fn(x_opt):
            proposed, _ = proposed_from_xopt(x_opt)
            return self.current_distr.kl_divergence(proposed).detach().numpy()

        def kl_constraint_fn_prime(x_opt):
            x_opt = torch.tensor(x_opt, requires_grad=True)
            _, x = proposed_from_xopt(x_opt)
            proposed = DomainRandDistribution.beta_from_stacked(bounds, x)
            kl_div, _, _ = self.current_distr.kl_divergence(
                proposed, requires_grad=True, q_params=x
            )
            grads = torch.autograd.grad(kl_div, x_opt)
            return np.concatenate([g.detach().numpy() for g in grads])

        # ---------------- performance (success-rate) constraint via IS -----------
        def performance_constraint_fn(x_opt, force_robust=None):
            proposed, _ = proposed_from_xopt(x_opt)
            is_ratio = torch.exp(
                proposed.pdf(dynamics, log=True, standardize=True)
                - self.current_distr.pdf(dynamics, log=True, standardize=True)
            )
            performance = torch.mean(is_ratio * perf_values)
            use_robust = (force_robust is True or self.robust_estimate) and (
                force_robust is not False
            )
            if use_robust:
                N = dynamics.shape[0]
                var = self.variance_IS_estimator(
                    dynamics, perf_values, proposed, self.current_distr
                )
                lcb, _ = self._get_ci(performance, torch.sqrt(var), N, self.alpha_ci)
                return lcb.detach().numpy()
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x_opt):
            x_opt = torch.tensor(x_opt, requires_grad=True)
            _, x = proposed_from_xopt(x_opt)
            proposed = DomainRandDistribution.beta_from_stacked(bounds, x)
            proposed_log_prob, _ = proposed.pdf(
                dynamics, log=True, requires_grad=True, standardize=True, to_params=x
            )
            is_ratio = torch.exp(
                proposed_log_prob
                - self.current_distr.pdf(dynamics, log=True, standardize=True)
            )
            performance = torch.mean(is_ratio * perf_values)
            if self.robust_estimate:
                N = dynamics.shape[0]
                var = self.variance_IS_estimator(
                    dynamics, perf_values, proposed, self.current_distr,
                    requires_grad=True, to_params=x,
                )
                lcb, _ = self._get_ci(performance, torch.sqrt(var), N, self.alpha_ci)
                grads = torch.autograd.grad(lcb, x_opt)
            else:
                grads = torch.autograd.grad(performance, x_opt)
            return np.concatenate([g.detach().numpy() for g in grads])

        # ---------------- objective: minimize KL(proposed || target) -------------
        def objective_fn(x_opt):
            x_opt = torch.tensor(x_opt, requires_grad=True)
            _, x = proposed_from_xopt(x_opt)
            proposed = DomainRandDistribution.beta_from_stacked(bounds, x)
            kl_div, _, _ = proposed.kl_divergence(
                self.target_distr, requires_grad=True, p_params=x
            )
            grads = torch.autograd.grad(kl_div, x_opt)
            return (
                float(kl_div.detach().numpy()),
                np.concatenate([g.detach().numpy() for g in grads]).astype(np.float64),
            )

        x0 = self.current_distr.get_stacked_params()
        x0_opt = DomainRandDistribution.inv_sigmoid(
            torch.tensor(x0), self.min_bound, self.max_bound
        ).numpy()

        info["est_success"] = float(performance_constraint_fn(x0_opt, force_robust=False))

        # Skip expansion until the success constraint is first satisfied
        if self.train_until_performance_lb and not self.train_until_done:
            if performance_constraint_fn(x0_opt) < self.alpha:
                if self.verbose:
                    print(
                        f"[DORAEMON] iter {self.current_iter}: success "
                        f"{info['est_success']:.3f} < alpha {self.alpha}; "
                        f"not expanding yet."
                    )
                self.current_iter += 1
                return info
            self.train_until_done = True

        # --------- backup problem: contract toward feasibility if needed --------
        # If the current distribution already violates the success constraint, the
        # expansion problem (Eq. 4) is infeasible. Following the paper (Eq. 6) we
        # first solve the inverted problem  max G(phi)  s.t.  KL(phi_i || phi) <= eps
        # to recover a feasible starting point (or, failing that, adopt the most
        # successful distribution within the trust region and stop -- i.e. contract).
        start_x_opt = x0_opt
        if float(performance_constraint_fn(x0_opt)) < self.alpha:
            feas_x_opt, feas_ok = self._solve_inverted(
                x0_opt,
                performance_constraint_fn,
                performance_constraint_fn_prime,
                kl_constraint_fn,
                kl_constraint_fn_prime,
                minimize,
                NonlinearConstraint,
            )
            if not feas_ok:
                if self.verbose:
                    print("[DORAEMON] backup (inverted) problem failed; keeping distribution.")
                self.current_iter += 1
                return info

            if float(performance_constraint_fn(feas_x_opt)) >= self.alpha:
                # feasible starting distribution found -> expand from there
                start_x_opt = feas_x_opt
            else:
                # no feasible distribution in the trust region -> adopt the
                # max-success (contracted) distribution and stop this iteration
                new_x = DomainRandDistribution.sigmoid(
                    torch.tensor(feas_x_opt), self.min_bound, self.max_bound
                ).numpy()
                curr_step_kl = float(kl_constraint_fn(feas_x_opt))
                self.current_distr.update_parameters(new_x)
                info["updated"] = True
                info["kl_step"] = curr_step_kl
                info["entropy"] = float(self.current_distr.entropy().item())
                info["kl_from_target"] = float(
                    self.current_distr.kl_divergence(self.target_distr).item()
                )
                info["est_success"] = float(
                    performance_constraint_fn(feas_x_opt, force_robust=False)
                )
                if self.verbose:
                    print(
                        f"[DORAEMON] iter {self.current_iter}: infeasible; contracted "
                        f"to max-success distribution within trust region "
                        f"(est_succ={info['est_success']:.3f})."
                    )
                self.current_iter += 1
                return info

        constraints = [
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound,
                jac=kl_constraint_fn_prime,
                keep_feasible=False,
            ),
            NonlinearConstraint(
                fun=performance_constraint_fn,
                lb=self.alpha - 1e-4,
                ub=np.inf,
                jac=performance_constraint_fn_prime,
                keep_feasible=False,
            ),
        ]

        try:
            result = minimize(
                objective_fn,
                start_x_opt,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6, "maxiter": 300},
            )
            new_x_opt = result.x
        except Exception as exc:  # numerical failure -> keep current distribution
            if self.verbose:
                print(f"[DORAEMON] optimization raised {exc!r}; keeping distribution.")
            self.current_iter += 1
            return info

        # Validate the optimum; revert if it did not improve while staying feasible
        if not result.success:
            old_f = objective_fn(start_x_opt)[0]
            feasible = [
                bool(c.lb <= float(c.fun(new_x_opt)) <= c.ub) for c in constraints
            ]
            if not (all(feasible) and result.fun < old_f):
                if self.verbose:
                    print(
                        "[DORAEMON] update unsuccessful; keeping starting parameters."
                    )
                new_x_opt = start_x_opt

        curr_step_kl = float(kl_constraint_fn(new_x_opt))
        new_x = DomainRandDistribution.sigmoid(
            torch.tensor(new_x_opt), self.min_bound, self.max_bound
        ).numpy()
        self.current_distr.update_parameters(new_x)

        info["updated"] = True
        info["kl_step"] = curr_step_kl
        info["entropy"] = float(self.current_distr.entropy().item())
        info["kl_from_target"] = float(
            self.current_distr.kl_divergence(self.target_distr).item()
        )
        info["est_success"] = float(
            performance_constraint_fn(new_x_opt, force_robust=False)
        )
        self.current_iter += 1
        return info
