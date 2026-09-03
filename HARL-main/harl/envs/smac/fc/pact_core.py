"""PACT for Formation Congestion -- the arithmetic, in pure numpy.

Kept free of pysc2 and torch so ``fc/selfcheck.py`` can certify every claim on a
machine with no StarCraft II installed, and so the environment wrapper and the
test can never drift apart.  Follows ``PACT_PIPELINE_SPEC.md`` section by
section; each class names the section it implements and each **[PAID]** note
reproduces a failure the spec paid for on POWER.

WHAT THE AGENT IS SOLVING HERE
------------------------------
Its delivered step is throttled by a deficit

    Delta_i(t) = u_i(t) * (1 - g(t))
               = c(t) * [ sum_{j != i} W[i,j] prox_ij cone_ij Phi_j  +  L_i^fix ]

with  c(t) = (1 - g(t)) / (K_i^0 g(t))  -- an unknown scalar that DRIFTS with the
hidden driver.  Project the bracket onto the declared operator and the whole
thing is ``Delta_i = beta*(t) . psi_i``, with ``beta*`` an r-vector the agent has
to track online.  That is exactly POWER's problem, so exactly POWER's machinery
applies: declared basis, RLS on the agent's own proprioceptive residual, and the
certified channel inverse driven by the estimate.

THE SENSOR.  ``Delta_i`` is the fraction of its ordered step the unit did not
travel -- ``1 - realized/commanded``.  A unit knows how far it ordered itself to
go and it can see how far it got, so this is odometry, not privilege.  It reports
the PAST (A.4): it says what step t cost while step t+1 is the one that needs
compensating, and closing that gap is the entire problem.
"""

import math

import numpy as np

# The divisor 1 - Delta may not be trusted below this: past it the channel is so
# throttled that the inverse would demand an unbounded command.  Guarded, and the
# guard is counted in `du_da_floor_frac` rather than silently applied.
DU_DA_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# Component 1 -- the declared basis        (PACT_PIPELINE_SPEC 2)
# --------------------------------------------------------------------------- #
class Basis:
    """Channels over the declared operator, with per-agent per-channel scaling.

    2.2 CHANNELS.  ``r = 1`` by DEFAULT and that is a measured choice, not
    laziness: POWER's r=2 strong/weak split was WORSE conditioned than a single
    weighted channel (3006 vs 810), because after per-channel normalisation both
    columns collapse to a weighted mean of peer exertion fractions and go near
    collinear.  ``r = 2`` here splits by coupling strength (which on 3s5z is
    exactly the same-band / cross-band split the operator's `band` factor
    creates) and carries the operator magnitudes as WEIGHTS INSIDE each channel,
    never as flat buckets.  The effective r is reported and dead channels are
    dropped and named (2.4).

    2.3 PER-AGENT PER-CHANNEL SCALING **[PAID]**::

        range[i, c] = sum_{j in c} W[i, j] * Phi_max        DECLARED
        x_ref[i, c] = 0.5 * range[i, c]                     uniform-random peers
        psi[i, c]   = (x[i, c] - x_ref[i, c]) / range[i, c] lands in [-0.5, 0.5]

    A single scale shared across agents drove ``cond(E[psi psi^T])`` to 72,148
    against 57, because an agent whose weak channel is ~100x smaller than its
    strong one contributes a near-zero column and the Gram goes singular.  Each
    agent runs its own estimator, so per-agent scaling leaks nothing.

    Centred on the GEOMETRIC reference (peers acting uniformly at random), never
    on the sample mean -- the sample mean is run data and would turn a declared
    model class into a fit.
    """

    def __init__(self, W, phi_max, r=1, min_coverage=1e-3):
        self.W = np.asarray(W, dtype=np.float64)
        self.n = self.W.shape[0]
        assert self.W.shape[0] == self.W.shape[1], "W must be square (agents x agents)"
        # 2.4: zero diagonal is ASSERTED, not argued.
        assert np.allclose(np.diag(self.W), 0.0), \
            "W[i, i] must be exactly 0 -- the own-effect is not coupling (A.2)"
        self.phi_max = float(phi_max)
        self.r_req = max(1, int(r))
        self.min_coverage = float(min_coverage)
        self.masks, self.ranges, self.r, self.dead = self._channels()

    def _channels(self):
        """Assign each agent's peers to channels by coupling strength."""
        n, r = self.n, self.r_req
        masks = np.zeros((n, r, n), dtype=np.float64)
        for i in range(n):
            w = self.W[i].copy()
            peers = np.where(w > 0.0)[0]
            if peers.size == 0:
                continue
            if r == 1:
                masks[i, 0, peers] = 1.0
                continue
            # Split at the geometric mean of this row's extremes: a scale-free
            # threshold that separates the same-band peers from the cross-band
            # ones without looking at any run data.
            lo, hi = float(w[peers].min()), float(w[peers].max())
            thr = math.sqrt(max(lo, 1e-300) * hi)
            edges = np.linspace(0.0, 1.0, r + 1)
            if r == 2:
                masks[i, 0, peers[w[peers] >= thr]] = 1.0      # strong / same band
                masks[i, 1, peers[w[peers] < thr]] = 1.0       # weak / cross band
            else:
                q = np.quantile(w[peers], edges)
                for c in range(r):
                    sel = peers[(w[peers] >= q[c]) & (w[peers] <= q[c + 1])]
                    masks[i, c, sel] = 1.0
        # DECLARED range: every peer in the channel at its declared maximum
        # exertion, at maximum proximity and dead ahead.  No run data.
        ranges = np.einsum("ij,icj->ic", self.W, masks) * self.phi_max
        # 2.4: drop channels whose coverage is negligible rather than forcing them
        # to survive, and NAME them.
        live = ranges > self.min_coverage * max(1e-12, float(ranges.max()))
        dead = [(int(i), int(c)) for i in range(self.n)
                for c in range(self.r_req) if not live[i, c]]
        eff = int(np.max(live.sum(axis=1))) if live.any() else 0
        return masks, ranges, max(1, eff), dead

    def no_coupling_agents(self):
        """Agents with no live coupling at all -- they stay at trust 0 forever
        and 2.4 asks for them to be listed explicitly."""
        return [int(i) for i in range(self.n) if float(self.ranges[i].sum()) <= 0.0]

    def psi(self, contrib, phi, live):
        """``psi[i, c]`` from this step's per-pair contributions.

        ``contrib[i, j] = W[i, j] * prox_ij * cone_ij^{m*}`` is supplied by the
        environment (it owns the geometry); ``phi`` is the exertion vector and
        ``live`` masks dead units out.
        """
        contrib = np.asarray(contrib, dtype=np.float64)
        ph = np.asarray(phi, dtype=np.float64) * np.asarray(live, dtype=np.float64)
        x = np.einsum("ij,icj->ic", contrib * ph[None, :], self.masks)
        rng = self.ranges
        out = np.zeros_like(x)
        ok = rng > 0.0
        out[ok] = (x[ok] - 0.5 * rng[ok]) / rng[ok]
        return out

    def report(self):
        return dict(r_requested=self.r_req, r_effective=self.r,
                    dead_channels=self.dead,
                    no_coupling_agents=self.no_coupling_agents(),
                    range_min=float(self.ranges[self.ranges > 0].min())
                    if np.any(self.ranges > 0) else 0.0,
                    range_max=float(self.ranges.max()))


def own_column(phi_i, phi_max):
    """The agent's own exertion column, scaled the same way as psi.

    4.2: the own column is CURRENT, because the observation already reflects the
    agent's own action and it knows its own action exactly.  4.3: it is carried as
    an explicit column so the reported quality metric is the LIFT of the peer
    channels over an intercept-plus-own null, never a raw R^2.
    """
    pm = max(1e-12, float(phi_max))
    return (np.asarray(phi_i, dtype=np.float64) - 0.5 * pm) / pm


# --------------------------------------------------------------------------- #
# Component 4 -- the estimator             (PACT_PIPELINE_SPEC 5)
# --------------------------------------------------------------------------- #
class RLS:
    """Recursive least squares with forgetting, and a covariance windup bound.

    5.1  mu = 0.9995 by default, MEASURED on POWER, not assumed.  Aggressive
    forgetting bought nothing there and injected noise straight into the
    coefficient the inverse divides by::

        mu      0.990  0.995  0.997  0.999  0.9995  0.9999
        corr    0.556  0.500  0.481  0.525  0.555   0.572
        dg_std  0.395  0.317  0.231  0.093  0.055   0.030

    RE-MEASURE PER ENVIRONMENT -- the optimum follows the drift rate, and
    ``fc/calibrate.py --sweep mu`` does it here.

    5.2  COVARIANCE WINDUP BOUND **[PAID]**.  Forgetting inflates unexcited
    directions by 1/mu on every update WITHOUT BOUND.  As the policy converges
    excitation dies and the estimator runs away: measured ``se(own_gain)`` across
    quarters 29 -> 4.9e3 -> 8.4e5 -> 1.4e8, with (1/0.9995)^83000 ~ 1e18.  Return
    tracked it exactly -- the method led in Q1-Q3 and lost 28.8 in Q4.  Rescaling
    preserves RELATIVE uncertainty while stopping the absolute scale diverging.
    """

    def __init__(self, dim, mu=0.9995, p0=1.0, p_max_mult=10.0, directional=False):
        self.dim = int(dim)
        self.mu = float(mu)
        self.p0 = float(p0)
        self.p_max = float(p_max_mult) * self.p0 * self.dim
        # Directional (Kulhavy) forgetting discounts information ONLY in the
        # subspace the data excited, so unexcited directions keep their prior
        # certainty.  Not the spec's default -- kept as a declared ablation
        # against the plain-forgetting-plus-trace-bound recipe of 5.1/5.2.
        self.directional = bool(directional)
        self.reset()

    def reset(self):
        self.P = np.eye(self.dim) * self.p0
        self.beta = np.zeros(self.dim)
        self.n_upd = 0
        self.n_clamp = 0
        self.n_restart = 0
        self.sigma2 = 1.0          # innovation variance EMA, for the std errors
        self.last_innov = 0.0

    def predict(self, psi):
        return float(np.dot(self.beta, np.asarray(psi, dtype=np.float64).reshape(-1)))

    def update(self, psi, y):
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        Pp = self.P @ psi
        s = float(psi @ Pp)
        e = float(y) - float(psi @ self.beta)
        if self.directional:
            den = 1.0 + s
            if den > 1e-12:
                K = Pp / den
                self.beta = self.beta + K * e
                self.P = self.P - np.outer(Pp, Pp) / den
                if s > 1e-12:
                    Pp2 = self.P @ psi
                    s2 = float(psi @ Pp2)
                    if s2 > 1e-12:
                        self.P = self.P + ((1.0 - self.mu) / self.mu) * (
                            np.outer(Pp2, Pp2) / s2)
        else:
            den = self.mu + s
            if den > 1e-12:
                K = Pp / den
                self.beta = self.beta + K * e
                self.P = (self.P - np.outer(K, Pp)) / self.mu
        self.P = 0.5 * (self.P + self.P.T)      # keep symmetric; the gates read P
        # --- 5.2, and note the ORDER: non-finite is tested FIRST, as a VALUE.
        tr = float(np.trace(self.P))
        if not np.isfinite(tr):
            self.P = np.eye(self.dim) * self.p0     # diverged: restart the prior
            self.n_restart += 1
        elif tr > self.p_max:
            self.P *= self.p_max / tr
            self.n_clamp += 1
        self.last_innov = e
        self.sigma2 = 0.999 * self.sigma2 + 0.001 * (e * e)
        self.n_upd += 1
        return self.beta

    def se(self, k):
        """Standard error of coefficient k -- the |t| > 3 check of 9's table."""
        v = float(self.P[k, k]) * max(1e-30, float(self.sigma2))
        return math.sqrt(v) if np.isfinite(v) and v >= 0.0 else float("inf")

    def trace(self):
        return float(np.trace(self.P))


# --------------------------------------------------------------------------- #
# 4.3 -- the null model is mandatory
# --------------------------------------------------------------------------- #
class FitGain:
    """Windowed one-step-ahead R^2 LIFT of the peer channels over the null.

    4.3 **[PAID]**.  A pooled R^2 of 0.9998 looked like a triumph on POWER until
    the intercept-only model was scored too and came in at 0.656; the honest
    quantity was 0.344.  Score PRIOR (one-step-ahead) predictions, never
    posterior fits, and skip a warmup before accumulating -- a cold start
    otherwise dominates both sums and their difference is noise.

    8.2 **[PAID]**.  Gate on this, not on a covariance proxy: POWER's Q1 ran
    applied_trust 0.095 with fit_gain 0.0006 -- the compensator on at 10% strength
    while its estimate was worthless -- and lost 6-14 return per iteration over
    that stretch.  And use a WINDOWED version: a cumulative one is held down
    forever by early negatives.
    """

    def __init__(self, lam=0.999, warmup=200):
        self.lam = float(lam)
        self.warmup = int(warmup)
        self.n = 0
        self.sse_full = 0.0
        self.sse_null = 0.0
        self.sst = 0.0
        self.ymean = 0.0

    def observe(self, y, pred_full, pred_null):
        self.n += 1
        self.ymean = self.lam * self.ymean + (1.0 - self.lam) * float(y)
        if self.n <= self.warmup:
            return
        lam = self.lam
        self.sse_full = lam * self.sse_full + (1 - lam) * (float(y) - float(pred_full)) ** 2
        self.sse_null = lam * self.sse_null + (1 - lam) * (float(y) - float(pred_null)) ** 2
        self.sst = lam * self.sst + (1 - lam) * (float(y) - self.ymean) ** 2

    def value(self):
        """``R2(full) - R2(null)``.  NaN, never an epsilon, when the target has no
        variance to explain -- 9: 'guard every ratio with NaN'."""
        if self.n <= self.warmup or self.sst <= 0.0:
            return float("nan")
        return float((self.sse_null - self.sse_full) / self.sst)

    def r2_full(self):
        if self.n <= self.warmup or self.sst <= 0.0:
            return float("nan")
        return float(1.0 - self.sse_full / self.sst)


# --------------------------------------------------------------------------- #
# Component 5 -- the channel inverse       (PACT_PIPELINE_SPEC 6)
# --------------------------------------------------------------------------- #
def compensation_delta(deficit, du_da, gain=1.0, max_delta=3.0):
    """THE channel inverse.  Every caller and every test goes through here.

    6.2 **[PAID]**.  The sign chain lives in ONE place.  Two places disagreeing on
    the sign once made a passing unit test assert the wrong convention, so this
    function takes ``du_da`` -- one unambiguous quantity, the sensitivity of the
    DELIVERED step to the commanded stride -- and refuses to accept a raw
    regression coefficient.  Written out::

        delivered(cmd) = base * cmd * (1 - Delta)
        du_da          = d(delivered)/d(cmd) / base = (1 - Delta)      always > 0
        deficit        = the delivered shortfall to cancel, in units of `base`
        d              = gain * deficit / du_da

    At full trust and a perfect estimate this is EXACT, not linearised:
    ``d = Delta/(1 - Delta)`` gives ``cmd = 1/(1 - Delta)`` and the delivered step
    is restored byte for byte -- the conjugacy the frontier argument needs.

    6.3 asks for a SIGNIFICANCE floor rather than a magnitude floor when the
    divisor is estimated.  Here it is ANALYTIC (6.1), so there is no estimation
    error to test; what remains is the physical guard that the channel is not so
    throttled that the inverse demands an unbounded command.

    Returns ``(delta, floored)`` -- ``floored`` says the guard fired, so it can be
    counted instead of silently applied.
    """
    d0 = float(du_da)
    if not np.isfinite(d0) or abs(d0) < DU_DA_FLOOR:
        return 0.0, True
    if not np.isfinite(float(deficit)):
        return 0.0, True
    d = float(gain) * float(deficit) / d0
    return float(np.clip(d, -float(max_delta), float(max_delta))), False


# --------------------------------------------------------------------------- #
# The per-agent compensator: 6.4, 7 and 8 wired together
# --------------------------------------------------------------------------- #
class AgentCompensator:
    """One agent's estimator, gates and correction.  Fully decentralized: it sees
    its own psi, its own deficit reading, and nothing about any peer's estimate.
    """

    def __init__(self, r, mu=0.9995, p0=1.0, max_trust=1.0, fit_floor=0.0,
                 ready_updates=200, level_tau=1.0, peer_mode="delta",
                 ff_gain=1.0, max_delta=3.0, fit_lam=0.999, fit_warmup=200,
                 u_active=0.05, directional=False, p_max_mult=10.0):
        self.r = int(r)
        self.dim = 2 + self.r                     # [1, own_col, psi_peer...]
        self.rls = RLS(self.dim, mu=mu, p0=p0, p_max_mult=p_max_mult,
                       directional=directional)
        self.fit = FitGain(lam=fit_lam, warmup=fit_warmup)
        self.max_trust = float(max_trust)
        self.fit_floor = float(fit_floor)
        self.ready_updates = int(ready_updates)
        self.ff_gain = float(ff_gain)
        self.max_delta = float(max_delta)
        # 6.4: compensate the FLUCTUATION, not the pedestal.
        self.level_tau = max(1.0, float(level_tau))
        # "delta"  -- the peer term carries the predicted one-step CHANGE, so it
        #             cannot double-count what the local feedforward already did.
        #             This is 6.4's level with tau = 1 and is the default here
        #             BECAUSE W is zero-diagonal on SMAC: unlike POWER, where the
        #             agent's own exertion is 77% of its own loading, here the
        #             peer sum IS essentially the whole excess, so a slow level
        #             would leave d_ff and d_peer cancelling the same quantity
        #             twice and burn the actuator budget on a constant.
        # "ema"    -- 6.4 literally: a slow EMA with the declared tau.  The
        #             ablation row.
        assert peer_mode in ("delta", "ema"), peer_mode
        self.peer_mode = peer_mode
        self.level = 0.0
        self.prev_ell = 0.0
        # 7: u_i recovered from the deficit needs the driver to be ACTIVE; below
        # this the ratio is meaningless and is skipped rather than epsilon-floored.
        self.u_active = float(u_active)
        self.u_hat = 0.0
        self.n_u = 0
        # diagnostics
        self.last = dict(ell_hat=0.0, ell_ctrl=0.0, d_peer=0.0, d_ff=0.0,
                         delta=0.0, trust=0.0, admissible=0.0, du_da=1.0,
                         floored=0.0, clipped=0.0)

    # -- estimator ---------------------------------------------------------- #
    def regressor(self, own_col, psi_peer_prev):
        """4.2: the regressor is ASYMMETRIC IN TIME.  Own column CURRENT, peer
        columns PREVIOUS.  Regressing y(t) on the whole of psi(t-1) lags the own
        column by one step and corrupts the very coefficient the inverse uses."""
        return np.concatenate(([1.0, float(own_col)],
                              np.asarray(psi_peer_prev, dtype=np.float64).reshape(-1)))

    def observe(self, own_col, psi_peer_prev, y):
        """One deficit reading.  Scores the PRIOR prediction before updating."""
        x = self.regressor(own_col, psi_peer_prev)
        pred_full = self.rls.predict(x)
        b = self.rls.beta
        pred_null = float(b[0] + b[1] * float(own_col))
        self.fit.observe(y, pred_full, pred_null)
        self.rls.update(x, y)
        return pred_full

    def peer_predict(self, own_col, psi_peer_now):
        """``ell_hat`` -- the peer-driven part of the deficit the NEXT step will
        suffer.  1's diagram: ``beta_hat[peer part] . psi``, the peer block only."""
        b = self.rls.beta
        return float(np.dot(b[2:], np.asarray(psi_peer_now,
                                              dtype=np.float64).reshape(-1)))

    # -- gates -------------------------------------------------------------- #
    def admissible(self):
        """8.1 **[PAID]**: separate WHETHER from HOW MUCH.

        RLS returns the LEAST-SQUARES prediction, and for a least-squares
        predictor the residual-minimising gain is exactly
        ``g* = Cov(ell_hat, ell)/Var(ell_hat) = 1``, because the LS prediction IS
        the conditional mean.  T4 pulls g* below 1 and estimation noise pulls it
        further -- to an INTERIOR optimum, not to nothing.  Multiplying three
        heuristic confidences produced applied_trust 0.024 against fit_gain 0.016,
        ~40x below the theoretical gain; binary admissibility x a calibrated
        constant took applied trust to 0.192 and delta_abs to 0.0084.

        8.3 **[PAID]**: the trace gate is a TRAP and is deliberately absent here.
        tr(P) is dominated by the LEAST excited direction, which forgetting
        inflates without bound, so once the policy converges it DISARMS a working
        compensator: applied trust 0.173 -> 0.003 while fit_r2 still read 0.9998
        and the learning curve looked healthy.  Nothing looked wrong.
        """
        fg = self.fit.value()
        ready = self.rls.n_upd >= self.ready_updates
        ok = ready and np.isfinite(fg) and fg > self.fit_floor
        return bool(ok), (fg if np.isfinite(fg) else float("nan"))

    # -- the correction ----------------------------------------------------- #
    def correction(self, psi_peer_now, one_minus_g_next, delta_meas, one_minus_g_now):
        """The commanded stride offset for the coming step.

        Returns ``(delta, info)``.  ``delta`` is added to a commanded stride of
        1.0, so ``delta == 0`` means the executed action is BYTE-IDENTICAL to the
        blind arm -- 1's floor property: a diverging estimate can fail to help, it
        must never do worse.
        """
        # --- 7: the analytic driver feedforward.  Closed form, no estimator, no
        # gate, nothing to converge -- and LOCAL, so it supports no coordination
        # claim.  It is logged separately (`ff_abs`) from the coordination term
        # (`peer_abs`) and the split is reported.  POWER measured 79% / 21%.
        if one_minus_g_now > self.u_active and np.isfinite(delta_meas):
            u_obs = float(delta_meas) / float(one_minus_g_now)
            self.u_hat = 0.9 * self.u_hat + 0.1 * u_obs if self.n_u else u_obs
            self.n_u += 1
        ff_excess = self.u_hat * float(one_minus_g_next)

        # --- 6.4: compensate the fluctuation, not the pedestal.
        ell_hat = self.peer_predict(0.0, psi_peer_now)
        if self.peer_mode == "ema":
            self.level += (1.0 / self.level_tau) * (ell_hat - self.level)
            ell_ctrl = ell_hat - self.level
        else:
            ell_ctrl = ell_hat - self.prev_ell
        self.prev_ell = ell_hat

        adm, fg = self.admissible()
        trust = self.max_trust if adm else 0.0

        # --- 6.1: the divisor is ANALYTIC, read off the channel, never learned.
        # Learning it failed completely on POWER: own_gain 0.13 with se 5.0, i.e.
        # |t| = 0.024, never once clearing a t>3 bar in 5004 logged windows,
        # because excitation dies as the policy converges (cond 108 -> 909).
        du_da = 1.0 - float(np.clip(ff_excess, 0.0, 1.0 - DU_DA_FLOOR))

        d_ff, fl1 = compensation_delta(self.ff_gain * ff_excess, du_da, 1.0,
                                       self.max_delta)
        d_peer, fl2 = compensation_delta(ell_ctrl, du_da, trust, self.max_delta)
        raw = d_ff + d_peer
        delta = float(np.clip(raw, -self.max_delta, self.max_delta))
        self.last = dict(ell_hat=float(ell_hat), ell_ctrl=float(ell_ctrl),
                         d_peer=float(d_peer), d_ff=float(d_ff), delta=float(delta),
                         trust=float(trust), admissible=float(adm), du_da=float(du_da),
                         floored=float(fl1 or fl2),
                         clipped=float(abs(raw - delta) > 1e-12),
                         fit_gain=float(fg), u_hat=float(self.u_hat))
        return delta, self.last
