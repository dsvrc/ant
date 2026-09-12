"""The lane-swerve channel: disturbance, sensor, estimator and inverse, per step.

Pure numpy.  The layer (``layer.py``) only reads GRF's raw observation and hands
this object positions, ball ownership and the agents' commanded actions; it
gets back the actions to send.  Everything the spec gates can therefore be
tested without a simulator (``selftest.py``), and the in-simulator suite
(``smoke.py``) has only to check that the layer wires it correctly.

THE CHANNEL, IN ONE PASS
--------------------------------------------------------------------------------
    intended_i   the heading the agent COMMANDED (its sticky compass direction)
    d_i          swerve the surface imposes, compass steps / step      PRIVATE
    e_i          which way, +1 / -1                                    PUBLIC
    c_i          = g_i * d_hat_i     the compensator's aim-off, clipped   (II.6)
    n_i          = d_i - c_i         the NET rotation still to enact

    acc_i       += n_i
    k_i          = round(acc_i);  acc_i -= k_i          first-order sigma-delta
    executed_i   = rotate(intended_i, k_i * e_i)

    y_i          = k_i + c_i          the sensor: executed - sent, along e_i (P-2.1)

WHY A SIGMA-DELTA MODULATOR AND NOT A COIN
--------------------------------------------------------------------------------
GRF's headings are 45 degrees apart; the surface's effect is a fraction of a
step.  A first-order sigma-delta turns a continuous magnitude into a stream of
whole compass steps whose running mean tracks it with error bounded by half a
step, and the engine's own inertia low-passes that stream into an effective
heading error of ``45 * d`` degrees.  It is deterministic -- no RNG is consumed,
so the blind and PACT arms see identical environments for identical actions --
and it is exactly the PWM a motor driver uses for the same reason.

THE INVERSE IS EXACT, AND THE FLOOR PROPERTY IS EXACT (II.6, P-7.1)
--------------------------------------------------------------------------------
* With ``c_i == d_i`` (the oracle) the net is exactly 0.0, the accumulator never
  moves, ``k_i == 0`` on every step and the executed action IS the commanded
  action -- the sigma = 0 trajectory bit for bit.  That is the channel inverse,
  and ``selftest`` / ``smoke`` assert it rather than argue it.
* With ``g == 0`` the correction is exactly 0.0 however wrong the estimate, so
  ``pactoff`` enacts exactly the rotations the blind arm enacts.
* A non-finite estimate is treated as no information (trust 0), and the
  correction is bounded by ``corr_clip``, so a diverged estimator can fail to
  help but cannot do worse than the host it wraps -- and cannot score well by
  accident through a large saturating kick (trap 11).

THE SENSOR IS PROPRIOCEPTION (P-2.1)
--------------------------------------------------------------------------------
The agent knows the heading it sent and the heading that was enacted -- the
same actuator-side residual ``simple_ns`` reads off the clipped force.  It
never sees another agent's residual (P-4.1); it sees only where its teammates
are, which every player on a pitch can see.

THE ESTIMATOR IS THE SHARED OBJECT
--------------------------------------------------------------------------------
``AgentRLS`` and ``rls_confidence_pred`` are imported from the vendored,
verbatim copy of URB's ``pact1/core.py`` (``harl/envs/smac/smac_ns/pact1_core``).
The one addition is a bound on covariance windup (BenchMARL's ``p_trace_max``,
trap 7), applied to the estimator's state from outside rather than by editing
it.  Learned trust (P-6.2) is NOT implementable on this channel -- the
correction is a deterministic transform of the sampled action, so
``d log pi / d g = 0`` -- and the arms are therefore ``off`` / ``fixed`` /
``oracle`` / ``intercept``.  ``simple_ns`` has the same limitation and states it.
"""

from dataclasses import dataclass

import numpy as np

from harl.envs.smac.smac_ns.pact1_core import AgentRLS, rls_confidence_pred

from .actions import (DIR_FIRST, IDLE, RELEASE_DIRECTION, is_direction, rotate)

__all__ = ["PactConfig", "BoundedRLS", "SwerveChannel"]


@dataclass(frozen=True)
class PactConfig:
    enabled: bool = False
    """False = the blind arm: no estimator, no correction, the dial only."""
    trust: str = "off"
    """``off`` (trust forced to 0, bit-identical to blind) or ``fixed``."""
    g_fixed: float = 0.9
    """P-5.1's inverted prior: near full reliance, gated by confidence."""
    oracle: bool = False
    """Hand the TRUE disturbance to the compensator -- the ceiling arm.  An arm
    INSIDE the channel, not a wrapper: computed outside it would be one step
    stale (trap 5)."""
    intercept_only: bool = False
    """Delete the peer channels; keep everything else.  The (B)-vs-(C) probe."""
    mu: float = 0.99
    """Forgetting factor, a 100-step memory.  Not inherited from URB's 0.999
    (trap 9): it has to average the sigma-delta quantisation noise AND track a
    6000-step bump, and 100 steps does both with a ~10% lag.  Declared; swept
    on prediction error, never on return (trap 10)."""
    p0: float = 10.0
    warmup: int = 50
    """Rows an agent's estimator must have absorbed before trust engages."""
    p_trace_max: float = 100.0
    """Covariance windup bound, a multiple of the initial trace.  0 disables."""


class BoundedRLS(AgentRLS):
    """URB's ``AgentRLS`` with BenchMARL's covariance bound applied to its state.

    RLS with forgetting divides P by mu every update; in a direction the data
    stops exciting P grows without bound and the prediction eventually overflows
    (measured on a real run: 5.7M non-finite predictions out of 12M).  When the
    trace exceeds ``p_trace_max * p0 * dim`` P is shrunk isotropically, which
    preserves symmetry and positive-definiteness.
    """

    def __init__(self, r, mu, p0, p_trace_max):
        super(BoundedRLS, self).__init__(r, mu=mu, p0=p0)
        self.p_trace_max = float(p_trace_max)
        self.n_bounded = 0

    def update(self, Phi, y):
        out = super(BoundedRLS, self).update(Phi, y)
        if self.p_trace_max > 0.0:
            cap = self.p_trace_max * self.p0 * self.r
            tr = float(np.trace(self.P))
            if np.isfinite(tr) and tr > cap:
                self.P *= cap / tr
                self.n_bounded += 1
        return out


class SwerveChannel(object):
    """All per-agent state of the dial and the compensator.  numpy only."""

    def __init__(self, n_agents, p, driver, coupling, pact, load_norm=None,
                 ref=None, scale=None, clock0=0):
        self.n = int(n_agents)
        self.p = p
        self.driver = driver
        self.coupling = coupling
        self.pact = pact
        self.r = int(coupling.r)
        self.dim = 1 if pact.intercept_only else 1 + self.r
        self.clock = int(clock0)
        self.load_norm = None if load_norm is None else float(load_norm)
        self.ref = None if ref is None else np.asarray(ref, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        # one estimator per agent -- P-4.1, decentralised; persists across episodes
        self.rls = [BoundedRLS(self.dim, pact.mu, pact.p0, pact.p_trace_max)
                    for _ in range(self.n)]
        self.rows = np.zeros(self.n, dtype=np.int64)
        # II.10 panel, decaying sums
        self._fit_sse = np.zeros(self.n)
        self._null_sse = np.zeros(self.n)
        self._ybar = np.zeros(self.n)
        self.fit_gain = np.zeros(self.n)
        self.beta_cos = np.zeros(self.n)
        self.beta_err = np.zeros(self.n)
        self._ring = np.zeros((256, self.dim))
        self._ring_n = 0
        self.cond_psi = float("nan")
        # NS-3.3 counters -- printed at close, and the run refuses to be a
        # severity arm if they are zero at sigma > 0
        self.n_steps = 0
        self.n_harmed = 0
        self.n_dial_live = 0
        self.n_rot_sent = 0
        self.n_controlled = 0
        self.n_stale = 0
        self.n_diverged = 0
        self.n_corr_clipped = 0
        self.n_clip_hits = 0
        self.reset_episode()

    # ------------------------------------------------------------------ lifecycle
    def references_ready(self):
        return self.load_norm is not None and self.ref is not None and self.scale is not None

    def set_references(self, load_norm, ref, scale):
        self.load_norm = float(load_norm)
        self.ref = np.asarray(ref, dtype=np.float64).reshape(self.r)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(self.r)

    def reset_episode(self):
        """Per-episode buffers.  The CLOCK and the ESTIMATORS persist (NS-3.4):
        the pitch does not dry because a training episode ended, and beta*
        drifts far more slowly than an episode lasts."""
        n = self.n
        self.intended = np.full(n, -1, dtype=np.int64)
        self.applied_rot = np.zeros(n, dtype=np.int64)
        self.acc = np.zeros(n)
        self.Q = np.zeros((n, self.r))
        self.S = np.zeros(n)
        self.x = np.zeros((n, self.r))
        self.d = np.zeros(n)
        self.e = np.ones(n)
        self.c = np.zeros(n)
        self.k = np.zeros(n, dtype=np.int64)
        self.y = np.full(n, np.nan)
        self.pred = np.zeros(n)
        self.conf = np.zeros(n)
        self.trust = np.zeros(n)
        self.psi = np.zeros((n, self.dim))
        self.controlled = np.zeros(n, dtype=bool)
        self.A = 0.0
        self.amp = 0.0
        self.spread = float("nan")

    # ------------------------------------------------------------------ the step
    def step(self, actions, pos, ball_owner=-1, clock=None):
        """One environment step.

        ``actions``   ``(n,)`` the agents' COMMANDED actions (GRF indices)
        ``pos``       ``(n, 2)`` teammates' positions from the LAST observation
        ``ball_owner`` index of the controlled agent holding the ball, or -1
        ``clock``     override of the persistent step clock (tests only)

        Returns the ``(n,)`` actions to send to the engine.
        """
        if not self.references_ready():
            raise RuntimeError("SwerveChannel.step before set_references")
        if clock is not None:
            self.clock = int(clock)
        n = self.n
        a = np.asarray(actions).reshape(n).astype(np.int64)
        pos = np.asarray(pos, dtype=np.float64).reshape(n, 2)

        # 1. the commanded heading is a STATE: a direction action sets it, a
        #    release clears it, everything else leaves it alone
        for i in range(n):
            if DIR_FIRST <= a[i] <= DIR_FIRST + 7:
                self.intended[i] = a[i] - DIR_FIRST
            elif a[i] == RELEASE_DIRECTION:
                self.intended[i] = -1
                self.applied_rot[i] = 0
                self.acc[i] = 0.0
        under_way = self.intended >= 0
        # a step on which the layer OWNS the heading: idle or a direction, with
        # a heading to rotate.  Ball actions and toggles pass through untouched.
        controlled = ((a == IDLE) | is_direction(a)) & under_way

        # 2. the public channels, from public geometry
        recv = np.ones(n)
        if 0 <= int(ball_owner) < n:
            recv[int(ball_owner)] = float(self.p.recv_ball)
        x, s = self.coupling.channels(pos, self.intended, recv)
        self.Q, self.S = self.coupling.filter(self.Q, self.S, x, s)
        self.Q[~under_way] = 0.0
        self.S[~under_way] = 0.0
        self.acc[~under_way] = 0.0
        self.x = x
        self.spread = self.coupling.spread(pos)

        # 3. the driver and the disturbance
        self.A = float(self.driver.A(self.clock))
        amp = float(self.driver.amplitude(self.clock))
        self.amp = amp
        if self.p.direct:
            #  THE (B) CONTROL: the same amplitude reaches everyone directly, in a
            #  fixed public direction, with no sum over j != i anywhere.  A lone
            #  player feels it.  PACT's peer channels then carry no information
            #  and the regression puts all of it in the intercept: fit gain over
            #  an intercept-only null collapses to ~0 -- the measurement the
            #  (B)/(C) classification rests on.
            d = np.full(n, amp)
            e = np.ones(n)
        else:
            beta = amp * self.driver.send                       # (r,) beta*(t)
            d = (self.Q @ beta) / self.load_norm
            e = np.where(self.S > 0.0, -1.0, 1.0)               # AWAY from the traffic
        d = np.where(under_way, d, 0.0)
        self.d, self.e = d, e

        # 4. the estimator and the correction
        psi = self.coupling.design(self.Q, self.ref, self.scale)
        if self.pact.intercept_only:
            psi = psi[:, :1]
        self.psi = psi
        pred = np.zeros(n)
        conf = np.zeros(n)
        trust = np.zeros(n)
        c = np.zeros(n)
        if self.pact.enabled:
            for i in range(n):
                pred[i] = float(self.rls[i].predict(psi[i]))
                conf[i] = float(rls_confidence_pred(self.rls[i].P, self.pact.p0,
                                                    self.dim, psi[i]))
            bad = ~(np.isfinite(pred) & np.isfinite(conf))
            if bad.any():
                pred[bad] = 0.0
                conf[bad] = 0.0
                self.n_diverged += int(bad.sum())
            if self.pact.oracle:
                pred = d.copy()
                conf = np.ones(n)
                ready = np.ones(n, dtype=bool)
            else:
                ready = self.rows >= int(self.pact.warmup)
            g = float(self.pact.g_fixed) if self.pact.trust == "fixed" else 0.0
            trust = np.where(ready, g, 0.0) * conf
            c = trust * pred
            lim = float(self.p.corr_clip)
            if lim > 0.0:
                clipped = np.abs(c) > lim
                self.n_corr_clipped += int(clipped.sum())
                c = np.clip(c, -lim, lim)
            c = np.where(controlled, c, 0.0)
        self.pred, self.conf, self.trust, self.c = pred, conf, trust, c

        # 5. the sigma-delta, on the steps the layer owns
        exec_a = a.copy()
        k = np.zeros(n, dtype=np.int64)
        y = np.full(n, np.nan)
        for i in range(n):
            if controlled[i]:
                self.acc[i] += d[i] - c[i]
                ki = int(np.floor(self.acc[i] + 0.5))
                self.acc[i] -= ki
                rot = int(ki * int(e[i]))
                head_exec = rotate(self.intended[i], rot)
                if DIR_FIRST <= a[i] <= DIR_FIRST + 7:
                    exec_a[i] = head_exec + DIR_FIRST
                    self.applied_rot[i] = rot
                elif rot != self.applied_rot[i]:
                    # idle, but the enacted rotation has to change: re-assert
                    exec_a[i] = head_exec + DIR_FIRST
                    self.applied_rot[i] = rot
                k[i] = ki
                y[i] = ki + c[i]
            elif self.applied_rot[i] != 0 and a[i] != RELEASE_DIRECTION:
                self.n_stale += 1            # a rotation persisted through a pass-through step
        self.k = k
        yc = np.clip(y, -float(self.p.y_clip), float(self.p.y_clip))
        fin = np.isfinite(y)
        self.n_clip_hits += int(np.sum(np.abs(y[fin]) > float(self.p.y_clip)))
        self.y = yc
        self.controlled = controlled

        # 6. identify: one row per acting agent, its OWN residual, paired with the
        #    row that produced it (trap 6) -- psi(t) built before acting, y(t)
        #    measured after; the prediction was scored BEFORE this update.
        if self.pact.enabled:
            tgt = self.true_beta()
            for i in range(n):
                if controlled[i] and fin[i]:
                    self._panel(i, pred[i], float(yc[i]), tgt)
                    self.rls[i].update(psi[i][None, :], np.array([yc[i]]))
                    self.rows[i] += 1
                    self._ring[self._ring_n % self._ring.shape[0]] = psi[i]
                    self._ring_n += 1
            if self._ring_n % 16 == 0:
                self._update_cond()

        # 7. NS-3.3 counters
        self.n_steps += 1
        self.n_controlled += int(controlled.sum())
        if np.any(k != 0):
            self.n_harmed += 1
        if np.any(d[controlled] > 0.0):
            self.n_dial_live += 1
        self.n_rot_sent += int(np.sum(exec_a != a))
        self.clock += 1
        return exec_a

    # ------------------------------------------------------------------ II.10
    def _panel(self, i, ahead, y, tgt, decay=0.99):
        """Score the one-step-ahead prediction against an INTERCEPT-ONLY null,
        and beta against the TRUTH (this instance is synthetic; beta* is known).

        beta is scored only while the driver is well up (A > 0.25).  beta* is
        proportional to A(t), so as A -> 0 the target vanishes and both the
        cosine and the relative error become divisions by nothing; the last
        reading taken on a live driver is held through the rest of the cycle.
        """
        dcy = decay
        self._ybar[i] = dcy * self._ybar[i] + (1 - dcy) * y
        self._fit_sse[i] = dcy * self._fit_sse[i] + (1 - dcy) * (y - ahead) ** 2
        self._null_sse[i] = dcy * self._null_sse[i] + (1 - dcy) * (y - self._ybar[i]) ** 2
        self.fit_gain[i] = 1.0 - self._fit_sse[i] / max(self._null_sse[i], 1e-12)
        if tgt is None or self.A <= 0.25:
            return                                   # nothing (yet) to recover
        tn = float(np.linalg.norm(tgt))
        if tn <= 1e-9:
            return
        bh = self.rls[i].beta
        den = float(np.linalg.norm(bh)) * tn
        self.beta_cos[i] = float(bh @ tgt) / den if den > 1e-12 else 0.0
        self.beta_err[i] = float(np.linalg.norm(bh - tgt)) / tn

    def true_beta(self):
        """The coefficient vector the regression SHOULD recover right now, in the
        centred/scaled coordinates of ``psi``.  None for the intercept arm.

            d = sum_m coef_m Q_m,    coef_m = amp * send_m / load_norm
              = sum_m coef_m ref_m  +  sum_m (coef_m scale_m) z_m
        """
        if self.pact.intercept_only:
            return None
        if self.p.direct:
            return np.concatenate([[self.amp], np.zeros(self.r)])
        coef = self.amp * self.driver.send / self.load_norm
        return np.concatenate([[float(coef @ self.ref)], coef * self.scale])

    def _update_cond(self):
        m = min(self._ring_n, self._ring.shape[0])
        if m < 2 * self.dim or self.dim < 2:
            return
        try:
            sv = np.linalg.svd(self._ring[:m], compute_uv=False)
            self.cond_psi = float(sv[0] / max(sv[-1], 1e-300))
        except np.linalg.LinAlgError:
            self.cond_psi = float("inf")

    # ------------------------------------------------------------------ read-out
    def info(self, i):
        """Per-agent, per-step diagnostics: the II.10 instrument panel."""
        fin = np.isfinite(self.x)
        return {
            "ns_sigma": float(self.p.severity),
            "ns_A": float(self.A),
            "ns_amp": float(self.amp),
            "ns_placebo": float(bool(self.driver.is_placebo(self.clock - 1))),
            "ns_d": float(self.d[i]),
            "ns_e": float(self.e[i]),
            "ns_k": float(self.k[i]),
            "ns_c": float(self.c[i]),
            "ns_y": float(self.y[i]),
            "ns_x_front": float(self.x[i, 0]),
            "ns_x_flank": float(self.x[i, 1]),
            "ns_x_std": float(np.std(self.x[fin])) if fin.any() else 0.0,
            "ns_pred": float(self.pred[i]),
            "ns_conf": float(self.conf[i]),
            "ns_trust": float(self.trust[i]),
            "ns_pred_err": float(abs(self.pred[i] - self.d[i])) if self.controlled[i] else float("nan"),
            "ns_fit_gain": float(self.fit_gain[i]),
            "ns_beta_cos": float(self.beta_cos[i]),
            "ns_beta_relerr": float(self.beta_err[i]),
            "ns_cond_psi": float(self.cond_psi),
            "ns_rows": float(self.rows[i]),
            "ns_clip_frac": float(self.n_clip_hits) / max(1.0, float(self.rows.sum())),
            "ns_controlled": float(self.controlled[i]),
            "ns_harmed": float(self.n_harmed),
            "ns_dial_live": float(self.n_dial_live),
            "ns_rot_sent": float(self.n_rot_sent),
            "ns_stale": float(self.n_stale),
            "ns_diverged": float(self.n_diverged),
            "ns_bounded": float(sum(r.n_bounded for r in self.rls)),
            "ns_corr_clipped": float(self.n_corr_clipped),
            "ns_spread": float(self.spread),
            "ns_steps": float(self.n_steps),
        }

    def close_report(self, tag="[GRF-NS]"):
        """NS-3.3: refuse to describe the run as a severity arm if nothing fired."""
        sigma = float(self.p.severity)
        # the DIAL must have fired (a non-zero disturbance reached a controlled
        # agent); whether HARM was enacted depends on the arm -- the oracle
        # cancels it by design, so a zero there is the channel inverse working.
        ok = (sigma <= 0.0) or (self.n_dial_live > 0)
        print("%s steps=%d  controlled_agent_steps=%d  dial_live_steps=%d  harmed_steps=%d  "
              "rotated_actions_sent=%d  stale=%d  diverged=%d  corr_clipped=%d  "
              "rows=%d  clip_hits=%d"
              % (tag, self.n_steps, self.n_controlled, self.n_dial_live, self.n_harmed,
                 self.n_rot_sent, self.n_stale, self.n_diverged, self.n_corr_clipped,
                 int(self.rows.sum()), self.n_clip_hits))
        if not ok:
            print("%s[REFUSE] severity is %.3f but the dial NEVER produced a non-zero "
                  "disturbance.  This run MUST NOT be reported as a severity arm "
                  "(NS-3.3)." % (tag, sigma))
        elif sigma > 0.0 and self.n_harmed == 0 and not self.pact.oracle:
            print("%s[WARN] the dial fired but no heading was ever rotated: the swerve "
                  "never accumulated to a whole compass step.  sigma is too small to "
                  "bite, or the compensator cancelled everything." % tag)
        return ok
