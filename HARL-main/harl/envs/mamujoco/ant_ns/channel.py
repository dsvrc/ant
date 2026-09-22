"""The trunk-reaction channel: disturbance, sensor, estimator and exact inverse.

Pure numpy.  The layer (``layer.py``) hands this object the commanded torques and
the torques the actuators delivered last step, and gets back the torques to put
on the actuators.  Everything the spec gates is therefore testable without
MuJoCo (``selftest.py``), and the in-simulator suite has only to check the
wiring.

THE CHANNEL, IN ONE PASS
--------------------------------------------------------------------------------
    a_i          the torque the POLICY commanded              (its action)
    e_i          the direction the trunk reaction arrives along   PUBLIC
    d_i          = e_i * (beta*(t) . x_i) / load_norm            PRIVATE
    c_i          = g_i * dhat_i          the feed-forward, clipped   (II.6)

    u_sent_i     = a_i - c_i * e_i
    u_exec_i     = clip( u_sent_i + d_i ,  the actuator's own range )
    y_i          = < u_exec_i - u_sent_i , e_i >     the sensor          (P-2.1)

WHY THE INVERSE IS EXACT HERE
--------------------------------------------------------------------------------
The disturbance is a torque and the action is a torque, so the correction lives
in the same space and subtracts (II.6's first row).  With ``c_i = |d_i|`` the
executed torque is EXACTLY the commanded one and the whole trajectory is the
sigma = 0 trajectory, bit for bit -- which is why ``oracle`` is a ceiling and not
a competitor, and why ``smoke.py`` can assert that identity rather than argue it.

It stops at the actuator's own range.  That clip is the HOST's (MuJoCo clamps
``ctrl`` to the model's ``ctrlrange``), it is applied here only so the executed
torque can be MEASURED and reported, and it is where sigma* comes from: past it
the machine cannot deliver the compensation however well it is estimated.

THE SENSOR IS PROPRIOCEPTION (P-2.1)
--------------------------------------------------------------------------------
A joint knows the torque it asked its drive for and the torque the drive
delivered -- motor current against command, which every servo already measures.
The residual projected on the public direction is that shortfall.  It never sees
another agent's residual (P-4.1); it sees only peers' executed torques, which a
machine's own bus already carries.

THE ESTIMATOR IS THE SHARED OBJECT
--------------------------------------------------------------------------------
``AgentRLS`` and ``rls_confidence_pred`` are imported from the verbatim vendored
copy of URB's ``pact1/core.py`` (``harl/envs/smac/smac_ns/pact1_core.py``) -- the
same object ``smac_ns`` and ``grf_ns`` use.  The one addition is BenchMARL's
covariance-windup bound (``p_trace_max``, the porting brief's trap 7), applied
to the estimator's state from outside rather than by editing it.

Learned trust (P-6.2) is NOT implementable on this channel: the correction is a
deterministic transform applied below the policy, so ``d log pi / d g = 0`` and
a trust head would receive no gradient.  The arms are therefore ``off`` /
``fixed`` / ``oracle`` / ``intercept``, exactly as in ``simple_ns`` and
``grf_ns``, and the README says so.
"""

from dataclasses import dataclass

import numpy as np

from harl.envs.smac.smac_ns.pact1_core import AgentRLS, rls_confidence_pred

from .structure import N_JOINTS

__all__ = ["PactConfig", "BoundedRLS", "TrunkChannel"]


@dataclass(frozen=True)
class PactConfig:
    enabled: bool = False
    """False = the blind arm: no estimator, no correction, the dial only."""
    trust: str = "off"
    """``off`` (forced to 0, bit-identical to blind) or ``fixed``."""
    g_fixed: float = 0.9
    """P-5.1's inverted prior: near full reliance, gated by prediction confidence."""
    oracle: bool = False
    """Hand the TRUE disturbance to the compensator -- the ceiling arm.  An arm
    INSIDE the channel, not a wrapper: computed outside it would be one step
    stale, which understated the ceiling enough on a previous instance that PACT
    appeared to BEAT it (the porting brief's trap 5)."""
    intercept_only: bool = False
    """Delete the peer channels, keep everything else.  The (B)-vs-(C) probe."""
    mu: float = 0.99
    """Forgetting factor, a ~100-step memory.  NOT inherited from URB's 0.999
    (trap 9): declared, and swept on PREDICTION ERROR, never on return (trap 10)."""
    p0: float = 10.0
    warmup: int = 50
    """Rows an agent's estimator must absorb before trust engages."""
    p_trace_max: float = 100.0
    """Covariance windup bound, a multiple of the initial trace.  0 disables."""


class BoundedRLS(AgentRLS):
    """URB's ``AgentRLS`` with BenchMARL's covariance bound applied to its state.

    RLS with forgetting divides P by mu every update, so in a direction the data
    stops exciting P grows without bound and the prediction eventually overflows
    (measured on a real run before this existed: 5.7M non-finite predictions out
    of 12M agent-steps, and an applied correction worth 4% of the disturbance --
    the method had not been beaten by the baseline, it had never run).  When the
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


class TrunkChannel(object):
    """All per-agent state of the dial and the compensator.  numpy only."""

    def __init__(self, p, driver, coupling, pact, load_norm=None, ref=None, scale=None,
                 clock0=0, ctrl_range=1.0):
        self.p = p
        self.driver = driver
        self.coupling = coupling
        self.pact = pact
        self.n = int(coupling.n)
        self.r = int(coupling.r)
        self.dim = 1 if pact.intercept_only else 1 + self.r
        self.clock = int(clock0)
        self.ctrl_range = np.broadcast_to(
            np.asarray(ctrl_range, dtype=np.float64), (N_JOINTS,)).copy()
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
        # NS-3.3 counters
        self.n_steps = 0
        self.n_dial_live = 0
        self.n_harmed = 0
        self.n_clipped = 0
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
        the drivetrain does not cool because a training episode ended, and beta*
        drifts far more slowly than an episode lasts."""
        n, r = self.n, self.r
        self.Q = np.zeros((r, N_JOINTS))
        self.tau_prev = np.zeros(N_JOINTS)
        self.e = np.zeros(N_JOINTS)
        self.x = np.zeros((n, r))
        self.mag = np.zeros(n)
        self.d = np.zeros(N_JOINTS)
        self.c = np.zeros(n)
        self.y = np.full(n, np.nan)
        self.pred = np.zeros(n)
        self.conf = np.zeros(n)
        self.trust = np.zeros(n)
        self.psi = np.zeros((n, self.dim))
        self.live = np.zeros(n, dtype=bool)
        self.A = 0.0
        self.amp = 0.0
        self.tau_rms = 0.0

    # ------------------------------------------------------------------ the step
    def step(self, a_cmd, clock=None):
        """One environment step.  ``a_cmd`` is the flat commanded torque ``(8,)``.

        Returns the flat torque to put on the actuators.
        """
        if not self.references_ready():
            raise RuntimeError("TrunkChannel.step before set_references")
        if clock is not None:
            self.clock = int(clock)
        a = np.asarray(a_cmd, dtype=np.float64).reshape(N_JOINTS)

        # 1. the public channels, from peers' EXECUTED torques last step
        self.Q = self.coupling.step_channels(self.Q, self.tau_prev)
        e, x = self.coupling.project(self.Q)
        self.e, self.x = e, x

        # 2. the driver
        self.A = float(self.driver.A(self.clock))
        amp = float(self.driver.amplitude(self.clock))
        self.amp = amp

        # 3. the disturbance
        d = np.zeros(N_JOINTS)
        mag = np.zeros(self.n)
        if self.p.direct:
            #  THE (B) CONTROL -- exogenous, not interaction-mediated.
            #
            #  Same driver, same amplitude, same channel, same reward, same
            #  ladder.  The ONLY change is that the disturbance no longer passes
            #  through the neighbours: every agent gets it directly, opposing its
            #  own commanded torque, with no sum over j != i anywhere.
            #
            #    * A LONE agent now feels it, so the N = 1 test separates the two
            #      cells by measurement rather than by argument.
            #    * PACT's PEER CHANNELS carry no information about it -- the
            #      disturbance is the same for every agent, so the regression puts
            #      all of it in the INTERCEPT and the class channels go to zero.
            #
            #  Note what that does and does not predict.  PACT still helps on (B):
            #  a single per-agent adaptive bias tracks a level shift perfectly
            #  well, so "the method fails on B" would be the wrong claim and this
            #  experiment would refute it.  The right claim is sharper: on (B)
            #  nothing about the problem is multi-agent, and `intercept` loses
            #  nothing; on (C) the peer channels are load-bearing and deleting
            #  them removes the recovery.
            for i, grp in enumerate(self.coupling.parts):
                idx = np.asarray(grp, dtype=np.int64)
                nrm = float(np.linalg.norm(a[idx]))
                if nrm > 1e-12:
                    e[idx] = -a[idx] / nrm
                    mag[i] = amp
                    d[idx] = e[idx] * amp
            self.e = e
        else:
            for i, grp in enumerate(self.coupling.parts):
                idx = np.asarray(grp, dtype=np.int64)
                mag[i] = float(self.driver.send @ x[i]) * amp / self.load_norm
                d[idx] = e[idx] * mag[i]
        self.d, self.mag = d, mag
        live = np.abs(mag) > 0.0
        self.live = live

        # 4. the estimator and the correction
        psi = self.coupling.design(x, self.ref, self.scale)
        if self.pact.intercept_only:
            psi = psi[:, :1]
        self.psi = psi
        pred = np.zeros(self.n)
        conf = np.zeros(self.n)
        trust = np.zeros(self.n)
        c = np.zeros(self.n)
        if self.pact.enabled:
            for i in range(self.n):
                pred[i] = float(self.rls[i].predict(psi[i]))
                conf[i] = float(rls_confidence_pred(self.rls[i].P, self.pact.p0,
                                                    self.dim, psi[i]))
            #  P-7.1, enforced rather than hoped for.  A non-finite estimate is
            #  treated as NO INFORMATION -- trust goes to exactly zero and the
            #  floor property returns the untouched host -- because a NaN here
            #  would propagate into the torque, the state and then the policy's
            #  own action, and MuJoCo would warn and the run would be garbage.
            bad = ~(np.isfinite(pred) & np.isfinite(conf))
            if bad.any():
                pred[bad] = 0.0
                conf[bad] = 0.0
                self.n_diverged += int(bad.sum())
            if self.pact.oracle:
                pred = mag.copy()
                conf = np.ones(self.n)
                ready = np.ones(self.n, dtype=bool)
            else:
                ready = self.rows >= int(self.pact.warmup)
            g = float(self.pact.g_fixed) if self.pact.trust == "fixed" else 0.0
            trust = np.where(ready, g, 0.0) * conf
            c = trust * pred
            lim = float(self.p.corr_clip) * float(np.max(self.ctrl_range))
            if lim > 0.0:
                self.n_corr_clipped += int(np.sum(np.abs(c) > lim))
                c = np.clip(c, -lim, lim)
        self.pred, self.conf, self.trust, self.c = pred, conf, trust, c

        # 5. send, execute, measure
        #
        #  THE NET ACTUATOR ERROR IS FORMED FIRST, AND THAT IS NOT A DETAIL.
        #  Writing this as ``clip((a - c*e) + d)`` is the same number in exact
        #  arithmetic and NOT the same in floating point: ``a - x + x`` differs
        #  from ``a`` by an ulp, so the oracle's trajectory would drift from the
        #  sigma = 0 trajectory in the last bits and the ceiling identity could
        #  only be asserted to a tolerance.  Forming ``net = d - c*e`` first
        #  makes it EXACTLY zero when the estimate is exactly right (the same
        #  product subtracted from itself), so ``u_exec = clip(a + 0.0) = a``
        #  bit for bit -- and the same for sigma = 0 and for trust = 0.
        u_sent = a.copy()
        net = np.zeros(N_JOINTS)
        for i, grp in enumerate(self.coupling.parts):
            idx = np.asarray(grp, dtype=np.int64)
            corr = c[i] * e[idx]
            u_sent[idx] = a[idx] - corr
            net[idx] = d[idx] - corr
        u_exec = np.clip(a + net, -self.ctrl_range, self.ctrl_range)
        raw = a + net
        clipped = np.abs(u_exec - raw) > 1e-12
        self.n_clipped += int(clipped.sum())

        y = np.full(self.n, np.nan)
        for i, grp in enumerate(self.coupling.parts):
            idx = np.asarray(grp, dtype=np.int64)
            if float(np.abs(e[idx]).max()) > 0.0:
                y[i] = float((u_exec[idx] - u_sent[idx]) @ e[idx])
        yc = np.clip(y, -float(self.p.y_clip), float(self.p.y_clip))
        fin = np.isfinite(y)
        self.n_clip_hits += int(np.sum(np.abs(y[fin]) > float(self.p.y_clip)))
        self.y = yc

        # 6. identify.  psi(t) was built from tau(t-1) and it is the row that
        #    PRODUCED y(t), so the pair is within the step and no lag is needed
        #    (the porting brief's trap 6 is about pairing y with the wrong row).
        #    The prediction was scored BEFORE this update, so fit_gain is an
        #    honest one-step-ahead number and not a fit to data already absorbed.
        if self.pact.enabled:
            tgt = self.true_beta()
            for i in range(self.n):
                if fin[i] and live[i]:
                    self._panel(i, pred[i], float(yc[i]), tgt)
                    self.rls[i].update(psi[i][None, :], np.array([yc[i]]))
                    self.rows[i] += 1
                    self._ring[self._ring_n % self._ring.shape[0]] = psi[i]
                    self._ring_n += 1
            if self._ring_n % 16 == 0:
                self._update_cond()

        # 7. counters and hand-off
        self.n_steps += 1
        if bool(live.any()):
            self.n_dial_live += 1
        if float(np.abs(u_exec - a).max()) > 1e-12:
            self.n_harmed += 1
        self.tau_rms = float(np.sqrt(np.mean(u_exec ** 2)))
        self.tau_prev = u_exec.copy()
        #  NS-3.4: one tick per environment step, and it is NEVER reset between
        #  episodes.  Forgetting this line froze A(t) at the starting phase, so
        #  the disturbance was a CONSTANT rather than a drift -- and every gate
        #  still passed, because a frozen driver satisfies all of them.
        self.clock += 1
        return u_exec

    # ------------------------------------------------------------------ II.10
    def _panel(self, i, ahead, y, tgt, decay=0.99):
        """Score the one-step-ahead prediction against an INTERCEPT-ONLY null,
        and beta against the TRUTH (this instance is synthetic, so beta* is known
        exactly -- by far the most direct answer to "is it identifying, or is the
        return moving for some other reason").

        beta is scored only while the driver is well up (A > 0.25): beta* is
        proportional to A(t), so as A -> 0 the target vanishes and both the
        cosine and the relative error become divisions by nothing.  The last
        reading on a live driver is held through the placebo.
        """
        d = decay
        self._ybar[i] = d * self._ybar[i] + (1 - d) * y
        self._fit_sse[i] = d * self._fit_sse[i] + (1 - d) * (y - ahead) ** 2
        self._null_sse[i] = d * self._null_sse[i] + (1 - d) * (y - self._ybar[i]) ** 2
        self.fit_gain[i] = 1.0 - self._fit_sse[i] / max(self._null_sse[i], 1e-12)
        if tgt is None or self.A <= 0.25:
            return
        tn = float(np.linalg.norm(tgt))
        if tn <= 1e-9:
            return
        bh = self.rls[i].beta
        den = float(np.linalg.norm(bh)) * tn
        self.beta_cos[i] = float(bh @ tgt) / den if den > 1e-12 else 0.0
        self.beta_err[i] = float(np.linalg.norm(bh - tgt)) / tn

    def true_beta(self):
        """The coefficient vector the regression SHOULD recover right now, in the
        centred and scaled coordinates of ``psi``.  None for the intercept arm.

            mag = sum_m coef_m x_m,   coef_m = amp * send_m / load_norm
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
        return {
            "ns_sigma": float(self.p.severity),
            "ns_A": float(self.A),
            "ns_amp": float(self.amp),
            "ns_placebo": float(bool(self.driver.is_placebo(max(0, self.clock - 1)))),
            "ns_d": float(abs(self.mag[i])),
            "ns_dmax": float(np.abs(self.d).max()),
            "ns_c": float(abs(self.c[i])),
            "ns_y": float(self.y[i]),
            "ns_x_hh": float(self.x[i, 0]),
            "ns_x_aa": float(self.x[i, 1]),
            "ns_x_cross": float(self.x[i, 2]),
            "ns_x_std": float(np.std(self.x)),
            "ns_pred": float(self.pred[i]),
            "ns_conf": float(self.conf[i]),
            "ns_trust": float(self.trust[i]),
            "ns_pred_err": (float(abs(self.pred[i] - self.mag[i]))
                            if self.live[i] else float("nan")),
            "ns_fit_gain": float(self.fit_gain[i]),
            "ns_beta_cos": float(self.beta_cos[i]),
            "ns_beta_relerr": float(self.beta_err[i]),
            "ns_cond_psi": float(self.cond_psi),
            "ns_rows": float(self.rows[i]),
            "ns_clip_frac": float(self.n_clip_hits) / max(1.0, float(self.rows.sum())),
            "ns_live": float(self.live[i]),
            "ns_dial_live": float(self.n_dial_live),
            "ns_harmed": float(self.n_harmed),
            "ns_clipped": float(self.n_clipped),
            "ns_diverged": float(self.n_diverged),
            "ns_bounded": float(sum(r.n_bounded for r in self.rls)),
            "ns_corr_clipped": float(self.n_corr_clipped),
            # P-8.1, the commons: compensating means pushing differently, and
            # pushing differently is exactly what loads your neighbours.  Logged,
            # NEVER acted on -- acting on it would make the method a mechanism
            # rather than a per-agent estimator and break P-4.1.
            "ns_tau_rms": float(self.tau_rms),
            "ns_steps": float(self.n_steps),
        }

    def close_report(self, tag="[ANT-NS]"):
        """NS-3.3: refuse to describe the run as a severity arm if nothing fired."""
        sigma = float(self.p.severity)
        # the DIAL must have fired (a non-zero disturbance reached an agent);
        # whether HARM survived to the actuator depends on the arm -- the oracle
        # cancels it by design, and a zero there is the channel inverse working.
        ok = (sigma <= 0.0) or (self.n_dial_live > 0)
        print("%s steps=%d  dial_live=%d  harmed=%d  actuator_clipped=%d  "
              "corr_clipped=%d  diverged=%d  rows=%d  sensor_clip_hits=%d"
              % (tag, self.n_steps, self.n_dial_live, self.n_harmed, self.n_clipped,
                 self.n_corr_clipped, self.n_diverged, int(self.rows.sum()),
                 self.n_clip_hits))
        if not ok:
            print("%s[REFUSE] severity is %.3f but the dial NEVER produced a non-zero "
                  "disturbance.  This run MUST NOT be reported as a severity arm "
                  "(NS-3.3)." % (tag, sigma))
        elif sigma > 0.0 and self.n_harmed == 0 and not self.pact.oracle:
            print("%s[WARN] the dial fired but the executed torque never differed from "
                  "the commanded one -- either sigma is too small to bite or the "
                  "compensator cancelled everything." % tag)
        return ok
