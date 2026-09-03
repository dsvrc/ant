"""PACT on SMAC -- the compensator, as the outermost environment wrapper.

    StarCraft2Env
      +-- FormationCongestionEnv       the NS dial, applied to EVERY arm
            +-- PactEnv                THIS FILE

PACT_PIPELINE_SPEC 1, and the two properties it says to preserve at all costs:

  * THE HOST RL IS UNTOUCHED.  No loss terms, no critic changes, no extra action
    dimensions, no obs augmentation.  ``--algo pact`` trains the stock MAPPO
    actor on the stock observation and action spaces, so every arm shares
    hyperparameters AND shares checkpoints, and an arm difference cannot be an
    algorithm difference.

  * THE FLOOR PROPERTY.  When the gates say inadmissible the commanded stride is
    exactly 1.0, which is byte-identical to the blind arm's order.  A diverging
    estimate can fail to help; it must never do worse.  ``selfcheck.py`` asserts
    this rather than trusting it.

THE LOOP, PER PACT ONE STEP
---------------------------
::

        obs_i --> pi (host RL, UNCHANGED) --> a_i           (direction; discrete)
                                              |
      peers' EXECUTED exertion (t-1) --> psi --+   exact arithmetic, no estimation
      own deficit Delta_i(t) ----------> y ----+   the sensor: 1 - realized/commanded
                                              v
         RLS(psi, y) --> beta_hat            tracks c(t), the driver's signature
                                              v
         ell_hat = beta_hat[peer part] . psi   minus its standing level (6.4)
         g       = max_trust if admissible else 0                        (8.1)
         d_peer  = g * ell_ctrl / (1 - Delta)          the channel inverse (6)
         d_ff    = ff_gain * u_hat*(1 - g_dial) / (1 - Delta)  analytic driver (7)
                                              v
         cmd_i   = clip(1 + d_peer + d_ff, cmd_min, cmd_max)

TIMING IS ASYMMETRIC AND THAT IS DELIBERATE (4.2).  The own column is CURRENT --
the agent knows its own exertion exactly.  The peer columns are PREVIOUS -- it
cannot observe peers' current actions, and using them would be an oracle.  The
command computed at the end of step t is what gets executed at step t+1, which is
precisely the one-step gap A.4 says is the problem.
"""

import numpy as np

from .pact_core import AgentCompensator, Basis, own_column


def _nanmean(a):
    """np.nanmean over an all-NaN slice both warns and returns NaN; the NaN is
    what spec 9 wants ("guard every ratio with NaN, never an epsilon") and the
    warning is noise, so take the NaN quietly."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _or(v, default):
    """A yaml `~` arrives as None, and dict.get would hand it straight through."""
    return float(default) if v in (None, "", "auto") else float(v)


class PactEnv:
    """The compensator.  Wraps a FormationCongestionEnv and drives its command."""

    def __init__(self, fc_env, args=None):
        self.fc = fc_env
        a = dict(args or {})
        self.n_agents = int(fc_env.n_agents)

        # Component 1: the declared basis, built from the NS layer's own operator.
        self.basis = Basis(fc_env.W, fc_env.phi_max,
                           r=int(a.get("pact_r", 1)),
                           min_coverage=float(a.get("pact_min_coverage", 1e-3)))
        self.r = self.basis.r_req

        # 5.1: mu is MEASURED per environment, not assumed -- "the optimum follows
        # the drift rate".  beta* here is proportional to (1-g)/g, so it drifts on
        # the DRIVER period; an estimator whose memory 1/(1-mu) is longer than that
        # learns the cycle average and cannot track the cycle.  POWER's 0.9995
        # (memory 2000 steps) is right for a driver measured in days and wrong for
        # one measured in tens of steps.  MEASURED here on the real 3s5z basis
        # (selfcheck T6 prints the table): the correlation between beta_hat and a
        # drifting beta* peaks at a memory of roughly period/12, and the constant
        # holds at both period 75 and 150, which is what makes it a heuristic
        # rather than a fit.  `fc/calibrate.py --sweep mu` re-measures it on the
        # live environment; T6 also confirms POWER's 0.9995 does NOT track here.
        mu = a.get("pact_mu", None)
        if mu in (None, "", "auto"):
            mu = float(np.clip(1.0 - 12.0 / max(12.0, fc_env.driver.period),
                               0.50, 0.9999))
        self.mu = float(mu)
        self.enabled = bool(int(a.get("pact", 1)))
        self.max_trust = float(a.get("pact_max_trust", 1.0))
        self.ff_gain = float(a.get("pact_ff_gain", 1.0))
        self.cmd_min = float(a.get("pact_cmd_min", 0.25))
        self.cmd_max = float(a.get("pact_cmd_max", 4.0))
        self.agents = [
            AgentCompensator(
                r=self.r,
                mu=self.mu,
                p0=float(a.get("pact_p0", 1.0)),
                max_trust=self.max_trust,
                fit_floor=float(a.get("pact_fit_floor", 0.0)),
                ready_updates=int(a.get("pact_ready_updates", 200)),
                level_tau=_or(a.get("pact_level_tau", None),
                             5.0 * fc_env.driver.period),
                peer_mode=str(a.get("pact_peer_mode", "delta")),
                ff_gain=self.ff_gain,
                max_delta=float(a.get("pact_max_delta", 3.0)),
                fit_lam=float(a.get("pact_fit_lam", 0.999)),
                fit_warmup=int(a.get("pact_fit_warmup", 200)),
                u_active=float(a.get("pact_u_active", 0.05)),
                directional=bool(int(a.get("pact_directional", 0))),
                p_max_mult=float(a.get("pact_p_max_mult", 10.0)),
            )
            for _ in range(self.n_agents)
        ]

        self.psi_prev = np.zeros((self.n_agents, self.r))
        self.cmd = np.ones(self.n_agents)
        self._fresh = True                 # skip one observe across an episode seam
        self._gram = np.zeros((self.n_agents, 2 + self.r, 2 + self.r))
        self._gram_n = 0
        self._last = [dict() for _ in range(self.n_agents)]
        self._sat = 0
        self._sat_n = 0
        self._banner()

        self.observation_space = fc_env.observation_space
        self.share_observation_space = fc_env.share_observation_space
        self.action_space = fc_env.action_space

    def _banner(self):
        rep = self.basis.report()
        print("[PACT] r_requested=%d r_effective=%d dead_channels=%d "
              "no_coupling_agents=%s range=[%.3f..%.3f]"
              % (rep["r_requested"], rep["r_effective"], len(rep["dead_channels"]),
                 rep["no_coupling_agents"], rep["range_min"], rep["range_max"]))
        print("[PACT] enabled=%d max_trust=%.3f ff_gain=%.3f mu=%.5f "
              "peer_mode=%s cmd=[%.2f..%.2f]"
              % (int(self.enabled), self.max_trust, self.ff_gain,
                 self.agents[0].rls.mu, self.agents[0].peer_mode,
                 self.cmd_min, self.cmd_max))
        if not self.enabled:
            print("[PACT] DISABLED -- this arm is byte-identical to the blind host.")

    # ------------------------------------------------------------------ #
    def _psi_now(self):
        return self.basis.psi(self.fc.contrib, self.fc.exertion.phi, self.fc._alive)

    def _learn_and_plan(self):
        """Score the prior prediction, update the estimator, and compute the
        command for the NEXT step."""
        psi = self._psi_now()
        y = self.fc.sensor()
        phi = self.fc.exertion.phi
        alive = self.fc._alive
        g_now = float(self.fc.g)
        # The driver has already advanced inside fc.step(), so this is g(t+1) --
        # analytic, computable from observable time, and the only thing the
        # feedforward needs beyond the agent's own stale sensor (7).
        g_next = float(self.fc.driver.g()[0])

        cmd = np.ones(self.n_agents)
        for i in range(self.n_agents):
            ag = self.agents[i]
            oc = own_column(phi[i], self.fc.phi_max)
            if not self._fresh and alive[i] > 0:
                ag.observe(oc, self.psi_prev[i], float(y[i]))
                x = ag.regressor(oc, self.psi_prev[i])
                self._gram[i] += np.outer(x, x)
            d, info = ag.correction(psi[i], 1.0 - g_next, float(y[i]), 1.0 - g_now)
            self._last[i] = info
            if self.enabled and alive[i] > 0:
                cmd[i] = float(np.clip(1.0 + d, self.cmd_min, self.cmd_max))
        self._gram_n += 1
        self.psi_prev = psi
        self._fresh = False
        self.cmd = cmd
        return cmd

    # ------------------------------------------------------------------ #
    def reset(self):
        out = self.fc.reset()
        # The bus is cold again: this episode's psi history is stale, so drop it.
        # The ESTIMATE (beta and its covariance) PERSISTS -- the driver drifts on a
        # timescale far longer than an episode, and forgetting it every reset would
        # throw away the only thing that is genuinely learnable across a run.
        self.psi_prev[:] = 0.0
        self.cmd[:] = 1.0
        self._fresh = True
        self.fc.set_command(self.cmd)
        return out

    def step(self, actions):
        # the command computed at the end of the PREVIOUS step is what executes now
        self.fc.set_command(self.cmd)
        out = self.fc.step(actions)
        # rail bookkeeping: a command that cannot buy any more delivered distance
        ma = float(getattr(self.fc.env, "_move_amount", 2.0))
        ordered = ma * self.fc.base_frac * self.fc.stride
        live = self.fc._alive > 0
        if live.any():
            self._sat += int(np.sum(ordered[live] >= self.fc.reach[live] - 1e-9))
            self._sat_n += int(live.sum())
        self._learn_and_plan()
        return self._inject(out)

    # ------------------------------------------------------------------ #
    def diagnostics(self):
        """9's table.  READ ``applied_trust`` AND ``delta_nonzero_frac`` BEFORE ANY
        OTHER NUMBER: they answer "was the method on at all?" and "is it acting?",
        and on POWER a silently disarmed compensator reported fit_r2 = 0.9998 and a
        healthy learning curve while being plain PPO.  Nothing looked wrong.

        Every ratio is guarded with NaN, never an epsilon.
        """
        live = self.fc._alive > 0 if hasattr(self.fc, "_alive") \
            else np.ones(self.n_agents, bool)
        n = int(live.sum())
        idx = [i for i in range(self.n_agents) if live[i]]
        def col(k, default=0.0):
            v = [float(self._last[i].get(k, default)) for i in idx]
            return np.asarray(v) if v else np.asarray([np.nan])
        trust = col("trust")
        delta = col("delta")
        dpeer = col("d_peer")
        dff = col("d_ff")
        fg = col("fit_gain", float("nan"))
        trP = np.asarray([self.agents[i].rls.trace() for i in idx]) if idx \
            else np.asarray([np.nan])
        upd = np.asarray([self.agents[i].rls.n_upd for i in idx]) if idx \
            else np.asarray([0])
        clamps = np.asarray([self.agents[i].rls.n_clamp for i in idx]) if idx \
            else np.asarray([0])
        se = np.asarray([self.agents[i].rls.se(1) for i in idx]) if idx \
            else np.asarray([np.nan])
        # cond of the REGRESSOR's Gram -- 12.6: beta may be predictable without
        # being decomposable where this is ill-conditioned, so report it BEFORE
        # claiming to identify beta itself.  2.4: non-finite is a VALUE and is
        # tested FIRST.
        cond = float("nan")
        if self._gram_n > 0 and idx:
            G = self._gram[idx].mean(axis=0) / max(1, self._gram_n)
            try:
                c = float(np.linalg.cond(G))
            except np.linalg.LinAlgError:
                c = float("inf")
            cond = c if np.isfinite(c) else float("inf")
        applied = _nanmean(trust) if n else float("nan")
        sev = float(self.fc.sigma_applied)
        state = "INERT" if sev <= 0.0 else ("ASLEEP" if applied <= 0.0 else "ALIVE")
        return {
            "pact_applied_trust": applied,
            "pact_delta_nonzero_frac": float(np.mean(np.abs(delta) > 1e-12))
            if n else float("nan"),
            "pact_delta_abs": float(np.mean(np.abs(delta))) if n else float("nan"),
            "pact_delta_clip_frac": float(np.mean(col("clipped"))) if n else float("nan"),
            "pact_ff_abs": float(np.mean(np.abs(dff))) if n else float("nan"),
            "pact_peer_abs": float(np.mean(np.abs(dpeer))) if n else float("nan"),
            "pact_fit_gain_now": _nanmean(fg) if n else float("nan"),
            "pact_cond_psi": cond,
            "pact_trP": float(np.mean(trP)),
            "pact_clamp_frac": (float(np.sum(clamps)) / float(max(1, np.sum(upd)))),
            "pact_own_gain_se": float(np.mean(se)),
            "pact_n_updates": float(np.mean(upd)),
            "pact_du_da": float(np.mean(col("du_da", 1.0))) if n else float("nan"),
            "pact_floor_frac": float(np.mean(col("floored"))) if n else float("nan"),
            "pact_sat_frac": (float(self._sat) / self._sat_n
                              if self._sat_n else float("nan")),
            "pact_cmd_mean": float(np.mean(self.cmd[live])) if n else float("nan"),
            "pact_state": state,
            "pact_enabled": float(self.enabled),
        }

    def _inject(self, out):
        if not isinstance(out, tuple) or len(out) < 5:
            return out
        out = list(out)
        infos = out[4]
        d = self.diagnostics()
        try:
            for i in range(len(infos)):
                if isinstance(infos[i], dict):
                    infos[i].update(d)
        except (TypeError, IndexError):
            pass
        out[4] = infos
        return tuple(out)

    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        try:
            fc = self.__dict__["fc"]
        except KeyError:
            raise AttributeError(name)
        return getattr(fc, name)
