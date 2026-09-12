"""The severity layer -- a MIXIN, below the method and above the host.

`PACT_NS_SPEC` NS-3.1: the dial sits below the method in the class hierarchy and
is read from the TASK configuration, never from a method's own block.

    StarCraft2Env                       stock, one no-op hook
      +-- SmacNsEnv(SeverityMixin, ..)  THIS FILE.  Every arm gets it, unmodified.
            +-- trust term in the actor  the method, and nothing else

A mixin rather than a wrapper because a wrapper cannot reach inside ``step``
between the engine tick and ``update_units`` -- and that single point is the only
place where harm can touch the reward, the observation, the termination test and
the logged record together.  NS-3.2 is bought once, there, instead of by patching
three call sites and hoping.

THE HARM CHANNEL (NS-1.4)
--------------------------------------------------------------------------------
Overkill.  When several allies pour damage into the same enemy, the part beyond
what that enemy can usefully absorb is wasted -- and the guard cycle shrinks how
much it can absorb.  The layer computes the wasted fraction and hands it back to
the engine as restored hit points on the target, through the debug channel the
host already uses.

    u_i         = peer damage on my target / its published capacity   (j != i)
    excess_i    = f(u_i/g)/f(u_i) - 1,   f(u) = 1 + alpha*u      the harm RATIO
    wasted_i    = dmg_i * excess_i / (1 + excess_i)    the part that did not land
    -> restore sum_i wasted_i onto that enemy

The harm is a RATIO against the same agent's nominal loading, not the performance
function itself.  Overkill already exists in stock StarCraft; that base cost is
the game, not the disturbance.  Only its AMPLIFICATION by a shrunken absorption
capacity is injected -- which is why ``g == 1`` gives exactly zero excess and
sigma = 0 is the stock task byte for byte.

THE REWARD FUNCTION IS UNTOUCHED, BYTE FOR BYTE.  No penalty is subtracted
anywhere; ``reward_battle`` is the host's own and reads the host's own health
deltas.  The team is paid exactly what it was paid before -- there is simply less
dead enemy to be paid for, because the medium delivered less.  That is NS-1.4's
requirement and the reason a penalty term is forbidden: a penalty is a changed
objective, and every comparison against a stock baseline would then be measuring
the objective change.

The restoration is issued through ``controller.debug`` and lands on the engine's
next tick, so the engine stays authoritative and nothing desynchronises.  A
one-step lag is physically right for a regeneration-like effect.

NS-3.3 -- FAIL LOUDLY
--------------------------------------------------------------------------------
``n_harmed`` and ``n_restored_hp`` are counted every step and printed at close.
A silently inert disturbance is the one failure mode indistinguishable from a
clean null result, in exactly the arm you most need to trust, so a run with
severity > 0 that harmed nothing refuses to describe itself as a severity arm.
"""

import numpy as np

from .coupling import Coupling
from .driver import GuardDriver
from .pact1_core import (
    AgentRLS,
    herd_index,
    relative_excess,
    rls_confidence_pred,
)

N_ACTIONS_NO_ATTACK = 6          # no-op, stop, N, S, E, W -- stock SMAC


class SeverityMixin(object):
    """The dial.  Mix in ABOVE ``StarCraft2Env`` so ``_ns_hook`` is overridden."""

    def __init__(self, args, **kwargs):
        a = dict(args or {})

        # *** SET BEFORE THE HOST'S __init__ RUNS. ***
        # StarCraft2Env declares its observation space inside its own __init__,
        # which calls self.get_obs_size() -- and that is OUR override.  Anything
        # an override reads must therefore exist before super() is entered, not
        # after it returns.  This is the whole hazard of a mixin over a host that
        # calls its own virtuals during construction, and `selftest.py`'s
        # `mixin_survives_host_calling_its_overrides_during_init` reproduces the
        # exact crash so it cannot come back silently.
        self.ns_augment = bool(int(a.get("ns_augment", 0)))
        self.ns_cost = None              # get_obs_agent guards on this being set

        super(SeverityMixin, self).__init__(args, **kwargs)

        # ---- read severity from the TASK config (NS-3.1) --------------------
        self.ns_sigma = float(a.get("ns_severity", 1.0))
        self.ns_on = bool(int(a.get("ns_on", 1)))
        self.driver = GuardDriver(
            period=int(a.get("ns_period", self.episode_limit)),
            guard_frac=float(a.get("ns_guard_frac", 0.5)),
            loss=float(a.get("ns_loss", 0.10)),
            mean_preserving=bool(int(a.get("ns_mean_preserving", 0))),
        )
        self.coupling = Coupling(self.map_name, self.n_agents, self.n_enemies,
                                 step_mul=self._step_mul,
                                 alpha=float(a.get("ns_alpha", 2.28)))
        # certified at construction, in every process -- these are cheap
        self.driver.certify()
        self._ns_startup_gates()

        # per-enemy guard sensitivity: declared, mean 1, clipped.  Makes the
        # derating unit-specific so the BINDING element can shift with severity.
        rng = np.random.RandomState(7000 + int(a.get("ns_seed", 0)))
        self.ns_sens = np.clip(1.0 + 0.5 * rng.randn(self.n_enemies), 0.4, 2.0)
        self.ns_sens /= self.ns_sens.mean()

        # ---- the estimator.  Pure identification; identical in every arm. ----
        self.ns_mu = float(a.get("ns_mu", 0.999))
        self.ns_p0 = float(a.get("ns_p0", 10.0))
        self.ns_clip = float(a.get("ns_clip", 10.0))
        self.ns_kappa = float(a.get("ns_kappa", 1.0))
        self.ns_intercept_only = bool(int(a.get("ns_intercept_only", 0)))
        # The `oracle` arm: steer on the TRUE excess instead of the estimate, so
        # it measures the ceiling the estimator is chasing.  It is an arm INSIDE
        # the layer, not a wrapper around it -- computed outside, it would see a
        # different trajectory and stop being a ceiling for this one.
        self.ns_oracle = bool(int(a.get("ns_oracle", 0)))
        # (ns_augment is set ABOVE, before super().__init__ -- see the note there.)
        # It appends the per-action predicted cost to the observation so the
        # policy's steering head can read it.  The head peels the tail off before
        # the base network, so every PACT-family arm has the identical base
        # network and `pactoff` is bit-identical to blind.  Off for the stock
        # baselines, which then run on byte-identical stock SMAC observations
        # inside the same dial.
        self._rls = [AgentRLS(self.coupling.r + 1, mu=self.ns_mu, p0=self.ns_p0)
                     for _ in range(self.n_agents)]

        # ---- clock persists across episodes (NS-3.4) -------------------------
        self.ns_clock = int(a.get("ns_phase0", 0))
        self._ns_reset_state()
        self.ns_n_harmed = 0
        self.ns_n_restored = 0.0
        self.ns_n_steps = 0
        self.ns_n_debug_fail = 0
        self.ns_n_dial_live = 0
        self.ns_tot_dealt = 0.0
        self.ns_tot_wasted = 0.0
        # II.10's headline pair, on PRIOR (one-step-ahead) predictions and scored
        # against an intercept-only null.  Raw R^2 is inflated by the per-agent
        # intercept memorising each agent's typical residual; the LIFT over that
        # null is what means "the peer channels explained something".
        self._fg_lam = 0.999
        self._fg = dict(n=0, sse_full=0.0, sse_null=0.0, sse_fc=0.0, sst=0.0,
                        ybar=0.0)
        self._ns_banner()

    # ------------------------------------------------------------------ gates
    def _ns_startup_gates(self):
        """Gates 1, 3 and 4 of II.9 -- all abort, none warn."""
        c = self.coupling
        rng = np.random.RandomState(0)
        for _ in range(32):
            t = rng.randint(-1, self.n_enemies, self.n_agents)
            f = (rng.rand(self.n_agents) < 0.8).astype(float)
            al = (rng.rand(self.n_agents) < 0.9).astype(float)
            v = c.channels(t, al, f)
            b = c.channels_bruteforce(t, al, f)
            assert np.allclose(v, b, atol=1e-12), (
                "GATE 1 FAILED: vectorised basis != brute-force definition")
        assert np.all(np.diag(c.W()) == 0.0), "GATE: operator diagonal is not zero"
        # GATE 3: a lone agent reads exactly zero peer load, at any severity
        solo = Coupling(self.map_name, 1, self.n_enemies, self._step_mul, c.alpha)
        u = solo.loading(np.array([0]), np.array([1.0]), np.array([1.0]), 1e-3)
        assert float(np.max(np.abs(u))) == 0.0, (
            "GATE 3 FAILED: a lone agent feels the medium -- this is category B")
        # GATE 4: the class order the basis uses is the environment's own
        assert list(c.classes) == sorted(set(c.enemy_names)), \
            "GATE 4 FAILED: class ordering does not match the roster"

    def _ns_banner(self):
        print(self.driver.banner())
        print(self.coupling.banner())
        print("[SMAC-NS] sigma=%.3f on=%d  clock persists across episodes  "
              "mu=%.4f p0=%.1f kappa=%.2f clip=%.1f intercept_only=%d"
              % (self.ns_sigma, int(self.ns_on), self.ns_mu, self.ns_p0,
                 self.ns_kappa, self.ns_clip, int(self.ns_intercept_only)))

    # ------------------------------------------------------------------ state
    def _ns_reset_state(self):
        n, K = self.n_agents, self.n_actions
        self.ns_target = np.full(n, -1, dtype=np.int64)
        self.ns_prev_target = np.full(n, -1, dtype=np.int64)
        self.ns_fired = np.zeros(n)
        self.ns_u = np.zeros(n)
        self.ns_excess = np.zeros(n)
        self.ns_y = np.full(n, np.nan)
        self.ns_psi_prev = np.zeros((n, self.coupling.r + 1))
        self.ns_cost = np.zeros((n, K))          # per-action predicted cost
        self.ns_cost_prev = np.zeros((n, K))     # last step's, for pred_gain
        self.ns_conf = np.zeros(n)
        self.ns_pred_full = np.full(n, np.nan)
        self.ns_clip_hits = 0
        self.ns_rows = 0
        # per-step damage accounting -- the single most diagnostic pair of
        # numbers for "is the NS actually biting?"
        self.ns_dmg_dealt = 0.0
        self.ns_dmg_wasted = 0.0

    def reset(self, *args, **kwargs):
        out = super(SeverityMixin, self).reset(*args, **kwargs)
        # NS-3.4: the clock is NOT reset.  The guard cycle does not restart
        # because a training episode ended, any more than weather does.  The
        # per-episode buffers are cleared; the ESTIMATOR persists too, because
        # beta* drifts far more slowly than an episode lasts.
        self._ns_reset_state()
        return out

    # ------------------------------------------------------------------ driver
    def ns_g(self):
        """Per-element capacity multiplier this step."""
        if not self.ns_on:
            return np.ones(self.n_enemies)
        return np.asarray(self.driver.g(self.ns_clock, self.ns_sigma, self.ns_sens),
                          dtype=np.float64).reshape(self.n_enemies)

    # ------------------------------------------------------------------ hook
    def _ns_hook(self, actions_int):
        """Apply the harm.  Runs after the engine tick, before ``update_units``."""
        n = self.n_agents
        alive = np.array([1.0 if self.agents[i].health > 0 else 0.0
                          for i in range(n)])
        tgt = np.full(n, -1, dtype=np.int64)
        for i, act in enumerate(actions_int):
            if alive[i] > 0 and int(act) >= N_ACTIONS_NO_ATTACK:
                tgt[i] = int(act) - N_ACTIONS_NO_ATTACK
        fired = (tgt >= 0).astype(float)
        self.ns_target, self.ns_fired = tgt, fired

        g = self.ns_g()
        # u at NOMINAL capacity; the dial enters only through the harm RATIO, so
        # sigma = 0 leaves the task byte-identical to stock SMAC (NS-2.1).
        u = self.coupling.loading(tgt, alive, fired, np.ones(self.n_enemies))
        g_i = np.array([g[t] if t >= 0 else 1.0 for t in tgt])
        exc = self.coupling.excess(u, g_i)
        self.ns_u, self.ns_excess = u, exc

        # ---- the sensor (P-2.1): the agent's OWN relative excess -------------
        # realized/nominal - 1, on the cost of delivering a unit of damage.  It is
        # proprioception: a unit knows what its shot should have done and what it
        # did.  Clipped by the DECLARED constant, and clip_frac is reported.
        y = np.full(n, np.nan)
        act_m = fired > 0
        if act_m.any():
            nominal = np.ones(int(act_m.sum()))
            realized = 1.0 + exc[act_m]
            y[act_m] = relative_excess(realized, nominal, clip=self.ns_clip)
            self.ns_clip_hits += int(np.sum(realized - 1.0 > self.ns_clip))
        self.ns_y = y

        # ---- identify: one RLS row per acting agent, its OWN residual --------
        psi = self.coupling.psi(tgt, alive, fired)
        if self.ns_intercept_only:
            psi[:, 1:] = 0.0                 # the `intercept` arm: peer channels off
        for i in range(n):
            if act_m[i] and np.isfinite(y[i]):
                # SCORE THE PRIOR PREDICTION BEFORE UPDATING ON IT.  Scoring the
                # posterior fit measures memorisation, not prediction.
                pf = float(self._rls[i].predict(psi[i]))
                pn = float(self._rls[i].beta[0])          # intercept-only null
                pc = float(self.ns_cost_prev[i, int(actions_int[i])])                     if int(actions_int[i]) >= N_ACTIONS_NO_ATTACK else np.nan
                self.ns_pred_full[i] = pf
                self._fg_observe(y[i], pf, pn, pc)
                self._rls[i].update(psi[i][None, :], np.array([y[i]]))
                self.ns_rows += 1
            self.ns_conf[i] = rls_confidence_pred(
                self._rls[i].P, self.ns_p0, self.coupling.r + 1, psi[i])
        self.ns_psi_prev = psi

        # ---- predict the cost of every candidate target, for the steering ----
        self._ns_predict_costs(tgt, alive, fired)

        # ---- what the squad actually did to the enemy line this step --------
        # self.enemies still holds the PREVIOUS step (update_units has not run),
        # and self._obs holds the current, so the difference is this step's
        # realized damage.  Comparing it against the DECLARED per-unit damage is
        # how we catch an operator whose arithmetic has drifted from the game.
        self.ns_dmg_dealt = self._ns_measure_damage()
        self.ns_dmg_wasted = float(np.sum(
            self.coupling.dmg[fired > 0] * (exc[fired > 0]
                                            / (1.0 + exc[fired > 0]))))
        self.ns_tot_dealt += self.ns_dmg_dealt
        self.ns_tot_wasted += self.ns_dmg_wasted

        # ---- apply the harm to the engine -----------------------------------
        if self.ns_on and self.ns_sigma > 0.0:
            self._ns_restore(tgt, alive, fired, exc)

        self.ns_prev_target = tgt.copy()
        self.ns_cost_prev = self.ns_cost.copy()
        self.ns_n_dial_live += int(float(np.min(g)) < 1.0)
        self.ns_clock += 1
        self.ns_n_steps += 1

    def _ns_measure_damage(self):
        """Realized damage on the enemy line this step, from the health deltas."""
        cur = {}
        try:
            for u in self._obs.observation.raw_data.units:
                cur[u.tag] = float(u.health) + float(u.shield)
        except Exception:
            return float("nan")
        tot = 0.0
        for e, unit in (getattr(self, "enemies", {}) or {}).items():
            if unit is None:
                continue
            was = float(unit.health) + float(unit.shield)
            now = cur.get(unit.tag, 0.0)          # absent => died this step
            if was > now:
                tot += was - now
        return tot

    def _fg_observe(self, y, pred_full, pred_null, pred_forecast):
        """Accumulate II.10's fit_gain / pred_gain on one-step-ahead predictions."""
        f = self._fg
        lam = self._fg_lam
        f["n"] += 1
        f["ybar"] = lam * f["ybar"] + (1 - lam) * float(y)
        if f["n"] <= 200:                 # skip a warmup: a cold start otherwise
            return                        # dominates both sums and their diff
        f["sse_full"] = lam * f["sse_full"] + (1 - lam) * (y - pred_full) ** 2
        f["sse_null"] = lam * f["sse_null"] + (1 - lam) * (y - pred_null) ** 2
        f["sst"] = lam * f["sst"] + (1 - lam) * (y - f["ybar"]) ** 2
        if np.isfinite(pred_forecast):
            f["sse_fc"] = lam * f["sse_fc"] + (1 - lam) * (y - pred_forecast) ** 2

    def _fg_value(self, key):
        """Guarded with NaN, never an epsilon: inside the placebo the target has
        no variance to explain and the ratio is meaningless."""
        f = self._fg
        if f["n"] <= 200 or f["sst"] <= 0.0:
            return float("nan")
        return float((f["sse_null"] - f[key]) / f["sst"])

    def _ns_predict_costs(self, tgt, alive, fired):
        """``cost_hat[i, k]`` over agent i's candidate ATTACK actions.

        Only attack actions carry a prediction; move/stop rows are left at zero
        and are masked out of the z-score by the steering channel, so the shift
        never reorders a move against an attack.
        """
        self.ns_cost[:] = 0.0
        opts = np.arange(self.n_enemies)
        for i in range(self.n_agents):
            if alive[i] <= 0:
                continue
            P = self.coupling.psi_options(i, tgt, alive, fired, opts)
            if self.ns_intercept_only:
                P[:, 1:] = 0.0                     # the `intercept` arm
            if self.ns_oracle:
                # the TRUE excess each option would incur -- the ceiling arm
                gg = self.ns_g()
                u_k = np.array([self._ns_u_if(i, k, tgt, alive, fired)
                                for k in opts])
                self.ns_cost[i, N_ACTIONS_NO_ATTACK:] = self.coupling.excess(u_k, gg)
            else:
                self.ns_cost[i, N_ACTIONS_NO_ATTACK:] = self._rls[i].predict(P)

    def _ns_u_if(self, i, k, tgt, alive, fired):
        """Nominal loading agent i would meet if it attacked element k."""
        ex = np.array(tgt, dtype=np.int64).copy()
        ex[i] = int(k)
        return float(self.coupling.loading(ex, alive, fired,
                                           np.ones(self.n_enemies))[i])

    def _ns_restore(self, tgt, alive, fired, exc):
        """Hand the wasted damage back to the engine as restored hit points."""
        waste = np.zeros(self.n_enemies)
        for i in range(self.n_agents):
            if fired[i] <= 0 or tgt[i] < 0:
                continue
            f = exc[i] / (1.0 + exc[i])              # the part that did not land
            waste[tgt[i]] += self.coupling.dmg[i] * f
        # imported here, not at module scope: I.7's conformance suite must run
        # with no StarCraft II installed, and it imports this module.  A missing
        # protobuf is counted as a refused harm write rather than raised, so the
        # run reports "the dial is not reaching the records" (NS-3.2) instead of
        # dying -- and the runner's [WARN] fires on the very first interval.
        try:
            from s2clientprotocol import debug_pb2 as d_pb
        except ImportError:
            self.ns_n_debug_fail += 1
            return

        cmds = []
        for e in range(self.n_enemies):
            if waste[e] <= 1e-9:
                continue
            unit = self.enemies.get(e, None)
            if unit is None or unit.health <= 0:
                continue
            name = self.coupling.enemy_names[e]
            from .coupling import UNIT_STATS
            max_life = float(UNIT_STATS[name]["life"])
            max_sh = float(UNIT_STATS[name]["shield"])
            give = float(waste[e])
            # shields first, then life -- both capped at the published maxima, so
            # the dial can never credit a unit with more than it ever had.
            if max_sh > 0 and unit.shield < max_sh:
                add = min(give, max_sh - float(unit.shield))
                cmds.append(d_pb.DebugCommand(unit_value=d_pb.DebugSetUnitValue(
                    unit_value=d_pb.DebugSetUnitValue.Shields,
                    value=float(unit.shield) + add, unit_tag=unit.tag)))
                give -= add
            if give > 1e-9 and unit.health < max_life:
                add = min(give, max_life - float(unit.health))
                cmds.append(d_pb.DebugCommand(unit_value=d_pb.DebugSetUnitValue(
                    unit_value=d_pb.DebugSetUnitValue.Life,
                    value=float(unit.health) + add, unit_tag=unit.tag)))
                give -= add
            self.ns_n_restored += float(waste[e]) - max(0.0, give)
        if not cmds:
            return
        try:
            self._controller.debug(cmds)
            self.ns_n_harmed += 1
        except Exception:                       # engine refused; count it loudly
            self.ns_n_debug_fail += 1

    # ------------------------------------------------------------ observation
    def get_obs_agent(self, agent_id):
        o = super(SeverityMixin, self).get_obs_agent(agent_id)
        if not self.ns_augment:
            return o
        # The host may call this during construction, before the per-step buffers
        # exist; a zero tail is the correct value then (no prediction has been
        # made yet) and it keeps the declared shape honest either way.
        cost = (np.zeros(self.n_actions, dtype=np.float32)
                if self.ns_cost is None else self.ns_cost[agent_id])
        return np.concatenate([np.asarray(o, dtype=np.float32),
                               np.asarray(cost, dtype=np.float32)])

    def get_obs_size(self):
        sz = super(SeverityMixin, self).get_obs_size()
        if not self.ns_augment:
            return sz
        sz = list(sz)
        sz[0] = int(sz[0]) + self.n_actions
        return sz

    # ------------------------------------------------------------------ step
    def step(self, actions):
        """Merge the NS diagnostics into every info dict.

        *** THIS OVERRIDE IS LOAD-BEARING. ***  Without it the layer runs, the
        harm lands and every ``ns_*`` column in the debug file reads NaN -- which
        is indistinguishable from "the dial never fired", in exactly the arm you
        most need to trust.  That is NS-3.3's failure mode arriving through the
        diagnostics rather than through the physics, and it cost a 440k-step run.
        """
        out = super(SeverityMixin, self).step(actions)
        if not isinstance(out, tuple) or len(out) < 5:
            return out
        out = list(out)
        infos = out[4]
        d = self.ns_info()
        try:
            for i in range(len(infos)):
                if isinstance(infos[i], dict):
                    infos[i].update(d)
                    infos[i]["ns_u_i"] = float(self.ns_u[i])
                    infos[i]["ns_excess_i"] = float(self.ns_excess[i])
                    infos[i]["ns_target_i"] = float(self.ns_target[i])
        except (TypeError, IndexError):
            pass
        out[4] = infos
        return tuple(out)

    # ------------------------------------------------------------------ report
    def ns_info(self):
        """II.10's instrument panel, in full.

        The design principle the spec insists on: "is the method working" and "is
        it winning" live in SEPARATE columns, because the method can work
        perfectly and still not win, and that is a statement about how much
        headroom the domain has rather than a bug.

        Every ratio is guarded with NaN, never an epsilon.
        """
        live = self.ns_fired > 0
        nz = int(live.sum())
        alive = np.array([1.0 if (self.agents.get(i) is not None
                                  and self.agents[i].health > 0) else 0.0
                          for i in range(self.n_agents)])

        def m(x):
            v = np.asarray(x, dtype=np.float64)[live] if nz else np.asarray([np.nan])
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        def mx(x):
            v = np.asarray(x, dtype=np.float64)[live] if nz else np.asarray([np.nan])
            v = v[np.isfinite(v)]
            return float(v.max()) if v.size else float("nan")

        g = self.ns_g()
        A = float(self.driver.A(self.ns_clock))
        beta = np.mean([r.beta for r in self._rls], axis=0)
        x = self.ns_psi_prev[:, 1:]
        # herd index over the fleet's CHOSEN targets -- P-8.1's commons signature.
        # Logged, never acted on: acting on it would make the method a mechanism
        # rather than a per-agent estimator and break decentralization (P-4.1).
        cnt = np.bincount(self.ns_target[self.ns_target >= 0],
                          minlength=self.n_enemies).astype(np.float64)
        switched = float(np.mean((self.ns_target != self.ns_prev_target)[live]))             if nz else float("nan")
        cond = float("nan")
        try:
            G = self.ns_psi_prev.T @ self.ns_psi_prev / max(1, self.n_agents)
            c = float(np.linalg.cond(G))
            cond = c if np.isfinite(c) else float("inf")
        except np.linalg.LinAlgError:
            cond = float("inf")

        info = {
            # ---- is the DIAL live?  (every arm, baselines included) ----------
            "ns_sigma": float(self.ns_sigma),
            "ns_on": float(self.ns_on),
            "ns_A": A,
            "ns_g": float(np.mean(g)),
            "ns_g_min": float(np.min(g)),
            "ns_placebo": float(bool(self.driver.is_placebo(self.ns_clock))),
            "ns_dial_ratio": (float(self.ns_n_dial_live) / self.ns_n_steps
                              if self.ns_n_steps else float("nan")),
            "ns_clock": float(self.ns_clock),
            # ---- did the harm actually REACH the game? (NS-3.2, NS-3.3) ------
            "ns_harmed": float(self.ns_n_harmed),
            "ns_restored": float(self.ns_n_restored),
            "ns_debug_fail": float(self.ns_n_debug_fail),
            "ns_dmg_dealt": float(self.ns_dmg_dealt),
            "ns_dmg_wasted": float(self.ns_dmg_wasted),
            "ns_waste_frac": (self.ns_dmg_wasted / self.ns_dmg_dealt
                              if self.ns_dmg_dealt > 1e-9 else float("nan")),
            "ns_waste_frac_cum": (self.ns_tot_wasted / self.ns_tot_dealt
                                  if self.ns_tot_dealt > 1e-9 else float("nan")),
            # ---- is the MEDIUM loaded?  (III.1 Q7 -- if not, nothing can bite)
            "ns_u": m(self.ns_u),
            "ns_u_max": mx(self.ns_u),
            "ns_excess": m(self.ns_excess),
            "ns_excess_max": mx(self.ns_excess),
            "ns_y": m(self.ns_y),
            "ns_fire_frac": (float(np.sum(live)) / float(np.sum(alive))
                             if np.sum(alive) > 0 else float("nan")),
            "ns_alive": float(np.sum(alive)),
            "ns_herd_index": float(herd_index(cnt)),
            "ns_switch_frac": switched,
            # ---- is the METHOD working?  (kept apart from "is it winning") ---
            "ns_fit_gain": self._fg_value("sse_full"),
            "ns_pred_gain": self._fg_value("sse_fc"),
            "ns_cond_psi": cond,
            "ns_conf": float(np.mean(self.ns_conf)),
            "ns_rows": float(self.ns_rows),
            "ns_clip_frac": (float(self.ns_clip_hits) / max(1, self.ns_rows)),
            "ns_innov": float(np.mean([r.innov for r in self._rls])),
            "ns_trP": float(np.mean([np.trace(r.P) for r in self._rls])),
            "ns_x_std": float(np.std(x)),
            "ns_x_mean": float(np.mean(x)),
            "ns_cost_std": float(np.std(self.ns_cost[:, N_ACTIONS_NO_ATTACK:])),
        }
        # Always emit the full set, NaN-padded.  The schema is then identical
        # across maps -- 3s5z has r = 2 so ns_beta3 / ns_x2 are honestly NaN,
        # while 1c3s5z (r = 3) fills them -- and the runner's schema check stays
        # an exact equality instead of needing a list of optional columns.
        for k in range(4):
            info["ns_beta%d" % k] = float(beta[k]) if k < len(beta) else float("nan")
        for k in range(3):
            info["ns_x%d" % k] = (float(np.mean(x[:, k])) if k < x.shape[1]
                                  else float("nan"))
        return info

    def ns_close_report(self):
        """NS-3.3: refuse to describe the run as a severity arm if nothing fired."""
        ok = (self.ns_sigma <= 0.0) or (self.ns_n_harmed > 0
                                        and self.ns_n_restored > 0.0)
        print("[SMAC-NS] steps=%d  harmed_steps=%d  hp_restored=%.1f  debug_fail=%d"
              % (self.ns_n_steps, self.ns_n_harmed, self.ns_n_restored,
                 self.ns_n_debug_fail))
        if not ok:
            print("[SMAC-NS][REFUSE] severity is %.3f but nothing was harmed. "
                  "This run MUST NOT be reported as a severity arm (NS-3.3)."
                  % self.ns_sigma)
        return ok


def make_smac_ns_env(args):
    """Build ``SmacNsEnv`` -- the host with the severity layer mixed in."""
    from ..StarCraft2_Env import StarCraft2Env

    class SmacNsEnv(SeverityMixin, StarCraft2Env):
        pass

    return SmacNsEnv(args)
