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

    u_i         = peer damage on my target / its remaining hit points  (j != i)
    excess_i    = f(u_i/g)/f(u_i) - 1,   f(u) = 1 + alpha*u      the harm RATIO
    w_e         = sum_i dmg_i f_i / sum_i dmg_i,  f_i = excess_i/(1+excess_i)
    -> restore w_e * (damage that actually LANDED on e this step)

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

The restore is written into the observation snapshot that ``update_units`` reads
next AND sent to the engine through ``controller.debug``, so the reward, the
observation, the termination test and the engine all see it on the same step.  It
never exceeds the damage that landed, so the net hit-point change of every enemy
stays non-negative -- which matters, because stock ``reward_battle`` takes the
``abs()`` of that change and would otherwise pay healing out as damage.  A volley
that kills its target is not restored; a death cannot be undone.

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

_PB = []                         # [module or None] once the import has been tried


def _debug_pb2():
    """``s2clientprotocol.debug_pb2``, or None without StarCraft II installed.

    Imported lazily (the offline suite imports this module) and the OUTCOME is
    cached: a failed import is not cached by Python, and retrying it walks
    sys.path on every env step."""
    if not _PB:
        try:
            from s2clientprotocol import debug_pb2
            _PB.append(debug_pb2)
        except ImportError:
            _PB.append(None)
    return _PB[0]


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
        # One URB "day" is one SMAC battle, and URB's reference cycle is 100 days
        # (PACT_NS_SPEC reference driver, P = 100).  So the default cycle is 100
        # episode limits -- the same convention the GRF port uses (12000 steps).
        # It was one episode limit (150 steps) on the first run, which put the
        # estimator's memory at ~14x the period: beta_hat converged to the cycle
        # AVERAGE, never read zero in the placebo, and the steering pushed agents
        # off focus fire on steps where there was no harm to avoid.
        self.driver = GuardDriver(
            period=int(a.get("ns_period", 100 * self.episode_limit)),
            guard_frac=float(a.get("ns_guard_frac", 0.5)),
            loss=float(a.get("ns_loss", 0.10)),
            mean_preserving=bool(int(a.get("ns_mean_preserving", 0))),
        )
        self.coupling = Coupling(self.map_name, self.n_agents, self.n_enemies,
                                 step_mul=self._step_mul,
                                 alpha=float(a.get("ns_alpha", 2.28)),
                                 cap_mode=str(a.get("ns_cap_mode", "remaining")))
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
        # FIX A: capacity is the target's CURRENT effective hit points.  "max"
        # restores the first run's definition as an ablation.
        self.ns_cap_mode = str(a.get("ns_cap_mode", "remaining"))
        # FIX B: the steering acts only when the options' predicted relative
        # excess actually differs -- by at least this much (std across the valid
        # options).  Declared, never tuned; sweep it in the ablation.  The POLICY
        # applies it (smac_ns/channel.py); the layer reads it only to report how
        # often it would bind.
        if "ns_snr_min" in a:
            raise ValueError(
                "ns_snr_min was removed: it compared the options' spread with ONE "
                "row's prediction error, which gates a well-identified ranking.  "
                "Use ns_steer_floor (a declared floor on the spread itself).")
        self.ns_steer_floor = float(a.get("ns_steer_floor", 0.01))
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
        # P-4.3: the estimator must be able to FOLLOW the driver.  An agent adds at
        # most one row per step, so 1/(1-mu) rows is a LOWER bound on its memory in
        # steps; if even that is a large fraction of the cycle, beta_hat converges
        # to the cycle average and the placebo never reads zero.  This repository
        # has paid for that twice (on_policy_pact_smac_runner._check_tracking).
        mem = 1.0 / max(1e-12, 1.0 - self.ns_mu)
        if mem > 0.25 * self.driver.period:
            print("[SMAC-NS][NOT-TRACKABLE] estimator memory >= %.0f steps is %.0f%% "
                  "of the %d-step driver period (limit 25%%).  beta_hat will average "
                  "the waveform and steer in the placebo.  Lengthen ns_period or "
                  "lower ns_mu -- and report it either way."
                  % (mem, 100.0 * mem / self.driver.period, self.driver.period),
                  flush=True)
        self.ns_rows_total = 0

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
        self.ns_tot_aimed = 0.0
        self.ns_tot_overkill = 0.0
        self.ns_tot_harm_design = 0.0
        self._innov_ema = np.full(self.n_agents, np.nan)
        # rolling design-matrix Gram over EVERY firing agent's regressor row, so
        # cond_psi describes the estimator's data rather than one step's 8 rows
        self._gram = np.zeros((self.coupling.r + 1, self.coupling.r + 1))
        self._gram_n = 0
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
        # GATE 1b: the one-pass option basis the steering uses every step equals
        # the per-agent definition, and the one-pass loading equals `loading`
        # with the agent's own target moved -- under live capacity as well
        for _ in range(16):
            t = rng.randint(-1, self.n_enemies, self.n_agents)
            f = (rng.rand(self.n_agents) < 0.8).astype(float)
            al = (rng.rand(self.n_agents) < 0.9).astype(float)
            cap = rng.uniform(0.0, 200.0, self.n_enemies)
            for cp in (None, cap):
                allp = c.psi_all_options(t, al, f, cap=cp)
                allu = c.loading_all_options(t, al, f, cap=cp)
                for i in range(self.n_agents):
                    ref = c.psi_options(i, t, al, f, np.arange(self.n_enemies), cap=cp)
                    assert np.allclose(allp[i], ref, atol=1e-12, rtol=0.0), (
                        "GATE 1b FAILED: one-pass option basis != per-agent basis")
                    for k in range(self.n_enemies):
                        ex = t.copy()
                        ex[i] = k
                        want = c.loading(ex, al, f, np.ones(self.n_enemies), cap=cp)[i]
                        # moving i's weight between bins re-orders a float sum,
                        # so agreement is to rounding, not to the bit
                        assert abs(allu[i, k] - want) <= 1e-9 * max(1.0, abs(want)), (
                            "GATE 1b FAILED: one-pass loading != loading()")

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
        self.ns_dmg_aimed = 0.0
        self.ns_overkill = 0.0
        self.ns_dmg_harm_design = 0.0
        self.ns_hp_next = None
        self.ns_switch_frac = float("nan")
        self.ns_gated = 0
        self.ns_gate_n = 0
        self.ns_step_gated = 0
        self.ns_step_gate_n = 0
        self.ns_snr = float("nan")
        self.ns_spread = float("nan")
        self.ns_cap_now = None
        self.ns_hp_now = None
        # NS-3.4: latch the step the harm was computed FOR.  The hook advances the
        # clock after applying the harm, so reading the counter in ns_info would
        # label every row with the NEXT step's driver -- an off-by-one that
        # misaligns the whole placebo column.
        self.ns_t_now = None
        self.ns_g_now = None

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
        self.ns_t_now, self.ns_g_now = int(self.ns_clock), g          # the latch
        # FIX A -- the capacity this volley was fired against: each target's
        # effective hit points GOING INTO the step.  self.enemies still holds the
        # previous step (update_units has not run), which is exactly that.
        # The live bar is read in EVERY mode: the stock-overkill observable and the
        # gate's attackable-enemy mask need it even when the coupling uses "max".
        hp = self._ns_live_hp()
        cap = None if self.ns_cap_mode == "max" else hp
        self.ns_hp_now, self.ns_cap_now = hp, cap
        # u at NOMINAL capacity; the dial enters only through the harm RATIO, so
        # sigma = 0 leaves the task byte-identical to stock SMAC (NS-2.1).
        u = self.coupling.loading(tgt, alive, fired, np.ones(self.n_enemies),
                                  cap=cap)
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
        psi = self.coupling.psi(tgt, alive, fired, cap=cap)
        if self.ns_intercept_only:
            psi[:, 1:] = 0.0                 # the `intercept` arm: peer channels off
        for i in range(n):
            if act_m[i] and np.isfinite(y[i]):
                # SCORE THE PRIOR PREDICTION BEFORE UPDATING ON IT.  Scoring the
                # posterior fit measures memorisation, not prediction.
                pf = float(self._rls[i].predict(psi[i]))
                pn = float(self._rls[i].beta[0])          # intercept-only null
                a_i = int(actions_int[i])
                pc = (float(self.ns_cost_prev[i, a_i])
                      if a_i >= N_ACTIONS_NO_ATTACK else np.nan)
                self.ns_pred_full[i] = pf
                self._fg_observe(y[i], pf, pn, pc)
                self._rls[i].update(psi[i][None, :], np.array([y[i]]))
                self.ns_rows += 1
                self.ns_rows_total += 1
                e_abs = abs(float(y[i]) - pf)
                self._innov_ema[i] = (e_abs if not np.isfinite(self._innov_ema[i])
                                      else 0.99 * self._innov_ema[i] + 0.01 * e_abs)
                self._gram = 0.999 * self._gram + 0.001 * np.outer(psi[i], psi[i])
                self._gram_n += 1
            self.ns_conf[i] = rls_confidence_pred(
                self._rls[i].P, self.ns_p0, self.coupling.r + 1, psi[i])
        self.ns_psi_prev = psi

        # ---- the enemy line AFTER the engine applied this step ----------------
        # self.enemies is still the step BEFORE (update_units has not run); the
        # tick's result is in self._obs.  None offline, where there is no engine.
        post = self._ns_post_units()
        landed = (self._ns_landed(post) if post is not None
                  else np.full(self.n_enemies, np.nan))
        self.ns_dmg_dealt = float(np.nansum(landed)) if post is not None else np.nan
        # the harm as DESIGNED -- aimed damage times the wasted fraction -- kept
        # beside the harm actually DELIVERED, which is less: a volley that kills
        # its target cannot be un-killed, and damage that never landed cannot be
        # handed back
        self.ns_dmg_harm_design = float(np.sum(
            self.coupling.dmg[fired > 0] * (exc[fired > 0] / (1.0 + exc[fired > 0]))))

        # ---- apply the harm: to the snapshot AND the engine, on THIS step ----
        restored = np.zeros(self.n_enemies)
        if post is not None and self.ns_on and self.ns_sigma > 0.0:
            restored = self._ns_restore(tgt, fired, exc, post, landed)
        self.ns_dmg_wasted = float(restored.sum())

        # ---- predict the cost of every candidate target, for the steering ----
        # against the bar the NEXT decision will actually face: after this step's
        # damage and restore.  Falls back to the pre-step bar offline.
        hp_next = self._ns_post_hp(post, hp)
        cap_next = None if self.ns_cap_mode == "max" else hp_next
        self._ns_predict_costs(tgt, alive, fired, cap_next, hp_next)
        self.ns_hp_next = hp_next

        # ---- FIX C: target switching, measured BEFORE prev_target is overwritten
        # (the first run read switch_frac = 0.000 for 3.2M steps because info was
        # built after the overwrite, which is impossible in SMAC)
        both = (tgt >= 0) & (self.ns_prev_target >= 0)
        self.ns_switch_frac = (float(np.mean(tgt[both] != self.ns_prev_target[both]))
                               if both.any() else float("nan"))

        # the STOCK overkill the game already has, before any dial: damage aimed at
        # an enemy beyond what it had left.  This is III.1's observable for
        # calibrating alpha once, by a stated procedure, instead of borrowing
        # URB's 2.28 -- which is what this run did.
        aimed = np.zeros(self.n_enemies)
        np.add.at(aimed, tgt[fired > 0], self.coupling.dmg[fired > 0])
        self.ns_dmg_aimed = float(aimed.sum())
        self.ns_overkill = float(np.sum(np.maximum(0.0, aimed - hp)))
        if np.isfinite(self.ns_dmg_dealt):
            self.ns_tot_dealt += self.ns_dmg_dealt
        self.ns_tot_wasted += self.ns_dmg_wasted
        self.ns_tot_aimed += self.ns_dmg_aimed
        self.ns_tot_overkill += self.ns_overkill
        self.ns_tot_harm_design += self.ns_dmg_harm_design

        self.ns_prev_target = tgt.copy()
        self.ns_cost_prev = self.ns_cost.copy()
        self.ns_n_dial_live += int(float(np.min(g)) < 1.0)
        self.ns_clock += 1
        self.ns_n_steps += 1

    def _ns_live_hp(self):
        """Each enemy's effective hit points going into this volley.  A dead enemy
        reads 0 (the host keeps it in ``self.enemies`` at health 0; the coupling
        floors it to 1).  An enemy the host has no record for reads its PUBLISHED
        maximum -- no information must mean "the first run's capacity", never
        "zero hp", which would clip every loading to U_CLIP and silently max out
        the harm."""
        hp = self.coupling.cap.astype(np.float64).copy()
        for e, unit in (getattr(self, "enemies", {}) or {}).items():
            if unit is not None and e < self.n_enemies:
                # update_units marks the dead by zeroing HEALTH on the last proto it
                # saw, which keeps that proto's shields -- a unit killed through
                # shields and life in one volley would otherwise read as alive
                hp[e] = (0.0 if float(unit.health) <= 0.0
                         else float(unit.health) + max(0.0, float(unit.shield)))
        return hp

    def _ns_post_units(self):
        """``{enemy index: unit proto AFTER this step's tick}``, living units only.

        SC2 drops a dead unit from ``raw_data.units``, so an index missing here
        died this step.  None when there is no observation to read (offline)."""
        try:
            units = self._obs.observation.raw_data.units
        except AttributeError:
            return None
        by_tag = {u.tag: u for u in units}
        out = {}
        for e, unit in (getattr(self, "enemies", {}) or {}).items():
            if unit is None or e >= self.n_enemies:
                continue
            now = by_tag.get(unit.tag, None)
            if now is not None and float(now.health) > 0.0:
                out[e] = now
        return out

    def _ns_landed(self, post):
        """Damage each enemy actually TOOK this step: its bar before the tick minus
        its bar after (all of what it had, if it died).  Regeneration is not
        negative damage."""
        landed = np.zeros(self.n_enemies)
        for e, unit in (getattr(self, "enemies", {}) or {}).items():
            if unit is None or e >= self.n_enemies:
                continue
            was = max(0.0, float(unit.health)) + max(0.0, float(unit.shield))
            if float(unit.health) <= 0.0:
                continue                                  # already dead before
            now = (float(post[e].health) + float(post[e].shield)) if e in post else 0.0
            landed[e] = max(0.0, was - now)
        return landed

    def _ns_post_hp(self, post, fallback):
        """Effective hit points after this step (and its restore); 0 for the dead."""
        if post is None:
            return fallback
        hp = np.zeros(self.n_enemies)
        for e, unit in post.items():
            hp[e] = float(unit.health) + float(unit.shield)
        return hp

    def _fg_observe(self, y, pred_full, pred_null, pred_forecast):
        """Accumulate II.10's fit_gain / pred_gain on one-step-ahead predictions."""
        f = self._fg
        lam = self._fg_lam
        f["n"] += 1
        f["ybar"] = lam * f["ybar"] + (1 - lam) * float(y)
        if f["n"] <= 2000:                # skip a warmup: a cold start otherwise
            return                        # dominates both sums and their diff.
        #                                   200 was too short -- the first run's
        #                                   Q1 fit_gain read 2.05, i.e. > 1.
        f["sse_full"] = lam * f["sse_full"] + (1 - lam) * (y - pred_full) ** 2
        f["sse_null"] = lam * f["sse_null"] + (1 - lam) * (y - pred_null) ** 2
        f["sst"] = lam * f["sst"] + (1 - lam) * (y - f["ybar"]) ** 2
        if np.isfinite(pred_forecast):
            f["sse_fc"] = lam * f["sse_fc"] + (1 - lam) * (y - pred_forecast) ** 2

    def _fg_value(self, key):
        """Guarded with NaN, never an epsilon: inside the placebo the target has
        no variance to explain and the ratio is meaningless."""
        f = self._fg
        if f["n"] <= 2000 or f["sst"] <= 0.0:
            return float("nan")
        return float((f["sse_null"] - f[key]) / f["sst"])

    def _ns_predict_costs(self, tgt, alive, fired, cap=None, hp=None):
        """``cost_hat[i, k]`` over agent i's candidate ATTACK actions -- the tail.

        Only attack actions carry a prediction.  Stop/move rows stay at zero, and
        the policy excludes them from the z-score (``pact_steer_from``), so the
        shift never reorders a move against an attack.  The first run did not
        exclude them, and zeros among ~0.003 attack costs read as the cheapest
        options: ~1.7 logits of anti-attack prior on every step.

        The layer does NOT gate.  The floor (``smac_ns.channel``) is applied by the
        policy, because the gate must see the same option set as the z-score and
        only the policy knows which attacks are in range at decision time.  What
        the layer reports -- ``ns_cost_spread`` and the would-gate counts -- is the
        same test over every LIVING enemy: an upper bound on the spread the policy
        sees, so a diagnostic, not the decision.

        (A first attempt at the gate compared the spread with ONE row's prediction
        error.  A ranking estimated from ~1000 rows is resolved far below
        single-row noise, and on a simulated engagement that test gated 100% of
        decisions at every severity.)
        """
        A0 = N_ACTIONS_NO_ATTACK
        self.ns_cost[:] = 0.0
        if self.ns_oracle:
            # the TRUE excess each option would incur -- the ceiling arm
            u_all = self.coupling.loading_all_options(tgt, alive, fired, cap)
            pred = self.coupling.excess(u_all, np.asarray(self.ns_g_now)[None, :])
        else:
            Phi = self.coupling.psi_all_options(tgt, alive, fired, cap)  # (n, m, r+1)
            if self.ns_intercept_only:
                Phi[:, :, 1:] = 0.0                    # the `intercept` arm
            B = np.stack([r.beta for r in self._rls])  # (n, r+1)
            pred = np.einsum("ikj,ij->ik", Phi, B)     # == AgentRLS.predict, per row
        live_e = ((np.asarray(hp) > 0) if hp is not None
                  else np.ones(self.n_enemies, dtype=bool))
        spreads, snrs = [], []
        self.ns_step_gated = 0
        self.ns_step_gate_n = 0
        for i in range(self.n_agents):
            if alive[i] <= 0:
                continue
            self.ns_cost[i, A0:] = pred[i]
            c = pred[i][live_e]
            if c.size < 2:
                continue
            sd = float(c.std())
            spreads.append(sd)
            noise = self._innov_ema[i]
            if np.isfinite(noise) and noise > 0.0:
                snrs.append(sd / float(noise))
            self.ns_gate_n += 1
            self.ns_step_gate_n += 1
            if sd < self.ns_steer_floor:              # would gate (diagnostic)
                self.ns_gated += 1
                self.ns_step_gated += 1
        self.ns_spread = float(np.mean(spreads)) if spreads else float("nan")
        self.ns_snr = float(np.mean(snrs)) if snrs else float("nan")

    def _ns_restore(self, tgt, fired, exc, post, landed):
        """Hand back the wasted part of the damage that LANDED, on this step.

        For each enemy that was shot and survived:

            wasted fraction  w_e = sum_i dmg_i f_i / sum_i dmg_i   over shooters of e
                             f_i = excess_i / (1 + excess_i)
            restore          r_e = w_e * landed_e          (<= landed_e, always)

        added on top of the unit's POST-tick shields, then life, each capped at its
        published maximum.  Written twice, deliberately:

          * into the observation snapshot ``self._obs`` -- ``update_units`` reads
            those protos next, so the reward, the observation and the termination
            test all see the restore ON THIS STEP; and
          * to the engine through ``controller.debug`` as an absolute value, so the
            engine's state agrees with the snapshot before the next tick.

        THREE WAYS THE FIRST VERSION WAS WRONG, each worth a sentence:
          1. It set the absolute value from ``self.enemies`` -- the PRE-tick protos.
             Setting shields to (pre-tick shields + waste) undid the entire step's
             shield damage on every shared target in the guard phase, whatever the
             waste: a hidden harm orders of magnitude above the logged 0.25%.
          2. It handed back AIMED damage, landed or not.  Stock ``reward_battle``
             takes ``abs()`` of the enemy's net hit-point change, so healing beyond
             what landed is paid out as if it were damage.  ``r_e <= landed_e``
             makes the step's net change on every enemy non-negative.
          3. It lagged the snapshot by a tick, so the reward saw the restore one
             step late -- as negative damage, through that same ``abs()``.

        A volley that killed its target is not restored: a death cannot be undone,
        so the harm on killing blows is lost.  ``ns_dmg_harm_design`` against
        ``ns_dmg_wasted`` reports how much.
        """
        from .coupling import UNIT_STATS

        restored = np.zeros(self.n_enemies)
        d_pb = _debug_pb2()
        cmds = []
        for e in range(self.n_enemies):
            on = (tgt == e) & (fired > 0)
            if not on.any() or e not in post or landed[e] <= 0.0:
                continue                 # not shot, killed this step, or nothing landed
            d = self.coupling.dmg[on]
            f = exc[on] / (1.0 + exc[on])
            give = float(np.sum(d * f) / max(1e-12, np.sum(d))) * float(landed[e])
            if give <= 1e-9:
                continue
            unit = post[e]
            name = self.coupling.enemy_names[e]
            max_life = float(UNIT_STATS[name]["life"])
            max_sh = float(UNIT_STATS[name]["shield"])
            sh0, hp0 = float(unit.shield), float(unit.health)
            add_sh = min(give, max(0.0, max_sh - sh0)) if max_sh > 0 else 0.0
            add_hp = min(give - add_sh, max(0.0, max_life - hp0))
            if add_sh + add_hp <= 1e-9:
                continue
            if add_sh > 0.0:
                unit.shield = sh0 + add_sh                   # the snapshot
            if add_hp > 0.0:
                unit.health = hp0 + add_hp
            restored[e] = add_sh + add_hp
            if d_pb is not None:                             # the engine
                if add_sh > 0.0:
                    cmds.append(d_pb.DebugCommand(unit_value=d_pb.DebugSetUnitValue(
                        unit_value=d_pb.DebugSetUnitValue.Shields,
                        value=sh0 + add_sh, unit_tag=unit.tag)))
                if add_hp > 0.0:
                    cmds.append(d_pb.DebugCommand(unit_value=d_pb.DebugSetUnitValue(
                        unit_value=d_pb.DebugSetUnitValue.Life,
                        value=hp0 + add_hp, unit_tag=unit.tag)))
        tot = float(restored.sum())
        if tot <= 0.0:
            return restored
        self.ns_n_restored += tot
        # imported lazily and cached (see _debug_pb2): the offline suite runs with
        # no StarCraft II.  A missing protobuf or controller is counted as a
        # refused write rather than raised, so the run reports "the dial is not
        # reaching the engine" (NS-3.2) instead of dying.
        if d_pb is None or getattr(self, "_controller", None) is None:
            self.ns_n_debug_fail += 1
            return restored
        try:
            self._controller.debug(cmds)
            self.ns_n_harmed += 1
        except Exception:                       # engine refused; count it loudly
            self.ns_n_debug_fail += 1
        return restored

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

        # the step the harm was computed FOR (NS-3.4), not the advanced counter
        t_h = self.ns_clock if self.ns_t_now is None else self.ns_t_now
        g = self.ns_g() if self.ns_g_now is None else self.ns_g_now
        A = float(self.driver.A(t_h))
        beta = np.mean([r.beta for r in self._rls], axis=0)
        x = self.ns_psi_prev[:, 1:]
        # herd index over the fleet's CHOSEN targets -- P-8.1's commons signature.
        # Logged, never acted on: acting on it would make the method a mechanism
        # rather than a per-agent estimator and break decentralization (P-4.1).
        cnt = np.bincount(self.ns_target[self.ns_target >= 0],
                          minlength=self.n_enemies).astype(np.float64)
        switched = self.ns_switch_frac      # measured in the hook (FIX C)
        cond = float("nan")
        if self._gram_n > 50:
            try:
                c = float(np.linalg.cond(self._gram))
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
            "ns_placebo": float(bool(self.driver.is_placebo(t_h))),
            "ns_dial_ratio": (float(self.ns_n_dial_live) / self.ns_n_steps
                              if self.ns_n_steps else float("nan")),
            "ns_clock": float(t_h),
            # ---- did the harm actually REACH the game? (NS-3.2, NS-3.3) ------
            "ns_harmed": float(self.ns_n_harmed),
            "ns_restored": float(self.ns_n_restored),
            "ns_debug_fail": float(self.ns_n_debug_fail),
            "ns_dmg_dealt": float(self.ns_dmg_dealt),
            "ns_dmg_wasted": float(self.ns_dmg_wasted),
            # per-step RATIOS are not logged: a mean of ratios is dominated by
            # steps where almost nothing landed (the first run read 0.31 against a
            # true 0.0025).  The runner forms waste_frac as a ratio of SUMS.
            "ns_dmg_aimed": float(self.ns_dmg_aimed),
            # designed harm (aimed x wasted fraction) against delivered harm
            # (ns_dmg_wasted): the gap is killing blows and shots that never landed
            "ns_dmg_harm_design": float(self.ns_dmg_harm_design),
            "ns_harm_delivery_cum": (self.ns_tot_wasted / self.ns_tot_harm_design
                                     if self.ns_tot_harm_design > 1e-9
                                     else float("nan")),
            "ns_overkill": float(self.ns_overkill),
            "ns_overkill_frac_cum": (self.ns_tot_overkill / self.ns_tot_aimed
                                     if self.ns_tot_aimed > 1e-9 else float("nan")),
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
            # the quantity the gate reads: std of predicted excess over each live
            # agent's attackable options, BEFORE gating, averaged over agents
            "ns_cost_spread": float(self.ns_spread),
            "ns_rank_snr": float(self.ns_snr),
            "ns_steer_gated_frac": (float(self.ns_gated) / self.ns_gate_n
                                    if self.ns_gate_n else float("nan")),
            # this step's raw gate counts, so the runner can split by driver phase
            "ns_step_gated": float(self.ns_step_gated),
            "ns_step_gate_n": float(self.ns_step_gate_n),
            # P-4.3: estimator memory in STEPS as a fraction of the driver period,
            # from the observed row rate.  Above ~0.25 beta_hat averages the cycle.
            "ns_mem_frac": ((1.0 / max(1e-12, 1.0 - self.ns_mu))
                            / (float(self.ns_rows_total)
                               / float(self.n_agents * self.ns_n_steps))
                            / float(self.driver.period)
                            if self.ns_rows_total > 0 and self.ns_n_steps > 0
                            else float("nan")),
            # the capacity the COUPLING actually used, over enemies still standing
            # (the published bar in "max" mode, the live bar in "remaining")
            "ns_cap_mean": (float(np.mean(
                self.coupling._capvec(self.ns_cap_now)[self.ns_hp_now > 0]))
                if self.ns_hp_now is not None and np.any(self.ns_hp_now > 0)
                else float("nan")),
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
