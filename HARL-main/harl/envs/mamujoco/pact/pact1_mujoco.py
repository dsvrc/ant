"""PACT-1 — peer-action compensation when the COUPLING ITSELF is unknown.

WHAT CHANGED, AND WHY
---------------------
Old PACT was handed the coupling operator W ("hips load hips, ankles load ankles"),
so it could rebuild the entire 8-dimensional disturbance shape and had to learn
exactly ONE number. That is why it read as being given the answer.

PACT-1 is handed only the model CLASS -- the three physical load paths exist -- and
must estimate how the payload's weight is distributed over them:

    d  <-  rho*d + (1-rho) * c(t) * W(theta(t)) * tau ,   W(theta) = sum_m theta_m B_m

Unknown: the r-vector  beta* = c(t)*theta(t),  drifting, tracked ONLINE and
DECENTRALIZED from each leg's own torque sensor. This is the standard parametric
contract -- know the class, fit the parameters -- and it is the line reviewers accept.

THE ESTIMATOR (the part that is genuinely new)
----------------------------------------------
Each leg measures its own realized torque, so it knows the load that ALREADY hit:

    d_meas(t) = (realized - commanded) + sensor noise          [env: pcr_d_meas]

That is informative but NOT an oracle. Compensating step t+1 needs d(t+1), and

    d(t+1) = rho*d(t)  +  (1-rho) * sum_m beta*_m * (B_m tau(t))
             \_______/     \_________________________________/
              MEASURED              the unknown lives HERE

So the leg regresses the innovation on the r known basis waveforms:

    y(t) := ( d_meas(t) - rho*d_meas(t-1) ) / (1-rho)  ~=  sum_m beta*_m (B_m tau(t-1))

r unknowns, k rows per step per agent. Recursive least squares with a forgetting
factor tracks beta* as it drifts. Each agent runs its OWN estimator on its OWN two
joints -- nothing is centralized.

THE PREDICTOR IS ANCHORED, WHICH IS BETTER THAN THE OLD x2
-----------------------------------------------------------
Old PACT rebuilt the whole disturbance by running the leak from scratch
(`x2 = leak_rho(sum of peers)`), so any parameter error compounded through the
recursion. PACT-1 anchors every prediction on a fresh measurement:

    d_hat(t+1) = rho * d_meas(t)  +  (1-rho) * sum_m beta_hat_m (B_m tau(t))

Only the (1-rho) innovation is exposed to estimator error; the accumulated part is
measured. The price is that sensor noise now enters the control path through the
rho*d_meas term -- a real trade, exposed as `predictor: anchored | arithmetic`.

WHAT THE POLICY LEARNS: TRUST, NOT GAIN
----------------------------------------
    u_i = clip( a_i  -  g_i * d_hat_i ,  -1, +1 ) ,    g_i = sigmoid(w_i) in [0,1]

The estimator does the estimating; RL decides how much to RELY on it, from the
return. One extra bounded action dimension per agent, exactly as before.

**The floor property is STRONGER than in old PACT.** With g=0 the executed torque is
clip(a) -- plain HAPPO -- no matter how wrong beta_hat is. The estimator sits
entirely OUTSIDE the worst-case control path, so a diverging estimate cannot drag
performance below blind; it can only fail to help.

THE PREDICTION THIS ARM EXISTS TO TEST
---------------------------------------
With W known and c hand-set, the best gain is exactly beta = c (measured: return
peaks at 0.45, residual bottoms at 0.0018 -- see pact/loop.py). With W UNKNOWN and
estimated online, theory says the optimum sits strictly BELOW c, because suppressing
the coupling also suppresses the excitation the estimator feeds on:

    g* < c,   and   (c - g*) grows with the drift rate and with sensor noise.

`pact1_debug.csv` logs g, beta_hat, the true beta*, and the tracking error, so the
gap is directly readable.

Enabled by ``env_args.pact1: true``. Prints one ``[PACT-1]`` banner per process.
"""

import os

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

_BANNER_SHOWN = False

# The env's own mixing normalisation. theta=(1/2,1/2,0) with this gain reproduces the
# legacy coupling exactly, so the basis below is the one ant.py actually applies.
_MIXNORM = 2.0


# ==========================================================================
#  Pure-numpy core -- no torch, no gym, no mujoco. test_pact1.py exercises
#  all of it without a simulator.
# ==========================================================================

def ant_basis(n_legs=4, mixnorm=_MIXNORM):
    """The r=3 load-path basis, as explicit (n x n) matrices.

    Must match ant.py's `_coupling` EXACTLY, including the mixnorm gain:

        hip_l   <- th0*(other hips)   + th2*(other ankles)
        ankle_l <- th1*(other ankles) + th2*(other hips)

    Every matrix is zero-diagonal in the agent (leg) block, so the category-C
    signature holds for every theta: at N=1 the sums are empty and d stays 0.
    """
    n = 2 * n_legs
    Bhh = np.zeros((n, n))
    Baa = np.zeros((n, n))
    Bx = np.zeros((n, n))
    for l in range(n_legs):
        for lp in range(n_legs):
            if l == lp:
                continue
            Bhh[2 * l, 2 * lp] = mixnorm            # hip <- other hips
            Baa[2 * l + 1, 2 * lp + 1] = mixnorm    # ankle <- other ankles
            Bx[2 * l, 2 * lp + 1] = mixnorm         # hip <- other ankles
            Bx[2 * l + 1, 2 * lp] = mixnorm         # ankle <- other hips
    return [Bhh, Baa, Bx]


def basis_waveforms(tau, B):
    """(B_1 tau, ..., B_r tau) stacked as (r, n) -- the regressor columns, and the
    per-basis 'what the others just did' that the predictor mixes with beta_hat.
    Pure arithmetic over the peers' EXECUTED torques (declared communication)."""
    tau = np.clip(np.asarray(tau, dtype=np.float64), -1.0, 1.0)
    return np.stack([b @ tau for b in B], axis=0)


class AgentRLS:
    """Per-agent recursive least squares with a forgetting factor.

    Estimates the r-vector beta* = c*theta from this leg's OWN k=2 residual rows.
    Decentralized by construction: agent i never sees another agent's residual, only
    the peers' executed actions (which it already shares to build the basis).

    The forgetting factor mu is the bias/variance dial of the tracking floor:
    small mu forgets fast (tracks drift, amplifies sensor noise), large mu averages
    hard (rejects noise, lags the drift). The floor is Theta(sqrt(noise * drift)) and
    cannot be removed by tuning -- only balanced.
    """

    def __init__(self, r, mu=0.999, p0=10.0, beta0=None):
        self.r = int(r)
        self.mu = float(mu)
        self.P = np.eye(self.r) * float(p0)
        self.beta = np.zeros(self.r) if beta0 is None else np.asarray(beta0, float).copy()
        self.innov = 0.0

    def update(self, Phi, y):
        """Phi: (k, r) regressor rows for this agent. y: (k,) targets."""
        Phi = np.atleast_2d(np.asarray(Phi, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        tot = 0.0
        for j in range(Phi.shape[0]):
            phi = Phi[j]
            Pphi = self.P @ phi
            denom = self.mu + float(phi @ Pphi)
            if denom < 1e-12:
                continue
            K = Pphi / denom
            e = float(y[j] - phi @ self.beta)
            self.beta = self.beta + K * e
            self.P = (self.P - np.outer(K, Pphi)) / self.mu
            self.P = 0.5 * (self.P + self.P.T)            # keep it symmetric
            tot += abs(e)
        self.innov = tot / max(1, Phi.shape[0])
        return self.beta


def predict_load(anchor, waves, beta_hat, rho):
    """d_hat(t+1) = rho*anchor + (1-rho)*sum_m beta_hat_m (B_m tau(t)).

    ``anchor`` is what the recursion is carried forward FROM, and it is the whole
    design choice:

      anchored   anchor = d_meas(t), the SENSOR. The accumulated part is measured,
                 so estimator error is exposed only through the (1-rho) innovation
                 and cannot compound through the recursion. Costs: rho * sensor
                 noise enters the control path.
      arithmetic anchor = d_hat(t), the previous PREDICTION. No sensor noise in the
                 loop, but parameter error compounds -- this is what old PACT did.

    The caller picks the anchor; the recursion is identical either way.
    """
    innov = (1.0 - rho) * (np.asarray(beta_hat, dtype=np.float64)[:, None] * waves).sum(0)
    return rho * np.asarray(anchor, dtype=np.float64) + innov


def compensate(a, g, d_hat, offsets):
    """u_i = clip(a_i - g_i * d_hat_i, -1, 1). g is the per-agent TRUST in [0,1]."""
    a = np.asarray(a, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    u = np.empty_like(a)
    for i in range(a.shape[0]):
        u[i] = a[i] - g[i] * d_hat[offsets[i]:offsets[i + 1]]
    return np.clip(u, -1.0, 1.0)


def trust_from_w(w, prev, ema, g_max=1.0, bias=2.2):
    """g_i = g_max*sigmoid(w_i + bias), EMA-smoothed.

    *** THE BIAS IS THE POINT, AND ITS ABSENCE COST A 10M-STEP RUN. ***

    Old PACT's gain beta had to TRACK c(t) -- hidden, phase-dependent -- so starting
    at beta_max/2 was a sensible hedge and the tracking was the hard part. PACT-1's
    estimator already supplies the magnitude, so this knob tracks NOTHING: the
    optimal trust is the constant 1.0 whenever the estimate is right. Initialising at
    half is therefore starting at half of a known-correct answer and asking a weak,
    noisy policy gradient to walk to 1.0 -- measured, it does not: over 10M steps
    trust went 0.485 -> 0.444 while cos(d_hat, d_true) sat at 0.99, i.e. the run spent
    its entire budget at HALF compensation with a CORRECT waveform.

    So the prior is inverted: bias=2.2 puts w=0 at sigmoid(2.2) = 0.90 of g_max --
    trust the estimator unless the return says otherwise. The floor property is
    untouched (w -> -inf still gives g=0 == blind); it is now a safety net rather
    than the starting point.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    target = g_max / (1.0 + np.exp(-np.clip(w + bias, -30.0, 30.0)))
    return ema * np.asarray(prev, dtype=np.float64) + (1.0 - ema) * target


def rls_confidence(P, p0, r):
    """How much the estimator believes itself, in [0,1], from its own covariance.

    Full trust from step 1 would be wrong while the RLS is still cold -- but that
    needs no hyperparameter, because the estimator already knows its uncertainty.
    conf = 1/(1 + tr(P)/p0) is 1/(1+r) at the prior and rises to 1 as P shrinks, so
    reliance ramps itself in exactly as fast as the estimate firms up.

    This is the honest version of a warmup: the covariance IS the trust prior, and
    the policy only modulates it.
    """
    return 1.0 / (1.0 + float(np.trace(P)) / max(1e-9, p0))


# ==========================================================================
class Pact1MujocoMulti(MujocoMulti):
    """MujocoMulti + online coupling identification + trust-weighted compensation."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("pact1_cfg", {}))

        self.rho = float(cfg.get("rho", 0.8))                  # env structural
        self.mu = float(cfg.get("forget", 0.999))              # RLS forgetting factor
        self.p0 = float(cfg.get("p0", 10.0))                   # RLS prior looseness
        self.g_max = float(cfg.get("g_max", 1.0))
        self.g_ema = float(cfg.get("g_ema", 0.9))
        # w=0 -> sigmoid(bias) of g_max. 2.2 => 0.90 (trust by default; see
        # trust_from_w). Set 0.0 to reproduce the half-trust init that stalled.
        self.g_bias = float(cfg.get("g_bias", 2.2))
        # gate reliance by the estimator's own covariance while it is still cold
        self.use_conf = bool(cfg.get("confidence_gate", True))
        self.anchored = bool(cfg.get("anchored", True))
        self.critic_payload = bool(cfg.get("critic_payload", False))  # CTDE arm
        self.freeze_beta = cfg.get("freeze_beta", None)        # ablation: hand-set beta

        self.act_dim_env = int(self.true_action_space[0].shape[0])
        self.n_act = int(sum(sp.shape[0] for sp in self.true_action_space))
        self._offsets = np.cumsum([0] + [sp.shape[0] for sp in self.true_action_space])
        self.B = ant_basis(self.n_agents)
        self.r = len(self.B)

        # --- per-agent method state ---
        self._rls = [AgentRLS(self.r, self.mu, self.p0) for _ in range(self.n_agents)]
        # policy-set trust, starting at sigmoid(g_bias) of g_max (0.90 by default)
        self._g_pol = np.full(
            self.n_agents, self.g_max / (1.0 + np.exp(-self.g_bias))
        )
        self._conf = np.full(self.n_agents, 1.0 / (1.0 + self.r))   # cold estimator
        self._g = self._g_pol * self._conf                          # what is applied
        self._d_meas = np.zeros(self.n_act)                    # last sensor reading
        self._d_meas_prev = np.zeros(self.n_act)
        self._waves_prev = np.zeros((self.r, self.n_act))      # regressor at t-1
        self._d_hat = np.zeros(self.n_act)                     # what t+1 will face
        self._have_prev = False
        self._payload_cache = 0.0

        # --- action space: +1 bounded dim per agent (the trust control w_i) ---
        low = np.asarray(self.action_space[0].low, dtype=np.float32)
        high = np.asarray(self.action_space[0].high, dtype=np.float32)
        self.action_space = tuple(
            Box(np.concatenate([low, [-1.0]]).astype(np.float32),
                np.concatenate([high, [1.0]]).astype(np.float32))
            for _ in range(self.n_agents)
        )

        # --- obs / share spaces (appended POST-normalization, native units) ---
        #  obs   += [d_hat_i (k), beta_hat_i (r), g_i (1), d_meas_i (k)]
        #  share += [d_hat_all (n), mean beta_hat (r), g_all (N)] (+ payload for CTDE)
        self._obs_aug = self.act_dim_env + self.r + 1 + self.act_dim_env
        self._share_aug = self.n_act + self.r + self.n_agents + (1 if self.critic_payload else 0)
        raw_o, raw_s = self.get_obs_size(), self.get_state_size()
        self.observation_space = [
            Box(-np.inf, np.inf, shape=(raw_o + self._obs_aug,)) for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(-np.inf, np.inf, shape=(raw_s + self._share_aug,)) for _ in range(self.n_agents)
        ]

        offset = int(cfg.get("pcr_clock_offset", 0))           # C4 de-aliased eval
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[PACT-1] WARNING: could not set pcr_clock_offset={offset}.")
        self._banner()

    def _banner(self):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        print(
            "\n" + "=" * 74
            + f"\n[PACT-1] r={self.r} rho={self.rho} forget={self.mu} "
            f"predictor={'anchored' if self.anchored else 'arithmetic'} "
            f"g_max={self.g_max}\n"
            "  The coupling operator W(theta) is UNKNOWN. The agent is given only the\n"
            "  three load-path basis matrices (the model CLASS) and must track the\n"
            "  drifting r-vector beta* = c*theta ONLINE, DECENTRALIZED, from its own\n"
            "  torque sensor. Per-agent RLS with forgetting; the policy learns only\n"
            "  TRUST g in [0,1]. Floor property: g=0 => clip(a) == blind HAPPO, for\n"
            "  ANY beta_hat, so a bad estimate cannot go below blind.\n"
            + ("  [CTDE] critic_payload=ON: true A(t) -> CRITIC share_obs only.\n"
               if self.critic_payload else "")
            + ("  [ABLATION] freeze_beta set: the estimator is BYPASSED.\n"
               if self.freeze_beta is not None else "")
            + "=" * 74, flush=True)

    # ------------------------------------------------------------------ augment
    def _augment_obs(self, obs_n):
        out = []
        for i in range(self.n_agents):
            sl = slice(self._offsets[i], self._offsets[i + 1])
            aug = np.concatenate([self._d_hat[sl], self._rls[i].beta,
                                  [self._g[i]], self._d_meas[sl]])
            out.append(np.concatenate([np.asarray(obs_n[i], dtype=np.float64), aug]))
        return out

    def _augment_state(self, state_n):
        beta_bar = np.mean([r.beta for r in self._rls], axis=0)
        aug = np.concatenate([self._d_hat, beta_bar, self._g])
        if self.critic_payload:
            aug = np.concatenate([aug, [self._payload_cache]])
        return [np.concatenate([np.asarray(s, dtype=np.float64), aug]) for s in state_n]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        a = actions[:, : self.act_dim_env]                       # desired torque
        w = actions[:, self.act_dim_env:].reshape(self.n_agents)  # trust control

        # 1) trust: the policy's setting, gated by the estimator's own confidence
        self._g_pol = trust_from_w(w, self._g_pol, self.g_ema, self.g_max, self.g_bias)
        if self.use_conf:
            self._conf = np.array([
                rls_confidence(self._rls[i].P, self.p0, self.r)
                for i in range(self.n_agents)
            ])
        else:
            self._conf = np.ones(self.n_agents)
        self._g = self._g_pol * self._conf

        # 2) compensate with the CACHED prediction (what this action faces)
        d_hat_used = self._d_hat.copy()
        u = compensate(a, self._g, d_hat_used, self._offsets)
        u_flat = u.reshape(-1)

        # 3) step
        obs_n, state_n, rewards, dones, infos, avail = super().step(u)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

        # 4) read the leg's own torque sensor (the ONE consumable info key)
        self._d_meas_prev = self._d_meas
        self._d_meas = np.asarray(
            info0.get("pcr_d_meas", np.zeros(self.n_act)), dtype=np.float64
        ).reshape(-1)

        # 5) IDENTIFY. The innovation from t-1 -> t regresses on the basis waveforms
        #    that were live at t-1.  y = (d(t) - rho*d(t-1))/(1-rho) ~ sum_m beta_m w_m.
        if self._have_prev and self.freeze_beta is None:
            y = (self._d_meas - self.rho * self._d_meas_prev) / max(1e-9, 1.0 - self.rho)
            for i in range(self.n_agents):
                sl = slice(self._offsets[i], self._offsets[i + 1])
                self._rls[i].update(self._waves_prev[:, sl].T, y[sl])

        # 6) the basis waveforms this step's EXECUTED torque just created
        waves = basis_waveforms(u_flat, self.B)

        # 7) PREDICT what the next action will face. The anchor is the sensor
        #    (default) or the previous prediction (the old, sensor-free recursion).
        anchor = self._d_meas if self.anchored else d_hat_used
        d_hat_new = np.zeros(self.n_act)
        for i in range(self.n_agents):
            sl = slice(self._offsets[i], self._offsets[i + 1])
            b_i = (np.full(self.r, float(self.freeze_beta))
                   if self.freeze_beta is not None else self._rls[i].beta)
            d_hat_new[sl] = predict_load(anchor[sl], waves[:, sl], b_i, self.rho)
        self._d_hat = d_hat_new
        self._waves_prev = waves
        self._have_prev = True
        self._payload_cache = float(info0.get("pcr_payload", self._payload_cache))

        # 8) diagnostics (logging only -- none of this reaches the control path)
        d_next = np.asarray(info0.get("pcr_d_next", np.full(self.n_act, np.nan)),
                            dtype=np.float64).reshape(-1)
        beta_true = np.asarray(info0.get("pcr_beta_true", np.full(self.r, np.nan)),
                               dtype=np.float64).reshape(-1)
        beta_hat = np.mean([r.beta for r in self._rls], axis=0)
        pre_clip = a.reshape(-1) - np.repeat(self._g, self.act_dim_env) * d_hat_used
        info0["pact_beta"] = float(np.mean(self._g))            # runner reads this
        info0["pact_dbeta"] = float(np.mean([r.innov for r in self._rls]))
        info0["pact_x2"] = self._d_hat.tolist()                 # gate: d_hat vs d_next
        info0["pact_d_next"] = d_next.tolist()
        info0["pact_x2_absmean"] = float(np.mean(np.abs(self._d_hat)))
        info0["pact_resid_absmean"] = (
            float(np.mean(np.abs(d_next - d_hat_used)))
            if np.all(np.isfinite(d_next)) else float("nan")
        )
        # PREDICTION ERROR -- both sides are "what the NEXT action faces", so this is
        # aligned. (It was previously compared against d_hat_used, the prediction of
        # what THIS step faced: off by one, and since d(t+1)=0.8d(t)+0.2(...) the
        # step-to-step change alone put the ratio near 0.5, which read as a broken
        # compensator when nothing was wrong. cos_x2_dnext was always aligned, which
        # is why the two columns disagreed.)
        info0["pact_resid_absmean"] = (
            float(np.mean(np.abs(d_next - self._d_hat)))
            if np.all(np.isfinite(d_next)) else float("nan")
        )
        # THE NUMBER THAT MATTERS: what the policy actually still feels after
        # compensation. delivered = clip(a - g*d_hat_used) + d_applied, so the
        # uncancelled disturbance is d_applied - g*d_hat_used. At perfect prediction
        # this is (1-g)*|d|, i.e. ~0.11*|d| at g=0.89 -- compare it against
        # pact1_felt_blind, the same quantity with no compensation at all.
        d_app = np.asarray(info0.get("pcr_d_applied", np.full(self.n_act, np.nan)),
                           dtype=np.float64).reshape(-1)
        g_flat = np.repeat(self._g, self.act_dim_env)
        if np.all(np.isfinite(d_app)):
            info0["pact1_felt"] = float(np.mean(np.abs(d_app - g_flat * d_hat_used)))
            info0["pact1_felt_blind"] = float(np.mean(np.abs(d_app)))
            info0["pact1_cancel_frac"] = 1.0 - (
                info0["pact1_felt"] / max(1e-12, info0["pact1_felt_blind"])
            )
        info0["pact_u_clip"] = float(np.mean(np.abs(pre_clip) > 1.0))
        info0["pact_u_minus_a"] = float(np.sqrt(np.mean((u_flat - a.reshape(-1)) ** 2)))
        info0["pact1_trust"] = float(np.mean(self._g))          # applied = policy*conf
        info0["pact1_trust_pol"] = float(np.mean(self._g_pol))  # what the POLICY set
        info0["pact1_conf"] = float(np.mean(self._conf))        # estimator confidence
        info0["pact1_trust_spread"] = float(np.max(self._g) - np.min(self._g))
        info0["pact1_beta_hat"] = beta_hat.tolist()
        info0["pact1_beta_err"] = (
            float(np.linalg.norm(beta_hat - beta_true))
            if np.all(np.isfinite(beta_true)) else float("nan")
        )
        info0["pact1_innov"] = float(np.mean([r.innov for r in self._rls]))
        return (self._augment_obs(obs_n), self._augment_state(state_n),
                rewards, dones, infos, avail)

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # The env zeroes d each episode, so the sensor history and the regressor are
        # stale -- drop them. The ESTIMATE persists (theta drifts on a timescale far
        # longer than an episode, so forgetting it every reset would be throwing away
        # the only thing that is genuinely learnable). Trust persists too.
        self._d_meas = np.zeros(self.n_act)
        self._d_meas_prev = np.zeros(self.n_act)
        self._waves_prev = np.zeros((self.r, self.n_act))
        self._d_hat = np.zeros(self.n_act)
        self._have_prev = False
        return self._augment_obs(obs_n), self._augment_state(state_n), avail
