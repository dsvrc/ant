"""PACT env wrapper — peer-action compensation with a trained gain (the method).

This is the WHOLE deployable method (spec §3.1). Unlike O-MAX it reads no
privileged env telemetry in the control path; unlike RECON it estimates nothing.
The disturbance the oracle was handed, ``d_i = c(t)·x2_i(t)`` with
``x2_i = leak_ρ(Σ_{j≠i} τ_j)``, splits into

  * ``x2_i`` — a leaky sum of the OTHER agents' executed torques. This is pure
    arithmetic: given the peers' executed torques (a declared, one-step-delayed
    communication of 2 scalars/agent), every agent runs the *env's own leak
    recursion* and obtains ``x2_i`` EXACTLY (no gain, no model, no estimator);
  * ``c(t) = A(t)·σ`` — the one slow, global, genuinely-hidden scalar.

So PACT compensates with the exact waveform and a single shared learned gain β:

    u_i = clip(a_i − β · x2_i,  −1, +1)

β is driven by ONE extra bounded action dimension per agent (Δβ ∈ [−δ, +δ]),
aggregated into a rate-limited global integrator ``β ← clip(β + δ·agg_i(w_i),
0, β_max)``. Perfect play is β(t) ≈ c(t): full cancellation at the payload peak,
zero at the trough. RL learns only this scalar's schedule, from the return.

Floor property (the anti-RECON guarantee): with β ≡ 0 the executed torque is
``clip(a)`` — plain HAPPO. There is no estimator in the control path to be wrong,
so no configuration can crater below blind (unit test T3).

Timing (mirrors O-MAX's proven cache contract, whose ``timing_err`` was 0):
the ``x2`` used to compensate action t is the value updated AFTER step t−1 — i.e.
``x2`` and the env's ``pcr_d_next`` are both "what action t+1 will face", so they
are proportional by c and the online gate ``corr(x2, pcr_d_next) → 1`` certifies
the arithmetic is wired right (index order, reset masking, one-step timing).

Communication honesty (spec §6): PACT is a *communication* method — peers share
their executed torques (O(N) scalars/step, one-step delay). This is declared, and
justified by the measured impossibility that precedes it (central AND local
*passive* identification both fail on the trained coordinated gait — the RECON
post-mortem's Wall A). The message is not learned; it is the provably minimal one
that makes the coupling waveform computable exactly.

Enabled by ``env_args.pact: true``. Prints one ``[PACT]`` banner per process.
"""

import os

import numpy as np
from gym.spaces import Box

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

_BANNER_SHOWN = False


# ==========================================================================
#  Pure-arithmetic core (module-level so the unit tests exercise it without a
#  simulator, torch, or mujoco — spec §5 sanity constraint).
# ==========================================================================
def coupling_sum(tau):
    """s_i = Σ_{j≠i} τ_j per joint-type — EXACTLY ant.py's category-C channel.

    Hips couple to hips, ankles to ankles: for the flat 8-vector
    ``[hip0,ank0, hip1,ank1, ...]`` this is ``hip.sum()−hip`` on the even indices
    and ``ank.sum()−ank`` on the odd ones.
    """
    tau = np.asarray(tau, dtype=np.float64)
    s = np.empty_like(tau)
    hip, ank = tau[0::2], tau[1::2]
    s[0::2] = hip.sum() - hip
    s[1::2] = ank.sum() - ank
    return s


def leak_step(x2, u_flat, rho):
    """x2 ← ρ·x2 + (1−ρ)·s(clip(u)) — the env's leak recursion WITHOUT the A·σ
    factor. Fed by the EXECUTED (clipped) torques, so it reproduces the env's
    liability waveform up to the scalar c = A·σ."""
    tau = np.clip(np.asarray(u_flat, dtype=np.float64), -1.0, 1.0)
    return rho * np.asarray(x2, dtype=np.float64) + (1.0 - rho) * coupling_sum(tau)


def pcr_d_step(d, u_flat, A, sigma, rho):
    """Reference for the env's TRUE liability recursion (used only by the unit
    test to check x2·c == d). Mirrors ant.py's ``self._d`` update exactly."""
    tau = np.clip(np.asarray(u_flat, dtype=np.float64), -1.0, 1.0)
    return rho * np.asarray(d, dtype=np.float64) + (1.0 - rho) * (A * sigma * coupling_sum(tau))


def beta_step(beta, w, delta, beta_max, driver="mean"):
    """Rate-limited global integrator for the shared gain.

    ``w`` are the per-agent control outputs (clipped to [−1,1]); ``driver``
    selects the credit path: ``"mean"`` (spec default) aggregates all agents,
    ``"agent0"`` lets a single designated agent drive β (a cleaner credit signal —
    the reviewer's change #2, kept as a config option). Returns (new_beta, dbeta).
    """
    w = np.clip(np.asarray(w, dtype=np.float64).reshape(-1), -1.0, 1.0)
    drive = float(w[0]) if driver == "agent0" else float(np.mean(w))
    dbeta = delta * drive
    return float(np.clip(beta + dbeta, 0.0, beta_max)), dbeta


def compensate(a, beta, x2, offsets):
    """u_i = clip(a_i − β_i·x2_i, −1, 1). ``a`` is (N, k); ``x2`` is the flat
    (n_act,) cache; ``offsets`` gives agent i's slice [offsets[i]:offsets[i+1]).
    ``beta`` may be a scalar (shared gain) or a per-agent (N,) array."""
    a = np.asarray(a, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    u = np.empty_like(a)
    for i in range(a.shape[0]):
        bi = float(beta) if beta.ndim == 0 else float(beta[i])
        u[i] = a[i] - bi * x2[offsets[i]:offsets[i + 1]]
    return np.clip(u, -1.0, 1.0)


def direct_beta(w, beta_max, prev, ema):
    """Per-agent DIRECT gain: β_i = β_max·sigmoid(w_i), EMA-smoothed toward that
    target. Returns (new_beta_vec, mean|Δ|). Starts (w≈0) at β_max/2 — partial
    compensation from step 1 — and is phase-dependent instantly (no global
    integrator to collapse). The policy learns to drive w_i by phase."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    target = beta_max / (1.0 + np.exp(-np.clip(w, -30.0, 30.0)))
    new = ema * np.asarray(prev, dtype=np.float64) + (1.0 - ema) * target
    return new, float(np.mean(np.abs(target - new)))


class PactMujocoMulti(MujocoMulti):
    """MujocoMulti + peer-action exact-waveform compensation with a learned gain."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        env_args = kwargs["env_args"]
        cfg = dict(env_args.get("pact_cfg", {}))

        # --- the real knobs (+ structural constants) ---
        self.rho = float(cfg.get("rho", env_args.get("pact_rho", 0.8)))          # env structural
        self.beta_max = float(cfg.get("beta_max", env_args.get("pact_beta_max", 0.6)))
        self.ema_hl = float(cfg.get("ema_hl", env_args.get("pact_ema_hl", 200.0)))
        # how the policy sets the gain:
        #   "direct"     — per-agent β_i = β_max·sigmoid(w_i), EMA-smoothed. Starts
        #                  at β_max/2 (partial comp from step 1) and is phase-
        #                  dependent instantly. USE WITH A RECURRENT POLICY so w_i
        #                  can be driven by the (within-episode ~constant) phase.
        #   "integrator" — one global rate-limited scalar β←clip(β+δ·agg(w),0,βmax)
        #                  (the original; collapses to 0 under a memoryless policy
        #                  because the trough penalty dominates the global gain).
        self.beta_mode = str(cfg.get("beta_mode", "direct"))
        assert self.beta_mode in ("direct", "integrator")
        self.delta = float(cfg.get("delta", env_args.get("pact_delta", 0.01)))     # integrator speed
        self.beta_driver = str(cfg.get("beta_driver", "mean"))                     # integrator agg
        assert self.beta_driver in ("mean", "agent0")
        self.beta_ema = float(cfg.get("beta_ema", 0.9))                            # direct smoothing
        # EMA retention/step for the |x2| chart feature (half-life ema_hl steps)
        self._ema_alpha = 0.5 ** (1.0 / max(1.0, self.ema_hl))

        # --- phase-signal options (the hidden driver is what β must track) ---
        # critic_payload: append the true payload A(t) to the CRITIC's share_obs
        #   ONLY (never the actor). Standard CTDE — training-only, execution stays
        #   fully decentralized — so it is honest, not an oracle. It sharpens the
        #   value function, hence the (weak) policy-gradient signal for the β knob.
        # actor_c_oracle: append the true c=A·σ to the ACTOR's obs. This IS
        #   privileged — a LABELED diagnostic arm that measures the ceiling of the
        #   β-mechanism when the phase is known (never the headline).
        self.critic_payload = bool(cfg.get("critic_payload", False))
        self.actor_c_oracle = bool(cfg.get("actor_c_oracle", False))
        self.severity = float(cfg.get("severity", os.environ.get("ANT_PCR_SEVERITY", 0.45)))
        self._payload_cache = 0.0

        # per-agent env action dim (2 for Ant 4x2) and flat joint offsets
        self.act_dim_env = int(self.true_action_space[0].shape[0])
        self.n_act = int(sum(sp.shape[0] for sp in self.true_action_space))       # 8
        self._offsets = np.cumsum([0] + [sp.shape[0] for sp in self.true_action_space])

        # --- method state (per env instance) ---
        self._x2 = np.zeros(self.n_act)             # coupling-waveform cache (resets per episode)
        self._beta = 0.0                            # global gain (integrator mode)
        # per-agent gains actually used. direct: start at β_max/2 (partial comp);
        # integrator: start at 0 (== blind). Persists across episodes (c persists).
        init_b = 0.5 * self.beta_max if self.beta_mode == "direct" else 0.0
        self._beta_vec = np.full(self.n_agents, init_b)
        self._ema_absx2 = np.zeros(self.n_agents)   # per-agent |x2| EMA (persists)

        # --- extend the ACTION space by one bounded control dim (Δβ) per agent.
        #     Declared BEFORE HARL sizes the actor/critic/buffer. The env still
        #     receives only the first act_dim_env torques (super().step truncates
        #     via true_action_space); this extra dim is consumed here to move β. ---
        self._extra_act = 1
        low = np.asarray(self.action_space[0].low, dtype=np.float32)
        high = np.asarray(self.action_space[0].high, dtype=np.float32)
        new_low = np.concatenate([low, -np.ones(self._extra_act, dtype=np.float32)])
        new_high = np.concatenate([high, np.ones(self._extra_act, dtype=np.float32)])
        self.action_space = tuple(Box(new_low, new_high) for _ in range(self.n_agents))

        # --- extend the OBS / SHARE spaces (post-normalization append, §3.3) ---
        #  obs   += [x2_i (k), beta (1), ema_absx2_i (1)]          = k+2 per agent
        #  share += [x2_all (n_act), beta (1), ema_absx2_all (N)]  = the global mirror
        self._obs_aug = self.act_dim_env + 2 + (1 if self.actor_c_oracle else 0)
        self._share_aug = self.n_act + 1 + self.n_agents + (1 if self.critic_payload else 0)
        raw_obs = self.get_obs_size()
        raw_share = self.get_state_size()
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(raw_obs + self._obs_aug,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(raw_share + self._share_aug,))
            for _ in range(self.n_agents)
        ]

        # C4 de-aliased eval: a per-eval-env payload-clock offset (never training)
        offset = int(cfg.get("pcr_clock_offset", 0))
        if offset:
            tgt = getattr(self.env, "unwrapped", self.env)
            try:
                tgt._clock = int(offset)
            except Exception:
                print(f"[PACT] WARNING: could not set pcr_clock_offset={offset}.")

        self._banner()

    def _banner(self):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        print(
            "\n" + "=" * 72
            + f"\n[PACT] beta_mode={self.beta_mode} rho={self.rho} beta_max={self.beta_max} "
            f"beta_ema={self.beta_ema} delta={self.delta}\n"
            "  Method: u = clip(a - beta * x2), x2 = leak_rho(sum of PEERS' executed\n"
            "  torques) computed EXACTLY by arithmetic; beta is the only learned part\n"
            "  (one extra bounded action dim). Declared communication: peers share\n"
            "  their executed torques (one-step delay). No privileged env info in the\n"
            "  control path; no estimator to be wrong (floor property: beta->0 == blind).\n"
            + (f"  [CTDE] critic_payload=ON: true A(t) -> CRITIC share_obs only "
               f"(training-only, honest).\n" if self.critic_payload else "")
            + (f"  [ORACLE ARM] actor_c_oracle=ON: true c=A*sigma -> ACTOR obs "
               f"(LABELED diagnostic, not the headline).\n" if self.actor_c_oracle else "")
            + "=" * 72,
            flush=True,
        )

    # ------------------------------------------------------------------ augment
    def _augment_obs(self, obs_n):
        """o_i ⊕ [x2_i, beta, ema_absx2_i] — appended AFTER MujocoMulti's per-step
        whole-vector normalization, so these arrive in native units (x2 ~ O(0.2),
        beta ∈ [0, beta_max], ema ~ O(0.2): all O(1), no scaling needed)."""
        out = []
        for i in range(self.n_agents):
            xi = self._x2[self._offsets[i]:self._offsets[i + 1]]
            aug = np.concatenate([xi, [self._beta_vec[i]], [self._ema_absx2[i]]])
            if self.actor_c_oracle:  # LABELED diagnostic arm: true c to the actor
                aug = np.concatenate([aug, [self._payload_cache * self.severity]])
            out.append(np.concatenate([np.asarray(obs_n[i], dtype=np.float64), aug]))
        return out

    def _augment_state(self, state_n):
        """share_obs ⊕ [x2_all, mean_beta, ema_absx2_all] — the global mirror for
        the central critic (standard CTDE; execution never uses share_obs)."""
        aug = np.concatenate([self._x2, [float(np.mean(self._beta_vec))], self._ema_absx2])
        if self.critic_payload:  # CTDE-honest: true payload to the CRITIC only
            aug = np.concatenate([aug, [self._payload_cache]])
        return [
            np.concatenate([np.asarray(state_n[i], dtype=np.float64), aug])
            for i in range(self.n_agents)
        ]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)             # (N, k+1)
        a = actions[:, : self.act_dim_env]                          # (N, k) desired torque
        w = actions[:, self.act_dim_env:].reshape(self.n_agents)    # (N,) gain control

        # 1) set the per-agent gain vector beta_vec from the policy's control dim
        if self.beta_mode == "direct":
            self._beta_vec, dbeta = direct_beta(
                w, self.beta_max, self._beta_vec, self.beta_ema
            )
        else:  # integrator: one global rate-limited scalar, replicated per agent
            self._beta, dbeta = beta_step(
                self._beta, w, self.delta, self.beta_max, self.beta_driver
            )
            self._beta_vec[:] = self._beta

        # 2) compensate the desired torque with the CACHED coupling waveform
        #    (x2 from the previous step = what this action faces), per-agent gain.
        x2_used = self._x2.copy()
        u = compensate(a, self._beta_vec, x2_used, self._offsets)   # (N, k), clipped

        # 3) step the underlying env with the k-dim compensated torque
        obs_n, state_n, rewards, dones, infos, avail = super().step(u)

        # 4) update the coupling-waveform cache from the EXECUTED torque, by the
        #    env's own leak recursion — this is x2 for the NEXT action.
        u_flat = u.reshape(-1)
        self._x2 = leak_step(self._x2, u_flat, self.rho)

        # 5) per-agent |x2| EMA chart feature (persistent)
        absx2_agent = np.array([
            float(np.mean(np.abs(self._x2[self._offsets[i]:self._offsets[i + 1]])))
            for i in range(self.n_agents)
        ])
        self._ema_absx2 = self._ema_alpha * self._ema_absx2 + (1.0 - self._ema_alpha) * absx2_agent

        # 6) diagnostics into info (logging + the arithmetic exactness gate).
        #    pact_x2 (post-update) and pcr_d_next are BOTH "what action t+1 faces",
        #    so corr(pact_x2, pcr_d_next) -> 1 certifies the wiring. HYGIENE:
        #    pcr_d_next is used only for these logging columns, never in control.
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        d_next = np.asarray(
            info0.get("pcr_d_next", np.full(self.n_act, np.nan)), dtype=np.float64
        ).reshape(-1)
        # payload for the optional phase-signal features (next obs' phase; c is slow)
        self._payload_cache = float(info0.get("pcr_payload", self._payload_cache))
        beta_flat = np.repeat(self._beta_vec, self.act_dim_env)      # per-joint gain
        beta_x2 = beta_flat * self._x2
        pre_clip = a.reshape(-1) - beta_flat * x2_used              # what railed before clip
        info0["pact_beta"] = float(np.mean(self._beta_vec))
        info0["pact_dbeta"] = float(dbeta)
        info0["pact_x2"] = self._x2.tolist()
        info0["pact_d_next"] = d_next.tolist()
        info0["pact_x2_absmean"] = float(np.mean(np.abs(self._x2)))
        info0["pact_resid_absmean"] = (
            float(np.mean(np.abs(d_next - beta_x2)))
            if np.all(np.isfinite(d_next)) else float("nan")
        )
        info0["pact_u_clip"] = float(np.mean(np.abs(pre_clip) > 1.0))
        info0["pact_u_minus_a"] = float(np.sqrt(np.mean((u_flat - a.reshape(-1)) ** 2)))

        return (
            self._augment_obs(obs_n),
            self._augment_state(state_n),
            rewards,
            dones,
            infos,
            avail,
        )

    # ----------------------------------------------------------------- reset
    def reset(self, **kwargs):
        obs_n, state_n, avail = super().reset(**kwargs)
        # d resets each episode (ant.py's reset_model zeroes _d), so x2 -> 0;
        # beta and the EMA PERSIST across episodes (c persists — spec §1.2).
        self._x2 = np.zeros(self.n_act)
        return self._augment_obs(obs_n), self._augment_state(state_n), avail
