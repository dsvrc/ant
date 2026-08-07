"""Tier-0 probe machinery  [campaign spec Part 4.0].

Pure evaluation: a probe interposes on the flat 8-dim action at the gym boundary
and rewrites it. **No gradient anywhere, no learning, no method component.**

The shim
--------
``MujocoMulti`` builds ``wrapped_env = NormalizedActions(TimeLimit(AntEnv))`` and
steps it with the concatenated 8-dim action, so interposing there sees exactly
the commanded torque vector::

    install_probe(mm, probe_fn)      # NOT `mm.wrapped_env = ProbeShim(...)` alone

Why the helper and not the bare assignment: ``MujocoMulti.reset()`` calls
``self.timelimit_env.reset()`` — a reference captured at construction — so it
**bypasses** ``wrapped_env`` entirely and ``ProbeShim.reset()`` would never fire.
``install_probe`` therefore also wraps ``mm.reset`` so per-episode probe state
(delay rings, EMAs, the DOB history) is cleared exactly when the env clears
``_d``. Getting this wrong silently leaks state across episodes.

``NormalizedActions`` is an identity map for Ant's [-1,1] bounds
(``a -> ((a+1)/2)*(high-low) + low = a``); ``install_probe`` asserts it once
rather than trusting the comment.

Probes (spec §4.0)
------------------
``identity``          control arm; must reproduce plain eval exactly.
``cancel``            ``a' = clip(a - beta * transform(d_applied), -1, 1)`` — the
                      feed-forward disturbance rejection of E2/E3.
``project_sumzero``   hips ``a[0,2,4,6] -= mean(...)``, ankles likewise, then clip.
                      Uses **no** d — the information-free escape (E4).

Transforms for ``cancel`` (the E3 degraded-information frontier)
---------------------------------------------------------------
``exact``       d itself (the E2 arm).
``delay:k``     d from k steps ago (ring buffer) — the lag budget.
``ema:h``       EMA of d with half-life h — the bandwidth budget.
``dc:h``        the DC (slow) component of d, i.e. an EMA of half-life h. By
                construction ``dc:64`` computes the same thing as ``ema:64``;
                the campaign keeps both cells because they answer different
                questions (how fast must an estimator be, vs is a *slow chart*
                sufficient at all — the ECL/c-conditioning retro-diagnosis). Do
                not "optimize away" the duplicate: the report reads them as
                separate axes (V8 vs the L2 leaf).
``noise:r``     d + N(0, (r * running-RMS(d))^2) — the accuracy budget.
``sign_leg``    per-leg mean magnitude x per-joint sign — a coarse quantization.
``dob:<npz>``   the E5-fitted causal linear filter, run online on **agent-local
                features only** (own obs history + own action history). This is
                the decentralized-feasibility certificate: no privileged input at
                run time. See §4.3 E3-DOB.

Standing rule (spec §4.0): **every probe is first run at ``FREEZE_A=0`` and must
reproduce the stationary return within CI.** A probe that degrades the stock env
is confounded and its results are void — fix the probe, do not reinterpret
(abort rule 5). ``project_sumzero`` is the one exception: its A=0 run *is* the
measurement (the cost of the constraint itself).

Self-test (no simulator, no torch)::

    python -m harl.envs.mamujoco.diag.probes --selftest
"""

import argparse
import os
import sys

import numpy as np

_N_ACT = 8
_N_LEGS = 4


# ==========================================================================
#  transforms  (stateful; reset() at every episode boundary)
# ==========================================================================
class Transform:
    """Base: maps the true d to what the controller gets to use."""

    name = "base"

    def __call__(self, d):
        raise NotImplementedError

    def reset(self):
        pass


class Exact(Transform):
    name = "exact"

    def __call__(self, d):
        return np.asarray(d, dtype=np.float64)


class Delay(Transform):
    """d from k steps ago. k=0 is exact. The pre-episode history is zeros, which
    is what the env's d actually is at an episode start."""

    def __init__(self, k):
        self.k = int(k)
        assert self.k >= 0, "delay k must be >= 0"
        self.name = f"delay:{self.k}"
        self.reset()

    def reset(self):
        self.ring = [np.zeros(_N_ACT) for _ in range(self.k + 1)]
        self.i = 0

    def __call__(self, d):
        self.ring[self.i] = np.asarray(d, dtype=np.float64).copy()
        self.i = (self.i + 1) % (self.k + 1)
        return self.ring[self.i].copy()      # the slot k steps old


class Ema(Transform):
    """EMA lowpass with the given half-life (in steps). h<=0 => exact."""

    def __init__(self, halflife, name=None):
        self.h = float(halflife)
        self.lam = 0.0 if self.h <= 0 else 0.5 ** (1.0 / self.h)
        self.name = name or f"ema:{self.h:g}"
        self.reset()

    def reset(self):
        self.state = np.zeros(_N_ACT)

    def __call__(self, d):
        d = np.asarray(d, dtype=np.float64)
        self.state = self.lam * self.state + (1.0 - self.lam) * d
        return self.state.copy()


class Dc(Ema):
    """The DC (slow) component of d. Same operation as ``ema`` at the same
    half-life — kept separate on purpose (see the module docstring)."""

    def __init__(self, halflife=64):
        super().__init__(halflife, name=f"dc:{float(halflife):g}")


class Noise(Transform):
    """d + Gaussian noise with sigma = rel_sigma * running-RMS(|d|).

    The RMS tracker is an EMA over the episode so sigma follows the disturbance's
    own scale (a fixed absolute sigma would mean nothing at the trough)."""

    def __init__(self, rel_sigma, halflife=64, seed=0):
        self.rel = float(rel_sigma)
        self.lam = 0.5 ** (1.0 / float(halflife))
        self.seed = int(seed)
        self.name = f"noise:{self.rel:g}"
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self):
        self.ms = 0.0            # running mean-square
        self.n = 0

    def __call__(self, d):
        d = np.asarray(d, dtype=np.float64)
        self.ms = self.lam * self.ms + (1.0 - self.lam) * float(np.mean(d * d))
        self.n += 1
        # de-bias the EMA at the start of an episode (else sigma starts at ~0)
        corr = 1.0 - self.lam ** self.n
        rms = float(np.sqrt(max(self.ms / max(corr, 1e-12), 0.0)))
        return d + self._rng.normal(0.0, self.rel * rms, size=d.shape)


class SignLeg(Transform):
    """Per-leg mean magnitude x per-joint sign: you know each joint's sign and its
    leg's average load, but not the per-joint magnitude."""

    name = "sign_leg"

    def __call__(self, d):
        d = np.asarray(d, dtype=np.float64)
        legs = d.reshape(_N_LEGS, 2)
        mag = np.abs(legs).mean(axis=1, keepdims=True)     # (4,1)
        return (np.sign(legs) * mag).reshape(-1)


class Dob(Transform):
    """The E5-fitted causal linear filter, applied online to agent-local features.

    Signature note: unlike every other transform this one **ignores its d
    argument** — that is the entire point. It reads only the per-agent history
    the policy itself could read (own obs, own commanded action), so a passing
    E3-DOB cell certifies a decentralized solution end-to-end. The shim feeds it
    the observations via ``observe()`` and the actually-commanded action via
    ``post_step()``.

    Timing (must match ``sysid.py`` exactly, or the certificate is meaningless):
    row t of the fit is ``features(o_{t-L+1..t}, a_{t-L+1..t}) -> d_next(t)``.
    At step t+1 the controller needs ``d_applied(t+1) == d_next(t)``, so it
    predicts from the window ending at ``(o_t, a_t)`` — everything already known
    at decision time. Hence ``observe()`` only *stashes* the current obs; the
    append happens in ``post_step`` after the action is committed.
    """

    def __init__(self, path):
        self.path = str(path)
        z = np.load(self.path, allow_pickle=False)
        self.W = z["W"]                     # (n_agents, D+1, act_dim), row 0 = intercept
        self.L = int(z["L"])
        self.feature_set = str(z["feature_set"])
        self.obs_dim = int(z["obs_dim"])
        self.act_dim = int(z["act_dim"])    # per-agent action dim (2 for Ant 4x2)
        self.n_agents = int(z["n_agents"])
        self.per_lag = int(z["per_lag"])    # sysid's layout; do NOT re-derive it
        self.mu = z["mu"] if "mu" in z.files else None       # (n_agents, D)
        self.sigma = z["sigma"] if "sigma" in z.files else None
        assert self.feature_set in ("F-loc", "F-loc-obs"), (
            f"E3-DOB is the DECENTRALIZED certificate: the filter must be fit on "
            f"agent-local features, got feature_set={self.feature_set!r}. A "
            f"CTDE (F-joint) filter is not admissible here (spec §4.3)."
        )
        assert self.W.shape[1] == self.L * self.per_lag + 1, (
            f"DOB filter shape {self.W.shape} disagrees with L={self.L} x "
            f"per_lag={self.per_lag} (+1 intercept) — refusing to run a filter whose "
            f"feature layout does not match sysid's."
        )
        # HARD decentralization guarantee: the teammates'-action columns must carry
        # EXACTLY zero weight. This slot is never filled at run time either, so a
        # mis-exported CTDE filter fails loudly here instead of silently turning
        # E3-DOB into a privileged arm.
        oth = np.concatenate([
            np.arange(j * self.per_lag + self.obs_dim + self.act_dim,
                      (j + 1) * self.per_lag) for j in range(self.L)
        ]).astype(int) if self.per_lag > self.obs_dim + self.act_dim else np.array([], int)
        if oth.size:
            leak = float(np.abs(self.W[:, 1:, :][:, oth, :]).max())
            assert leak == 0.0, (
                f"DOB filter has non-zero weight ({leak:g}) on TEAMMATES' action "
                f"columns — that is not a decentralized filter (spec §4.3)."
            )
        self.name = f"dob:{os.path.basename(self.path)}"
        self.reset()

    def reset(self):
        self.obs_hist = []      # newest last
        self.act_hist = []
        self._pending_obs = None

    def observe(self, obs_list):
        """Called by the shim at the START of a step with the CURRENT per-agent obs
        (i.e. o_t, the policy's own input). Stashed, not appended — see the timing
        note above."""
        self._pending_obs = [np.asarray(o, dtype=np.float64) for o in obs_list]

    def post_step(self, commanded_flat_a, info):
        """Called by the shim AFTER the step with the action actually commanded
        (post-probe: that is what the env's recursion integrates)."""
        if self._pending_obs is None:
            return
        self.obs_hist.append(self._pending_obs)
        self.act_hist.append(np.asarray(commanded_flat_a, dtype=np.float64).copy())
        if len(self.obs_hist) > self.L:
            self.obs_hist.pop(0)
            self.act_hist.pop(0)
        self._pending_obs = None

    def _features(self, agent):
        """Assemble agent-local features from the ring, lag-major and zero-padded
        at an episode start — byte-for-byte the layout sysid.py fits on
        (``[obs | own_act | other_acts]`` per lag, newest at lag 0).

        The ``other_acts`` slots are deliberately left at zero: this is the
        decentralized certificate, so teammates' actions are not read at run time
        even though the layout reserves room for them (the constructor asserts
        their weights are zero, so leaving them out changes nothing)."""
        x = np.zeros(self.L * self.per_lag)
        n = len(self.obs_hist)
        for j in range(min(n, self.L)):
            # lag j = j steps back from the newest; newest is lag 0
            o = self.obs_hist[n - 1 - j][agent]
            base = j * self.per_lag
            x[base:base + self.obs_dim] = o[: self.obs_dim]
            if self.feature_set == "F-loc":
                a = self.act_hist[n - 1 - j]
                x[base + self.obs_dim: base + self.obs_dim + self.act_dim] = \
                    a[agent * self.act_dim: (agent + 1) * self.act_dim]
        return x

    def __call__(self, d):
        out = np.zeros(_N_ACT)
        if not self.obs_hist:
            return out                      # episode start: the true d is 0 too
        for i in range(self.n_agents):
            x = self._features(i)
            if self.mu is not None:
                x = (x - self.mu[i]) / self.sigma[i]
            out[i * self.act_dim:(i + 1) * self.act_dim] = \
                self.W[i][0] + x @ self.W[i][1:]
        return out


_TRANSFORMS = {
    "exact": lambda arg: Exact(),
    "delay": lambda arg: Delay(int(float(arg))),
    "ema": lambda arg: Ema(float(arg)),
    "dc": lambda arg: Dc(float(arg) if arg else 64),
    "noise": lambda arg: Noise(float(arg)),
    "sign_leg": lambda arg: SignLeg(),
    "dob": lambda arg: Dob(arg),
}


def make_transform(spec):
    """``exact`` | ``delay:4`` | ``ema:16`` | ``dc:64`` | ``noise:0.2`` |
    ``sign_leg`` | ``dob:/path/dob_filter.npz``"""
    if spec is None or spec == "":
        return Exact()
    name, _, arg = str(spec).partition(":")
    if name not in _TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; "
                         f"choose from {sorted(_TRANSFORMS)}")
    return _TRANSFORMS[name](arg)


# ==========================================================================
#  probes
# ==========================================================================
class Probe:
    name = "base"
    uses_d = False

    def __call__(self, a, d_applied):
        raise NotImplementedError

    def reset(self):
        pass


class Identity(Probe):
    """Control arm. Must reproduce plain eval EXACTLY (bitwise) — self-checked."""

    name = "identity"

    def __call__(self, a, d_applied):
        return a


class Cancel(Probe):
    """``a' = clip(a - beta * transform(d_applied), -1, 1)``.

    ``d_applied`` is the d that will be added to THIS step's torque (the shim
    carries it over from the previous step's ``pcr_d_next``), so the cancellation
    is genuinely feed-forward: with beta=1 and no clipping the env delivers
    exactly ``a``.

    Note the ctrl cost is charged on the COMMANDED torque, i.e. on ``a'`` — the
    canceller pays for the torque it commands. That is physically correct (the
    probe *is* the controller) and is part of what E2 measures; the recorder logs
    ``r_ctrl`` so the report can separate it from the achievement term.
    """

    uses_d = True

    def __init__(self, beta=1.0, transform="exact"):
        self.beta = float(beta)
        self.transform = (transform if isinstance(transform, Transform)
                          else make_transform(transform))
        self.name = f"cancel:beta={self.beta:g},transform={self.transform.name}"

    def reset(self):
        self.transform.reset()

    def observe(self, obs_list):
        if hasattr(self.transform, "observe"):
            self.transform.observe(obs_list)

    def post_step(self, commanded_flat_a, info):
        if hasattr(self.transform, "post_step"):
            self.transform.post_step(commanded_flat_a, info)

    def __call__(self, a, d_applied):
        d_hat = self.transform(d_applied)
        return np.clip(a - self.beta * d_hat, -1.0, 1.0)


class ProjectSumZero(Probe):
    """Hips ``a[0,2,4,6] -= mean(a[0,2,4,6])``, ankles likewise, then clip.

    Information-free: reads no d at all. On the sum-zero manifold the mode algebra
    gives ``s_i = -tau_i`` exactly, i.e. the cross-coupling collapses to a private,
    self-inflicted, perfectly predictable gain droop (E4).

    Order matters: project THEN clip. Clipping can re-break the sum-zero property;
    the recorder's ``sumzero_resid`` column measures how much (logged, not hidden).
    """

    name = "project_sumzero"

    def __call__(self, a, d_applied):
        a = np.asarray(a, dtype=np.float64).copy()
        a[0::2] -= a[0::2].mean()
        a[1::2] -= a[1::2].mean()
        return np.clip(a, -1.0, 1.0)


def make_probe(spec):
    """``identity`` | ``project_sumzero`` |
    ``cancel:beta=1.0[,transform=delay:4]``"""
    spec = str(spec)
    name, _, rest = spec.partition(":")
    kw = {}
    if rest:
        for part in rest.split(","):
            k, _, v = part.partition("=")
            kw[k.strip()] = v.strip()
    if name == "identity":
        return Identity()
    if name == "project_sumzero":
        return ProjectSumZero()
    if name == "cancel":
        return Cancel(beta=float(kw.get("beta", 1.0)),
                      transform=kw.get("transform", "exact"))
    raise ValueError(f"unknown probe {name!r}; choose identity | cancel | "
                     f"project_sumzero")


# ==========================================================================
#  the shim
# ==========================================================================
class ProbeShim:
    """Interposes on the flat 8-dim action at the gym boundary.

    Install via ``install_probe(mm, probe_fn)`` — do not assign ``mm.wrapped_env``
    by hand (see the module docstring: reset would not reach the probe).
    """

    _OWN = ("inner", "probe_fn", "_d_next_prev", "_obs_fn", "n_steps")

    def __init__(self, inner, probe_fn, obs_fn=None):
        self.inner = inner
        self.probe_fn = probe_fn
        self._obs_fn = obs_fn
        self._d_next_prev = np.zeros(_N_ACT)
        self.n_steps = 0

    def step(self, flat_a):
        if self._obs_fn is not None and hasattr(self.probe_fn, "observe"):
            self.probe_fn.observe(self._obs_fn())
        a2 = self.probe_fn(np.asarray(flat_a, dtype=np.float64), self._d_next_prev)
        a2 = np.asarray(a2, dtype=np.float64)
        obs, rew, done, info = self.inner.step(a2)
        self._d_next_prev = np.asarray(
            info.get("pcr_d_next", np.zeros(_N_ACT)), dtype=np.float64
        ).copy()
        # expose what the probe actually commanded, for the recorder and for DOB
        info["probe_commanded_action"] = a2
        if hasattr(self.probe_fn, "post_step"):
            self.probe_fn.post_step(a2, info)
        self.n_steps += 1
        return obs, rew, done, info

    def reset_probe_state(self):
        """Clear per-episode probe state. Called by ``install_probe``'s reset hook
        (MujocoMulti.reset bypasses this object, so it cannot be reset() alone)."""
        self._d_next_prev = np.zeros(_N_ACT)   # matches reset_model's self._d[:] = 0
        if hasattr(self.probe_fn, "reset"):
            self.probe_fn.reset()

    def reset(self, **kw):
        self.reset_probe_state()
        return self.inner.reset(**kw)

    def __getattr__(self, name):
        if name in ProbeShim._OWN:
            raise AttributeError(name)
        return getattr(self.inner, name)


def install_probe(mm, probe_fn, check_normalized_actions=True):
    """Install ``probe_fn`` on a constructed ``MujocoMulti``.

    Does three things the bare assignment does not:
      1. asserts ``NormalizedActions`` is the identity for Ant's [-1,1] bounds,
         so the probe's output reaches the env unmodified (asserted once);
      2. wraps ``mm.reset`` so per-episode probe state is cleared (MujocoMulti
         resets via its captured ``timelimit_env`` reference, never via
         ``wrapped_env``);
      3. gives the probe an obs source, for the DOB transform only.

    Returns the shim.
    """
    if check_normalized_actions:
        space = mm.env.action_space
        lo, hi = np.asarray(space.low, float), np.asarray(space.high, float)
        probe_a = np.linspace(-1.0, 1.0, len(lo))
        mapped = ((probe_a + 1.0) / 2.0) * (hi - lo) + lo
        assert np.allclose(mapped, probe_a, atol=1e-12), (
            "NormalizedActions is NOT the identity for this action space "
            f"(low={lo}, high={hi}); the probe's action would be remapped before "
            "reaching the env and every Tier-0 number would be wrong."
        )

    shim = ProbeShim(mm.wrapped_env, probe_fn, obs_fn=lambda: mm.get_obs())
    mm.wrapped_env = shim
    _orig_reset = mm.reset

    def _reset(**kw):
        shim.reset_probe_state()
        return _orig_reset(**kw)

    mm.reset = _reset
    return shim


# ==========================================================================
#  self-test: a synthetic PCR recursion (no simulator, no torch)
# ==========================================================================
class _SyntheticPCR:
    """The EXACT ant_diag disturbance channel with the mujoco part replaced by a
    stub 'plant'. Copied (not imported) from the ECL test harness's stream
    generator, per spec §11.1 item 4.

        d      <- rho*d + (1-rho) * A*sigma * s(tau)      s_i = sum_{j!=i} tau_j
        delivered = clip(tau + d_applied, -1, 1)
        obs    = a deterministic readout of `delivered` with a wandering,
                 sign-flipping plant gain (the ratiometric stressor) + noise.

    Exposes the same info keys as ant_diag so the shim is exercised for real.
    """

    def __init__(self, severity=0.9, rho=0.8, freeze_a=1.0, obs_dim=32, seed=0,
                 ep_len=200, normalize_obs=True):
        self.severity, self.rho, self.freeze_a = severity, rho, freeze_a
        self.obs_dim, self.ep_len = obs_dim, ep_len
        # normalize_obs=True mimics MujocoMulti.get_obs()'s per-timestep whole-vector
        # normalization; False exposes the raw state, which makes the PCR recursion
        # exactly linear in the features — the setting sysid's self-test needs to
        # test the FITTER rather than the env's observability.
        self.normalize_obs = bool(normalize_obs)
        self.rng = np.random.default_rng(seed)
        self.action_space = type("S", (), {"low": -np.ones(8), "high": np.ones(8)})()
        self.reset()

    def reset(self, **kw):
        self._d = np.zeros(_N_ACT)
        self._clock = 0
        self._t = 0
        self._last_delivered = np.zeros(_N_ACT)
        return self._obs()

    def _obs(self):
        o = np.zeros(self.obs_dim)
        g = 1.0 + 0.3 * np.sin(0.01 * self._clock + np.arange(_N_ACT))
        g[1::2] *= -1.0
        o[:_N_ACT] = g * self._last_delivered
        o[_N_ACT:2 * _N_ACT] = self._last_delivered
        o[2 * _N_ACT] = 1.0
        return o

    def get_obs(self):
        """Per-agent obs, mimicking MujocoMulti: shared state + agent one-hot,
        whole-vector normalized (unless normalize_obs=False)."""
        state = self._obs()
        out = []
        for a in range(_N_LEGS):
            oh = np.zeros(_N_LEGS)
            oh[a] = 1.0
            v = np.concatenate([state, oh])
            if self.normalize_obs:
                v = (v - v.mean()) / (v.std() + 1e-12)
            out.append(v)
        return out

    def step(self, a):
        tau = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        d_applied = self._d.copy()
        raw = tau + self._d
        delivered = np.clip(raw, -1.0, 1.0)
        self._last_delivered = delivered
        self._clock += 1
        self._t += 1
        A = self.freeze_a
        hip, ank = tau[0::2], tau[1::2]
        s = np.empty_like(tau)
        s[0::2] = hip.sum() - hip
        s[1::2] = ank.sum() - ank
        self._d = self.rho * self._d + (1.0 - self.rho) * (A * self.severity * s)
        rew = float(np.sum(delivered)) - 0.5 * float(np.square(a).sum())
        done = self._t >= self.ep_len
        info = dict(pcr_payload=float(A), pcr_load=float(np.abs(self._d).mean()),
                    pcr_loadmax=float(np.abs(self._d).max()),
                    pcr_d_applied=d_applied, pcr_d_next=self._d.copy(),
                    pcr_sat_frac=float(np.mean(np.abs(raw) > 1.0)),
                    pcr_clock=int(self._clock),
                    delivered=delivered, s=s)
        return self._obs(), rew, done, info


class _FakeMM:
    """Minimal MujocoMulti stand-in: the same wrapped_env/reset/get_obs contract
    the real one has, including the reset-bypasses-wrapped_env quirk that
    install_probe exists to work around."""

    def __init__(self, env):
        self.wrapped_env = env
        self.env = env
        self.timelimit_env = env

    def get_obs(self):
        return self.wrapped_env.get_obs() if hasattr(self.wrapped_env, "get_obs") \
            else self.env.get_obs()

    def reset(self, **kw):
        return self.timelimit_env.reset(**kw)     # bypasses wrapped_env, as in HARL

    def step(self, a):
        return self.wrapped_env.step(a)


def _rollout(mm, policy, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    mm.reset()
    rec = {k: [] for k in ("a_nom", "a_cmd", "delivered", "d_app", "s", "rew")}
    for t in range(n_steps):
        a = policy(t, rng)
        obs, rew, done, info = mm.step(a)
        rec["a_nom"].append(a.copy())
        rec["a_cmd"].append(info.get("probe_commanded_action", a).copy())
        rec["delivered"].append(info["delivered"].copy())
        rec["d_app"].append(info["pcr_d_applied"].copy())
        rec["s"].append(info["s"].copy())
        rec["rew"].append(rew)
        if done:
            mm.reset()
    return {k: np.asarray(v) for k, v in rec.items()}


def _gait(t, rng):
    """A gait with a NON-vanishing common mode, so the coupling is real.

    Phases are deliberately NOT the natural [0, pi/2, pi, 3pi/2] quadrature: those
    sum to zero, which would make sum(hip) == 0 and hence s_i == -tau_i for free —
    T4 would then pass without ProjectSumZero doing anything, and T2 would never
    reach the actuator rails. Uneven phases + a DC offset give a large common mode
    (the difference-DC amplification E2 is about) and genuine railing."""
    phi = np.array([0.0, 0.7, 1.4, 2.1])
    hip = 0.45 * np.sin(0.3 * t + phi) + 0.15
    ank = 0.30 * np.sin(0.3 * t + phi + 0.5) - 0.10
    a = np.zeros(_N_ACT)
    a[0::2], a[1::2] = hip, ank
    return np.clip(a + 0.02 * rng.standard_normal(_N_ACT), -1.0, 1.0)


def selftest():
    from harl.envs.mamujoco.diag.report_io import DebugReport

    rep = DebugReport(os.path.join("diag_out", "v0", "v0_probes.md"),
                      title="V0 — probe self-test",
                      subtitle="synthetic PCR recursion; no simulator, no torch")
    ok_all = True

    # ---- T1: identity probe reproduces the un-shimmed stream exactly ----------
    rep.h2("T1 — identity probe is bitwise transparent")
    base = _rollout(_FakeMM(_SyntheticPCR(seed=1)), _gait, 400, seed=5)
    mm = _FakeMM(_SyntheticPCR(seed=1))
    install_probe(mm, Identity(), check_normalized_actions=False)
    shimmed = _rollout(mm, _gait, 400, seed=5)
    err = max(float(np.max(np.abs(base[k] - shimmed[k])))
              for k in ("delivered", "d_app", "rew"))
    ok = err == 0.0
    ok_all &= ok
    rep.line(f"  max|Δ(delivered, d_applied, reward)| = {err:.3e} (must be exactly 0)")
    rep.verdict("T1 identity transparency", ok)

    # ---- T2: cancel(beta=1, exact) delivers the nominal action ---------------
    rep.h2("T2 — cancel(beta=1, exact) achieves feed-forward rejection")
    mm = _FakeMM(_SyntheticPCR(seed=1))
    install_probe(mm, Cancel(1.0, "exact"), check_normalized_actions=False)
    r = _rollout(mm, _gait, 400, seed=5)
    # where neither the command nor the nominal action is railed, delivered == a
    free = (np.abs(r["a_cmd"]) < 1.0 - 1e-9) & (np.abs(r["a_nom"]) <= 1.0)
    err = float(np.max(np.abs(r["delivered"] - r["a_nom"])[free])) if free.any() \
        else float("nan")
    ok = bool(free.any()) and err < 1e-12
    ok_all &= ok
    rep.line(f"  unrailed joints: max|delivered - a_nominal| = {err:.3e} (must be ~0)")
    rep.line(f"  railed fraction = {1.0 - float(free.mean()):.3f} "
             f"(where the canceller runs out of actuator authority — this is the "
             f"saturation geometry E2's beta-grid measures on the real ant)")
    rep.line(f"  vs identity: mean|d_applied| {np.abs(base['d_app']).mean():.4f} "
             f"-> {np.abs(r['d_app']).mean():.4f}")
    rep.verdict("T2 exact cancellation", ok)

    # ---- T3: delay(0) == exact; lag degrades the rejection -------------------
    rep.h2("T3 — delay(k): k=0 is exact, and rejection degrades monotonically in k")
    resid = {}
    for k in (0, 1, 2, 4, 8, 16, 32):
        mm = _FakeMM(_SyntheticPCR(seed=1))
        install_probe(mm, Cancel(1.0, f"delay:{k}"), check_normalized_actions=False)
        rk = _rollout(mm, _gait, 400, seed=5)
        resid[k] = float(np.mean(np.abs(rk["delivered"] - rk["a_nom"])))
    mm = _FakeMM(_SyntheticPCR(seed=1))
    install_probe(mm, Cancel(1.0, "exact"), check_normalized_actions=False)
    r0 = _rollout(mm, _gait, 400, seed=5)
    exact_resid = float(np.mean(np.abs(r0["delivered"] - r0["a_nom"])))
    ok = abs(resid[0] - exact_resid) < 1e-12 and resid[32] > resid[1] > resid[0] - 1e-12
    ok_all &= ok
    rep.table(["delay k", "mean|delivered - a_nom|"],
              [[k, f"{v:.5f}"] for k, v in resid.items()])
    rep.line(f"  exact = {exact_resid:.5f}; delay:0 = {resid[0]:.5f} (must match)")
    rep.verdict("T3 delay ring", ok)

    # ---- T4: project_sumzero => s_i == -tau_i exactly ------------------------
    rep.h2("T4 — project_sumzero collapses the coupling to s_i = -tau_i")
    mm = _FakeMM(_SyntheticPCR(seed=1))
    install_probe(mm, ProjectSumZero(), check_normalized_actions=False)
    r = _rollout(mm, _gait, 400, seed=5)
    # only where the post-projection clip did not re-break sum-zero
    unrailed = np.all(np.abs(r["a_cmd"]) < 1.0 - 1e-9, axis=1)
    err = float(np.max(np.abs(r["s"] + r["a_cmd"])[unrailed]))
    ok = err < 1e-9 and unrailed.mean() > 0.5
    ok_all &= ok
    rep.line(f"  unrailed steps ({unrailed.mean():.2f} of all): "
             f"max|s + tau| = {err:.3e} (must be ~0)")
    rep.line(f"  => on the sum-zero manifold the cross-coupling is a PRIVATE, "
             f"self-inflicted, perfectly predictable gain droop (E4's premise).")
    rep.verdict("T4 sum-zero mode algebra", ok)

    # ---- T5: transforms are shape/determinism clean --------------------------
    rep.h2("T5 — transform library: shapes, run-determinism, reset semantics")
    ok = True
    d = np.array([0.4, -0.2, 0.1, 0.5, -0.3, 0.2, 0.05, -0.5])
    stream = [d * (1 + 0.1 * i) for i in range(20)]
    for spec in ("exact", "delay:3", "ema:16", "dc:64", "noise:0.2", "sign_leg"):
        # determinism = two FRESH instances agree (noise seeds at construction, so
        # a whole run is reproducible from the run seed). Not tested via reset():
        # reset() deliberately does NOT reseed Noise — every episode must draw
        # fresh noise, else 40 episodes would share one correlated noise sequence
        # and the E3 noise-budget CI would be a lie.
        out1 = [make_transform(spec)(x).copy() for x in stream[:1]]
        t1, t2 = make_transform(spec), make_transform(spec)
        s1 = [t1(x).copy() for x in stream]
        s2 = [t2(x).copy() for x in stream]
        same = all(np.allclose(x, y) for x, y in zip(s1, s2))
        shape = all(o.shape == (8,) for o in s1)
        ok &= same and shape
        rep.line(f"  {spec:<12} shape8={shape} fresh-instance determinism={same}")
    # reset() must restore the deterministic transforms to a fresh state
    for spec in ("delay:3", "ema:16", "dc:64"):
        t = make_transform(spec)
        for x in stream:
            t(x)
        t.reset()
        after = t(stream[0])
        fresh = make_transform(spec)(stream[0])
        good = np.allclose(after, fresh)
        ok &= good
        rep.line(f"  {spec:<12} reset() -> fresh state: {good}")
    sl = SignLeg()(d)
    legs = d.reshape(4, 2)
    want = (np.sign(legs) * np.abs(legs).mean(axis=1, keepdims=True)).reshape(-1)
    ok &= bool(np.allclose(sl, want))
    rep.line(f"  sign_leg matches per-leg mean-magnitude x sign: "
             f"{np.allclose(sl, want)}")
    # dc:h and ema:h compute the same thing by construction — assert it, so the
    # report's V8/L2 readings are known to be the same operation asked twice.
    same_dc = np.allclose([make_transform("dc:64")(x) for x in stream][-1],
                          [make_transform("ema:64")(x) for x in stream][-1])
    rep.line(f"  dc:64 == ema:64 (by construction, kept as separate cells on "
             f"purpose): {same_dc}")
    ok_all &= ok
    rep.verdict("T5 transform library", ok)

    # ---- T6: install_probe resets probe state on MujocoMulti.reset() ---------
    rep.h2("T6 — install_probe clears probe state on reset (the bypass fix)")
    mm = _FakeMM(_SyntheticPCR(seed=1, ep_len=50))
    probe = Cancel(1.0, "delay:4")
    shim = install_probe(mm, probe, check_normalized_actions=False)
    _rollout(mm, _gait, 60, seed=5)             # crosses one episode boundary
    mm.reset()
    ring_clear = all(np.all(x == 0) for x in probe.transform.ring)
    dprev_clear = bool(np.all(shim._d_next_prev == 0))
    ok = ring_clear and dprev_clear
    ok_all &= ok
    rep.line(f"  delay ring cleared on reset: {ring_clear}")
    rep.line(f"  shim d_next_prev cleared on reset: {dprev_clear}")
    rep.line("  (MujocoMulti.reset() calls timelimit_env.reset() directly — without "
             "install_probe's hook this leaks state across episodes and silently "
             "corrupts every Tier-0 number.)")
    rep.verdict("T6 reset hook", ok)

    rep.h2("SUMMARY")
    rep.verdict("V0 probe self-test", ok_all)
    rep.close()
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="PCR Tier-0 probe library.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic-PCR unit tests (no simulator needed)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
