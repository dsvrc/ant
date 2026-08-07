"""TIER-0 PROBE #2 — the compensation loop gain  (theorem T4)

Runs against an EXISTING B0 checkpoint.  No training, no env change.  Replaces
`excite.py` as the load-bearing Tier-0 measurement after excite.py REFUTED T2
(see PACT2_upgrade_spec.md §7 for that result and why the basis-rank story cannot
be rescued).

THE CLAIM UNDER TEST
--------------------
PACT's exertion functional reads the EXECUTED torque, so compensating changes the
very quantity being compensated.  The wrapper already does this (`leak_step` is fed
`clip(u)`), so the loop exists in the deployed method and has never been analysed.

With uniform gain beta and the env's coupling operator W:

    tau = a - beta*x2 ,   x2 <- rho*x2 + (1-rho)*W tau
    steady state:  (I + beta*W) x2 = W a   =>   x2 = (I + beta*W)^{-1} W a

so per eigenvalue lambda of W the coupling is scaled, relative to blind, by

    T4:      ratio(lambda; beta)  =  1 / (1 + beta*lambda)

For Ant's per-joint-type W = J - I over 4 legs:  spec(W) = {3, -1, -1, -1}.

    common mode      (lambda = +3):  1/(1+3beta)  -- ATTENUATED. a public good.
    differential     (lambda = -1):  1/(1-beta)   -- AMPLIFIED,  pole at beta = 1.
                                                     a public BAD.

*** At beta = c = 0.45 (perfect cancellation of the delivered disturbance) the
    differential-mode coupling is predicted to be 1.82x LARGER than blind. ***

Why it matters, in three directions:

  * it DERIVES the "loop-gain twist" the pipeline doc only warns about, with a
    sharp constant, and predicts the measured best_beta crossover (1.0 -> 0.25 as
    sigma goes 0.5 -> 1.0);
  * it gives sigma* a second closed-form boundary, sigma_loop = 1/|lambda_min(W)|
    (= 1 here), independent of the actuator-saturation boundary;
  * one agent's gain helps every peer in the common mode and harms them in the
    differential modes, which each agent does not internalise -- so the gains form
    a game whose Nash under-compensates the common mode.  That is the mathematical
    motivation for the CTDE critic, i.e. for the 5500 -> 6000 lift you measured.

PRE-REGISTERED PREDICTIONS (write these down before running)
------------------------------------------------------------
  Q1  measured ratio for lambda=+3 tracks 1/(1+3beta) within ~15%.
  Q2  measured ratio for lambda=-1 tracks 1/(1-beta) and EXCEEDS 1 for every
      beta > 0 -- i.e. compensation makes the differential coupling worse.
  Q3  the amplification grows sharply toward beta -> 1 (the pole).
  Q4  residual |delivered - a| is minimised near beta = c, NOT at beta -> 1.

Q2 is the whole result.  If ratio(-1) <= 1 the loop is not behaving as derived and
T4 must be re-derived WITH clipping (the derivation above ignores the +/-1 rail and
the leak transient; the `clip_frac` column reports how much of the sweep was railed,
which is the first thing to check if the fit is poor at large beta).

THE DRIVER IS FROZEN (this probe's one methodological requirement).  ant.py's clock
persists across episodes AND across env instances, so successive cells otherwise sit
at different phases of A(t) -- which is what made excite.py's returns bounce
3969..5531 with no trend.  Every cell here pins A to --freeze_a, so cells differ only
by beta.  c = freeze_a * severity is then known exactly and printed.

USAGE
-----
    python -m harl.envs.mamujoco.pact.loop --selftest          # no simulator needed

    python -m harl.envs.mamujoco.pact.loop \
        --load_config results/.../B0/config.json \
        --model_dir   results/.../B0/models \
        --betas 0,0.15,0.3,0.45,0.6,0.8 --freeze_a 1.0 --episodes 20
"""

import argparse
import json
import os

import numpy as np

RHO_DEFAULT = 0.8


# ==========================================================================
#  Pure-numpy core (no torch / gym / mujoco) -- exercised by --selftest.
# ==========================================================================

def ant_W(n_legs=4):
    """The DEPLOYED coupling operator: hips couple to hips, ankles to ankles.

    Exactly ant.py's `s[0::2] = hip.sum()-hip ; s[1::2] = ank.sum()-ank`, written
    as a matrix so it can be eigendecomposed.  Symmetric, zero-diagonal in the
    agent-block sense (leg l's rows never read leg l).
    """
    n = 2 * n_legs
    W = np.zeros((n, n))
    for l in range(n_legs):
        for lp in range(n_legs):
            if l == lp:
                continue
            W[2 * l, 2 * lp] = 1.0
            W[2 * l + 1, 2 * lp + 1] = 1.0
    return W


def modal_projectors(W, tol=1e-8):
    """Group W's spectrum into distinct eigenvalues and return their projectors.

    Done numerically rather than hard-coded so the probe stays correct if the
    coupling basis is changed (declaration #2). Returns [(lambda, P), ...] sorted
    by descending lambda.
    """
    w, V = np.linalg.eigh((W + W.T) / 2.0)
    out = []
    i = 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and abs(w[j + 1] - w[i]) < tol:
            j += 1
        Vk = V[:, i:j + 1]
        out.append((float(w[i]), Vk @ Vk.T))
        i = j + 1
    out.sort(key=lambda t: -t[0])
    return out


def predicted_ratio(lam, beta):
    """T4: the steady-state coupling gain relative to blind, per mode."""
    den = 1.0 + beta * lam
    return float("inf") if abs(den) < 1e-12 else 1.0 / den


def leak_step(x2, u, W, rho):
    return rho * x2 + (1.0 - rho) * (W @ u)


class ModalAccumulator:
    """Accumulates E||P_lambda x2|| per mode, plus clipping and residual stats."""

    def __init__(self, modes, c):
        self.modes = modes
        self.c = c
        self.energy = np.zeros(len(modes))
        self.clip = 0.0
        self.resid = 0.0
        self.T = 0

    def add(self, x2, pre_clip, a, u):
        for k, (_lam, P) in enumerate(self.modes):
            self.energy[k] += float(np.linalg.norm(P @ x2))
        self.clip += float(np.mean(np.abs(pre_clip) > 1.0))
        # delivered = clip(u) + c*x2 ; the policy intended a
        self.resid += float(np.mean(np.abs(u + self.c * x2 - a)))
        self.T += 1

    def result(self):
        T = max(1, self.T)
        return dict(energy=(self.energy / T).tolist(),
                    clip_frac=self.clip / T, resid=self.resid / T, n_steps=self.T)


# ==========================================================================
#  Self-test: reproduce T4 on a synthetic linear plant, no simulator.
# ==========================================================================

def selftest():
    print("=" * 78)
    print("T4 SELFTEST  (pure numpy: no mujoco, no torch, no checkpoint)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    W = ant_W(4)
    n = W.shape[0]

    # --- 1. the operator is the deployed one ---------------------------------
    v = rng.normal(size=n)
    ref = np.empty(n)
    hip, ank = v[0::2], v[1::2]
    ref[0::2] = hip.sum() - hip
    ref[1::2] = ank.sum() - ank
    assert np.allclose(W @ v, ref)
    print("[1] W reproduces ant.py's coupling_sum exactly                     OK")

    # --- 2. spectrum ----------------------------------------------------------
    modes = modal_projectors(W)
    print("[2] spec(W): " + ", ".join(f"lambda={l:+.1f} (mult {int(round(np.trace(P)))})"
                                      for l, P in modes))
    assert len(modes) == 2, modes
    print("    -> common mode lambda=+3 (attenuated), differential lambda=-1 "
          "(AMPLIFIED, pole at beta=1)")

    # --- 3. T4 on a synthetic plant (no clipping) -----------------------------
    T, rho = 20000, RHO_DEFAULT
    print("\n[3] T4 -- measured vs predicted steady-state modal gain (no clipping)")
    print(f"    {'beta':>7}" + "".join(f"{('lam=%+d meas' % l):>14}{('pred'):>9}"
                                       for l, _ in modes))
    A = rng.normal(0, 0.3, size=(T, n))
    base = None
    for beta in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8):
        x2 = np.zeros(n)
        acc = np.zeros(len(modes))
        for t in range(T):
            u = A[t] - beta * x2                       # NO clip: tests the algebra
            x2 = leak_step(x2, u, W, rho)
            if t > 500:                                 # discard the leak transient
                for k, (_l, P) in enumerate(modes):
                    acc[k] += np.linalg.norm(P @ x2)
        acc /= (T - 500)
        if base is None:
            base = acc.copy()
        row = f"    {beta:>7.2f}"
        for k, (l, _P) in enumerate(modes):
            row += f"{acc[k]/base[k]:>14.3f}{predicted_ratio(l, beta):>9.3f}"
        print(row)
    print("    -> if the two columns agree per mode, T4's algebra is right and the")
    print("       only open question is how much the +/-1 rail bends it in the env.")
    print("\n" + "=" * 78)
    print("SELFTEST DONE. Now run the real probe: the question is whether the")
    print("TRAINED GAIT + clipping still follow 1/(1+beta*lambda).")
    print("=" * 78)


# ==========================================================================
#  Real probe.
# ==========================================================================

def build_env(cfg_path):
    from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_args = dict(cfg["env_args"])
    for k in ("pact", "diag", "echor", "recon", "omax"):
        env_args.pop(k, None)
    return MujocoMulti(env_args=env_args), cfg


def build_actors(cfg, env, model_dir, device):
    import torch
    from harl.algorithms.actors import ALGO_REGISTRY

    algo = cfg["main_args"]["algo"]
    algo = "happo" if algo in ("pact", "omax", "oracle") else algo
    merged = {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]}
    actors = []
    for i in range(env.n_agents):
        ag = ALGO_REGISTRY[algo](
            merged, env.observation_space[i], env.action_space[i], device=device
        )
        ag.actor.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_agent{i}.pt"),
                       map_location=device, weights_only=False)
        )
        ag.prep_rollout()
        actors.append(ag)
    return actors


def inner_ant(env):
    """The deployed AntEnv underneath MujocoMulti's NormalizedActions(TimeLimit(.))."""
    e = getattr(env, "env", env)
    return getattr(e, "unwrapped", e)


def pin_driver(env, freeze_a):
    """Hold A(t) at a constant so every cell differs only by beta.

    ant.py snapshots the knobs per instance (`self._freeze_a`), so setting it on the
    live object is the documented way to freeze an already-built env. Returns True
    if the pin took, False if the deployed ant.py predates the freeze knob.
    """
    tgt = inner_ant(env)
    if not hasattr(tgt, "_freeze_a"):
        return False
    tgt._freeze_a = None if freeze_a is None else float(freeze_a)
    tgt._clock = 0
    return True


def replay_modal(trace, W, modes, beta, rho, clip=True):
    """OPEN-LOOP transfer: replay a FIXED action trace through the loop at gain beta.

    The closed-loop sweep conflates two things -- the loop gain 1/(1+beta*lambda),
    and the fact that the POLICY REACTS: at beta=0 it is flailing under a full
    disturbance, at beta=c it runs its clean stationary gait. Different `a` means a
    different `W a` means the ratio is not the transfer function.

    Replaying one recorded `a` sequence at every beta holds the gait fixed, so what
    is left is exactly T4's algebra plus the +/-1 rail. This is the cell that should
    match the prediction to within clip_frac; the closed-loop table then shows how
    much the gait's own adaptation absorbs. Two panels, and the gap between them is
    itself a result (the loop gain a trained policy actually experiences is softer
    than the open-loop one).

    `trace` is a list of per-episode (T_ep, n) arrays of the RAW policy actions.
    Pure numpy -- no env, no torch. With ``clip=False`` the +/-1 rail is disabled, so
    the result is the EXACT linear-filter answer and the gap to the clipped run is
    attributable to saturation alone. Returns (rms_energies, clip_frac).
    """
    energy = np.zeros(len(modes))
    clipped = 0.0
    T = 0
    for ep in trace:
        x2 = np.zeros(ep.shape[1])
        for t in range(ep.shape[0]):
            pre = ep[t] - beta * x2
            u = np.clip(pre, -1.0, 1.0) if clip else pre
            x2 = leak_step(x2, u, W, rho)
            if t > 20:                       # discard the leak transient, as in rollout()
                for k, (_l, P) in enumerate(modes):
                    energy[k] += float(P @ x2 @ (P @ x2))   # squared norm -> RMS
                clipped += float(np.mean(np.abs(pre) > 1.0))
                T += 1
    T = max(1, T)
    return np.sqrt(energy / T), clipped / T


def spectral_ratio(trace, W, modes, beta, rho):
    """ANALYTIC prediction that accounts for the gait's spectrum, not just DC.

    The leak is a first-order filter, so per mode lambda

        H_lam(z; beta) = (1-rho) / (z - rho + (1-rho)*beta*lam)

    and the ratio to blind is  (z-rho)/(z-rho+(1-rho)*beta*lam).  At z=1 this is the
    DC value 1/(1+beta*lam); at z=e^{i w} the magnitude |z-rho| grows, so the ratio is
    pulled TOWARD 1 -- the loop bites hardest on SLOW content.  That matters here
    because the driver c(t) is slow by construction (period 40000), so the loop is
    strongest exactly where the non-stationarity lives.

    Weighting |H|^2 by the measured power spectrum of the mode-projected drive
    P_lam W a gives the RMS ratio with NO free parameters:

        ratio_lam = sqrt( sum_w |H_lam(w;beta)|^2 S_lam(w) / sum_w |H_lam(w;0)|^2 S_lam(w) )
    """
    out = np.zeros(len(modes))
    for k, (lam, P) in enumerate(modes):
        num = den = 0.0
        for ep_full in trace:
            # Match replay_modal's warmup discard EXACTLY. Episode starts are
            # atypical (the Ant settles from a randomized pose), so including them
            # here but not there is a systematic mismatch, not a rounding detail.
            ep = ep_full[21:]
            if ep.shape[0] < 32:
                continue
            # NB: do NOT de-mean. The DC bin carries real power that the simulation
            # includes, and it is exactly where the loop gain is largest. A constant
            # mean is perfectly periodic, so it leaks nothing; removing it would bias
            # the prediction toward high frequency and toward 1.
            v = (P @ (W @ ep.T)).T                      # (T_ep, n) mode-projected drive
            V = np.fft.rfft(v, axis=0)
            S = np.sum(np.abs(V) ** 2, axis=1)          # power per frequency bin
            w = 2.0 * np.pi * np.arange(len(S)) / ep.shape[0]
            z = np.exp(1j * w)
            h_b = (1.0 - rho) / (z - rho + (1.0 - rho) * beta * lam)
            h_0 = (1.0 - rho) / (z - rho)
            num += float(np.sum(np.abs(h_b) ** 2 * S))
            den += float(np.sum(np.abs(h_0) ** 2 * S))
        out[k] = np.sqrt(num / den) if den > 0 else float("nan")
    return out


def rollout(env, actors, W, modes, beta, c, episodes, rho, hidden, rec_n,
            seed, max_steps, freeze_a, trace=None):
    import torch
    from harl.utils.trans_tools import _t2n

    n_agents = env.n_agents
    dims = [sp.shape[0] for sp in env.true_action_space]
    n = int(sum(dims))
    off = np.cumsum([0] + dims)
    acc = ModalAccumulator(modes, c)
    rng = np.random.default_rng(seed)
    rets = []

    for _ep in range(episodes):
        pin_driver(env, freeze_a)          # re-pin: reset_model does not touch _clock
        obs, _st, _av = env.reset()
        pin_driver(env, freeze_a)
        obs = np.array(obs, dtype=np.float32)
        rnn = np.zeros((n_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)
        x2 = np.zeros(n)                    # resets per episode, as the env's d does
        ep_a = [] if trace is not None else None
        ret = 0.0
        for t in range(max_steps):
            if actors is None:
                a_list = [rng.uniform(-1, 1, size=dims[i]) for i in range(n_agents)]
            else:
                a_list = []
                with torch.no_grad():
                    for i in range(n_agents):
                        act, r_ = actors[i].act(
                            obs[i:i + 1], rnn[i:i + 1], masks[i:i + 1],
                            None, deterministic=True,
                        )
                        rnn[i] = _t2n(r_)[0]
                        a_list.append(_t2n(act)[0][:dims[i]])
            a = np.clip(np.concatenate(a_list), -1.0, 1.0)

            if ep_a is not None:
                ep_a.append(a.copy())       # the RAW policy action, pre-compensation

            pre_clip = a - beta * x2                    # what railed, if anything
            u = np.clip(pre_clip, -1.0, 1.0)
            if t > 20:                                  # skip the leak transient
                acc.add(x2, pre_clip, a, u)
            x2 = leak_step(x2, u, W, rho)

            obs_l, _s, rew, dones, _infos, _av = env.step(
                [u[off[i]:off[i + 1]] for i in range(n_agents)]
            )
            ret += float(np.asarray(rew).reshape(-1)[0])
            if bool(np.all(dones)):
                break
            obs = np.array(obs_l, dtype=np.float32)
        rets.append(ret)
        if ep_a is not None and len(ep_a) > 21:
            trace.append(np.asarray(ep_a))

    out = acc.result()
    out["return"] = float(np.mean(rets))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--load_config", type=str, default=None)
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--betas", type=str, default="0,0.15,0.3,0.45,0.6,0.8")
    p.add_argument("--freeze_a", type=float, default=1.0,
                   help="hold A(t) here (1.0 = the driver peak). c = freeze_a*severity.")
    p.add_argument("--severity", type=float, default=None,
                   help="override sigma for the c used in the residual column only")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--rho", type=float, default=RHO_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay_beta", type=float, default=None,
                   help="beta whose gait is recorded for the OPEN-LOOP replay pass "
                        "(default: the cell nearest c, i.e. the clean stationary gait)")
    p.add_argument("--out", type=str, default="loop_probe.csv")
    args = p.parse_args()

    if args.selftest or args.load_config is None or args.model_dir is None:
        selftest()
        if args.load_config is None and not args.selftest:
            print("\n(no --load_config/--model_dir: ran the selftest only)")
        return

    import csv
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, cfg = build_env(args.load_config)
    dims = [sp.shape[0] for sp in env.true_action_space]
    if set(dims) != {2}:
        raise SystemExit(f"ant_W() is the Ant hip/ankle operator; per-agent dims {dims}.")
    W = ant_W(env.n_agents)
    modes = modal_projectors(W)
    hidden = int(cfg["algo_args"]["model"]["hidden_sizes"][-1])
    rec_n = int(cfg["algo_args"]["model"]["recurrent_n"])
    betas = [float(s) for s in args.betas.split(",") if s.strip()]

    sigma = args.severity
    if sigma is None:
        sigma = float(getattr(inner_ant(env), "_severity",
                              os.environ.get("ANT_PCR_SEVERITY", 0.45)))
    c = args.freeze_a * sigma
    pinned = pin_driver(env, args.freeze_a)

    print("=" * 96)
    print(f"T4 LOOP PROBE   freeze_a={args.freeze_a}  sigma={sigma}  => c={c:.3f}   "
          f"rho={args.rho}  episodes={args.episodes}")
    print(f"driver pinned: {'YES' if pinned else '*** NO -- deployed ant.py has no _freeze_a; cells are NOT comparable ***'}")
    print("spec(W): " + ", ".join(f"lambda={l:+.1f}(x{int(round(np.trace(P)))})"
                                  for l, P in modes))
    print("Q2 is the result: the lambda=-1 ratio must EXCEED 1 -- compensation makes")
    print("the differential-mode coupling WORSE. Predicted 1/(1-beta).")
    print("=" * 96)
    hdr = f"{'beta':>7}"
    for l, _ in modes:
        hdr += f"{('lam%+d meas' % l):>13}{'pred':>8}{'err%':>7}"
    hdr += f"{'clip':>7}{'resid':>9}{'return':>10}"
    print(hdr)
    print("-" * 96)

    actors = build_actors(cfg, env, args.model_dir, device)
    # record the gait ONCE, at the beta whose cell runs the clean (compensated) gait,
    # so the open-loop replay pass below holds `a` fixed across the sweep.
    rec_beta = min(betas, key=lambda b: abs(b - (c if args.replay_beta is None
                                                 else args.replay_beta)))
    trace = []
    rows, base = [], None
    for beta in betas:
        s = rollout(env, actors, W, modes, beta, c, args.episodes, args.rho,
                    hidden, rec_n, args.seed, args.max_steps, args.freeze_a,
                    trace=(trace if beta == rec_beta else None))
        e = np.array(s["energy"])
        if base is None:
            base = e.copy()
        line = f"{beta:>7.2f}"
        rec = dict(beta=beta, clip_frac=s["clip_frac"], resid=s["resid"],
                   ret=s["return"])
        for k, (l, _P) in enumerate(modes):
            meas = e[k] / max(base[k], 1e-12)
            pred = predicted_ratio(l, beta)
            err = 100.0 * (meas - pred) / pred if np.isfinite(pred) and pred else float("nan")
            line += f"{meas:>13.3f}{pred:>8.3f}{err:>7.1f}"
            rec[f"meas_lam{l:+.0f}"] = meas
            rec[f"pred_lam{l:+.0f}"] = pred
        line += f"{s['clip_frac']:>7.3f}{s['resid']:>9.4f}{s['return']:>10.1f}"
        print(line)
        rows.append(rec)

    # ---- OPEN-LOOP pass: same gait at every beta, so only T4's algebra is left ----
    if trace:
        n_steps = sum(len(e) for e in trace)
        print("\n" + "=" * 96)
        print(f"OPEN LOOP -- gait recorded once at beta={rec_beta} ({len(trace)} episodes, "
              f"{n_steps} steps) and replayed at every beta.")
        print("The closed-loop table above conflates the loop gain with the policy's own "
              "reaction; this one does not.")
        print("=" * 96)
        # four numbers per cell:
        #   railed   replay WITH the +/-1 rail        (what the env actually does)
        #   linear   replay WITHOUT it                (the exact linear-filter answer)
        #   spectral analytic, gait spectrum weighted (closed form, no free parameters)
        #   DC       1/(1+beta*lam)                   (the steady-state approximation)
        # linear ~= spectral certifies the transfer-function derivation; the DC column
        # is what the naive steady-state formula would have claimed.
        e0r = replay_modal(trace, W, modes, 0.0, args.rho, clip=True)[0]
        e0l = replay_modal(trace, W, modes, 0.0, args.rho, clip=False)[0]

        # --- the constant the GAIN GAME turns on -------------------------------
        # Raising beta_i attenuates the common mode (helps every peer) and amplifies
        # the differential modes (harms every peer). Which effect dominates -- hence
        # whether Nash OVER- or UNDER-compensates relative to the social optimum --
        # is decided by how the gait's coupling energy splits across the spectrum.
        # This is a measured property of B0's gait, not a modelling assumption.
        share = (e0l ** 2) / max(float(np.sum(e0l ** 2)), 1e-30)
        print("\n  MODAL ENERGY SPLIT of the uncompensated coupling (beta=0):")
        for k, (l, P) in enumerate(modes):
            print(f"    lambda={l:+.1f} (mult {int(round(np.trace(P)))}): "
                  f"{share[k]:6.1%} of ||x2||^2")
        print("    -> this ratio sets the SIGN of the compensation externality, i.e.")
        print("       whether the gain game's Nash over- or under-compensates.")
        for k, (l, _P) in enumerate(modes):
            print(f"\n  mode lambda = {l:+.1f}")
            print(f"  {'beta':>7}{'railed':>10}{'linear':>10}{'spectral':>10}"
                  f"{'DC':>10}{'lin-vs-spec%':>14}{'clip':>8}")
            for beta in betas:
                er, cf = replay_modal(trace, W, modes, beta, args.rho, clip=True)
                el, _ = replay_modal(trace, W, modes, beta, args.rho, clip=False)
                sp = spectral_ratio(trace, W, modes, beta, args.rho)[k]
                rr = er[k] / max(e0r[k], 1e-12)
                rl = el[k] / max(e0l[k], 1e-12)
                dc = predicted_ratio(l, beta)
                gap = 100.0 * (rl - sp) / sp if sp else float("nan")
                print(f"  {beta:>7.2f}{rr:>10.3f}{rl:>10.3f}{sp:>10.3f}{dc:>10.3f}"
                      f"{gap:>14.1f}{cf:>8.3f}")
                for r_ in rows:
                    if r_["beta"] == beta:
                        r_[f"ol_railed_lam{l:+.0f}"] = rr
                        r_[f"ol_linear_lam{l:+.0f}"] = rl
                        r_[f"ol_spectral_lam{l:+.0f}"] = sp
                        r_["ol_clip"] = cf
        print("\n  Read: linear ~= spectral => the transfer function")
        print("        H_lam(z;beta)/H_lam(z;0) = (z-rho)/(z-rho+(1-rho)*beta*lam)")
        print("        is exact, and the DC column's error is the FREQUENCY effect:")
        print("        the loop bites hardest on SLOW content -- which is precisely")
        print("        where the driver c(t) lives (period 40000). railed-vs-linear")
        print("        is the saturation contribution, and it is the sigma_sat channel.")
        print("-" * 96)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r_ in rows for k in r_})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.abspath(args.out)}")

    # ---------------------------- verdict ------------------------------------
    print("\nVERDICT")
    lam_neg = [l for l, _ in modes if l < 0]
    if lam_neg:
        key = f"meas_lam{lam_neg[0]:+.0f}"
        amp = [(r["beta"], r[key]) for r in rows if r["beta"] > 0]
        worst = max(amp, key=lambda t: t[1]) if amp else (0, 0)
        all_above = all(v > 1.0 for _b, v in amp)
        print(f"  Q2  differential-mode ratio at beta>0: "
              f"{['%.2f' % v for _b, v in amp]}  (max {worst[1]:.2f} at beta={worst[0]})")
        print("      " + ("CONFIRMED -- compensation AMPLIFIES the differential coupling. "
                          "T4 holds: the loop is real, sigma_loop is a genuine second "
                          "frontier boundary, and the gain game / PoA argument stands."
                          if all_above else
                          "NOT CONFIRMED -- re-derive T4 WITH the +/-1 rail before "
                          "building on it; check the clip column first."))
    errs = [abs(r.get("meas_lam+3", np.nan) - r.get("pred_lam+3", np.nan))
            / max(r.get("pred_lam+3", 1e-9), 1e-9) for r in rows if r["beta"] > 0]
    errs = [e for e in errs if np.isfinite(e)]
    if errs:
        print(f"  Q1  common-mode fit: mean |error| = {100*np.mean(errs):.1f}%  "
              f"{'CONFIRMED' if np.mean(errs) < 0.15 else 'off-prediction'}")
    best = min(rows, key=lambda r: r["resid"])
    print(f"  Q4  residual minimised at beta={best['beta']:.2f} (c={c:.3f})  "
          f"{'CONFIRMED' if abs(best['beta']-c) <= 0.16 else 'off-prediction'}")
    mx = max(r["clip_frac"] for r in rows)
    if mx > 0.05:
        print(f"  !!  max clip_frac = {mx:.3f}: the rail is active, so the linear "
              f"derivation is only approximate at the top of the beta grid.")


if __name__ == "__main__":
    main()
