"""TIER-0 PROBE — does coordination destroy identifiability?  (theorem T2)

Runs against an EXISTING B0 checkpoint.  No training, no env modification, no NS
change.  It measures the one assumption the whole PACT-D upgrade rests on, and it
measures the proposed fix at the same time.

THE CLAIM UNDER TEST
--------------------
Upgrade the NS so the coupling operator is unknown inside a KNOWN basis:

    l(t+1) = rho*l(t) + (1-rho) * c(t) * W(theta) * Phi(u(t)),   W(theta) = sum_m theta_m B_m

Each B_m is known and zero-diagonal, so each waveform x_m is computable exactly from
shared peer actions.  The unknown collapses to the r-vector  beta* = c*theta  (T1).
Recovering it from the residual is a linear regression whose regressor is

    Psi(t) = [ B_1 Phi(u(t)) , ... , B_r Phi(u(t)) ]  in R^{n x r}

so identifiability is governed by  lambda_min( E[Psi^T Psi] ).

    T2:  on a COORDINATED policy the joint action process concentrates on a
         low-dimensional gait manifold, so the columns of Psi become collinear and
         lambda_min collapses.  *** The better the team coordinates, the less
         identifiable the coupling.  Cooperation destroys the information that
         cooperation needs. ***

    T3:  dither injected THROUGH the compensator restores excitation:
         lambda_min >= lambda_gait + eps^2 * lambda_min(Xi).
         With beta_hat = 0 (a B0 checkpoint compensates nothing) the dithered
         executed torque is exactly  u = clip(a - eps * sum_m xi_m(t) * x_m),
         which is what this script rolls out.  So the SAME script that confirms the
         obstruction also validates the fix -- before any training.

WHAT TO EXPECT (write these down before running -- pre-registered predictions)
------------------------------------------------------------------------------
  P1  lam_min_norm(B0, eps=0)  <<  lam_min_norm(random, eps=0).          [T2]
  P2  lam_min_norm falls monotonically along a training-checkpoint ladder. [T2]
  P3  lam_min_norm(B0, eps) rises ~ eps^2 and the fitted slope is O(1).   [T3]
  P4  eff_rank(Psi Gram) on B0 is < r; on random it is ~ r.               [T2]

If P1/P2 fail -- lambda_min is healthy on the trained gait -- then T2 is FALSE, the
coupling is passively identifiable, and PACT-D's dither has nothing to buy.  Find that
out now, for the price of one eval, not after re-running the ladder.

SCALE-FREENESS MATTERS.  A trained policy may simply act smaller than a random one,
which would shrink lambda_min for a trivial reason.  Every headline number here is
normalized:  lam_min_norm = r * lambda_min / trace(G)  in [0, 1], equal to 1 iff the
Gram is isotropic.  Read that, plus cond and eff_rank -- never raw lambda_min.

USAGE
-----
    # arithmetic certificate, no simulator, no checkpoint (run this first):
    python -m harl.envs.mamujoco.pact.excite --selftest

    # the real probe (server; mujoco + a trained B0):
    python -m harl.envs.mamujoco.pact.excite \
        --load_config results/.../B0/config.json \
        --model_dirs  results/.../B0/models \
        --labels      B0 \
        --random --episodes 20 --eps 0,0.02,0.05,0.1,0.2

    # the coordination LADDER (P2) -- several checkpoints from one run:
    python -m harl.envs.mamujoco.pact.excite --load_config .../config.json \
        --model_dirs .../models_1M,.../models_4M,.../models_10M \
        --labels 1M,4M,10M --random

Set ANT_PCR_SEVERITY before launching to choose whether the gait is measured with the
NS on or off (both are informative: off = pure gait rank, on = the realistic
operating condition).  The probe never reads any pcr_* info key.
"""

import argparse
import json
import os

import numpy as np

# ==========================================================================
#  Pure-numpy core -- no torch, no gym, no mujoco.  --selftest exercises all
#  of it, so the arithmetic can be certified on a machine with no simulator.
# ==========================================================================

RHO_DEFAULT = 0.8       # the env's structural leak (ant.py _RHO)


def ant_basis(n_legs=4, normalize=True):
    """The r=3 coupling basis for Ant 4x2, as explicit (n x n) matrices.

    Flat joint order is [hip0, ank0, hip1, ank1, ...], agent (leg) l owning
    indices (2l, 2l+1).  Every basis matrix is ZERO-DIAGONAL in the agent-block
    sense -- leg l's rows never read leg l's own entries -- which is what keeps
    the family category-C at every theta (N=1 => empty sum => l == 0).

      B_hh  hip  of leg l  <- hips   of legs != l
      B_aa  ankle of leg l <- ankles of legs != l
      B_x   hip  of leg l  <- ankles of legs != l,  and ankle <- hips  (cross-rail)

    The CURRENT env is exactly  W = B_hh + B_aa  (i.e. theta ~ (1,1,0)), so the
    upgrade strictly contains the baseline.

    Frobenius-normalized by default: the B_m have different sparsity (B_x has
    twice the nonzeros), and without normalization lambda_min would report that
    rather than the geometry.  This is a reparameterization of theta and does not
    change the span.
    """
    n = 2 * n_legs
    Bhh = np.zeros((n, n))
    Baa = np.zeros((n, n))
    Bx = np.zeros((n, n))
    for l in range(n_legs):
        for lp in range(n_legs):
            if l == lp:
                continue
            Bhh[2 * l, 2 * lp] = 1.0
            Baa[2 * l + 1, 2 * lp + 1] = 1.0
            Bx[2 * l, 2 * lp + 1] = 1.0
            Bx[2 * l + 1, 2 * lp] = 1.0
    B = [Bhh, Baa, Bx]
    if normalize:
        B = [b / np.linalg.norm(b) for b in B]
    return B


def check_zero_diagonal(B, n_legs=4):
    """Category-C certificate: no basis matrix lets an agent load its own bus."""
    for k, b in enumerate(B):
        for l in range(n_legs):
            blk = b[2 * l:2 * l + 2, 2 * l:2 * l + 2]
            if np.any(np.abs(blk) > 1e-12):
                raise AssertionError(
                    f"B[{k}] has a nonzero self-block at leg {l}: not category-C."
                )
    return True


def hadamard(n):
    """Sylvester Hadamard matrix (n a power of 2). Rows 1.. are the dither probes:
    zero-mean, mutually orthogonal, and self-known to each agent."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


class Dither:
    """Orthogonal probe sequence xi_m(t), m = 1..r.

    Row 0 of the Hadamard matrix is all-ones and is deliberately SKIPPED: a DC
    probe would bias the executed torque instead of exciting it.  Orthogonality
    over each length-H block is what lets agent i correlate its own residual
    against its own probe and read off one basis coefficient at a time.
    """

    def __init__(self, r):
        size = 1
        while size < r + 1:
            size *= 2
        self.H = hadamard(size)
        self.size = size
        self.r = r

    def __call__(self, t):
        return np.array([self.H[m + 1, t % self.size] for m in range(self.r)])

    def gram(self, T):
        X = np.array([self(t) for t in range(T)])
        return X.T @ X / max(1, T)


def psi(u, B):
    """Psi(t) = [B_1 u, ..., B_r u]  in R^{n x r} -- the identification regressor."""
    return np.stack([b @ u for b in B], axis=1)


def spectrum(G):
    """Scale-free readouts of the Gram matrix G = E[Psi^T Psi].

    lam_min_norm  r*lambda_min/trace in [0,1]; 1 iff isotropic.  THE headline.
    cond          lambda_max/lambda_min.
    eff_rank      exp(entropy of the normalized spectrum) -- how many directions
                  are actually excited.  < r means a genuinely degenerate regressor.
    """
    w = np.linalg.eigvalsh((G + G.T) / 2.0)
    w = np.clip(w, 0.0, None)
    tr = float(w.sum())
    r = len(w)
    lam_min = float(w[0])
    lam_max = float(w[-1])
    if tr <= 1e-30:
        return dict(lam_min=0.0, lam_max=0.0, lam_min_norm=0.0,
                    cond=float("inf"), eff_rank=0.0, eig=w.tolist())
    p = w / tr
    nz = p[p > 1e-15]
    return dict(
        lam_min=lam_min,
        lam_max=lam_max,
        lam_min_norm=r * lam_min / tr,
        cond=(lam_max / lam_min) if lam_min > 1e-30 else float("inf"),
        eff_rank=float(np.exp(-(nz * np.log(nz)).sum())),
        eig=w.tolist(),
    )


class GramAccumulator:
    """Streams  G = (1/T) sum_t Psi(t)^T Psi(t)  plus the action covariance."""

    def __init__(self, B, n):
        self.B = B
        self.r = len(B)
        self.G = np.zeros((self.r, self.r))
        self.C = np.zeros((n, n))
        self.T = 0

    def add(self, u):
        P = psi(u, self.B)
        self.G += P.T @ P
        self.C += np.outer(u, u)
        self.T += 1

    def result(self):
        T = max(1, self.T)
        out = spectrum(self.G / T)
        out["n_steps"] = self.T
        out["act_eff_rank"] = spectrum(self.C / T)["eff_rank"]
        out["act_rms"] = float(np.sqrt(np.trace(self.C / T) / self.C.shape[0]))
        return out


def dithered_execute(a_flat, x, B, dith, t, eps, beta_hat=None):
    """The executed torque under dithered compensation, exactly as PACT-D emits it:

        u = clip( a - sum_m (beta_hat_m + eps*xi_m(t)) * x_m ,  -1, +1 )

    With beta_hat = 0 (a B0 checkpoint compensates nothing) this reduces to the
    pure-excitation case  u = clip(a - eps * sum_m xi_m(t) * x_m)  -- which is why a
    frozen B0 checkpoint is enough to measure T3's excitation gain.  Returns the
    executed torque; the caller advances the x_m leak with it.
    """
    if beta_hat is None:
        beta_hat = np.zeros(len(B))
    g = np.asarray(beta_hat, dtype=np.float64) + eps * dith(t)
    return np.clip(a_flat - (g[:, None] * x).sum(axis=0), -1.0, 1.0)


def leak_step_multi(x, u, B, rho):
    """x_m <- rho*x_m + (1-rho)*B_m u, for every m at once.  x is (r, n)."""
    return rho * x + (1.0 - rho) * np.stack([b @ u for b in B], axis=0)


# ==========================================================================
#  Synthetic self-test -- certifies the arithmetic AND the two predictions,
#  with no simulator and no checkpoint.  Run this before touching the server.
# ==========================================================================

def _synth(kind, T, n, rng):
    """Two action processes at the SAME rms but with different geometry.

    'coordinated'  a fixed gait: every joint is a fixed function of ONE phase, so
                   the action process lives on a 1-D manifold -- the caricature of
                   what T2 says a trained cooperative policy does.
    'iid'          isotropic noise: full-rank excitation, the caricature of an
                   untrained policy.

    Matched rms is the whole point: if the two differed in magnitude, any gap in
    lambda_min would be trivial rather than geometric.
    """
    if kind == "iid":
        out = rng.normal(0, 0.35, size=(T, n))
    else:
        # one phase drives every joint through fixed per-joint offsets
        ph = np.linspace(0, 40 * np.pi, T)
        off = np.linspace(0, 2 * np.pi, n, endpoint=False)
        out = (0.35 * np.sqrt(2.0)) * np.sin(ph[:, None] + off[None, :])
    return np.clip(out, -1.0, 1.0)


def selftest():
    print("=" * 74)
    print("TIER-0 SELFTEST  (pure numpy: no mujoco, no torch, no checkpoint)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    B = ant_basis(4)
    n, r = 8, len(B)

    # --- 1. category-C certificate -------------------------------------------
    check_zero_diagonal(B)
    print("[1] basis is zero-diagonal (category-C preserved at every theta)   OK")

    # --- 2. the upgrade contains the baseline --------------------------------
    Braw = ant_basis(4, normalize=False)
    v = rng.normal(size=n)
    ref = np.empty(n)                       # ant.py's coupling_sum, inlined
    hip, ank = v[0::2], v[1::2]
    ref[0::2] = hip.sum() - hip
    ref[1::2] = ank.sum() - ank
    got = Braw[0] @ v + Braw[1] @ v
    assert np.allclose(ref, got), (ref, got)
    print("[2] B_hh + B_aa reproduces ant.py's coupling_sum exactly            OK")
    print("    => theta=(1,1,0) IS the current env; the upgrade is a superset.")

    # --- 3. dither probes are orthogonal and zero-mean ------------------------
    d = Dither(r)
    Xi = d.gram(4 * d.size)
    assert np.allclose(Xi, np.eye(r), atol=1e-12), Xi
    assert np.allclose(np.array([d(t) for t in range(d.size)]).sum(0), 0.0)
    print(f"[3] dither: {r} orthonormal zero-mean probes, period {d.size}       OK")

    # --- 4. T2: coordination collapses lambda_min ----------------------------
    T = 4000
    rows = []
    for kind in ("iid", "coordinated"):
        A = _synth(kind, T, n, rng)
        acc = GramAccumulator(B, n)
        x = np.zeros((r, n))
        for t in range(T):
            u = dithered_execute(A[t], x, B, d, t, 0.0)
            acc.add(u)
            x = leak_step_multi(x, u, B, RHO_DEFAULT)
        rows.append((kind, acc.result()))
    print("\n[4] T2 -- passive identifiability vs coordination")
    print(f"    {'process':<14}{'lam_min_norm':>14}{'cond':>12}{'eff_rank':>10}"
          f"{'act_rms':>10}")
    for k, s in rows:
        print(f"    {k:<14}{s['lam_min_norm']:>14.3e}{s['cond']:>12.3g}"
              f"{s['eff_rank']:>10.3f}{s['act_rms']:>10.3f}")
    iid_s, coo_s = rows[0][1], rows[1][1]
    ok2 = coo_s["lam_min_norm"] < 0.1 * iid_s["lam_min_norm"]
    print(f"    -> coordinated is {iid_s['lam_min_norm']/max(coo_s['lam_min_norm'],1e-300):.3g}x "
          f"less identifiable at matched action RMS   "
          f"{'OK (T2 mechanism reproduced)' if ok2 else 'UNEXPECTED'}")

    # --- 5. T3: dither restores excitation, ~eps^2 ---------------------------
    print("\n[5] T3 -- excitation recovered by dither (on the coordinated gait)")
    print(f"    {'eps':>8}{'lam_min_norm':>16}{'eff_rank':>10}{'|u-a| rms':>12}")
    A = _synth("coordinated", T, n, rng)
    eps_grid = [0.0, 0.02, 0.05, 0.1, 0.2]
    lm = []
    for eps in eps_grid:
        acc = GramAccumulator(B, n)
        x = np.zeros((r, n))
        dev = 0.0
        for t in range(T):
            u = dithered_execute(A[t], x, B, d, t, eps)
            dev += float(np.mean((u - A[t]) ** 2))
            acc.add(u)
            x = leak_step_multi(x, u, B, RHO_DEFAULT)
        s = acc.result()
        lm.append(s["lam_min_norm"])
        print(f"    {eps:>8.3f}{s['lam_min_norm']:>16.3e}{s['eff_rank']:>10.3f}"
              f"{np.sqrt(dev/T):>12.4f}")
    ok3 = lm[-1] > 10 * max(lm[0], 1e-300)
    print(f"    -> dither raises lam_min_norm by "
          f"{lm[-1]/max(lm[0],1e-300):.3g}x at eps={eps_grid[-1]}   "
          f"{'OK (T3 mechanism reproduced)' if ok3 else 'UNEXPECTED'}")
    if len(lm) >= 3 and lm[1] > 0:
        # eps^2 scaling check on the small-eps end
        p = np.polyfit(np.log(eps_grid[1:]), np.log(np.maximum(lm[1:], 1e-300)), 1)[0]
        print(f"    -> log-log slope of lam_min_norm vs eps = {p:.2f} (theory: 2.0)")

    print("\n" + "=" * 74)
    print("SELFTEST DONE.  The synthetic caricature reproduces T2 and T3.")
    print("This certifies the ARITHMETIC and the MECHANISM -- not the claim.")
    print("The claim is about the REAL trained gait: run the probe on B0.")
    print("=" * 74)


# ==========================================================================
#  Real probe -- rolls a frozen checkpoint and measures the same quantities.
# ==========================================================================

def build_env(cfg_path):
    from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_args = dict(cfg["env_args"])
    if env_args.pop("pact", False):
        print("[probe] NOTE: config has pact=true; the probe measures the BASE gait, "
              "so the PACT wrapper is disabled and the extra beta action dim is "
              "ignored. Point --load_config at a B0/blind run for the cleanest read.")
    for k in ("diag", "echor", "recon", "omax"):
        env_args.pop(k, None)
    return MujocoMulti(env_args=env_args), cfg


def build_actors(cfg, env, model_dir, device):
    import torch
    from harl.algorithms.actors import ALGO_REGISTRY

    algo = cfg["main_args"]["algo"]
    algo = "happo" if algo in ("pact", "omax", "oracle") else algo
    aa = cfg["algo_args"]
    merged = {**aa["model"], **aa["algo"]}
    actors = []
    for i in range(env.n_agents):
        ag = ALGO_REGISTRY[algo](
            merged, env.observation_space[i], env.action_space[i], device=device
        )
        sd = torch.load(os.path.join(model_dir, f"actor_agent{i}.pt"),
                        map_location=device)
        ag.actor.load_state_dict(sd)
        ag.prep_rollout()
        actors.append(ag)
    return actors, merged


def rollout(env, actors, B, eps, episodes, rho, hidden, rec_n, seed, max_steps):
    """Roll the (frozen) policy with dithered compensation and accumulate the Gram.

    actors=None  =>  uniform random actions (the full-excitation reference).
    """
    import torch
    from harl.utils.trans_tools import _t2n

    n_agents = env.n_agents
    dims = [sp.shape[0] for sp in env.true_action_space]
    n = int(sum(dims))
    off = np.cumsum([0] + dims)
    d = Dither(len(B))
    acc = GramAccumulator(B, n)
    rng = np.random.default_rng(seed)
    rets = []
    t_glob = 0

    for _ep in range(episodes):
        obs, _st, _av = env.reset()
        obs = np.array(obs, dtype=np.float32)
        rnn = np.zeros((n_agents, rec_n, hidden), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)
        x = np.zeros((len(B), n))          # per-basis waveform cache, resets per episode
        ret = 0.0
        for _t in range(max_steps):
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
            a_flat = np.clip(np.concatenate(a_list), -1.0, 1.0)

            u = dithered_execute(a_flat, x, B, d, t_glob, eps)
            acc.add(u)
            x = leak_step_multi(x, u, B, rho)
            t_glob += 1

            step_a = [u[off[i]:off[i + 1]] for i in range(n_agents)]
            obs_l, _s, rew, dones, _infos, _av = env.step(step_a)
            ret += float(np.asarray(rew).reshape(-1)[0])
            if bool(np.all(dones)):
                break
            obs = np.array(obs_l, dtype=np.float32)
        rets.append(ret)

    out = acc.result()
    out["return"] = float(np.mean(rets))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="pure-numpy certificate; no simulator or checkpoint needed")
    p.add_argument("--load_config", type=str, default=None)
    p.add_argument("--model_dirs", type=str, default="",
                   help="comma-separated checkpoint dirs (a training ladder tests P2)")
    p.add_argument("--labels", type=str, default="")
    p.add_argument("--random", action="store_true",
                   help="also measure a uniform-random policy (the T2 reference)")
    p.add_argument("--eps", type=str, default="0,0.02,0.05,0.1,0.2")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--rho", type=float, default=RHO_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="excite_probe.csv")
    args = p.parse_args()

    if args.selftest or args.load_config is None:
        selftest()
        if args.load_config is None and not args.selftest:
            print("\n(no --load_config given: ran the selftest only)")
        return

    import csv
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, cfg = build_env(args.load_config)
    _dims = [sp.shape[0] for sp in env.true_action_space]
    if set(_dims) != {2}:
        raise SystemExit(
            f"ant_basis() is the Ant 4x2 hip/ankle basis and needs 2 joints per agent; "
            f"this env has per-agent action dims {_dims}. Declare the basis for this "
            f"partition before probing it (spec §1, declaration #4)."
        )
    B = ant_basis(env.n_agents)
    check_zero_diagonal(B, env.n_agents)
    hidden = int(cfg["algo_args"]["model"]["hidden_sizes"][-1])
    rec_n = int(cfg["algo_args"]["model"]["recurrent_n"])
    eps_grid = [float(s) for s in args.eps.split(",") if s.strip() != ""]

    dirs = [s for s in args.model_dirs.split(",") if s.strip()]
    labels = [s for s in args.labels.split(",") if s.strip()]
    if len(labels) != len(dirs):
        labels = [os.path.basename(os.path.dirname(d.rstrip("/\\"))) or f"ckpt{i}"
                  for i, d in enumerate(dirs)]
    arms = list(zip(labels, dirs))
    if args.random:
        arms.append(("random", None))

    print("=" * 96)
    print(f"TIER-0 PROBE  severity={os.environ.get('ANT_PCR_SEVERITY', '(env default)')}  "
          f"rho={args.rho}  r={len(B)}  episodes={args.episodes}")
    print("Reading lam_min_norm (scale-free, in [0,1]).  T2 predicts it COLLAPSES as "
          "the policy coordinates;")
    print("T3 predicts dither restores it ~ eps^2.  Raw lam_min is reported only for "
          "completeness.")
    print("=" * 96)
    hdr = (f"{'arm':<14}{'eps':>7}{'lam_min_norm':>15}{'cond':>11}{'eff_rank':>10}"
           f"{'act_eff_rank':>14}{'act_rms':>9}{'return':>10}")
    print(hdr)
    print("-" * 96)

    rows = []
    for label, md in arms:
        actors = None if md is None else build_actors(cfg, env, md, device)[0]
        for eps in eps_grid:
            s = rollout(env, actors, B, eps, args.episodes, args.rho,
                        hidden, rec_n, args.seed, args.max_steps)
            print(f"{label:<14}{eps:>7.3f}{s['lam_min_norm']:>15.4e}{s['cond']:>11.4g}"
                  f"{s['eff_rank']:>10.3f}{s['act_eff_rank']:>14.3f}"
                  f"{s['act_rms']:>9.3f}{s['return']:>10.1f}")
            rows.append(dict(arm=label, eps=eps, **{k: v for k, v in s.items()
                                                    if k != "eig"}))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-" * 96)
    print(f"wrote {os.path.abspath(args.out)}")

    # ---- the verdict, stated against the pre-registered predictions ----------
    def at(arm, eps):
        for r_ in rows:
            if r_["arm"] == arm and abs(r_["eps"] - eps) < 1e-12:
                return r_
        return None

    print("\nVERDICT")
    base = [a for a, _ in arms if a != "random"]
    rnd = at("random", 0.0)
    if rnd is not None and base:
        b0 = at(base[-1], 0.0)
        ratio = rnd["lam_min_norm"] / max(b0["lam_min_norm"], 1e-300)
        print(f"  P1  random / {base[-1]} at eps=0 : {ratio:.3g}x")
        print("      " + ("CONFIRMED -- the trained gait is far less identifiable. T2 holds; "
                          "the dither is load-bearing and PACT-D is justified."
                          if ratio > 10 else
                          "NOT CONFIRMED -- the trained gait is comparably identifiable. "
                          "T2 is FALSE as stated: passive RLS may suffice and the dual-control "
                          "story needs rethinking BEFORE any retraining."))
    if len(base) >= 2:
        seq = [at(a, 0.0)["lam_min_norm"] for a in base]
        mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        print(f"  P2  ladder {base} : {['%.2e' % v for v in seq]}  "
              f"{'monotone decreasing -- CONFIRMED' if mono else 'NOT monotone'}")
    for a in base:
        e = [r_ for r_ in rows if r_["arm"] == a and r_["eps"] > 0]
        if len(e) >= 2:
            sl = np.polyfit(np.log([r_["eps"] for r_ in e]),
                            np.log([max(r_["lam_min_norm"], 1e-300) for r_ in e]), 1)[0]
            print(f"  P3  {a}: log-log slope of lam_min_norm vs eps = {sl:.2f} "
                  f"(theory 2.0)  {'CONFIRMED' if 1.4 < sl < 2.6 else 'off-prediction'}")
    for a in base:
        r_ = at(a, 0.0)
        print(f"  P4  {a}: eff_rank = {r_['eff_rank']:.3f} of r={len(B)}  "
              f"{'CONFIRMED (degenerate)' if r_['eff_rank'] < len(B) - 0.5 else 'full rank'}")


if __name__ == "__main__":
    main()
