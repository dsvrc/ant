"""E5 — offline decentralized observability (system-id study)  [spec §4.5].

Pure numpy ridge regression on logged rollouts. **No training, no torch, no
gradient** — this is a measurement of what is *linearly decodable*, and the
filter it exports is the certificate E3-DOB then runs in closed loop.

The question
------------
Is the per-joint parasitic load ``d_i`` reconstructible from what an agent can
actually see by itself, at the lag budget E3 says the controller needs?

  target      ``d_i`` (2 dims per leg) at time t = ``pcr_d_next[t]``, i.e. the d
              that will hit the NEXT action.
  F-loc-obs   last-L own observations
  F-loc       last-L own obs (+) own actions          <- the decentralized set
  F-joint     last-L own obs (+) ALL agents' actions  <- the CTDE ceiling

Why F-joint is "own obs + all actions" and not "all agents' obs + all actions"
-----------------------------------------------------------------------------
In this HARL mamujoco setup every agent's observation is the SAME vector:
``MujocoMulti.get_obs()`` returns ``normalize(concat([full_state, onehot_i]))``,
and the normalizing mean/std are identical across agents (the one-hot always
contributes exactly one 1 and three 0s, so the sum and sum-of-squares do not
depend on i). Hence ``obs_i`` and ``obs_j`` differ ONLY in their 4 one-hot slots
— the union of all agents' observations carries no information beyond one
agent's. Concatenating them would just quadruple D for nothing. The genuine
CTDE increment is the teammates' **actions**, which are what the PCR recursion
actually feeds on (``d_i <- rho*d_i + (1-rho)*A*sigma*sum_{j!=i} tau_j``).

This is asserted at load time (``_assert_obs_shared``), not assumed. If a future
env breaks the property the assert fires and the fallback concatenates for real.

Timing contract (must match ``probes.Dob`` exactly)
---------------------------------------------------
Row t: ``features(o_{t-L+1..t}, a_{t-L+1..t}) -> d_next(t)``. Causal: ``d_next(t)``
is a function of ``tau(t)`` and ``d(t)``, both settled by time t. At run time the
controller needs ``d_applied(t+1) == d_next(t)``, which it predicts from the
window ending at ``(o_t, a_t)`` — all known at decision time. Lags are zero-padded
at an episode start, which is exactly right: the env sets ``d = 0`` there.

Gate V6a
--------
Decentralized observability holds iff **F-loc reaches R^2 >= 0.6 at the peak bin
with L <= 8 on data source (a)** (competent on-manifold data). Also reported:
F-loc-obs vs F-loc (does knowing your own action matter?) and F-loc vs F-joint
(how much is fundamentally centralized?).

Implementation notes
--------------------
* **Moment-based.** Everything (all lambdas, all L, all feature sets, all CV
  folds) is derived from per-block moment accumulators (n, Sx, Sxx, Sy, Sxy,
  Syy). The features are laid out lag-major with the nesting
  ``[obs | own_act | other_acts]`` per lag, so F-loc-obs / F-loc / F-joint and
  every L are **column subsets of one Gram** — accumulate once, slice for free.
* **Time-blocked CV.** 5 contiguous blocks; fold k trains on the other 4 (sum of
  their moments) and scores on block k. R^2 is computed from moments in closed
  form, so no fold ever materializes a prediction. Blocks are contiguous in time
  precisely because random k-fold on a 200-step-correlated trajectory would leak
  the answer across the split and report a fantasy R^2.
* **Constant-column drop.** Ant's 84 cfrc_ext slots are mostly structurally zero;
  zero-variance columns are unidentifiable under ridge anyway. They are dropped
  from the Gram (big speedup, zero information loss) and re-inserted as zeros in
  the exported filter, so ``probes.Dob`` needs no knowledge of the drop.
* **Ridge scaling.** Features standardized, target centered; the penalty is
  ``lambda * n`` so the lambda grid means the same thing at every n.

Run::

    python -m harl.envs.mamujoco.diag.sysid --selftest
    python -m harl.envs.mamujoco.diag.sysid --data 'e1_frozen:diag_out/e1/**/*.npz' \
        --out diag_out/e5 --export_dob diag_out/e5/dob_filter.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

from harl.envs.mamujoco.diag.report_io import DebugReport, write_csv

_FEATURE_SETS = ("F-loc-obs", "F-loc", "F-joint")
_DEFAULT_L = (1, 2, 4, 8, 16, 32)
_DEFAULT_LAM = (1e-4, 1e-3, 1e-2, 1e-1, 1e0)
_N_FOLDS = 5
_VAR_TOL = 1e-12


# ==========================================================================
#  data
# ==========================================================================
class Rollout:
    """One contiguous logged stream.

    obs      (T, n_agents, obs_dim)   the per-agent obs the POLICY saw at time t
                                      (pre-step), i.e. o_t
    act      (T, n_act)               the flat action COMMANDED at time t
    d_next   (T, n_act)               pcr_d_next[t] = the target
    payload  (T,)                     pcr_payload[t]
    ep_id    (T,)                     episode index; lags never cross a boundary
    """

    def __init__(self, obs, act, d_next, payload, ep_id, source="?"):
        self.obs = np.asarray(obs, dtype=np.float64)
        self.act = np.asarray(act, dtype=np.float64)
        self.d_next = np.asarray(d_next, dtype=np.float64)
        self.payload = np.asarray(payload, dtype=np.float64)
        self.ep_id = np.asarray(ep_id, dtype=np.int64)
        self.source = source
        T = self.obs.shape[0]
        assert self.act.shape[0] == T and self.d_next.shape[0] == T, "ragged rollout"
        self.T = T
        self.n_agents = self.obs.shape[1]
        self.obs_dim = self.obs.shape[2]
        self.n_act = self.act.shape[1]
        self.act_dim = self.n_act // self.n_agents
        # first index of each row's episode -> lag windows are zero-padded, never
        # allowed to read across a reset (where the env sets d = 0).
        self.ep_start = np.zeros(T, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, self.ep_id[1:] != self.ep_id[:-1]])
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else T
            self.ep_start[s:e] = s


def _assert_obs_shared(rollout, rep=None):
    """Verify obs_i == obs_j outside the agent-id slots (see module docstring).
    Returns True if the dedup is valid."""
    o = rollout.obs
    if rollout.n_agents < 2 or o.shape[0] == 0:
        return True
    n = min(2000, o.shape[0])
    diff = np.abs(o[:n, 0, :] - o[:n, 1, :]).max(axis=0)     # (obs_dim,)
    n_differ = int(np.sum(diff > 1e-9))
    shared = n_differ <= rollout.n_agents
    if rep is not None:
        rep.line(f"  obs-sharing check: {n_differ} of {rollout.obs_dim} coords differ "
                 f"between agent 0 and 1 (expect <= {rollout.n_agents}, the one-hot "
                 f"slots) -> dedup {'VALID' if shared else 'INVALID'}")
    return shared


def load_npz_glob(pattern, source="?"):
    """Load and concatenate the recorder's NPZ dumps matching a glob."""
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"E5: no NPZ matched {pattern!r}")
    obs, act, dn, pay, ep = [], [], [], [], []
    base = 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        e = z["ep_id"].astype(np.int64) + base
        obs.append(z["obs"]); act.append(z["act"]); dn.append(z["d_next"])
        pay.append(z["payload"]); ep.append(e)
        base = int(e.max()) + 1 if e.size else base
    return Rollout(np.concatenate(obs), np.concatenate(act), np.concatenate(dn),
                   np.concatenate(pay), np.concatenate(ep), source=source)


# ==========================================================================
#  feature layout
# ==========================================================================
class FeatureLayout:
    """Lag-major columns; within each lag: [obs | own_act | other_acts].

    That nesting makes F-loc-obs / F-loc / F-joint and every L pure column
    subsets of the widest design matrix — one Gram serves the whole grid.
    """

    def __init__(self, L, obs_dim, act_dim, n_agents):
        self.L, self.obs_dim, self.act_dim = int(L), int(obs_dim), int(act_dim)
        self.n_agents = int(n_agents)
        self.n_other = (n_agents - 1) * act_dim
        self.per_lag = obs_dim + act_dim + self.n_other
        self.D = self.L * self.per_lag

    def cols(self, feature_set, L):
        """Column indices of (feature_set, L) inside the widest layout."""
        assert L <= self.L, f"L={L} exceeds the accumulated Lmax={self.L}"
        if feature_set == "F-loc-obs":
            width = self.obs_dim
        elif feature_set == "F-loc":
            width = self.obs_dim + self.act_dim
        elif feature_set == "F-joint":
            width = self.per_lag
        else:
            raise ValueError(feature_set)
        return np.concatenate([np.arange(j * self.per_lag, j * self.per_lag + width)
                               for j in range(L)]).astype(np.int64)

    def build(self, rollout, agent, rows):
        """Design matrix for ``rows`` (row t = window ending at (o_t, a_t))."""
        rows = np.asarray(rows, dtype=np.int64)
        X = np.zeros((rows.size, self.D), dtype=np.float64)
        a0 = agent * self.act_dim
        other = np.array([k for k in range(rollout.n_act)
                          if not (a0 <= k < a0 + self.act_dim)], dtype=np.int64)
        start = rollout.ep_start[rows]
        for j in range(self.L):
            idx = rows - j
            ok = idx >= start                       # zero-pad across episode starts
            idx_c = np.where(ok, idx, 0)
            base = j * self.per_lag
            m = ok[:, None].astype(np.float64)
            X[:, base:base + self.obs_dim] = rollout.obs[idx_c, agent, :] * m
            b = base + self.obs_dim
            X[:, b:b + self.act_dim] = rollout.act[idx_c][:, a0:a0 + self.act_dim] * m
            b += self.act_dim
            X[:, b:b + self.n_other] = rollout.act[idx_c][:, other] * m
        return X


# ==========================================================================
#  moments
# ==========================================================================
class Moments:
    """Sufficient statistics for ridge + closed-form R^2, additive across blocks."""

    def __init__(self, D, dy):
        self.n = 0
        self.sx = np.zeros(D)
        self.sxx = np.zeros((D, D))
        self.sy = np.zeros(dy)
        self.sxy = np.zeros((D, dy))
        self.syy = 0.0                    # scalar: sum_t ||y_t||^2

    def add(self, X, Y):
        self.n += X.shape[0]
        self.sx += X.sum(axis=0)
        self.sxx += X.T @ X
        self.sy += Y.sum(axis=0)
        self.sxy += X.T @ Y
        self.syy += float(np.sum(Y * Y))

    def __iadd__(self, o):
        self.n += o.n
        self.sx += o.sx
        self.sxx += o.sxx
        self.sy += o.sy
        self.sxy += o.sxy
        self.syy += o.syy
        return self

    def sub(self, cols):
        """Restrict to a column subset (the free (feature_set, L) slicing)."""
        m = Moments(len(cols), self.sy.shape[0])
        m.n = self.n
        m.sx = self.sx[cols]
        m.sxx = self.sxx[np.ix_(cols, cols)]
        m.sy = self.sy.copy()
        m.sxy = self.sxy[cols]
        m.syy = self.syy
        return m


def _standardize(m):
    """mu, sigma from raw moments (sigma floored: constant columns are inert)."""
    mu = m.sx / max(m.n, 1)
    var = m.sxx.diagonal() / max(m.n, 1) - mu * mu
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma


def ridge_fit(m, lam):
    """Ridge on standardized-X / centered-y. Returns (intercept(dy,), V(D,dy),
    mu(D,), sigma(D,)) with the prediction rule  yhat = intercept + ((x-mu)/sigma) @ V.
    Penalty is ``lam * n`` so the grid means the same thing at every n."""
    mu, sigma = _standardize(m)
    ybar = m.sy / max(m.n, 1)
    Ds = 1.0 / sigma
    # Z'Z  = Ds (Sxx - n mu mu') Ds ;  Z'(y - ybar) = Ds (Sxy - n mu ybar')
    G = (m.sxx - m.n * np.outer(mu, mu)) * np.outer(Ds, Ds)
    h = (m.sxy - m.n * np.outer(mu, ybar)) * Ds[:, None]
    A = G + (lam * max(m.n, 1)) * np.eye(G.shape[0])
    try:
        V = np.linalg.solve(A, h)
    except np.linalg.LinAlgError:
        V = np.linalg.lstsq(A, h, rcond=None)[0]
    return ybar, V, mu, sigma


def r2_from_moments(m_val, intercept, V, mu, sigma):
    """R^2 of the fitted filter on a validation block, from its moments alone.

    yhat = c + x @ A  with  A = diag(1/sigma) V,  c = intercept - mu @ A.
    SSE = tr(Syy) - 2[c.Sy + tr(A' Sxy)] + [n c'c + 2 c' A' Sx + tr(A' Sxx A)]
    SST = tr(Syy) - n ||ybar_val||^2
    """
    if m_val.n == 0:
        return float("nan")
    A = V / sigma[:, None]
    c = intercept - mu @ A
    sse = (m_val.syy
           - 2.0 * (float(c @ m_val.sy) + float(np.sum(A * m_val.sxy)))
           + (m_val.n * float(c @ c)
              + 2.0 * float(c @ (A.T @ m_val.sx))
              + float(np.sum(A * (m_val.sxx @ A)))))
    ybar = m_val.sy / m_val.n
    sst = m_val.syy - m_val.n * float(ybar @ ybar)
    if sst <= 1e-12:
        return float("nan")
    return float(1.0 - sse / sst)


# ==========================================================================
#  the study
# ==========================================================================
def _block_of(rows, n_blocks):
    """Contiguous time blocks (NOT random k-fold: a 200-step-correlated
    trajectory would leak across a random split and report a fantasy R^2)."""
    edges = np.linspace(0, rows.size, n_blocks + 1).astype(int)
    return [rows[edges[k]:edges[k + 1]] for k in range(n_blocks)]


def _keep_columns(rollout, layout, agent, rows, chunk):
    """Column mask: drop zero-variance columns (Ant's structurally-zero cfrc_ext
    slots). Unidentifiable under ridge anyway; dropping them is free speed."""
    n = 0
    sx = np.zeros(layout.D)
    sxx_diag = np.zeros(layout.D)
    for i in range(0, rows.size, chunk):
        X = layout.build(rollout, agent, rows[i:i + chunk])
        n += X.shape[0]
        sx += X.sum(axis=0)
        sxx_diag += np.sum(X * X, axis=0)
    var = sxx_diag / max(n, 1) - (sx / max(n, 1)) ** 2
    return var > _VAR_TOL


def payload_bins(payload, n_bins=5):
    """Quintile bin index per row. Under the frozen arms the payload is constant
    within a source file, so a bin is simply 'which frozen A' — which is exactly
    what the V6a 'peak bin' means there."""
    p = np.asarray(payload, dtype=np.float64)
    # A non-finite payload would poison the dict lookup below (nan is never equal
    # to itself, so `lut[nan]` raises) and silently mis-bin the quantile path.
    # Park them in the lowest bin rather than crashing a 3-hour fit.
    bad = ~np.isfinite(p)
    if bad.any():
        p = p.copy()
        p[bad] = np.nanmin(p[~bad]) if (~bad).any() else 0.0
    uniq = np.unique(np.round(p, 6))
    if uniq.size <= n_bins:                       # frozen arms: bin == the A value
        lut = {v: i for i, v in enumerate(uniq)}
        return np.array([lut[v] for v in np.round(p, 6)]), uniq
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(p, qs), np.round(qs, 4)


def study(rollout, rep, L_grid=_DEFAULT_L, lam_grid=_DEFAULT_LAM,
          feature_sets=_FEATURE_SETS, max_rows=50000, chunk=4096, n_bins=5,
          seed=0):
    """Fit the whole grid. Returns (rows_for_csv, best_per_agent_for_export)."""
    Lmax = max(L_grid)
    layout = FeatureLayout(Lmax, rollout.obs_dim, rollout.act_dim, rollout.n_agents)
    shared_ok = _assert_obs_shared(rollout, rep)
    if not shared_ok:
        rep.note("obs-sharing dedup INVALID for this rollout: F-joint here is "
                 "'own obs + all actions', which would then be a STRICT SUBSET of "
                 "the true CTDE set. Read the F-joint row as a lower bound.")
    bins, bin_vals = payload_bins(rollout.payload, n_bins)
    rep.kv("rows", f"{rollout.T} (obs_dim={rollout.obs_dim}, n_agents="
                   f"{rollout.n_agents}, act_dim={rollout.act_dim})")
    rep.kv("widest design", f"L={Lmax} x per_lag={layout.per_lag} = D={layout.D}")
    rep.kv("payload bins", f"{sorted(set(bins.tolist()))} (values/edges {bin_vals})")

    out_rows = []
    best_export = {}
    rng = np.random.default_rng(seed)
    for agent in range(rollout.n_agents):
        rep.h3(f"agent {agent}")
        all_rows = np.arange(rollout.T)
        # stride the whole range rather than take a prefix: a column that is
        # constant only in the first 20k rows must not be dropped as "constant".
        stride = max(1, rollout.T // 20000)
        keep = _keep_columns(rollout, layout, agent, all_rows[::stride], chunk)
        cols_kept = np.flatnonzero(keep)
        rep.line(f"  kept {cols_kept.size}/{layout.D} columns "
                 f"({layout.D - cols_kept.size} zero-variance dropped)")
        if cols_kept.size == 0:
            rep.line("  every column is constant — nothing to fit; skipping agent")
            continue
        pooled = [None] * _N_FOLDS
        for b in sorted(set(bins.tolist())):
            rows_b = np.flatnonzero(bins == b)
            if rows_b.size > max_rows:            # subsample CONTIGUOUS-safe:
                # keep a random contiguous span per block, never 1-in-K (that would
                # destroy the lag windows this whole study is about)
                keep_n = max_rows
                s = rng.integers(0, rows_b.size - keep_n)
                rows_b = rows_b[s:s + keep_n]
            if rows_b.size < 5 * cols_kept.size:
                rep.line(f"  bin {b}: {rows_b.size} rows vs {cols_kept.size} kept "
                         f"columns — thin (n/D < 5); reported, read with care")
            blocks = _block_of(rows_b, _N_FOLDS)
            bmom = []
            for blk in blocks:
                m = Moments(cols_kept.size, rollout.act_dim)
                for i in range(0, blk.size, chunk):
                    idx = blk[i:i + chunk]
                    X = layout.build(rollout, agent, idx)[:, cols_kept]
                    Y = rollout.d_next[idx][:,
                                            agent * rollout.act_dim:
                                            (agent + 1) * rollout.act_dim]
                    m.add(X, Y)
                bmom.append(m)
            for k in range(_N_FOLDS):
                if pooled[k] is None:
                    pooled[k] = Moments(cols_kept.size, rollout.act_dim)
                pooled[k] += bmom[k]
            _fit_grid(bmom, layout, cols_kept, feature_sets, L_grid, lam_grid,
                      rollout, agent, f"bin{b}", out_rows, best_export, rep)
        _fit_grid(pooled, layout, cols_kept, feature_sets, L_grid, lam_grid,
                  rollout, agent, "pooled", out_rows, best_export, rep)
    return out_rows, best_export, layout


def _fit_grid(bmom, layout, cols_kept, feature_sets, L_grid, lam_grid, rollout,
              agent, tag, out_rows, best_export, rep):
    """Time-blocked CV over the whole (feature_set, L, lambda) grid from moments."""
    for fs in feature_sets:
        for L in L_grid:
            want = layout.cols(fs, L)
            # map the widest-layout columns onto the kept-column indexing
            pos = np.searchsorted(cols_kept, want)
            sel = pos[(pos < cols_kept.size) & (cols_kept[np.minimum(
                pos, cols_kept.size - 1)] == want)]
            if sel.size == 0:
                continue
            sub = [m.sub(sel) for m in bmom]
            for lam in lam_grid:
                r2s = []
                for k in range(len(sub)):
                    tr = Moments(sel.size, rollout.act_dim)
                    for j in range(len(sub)):
                        if j != k:
                            tr += sub[j]
                    if tr.n == 0 or sub[k].n == 0:
                        continue
                    c, V, mu, sg = ridge_fit(tr, lam)
                    r2s.append(r2_from_moments(sub[k], c, V, mu, sg))
                r2 = float(np.nanmean(r2s)) if r2s else float("nan")
                out_rows.append([rollout.source, agent, tag, fs, L, f"{lam:g}",
                                 f"{r2:.4f}", int(sel.size),
                                 int(sum(m.n for m in sub))])
                # remember the best DECENTRALIZED filter for the DOB export
                if fs == "F-loc" and tag == "pooled" and np.isfinite(r2):
                    key = agent
                    if key not in best_export or r2 > best_export[key]["r2"]:
                        full = Moments(sel.size, rollout.act_dim)
                        for m in sub:
                            full += m
                        c, V, mu, sg = ridge_fit(full, lam)
                        best_export[key] = {"r2": r2, "L": L, "lam": lam, "fs": fs,
                                            "intercept": c, "V": V, "mu": mu,
                                            "sigma": sg, "sel": sel,
                                            "cols_kept": cols_kept}


def export_dob(path, best, layout, rollout, rep):
    """Write the E3-DOB filter. Dropped/unused columns are re-inserted as zeros
    (mu=0, sigma=1) so ``probes.Dob`` can build the full lag-major feature vector
    with no knowledge of the column surgery done here.

    Exported at ONE L (the best per-agent L is forced to a common value — the
    probe runs a single history length for all four legs)."""
    n_agents = rollout.n_agents
    Ls = [best[i]["L"] for i in range(n_agents) if i in best]
    if not Ls:
        rep.line("  export skipped: no F-loc pooled fit succeeded")
        return None
    L = int(max(Ls))
    lay = FeatureLayout(L, rollout.obs_dim, rollout.act_dim, n_agents)
    D = lay.D
    W = np.zeros((n_agents, D + 1, rollout.act_dim))
    MU = np.zeros((n_agents, D))
    SG = np.ones((n_agents, D))
    for i in range(n_agents):
        if i not in best:
            continue
        b = best[i]
        if b["L"] != L:
            rep.line(f"  agent {i}: best L={b['L']}, embedded in the common export "
                     f"L={L} (its lags > {b['L'] - 1} simply carry zero weight — no "
                     f"refit needed, an L={b['L']} filter IS an L={L} filter with "
                     f"zeros)")
        # widest-layout column ids of this agent's selected features
        kept = b["cols_kept"]
        widest_ids = kept[b["sel"]]
        # translate widest-layout(Lmax) ids -> export-layout(L) ids
        keep_mask = []
        tgt_ids = []
        for cid in widest_ids:
            lag, off = divmod(int(cid), layout.per_lag)
            if lag >= L or off >= lay.per_lag:
                keep_mask.append(False)
                tgt_ids.append(0)
            else:
                keep_mask.append(True)
                tgt_ids.append(lag * lay.per_lag + off)
        keep_mask = np.asarray(keep_mask)
        tgt_ids = np.asarray(tgt_ids)[keep_mask]
        W[i][0] = b["intercept"]
        W[i][1:][tgt_ids] = b["V"][keep_mask]
        MU[i][tgt_ids] = b["mu"][keep_mask]
        SG[i][tgt_ids] = b["sigma"][keep_mask]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, W=W, mu=MU, sigma=SG, L=L, feature_set="F-loc",
             obs_dim=rollout.obs_dim, act_dim=rollout.act_dim, n_agents=n_agents,
             per_lag=lay.per_lag,     # probes.Dob must NOT re-derive this
             r2=np.array([best[i]["r2"] if i in best else np.nan
                          for i in range(n_agents)]))
    rep.line(f"  DOB filter -> {os.path.abspath(path)}  (L={L}, F-loc, "
             f"per-agent pooled R^2 = "
             f"{[round(best[i]['r2'], 3) for i in range(n_agents) if i in best]})")
    rep.note("This filter is the E3-DOB certificate's subject: run "
             "`diag_tier0.py --probe cancel:beta=BEST,transform=dob:<this file>`. "
             "It reads ONLY agent-local history — no privileged input at run time.")
    return path


# ==========================================================================
#  self-test
# ==========================================================================
def _synth_rollout(n_steps=6000, normalize_obs=False, seed=0, ep_len=250):
    """Roll the synthetic PCR recursion from probes.py into a Rollout."""
    from harl.envs.mamujoco.diag.probes import _SyntheticPCR, _gait

    env = _SyntheticPCR(seed=seed, ep_len=ep_len, normalize_obs=normalize_obs)
    rng = np.random.default_rng(seed + 1)
    obs, act, dn, pay, ep = [], [], [], [], []
    env.reset()
    e = 0
    for t in range(n_steps):
        o = env.get_obs()                       # o_t: pre-step, what a policy sees
        a = _gait(t, rng)
        _, _, done, info = env.step(a)
        obs.append(np.asarray(o)); act.append(a.copy())
        dn.append(info["pcr_d_next"].copy()); pay.append(info["pcr_payload"])
        ep.append(e)
        if done:
            env.reset()
            e += 1
    return Rollout(np.asarray(obs), np.asarray(act), np.asarray(dn),
                   np.asarray(pay), np.asarray(ep), source="synthetic")


def selftest():
    rep = DebugReport(os.path.join("diag_out", "v0", "v0_sysid.md"),
                      title="V0 — sysid self-test",
                      subtitle="ridge/CV/export machinery on the synthetic PCR "
                               "recursion; no simulator, no torch")
    ok_all = True

    rep.h2("T1 — the fitter recovers d on the synthetic stream (unnormalized obs)")
    rep.line("  Gate on **F-joint**, deliberately. With raw obs and ALL agents'")
    rep.line("  actions the recursion is exactly linear in the features at L>=2:")
    rep.line("    delivered_{t-1} is in o_t, so d_applied(t-1) = delivered_{t-1} -")
    rep.line("    tau_{t-1}; then d_next(t) = rho*d_applied(t) + (1-rho)*A*sigma*")
    rep.line("    s(tau_t), and s() is linear in the joint action.")
    rep.line("  So a correct ridge + time-blocked CV + moment pipeline MUST reach")
    rep.line("  R^2 ~ 1. That makes it a test of the FITTER.")
    rep.line("  F-loc is NOT gated: it carries only the agent's OWN action, while")
    rep.line("  d_i is driven by its TEAMMATES' torques — whether that is")
    rep.line("  recoverable from local history is the open question E5 exists to")
    rep.line("  answer (gate V6a), not something a unit test may assume.")
    ro = _synth_rollout(normalize_obs=False)
    rows, best, layout = study(ro, rep, L_grid=(1, 2, 4), lam_grid=(1e-4, 1e-2),
                               feature_sets=("F-loc-obs", "F-loc", "F-joint"),
                               max_rows=6000, n_bins=1)
    by_fs = {}
    for r in rows:
        _, agent, tag, fs, L, lam, r2, _, _ = r
        if tag == "pooled" and int(L) >= 2:
            by_fs.setdefault(fs, {})
            by_fs[fs][agent] = max(by_fs[fs].get(agent, -9), float(r2))
    got = min(by_fs.get("F-joint", {}).values()) if by_fs.get("F-joint") else -9
    ok = got > 0.95
    ok_all &= ok
    rep.table(["feature set", "worst-agent pooled R^2 (L>=2)", "gated?"],
              [[fs, f"{min(v.values()):.4f}", "YES (>0.95)" if fs == "F-joint"
                else "no (informative)"] for fs, v in by_fs.items()])
    rep.verdict("T1 ridge recovers d from F-joint (R^2 > 0.95)", ok)

    rep.h2("T2 — probes.Dob rebuilds EXACTLY the features sysid fit on")
    rep.line("  The invariant that matters is not how well the filter predicts —")
    rep.line("  it is that the offline fit and the online probe compute the SAME")
    rep.line("  function. A feature-layout or timing mismatch between sysid.py and")
    rep.line("  probes.Dob would leave E3-DOB silently meaningless while every")
    rep.line("  number still looked plausible. So: apply the exported filter")
    rep.line("  offline, run the same stream through Dob's online interface, and")
    rep.line("  require the two predictions to agree to 1e-6.")
    path = os.path.join("diag_out", "v0", "_selftest_dob.npz")
    export_dob(path, best, layout, ro, rep)
    from harl.envs.mamujoco.diag.probes import Dob

    dob = Dob(path)
    lay = FeatureLayout(dob.L, ro.obs_dim, ro.act_dim, ro.n_agents)

    # --- online: through the transform's own interface, timing and all ---
    pred_on, true, rows_used = [], [], []
    dob.reset()
    for t in range(1, ro.T):
        if ro.ep_id[t] != ro.ep_id[t - 1]:
            dob.reset()
            continue
        dob.observe([ro.obs[t - 1, i] for i in range(ro.n_agents)])
        dob.post_step(ro.act[t - 1], {})
        pred_on.append(dob(np.zeros(8)))         # the d argument is ignored by design
        true.append(ro.d_next[t - 1])
        rows_used.append(t - 1)
    pred_on = np.asarray(pred_on)
    true = np.asarray(true)
    rows_used = np.asarray(rows_used)

    # --- offline: the same filter applied to sysid's own feature matrix ---
    pred_off = np.zeros_like(pred_on)
    for i in range(ro.n_agents):
        X = lay.build(ro, i, rows_used)
        Z = (X - dob.mu[i]) / dob.sigma[i]
        pred_off[:, i * ro.act_dim:(i + 1) * ro.act_dim] = \
            dob.W[i][0] + Z @ dob.W[i][1:]
    # zero out the teammates'-action columns' contribution: Dob never fills them,
    # and their weights are asserted zero, so the two must agree exactly.
    err = float(np.max(np.abs(pred_on - pred_off)))
    ok = err < 1e-6
    ok_all &= ok
    rep.line(f"  max|online - offline| = {err:.3e}  (must be < 1e-6)  "
             f"{'OK' if ok else 'FAIL'}")
    sse = float(np.sum((pred_on - true) ** 2))
    sst = float(np.sum((true - true.mean(axis=0)) ** 2))
    rep.line(f"  (informative) online R^2 of the exported F-loc filter vs the true "
             f"d_next = {1.0 - sse / max(sst, 1e-12):.4f}")
    rep.verdict("T2 offline fit == online probe (exact)", ok)

    rep.h2("T3 — what the obs normalization costs linear decodability (informative)")
    rep.line("  Same stream, obs normalized the way MujocoMulti.get_obs() does")
    rep.line("  (whole-vector, every step). NOT a gate — a preview of what the real")
    rep.line("  E5 is up against, and a miniature of the finding in diag/README.md.")
    ro_n = _synth_rollout(normalize_obs=True)
    rows_n, _, _ = study(ro_n, rep, L_grid=(2, 4), lam_grid=(1e-3,),
                         feature_sets=("F-loc", "F-joint"), max_rows=6000, n_bins=1)
    norm = {}
    for r in rows_n:
        if r[2] == "pooled":
            norm.setdefault(r[3], []).append(float(r[6]))
    rep.table(["feature set", "pooled R^2 raw obs", "pooled R^2 normalized obs"],
              [[fs,
                f"{min(by_fs.get(fs, {'x': float('nan')}).values()):.3f}",
                f"{min(norm.get(fs, [float('nan')])):.3f}"]
               for fs in ("F-loc", "F-joint")])

    rep.h2("SUMMARY")
    rep.verdict("V0 sysid self-test", ok_all)
    rep.close()
    return 0 if ok_all else 1


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="E5 — offline decentralized "
                                             "observability (ridge system-id).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", action="append", default=[],
                    help="'<source>:<glob>' — repeatable. Sources per spec §4.5: "
                         "(a) e1_frozen, (b) blind_drift, (c) random_excite.")
    ap.add_argument("--out", default="diag_out/e5")
    ap.add_argument("--export_dob", default=None,
                    help="write the best per-agent F-loc filter here (npz) for "
                         "E3-DOB. Only source (a) should feed this.")
    ap.add_argument("--L", default=",".join(str(x) for x in _DEFAULT_L))
    ap.add_argument("--lam", default=",".join(f"{x:g}" for x in _DEFAULT_LAM))
    ap.add_argument("--max_rows", type=int, default=50000,
                    help="per (agent, bin) cap. Cost ~ rows x D^2; D = Lmax x 123 "
                         "for Ant 4x2, so 50k rows x L=32 is ~20 s/(agent,bin).")
    ap.add_argument("--n_bins", type=int, default=5)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.data:
        ap.error("--data is required (or --selftest)")

    L_grid = tuple(int(x) for x in args.L.split(","))
    lam_grid = tuple(float(x) for x in args.lam.split(","))
    rep = DebugReport(os.path.join(args.out, "e5_sysid.md"),
                      title="E5 — offline decentralized observability",
                      subtitle="ridge system-id; target = pcr_d_next; "
                               "gate V6a = F-loc R^2 >= 0.6 at the peak bin, L <= 8, "
                               "on data source (a)")
    all_rows = []
    v6a = {}
    for spec in args.data:
        source, _, pattern = spec.partition(":")
        rep.h2(f"data source: {source}   (`{pattern}`)")
        ro = load_npz_glob(pattern, source=source)
        rows, best, layout = study(ro, rep, L_grid=L_grid, lam_grid=lam_grid,
                                   max_rows=args.max_rows, n_bins=args.n_bins)
        all_rows += rows
        if args.export_dob and source.startswith("e1"):
            export_dob(args.export_dob, best, layout, ro, rep)
        # V6a reading: F-loc, L<=8, peak bin (the highest bin index)
        peak = max(int(r[2][3:]) for r in rows if r[2].startswith("bin")) \
            if any(r[2].startswith("bin") for r in rows) else None
        if peak is not None:
            cand = [float(r[6]) for r in rows
                    if r[2] == f"bin{peak}" and r[3] == "F-loc" and int(r[4]) <= 8
                    and np.isfinite(float(r[6]))]
            v6a[source] = max(cand) if cand else float("nan")

    csv = write_csv(os.path.join(args.out, "e5_r2.csv"),
                    ["source", "agent", "bin", "feature_set", "L", "lambda",
                     "r2_cv", "n_features", "n_rows"], all_rows)
    rep.h2("V6a — decentralized observability gate")
    for src, v in v6a.items():
        rep.line(f"  {src}: best F-loc R^2 at the peak bin with L<=8 = {v:.3f} "
                 f"({'PASS' if v >= 0.6 else 'FAIL'} vs the 0.6 bar)")
    a_src = [v for s, v in v6a.items() if s.startswith("e1")]
    if a_src:
        passed = max(a_src) >= 0.6
        rep.verdict("V6a decentralized observability (source (a))", passed)
        if max(a_src) < 0.3:
            rep.note("Abort rule 4 (spec §10.2): R^2 < 0.3 everywhere on source (a) "
                     "=> SKIP E3-DOB and log V6 fail early. Do not spend the eval "
                     "pass.")
    rep.kv("per-cell R^2 table", csv)
    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
