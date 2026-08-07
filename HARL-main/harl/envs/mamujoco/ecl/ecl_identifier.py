"""ECL central identifier — v2 clip-aware known-structure fit (spec C1).

v1 used a two-coefficient ratio `b2/b1 = c`. The 10M run showed it never locked
(corr(c_now, payload) ≈ 0.12; scale ~10x under-read; inverted-U in payload). Root
cause (spec F5): the real channel is `delivered = clip(τ + d, −1, 1)` with
`d = c·x2`, and at trained gaits (a) clip-censoring zeroes ∂y/∂d on the rails
exactly where c must be read, and (b) near-sum-zero gaits fold the coupling into
an effective own-gain. A *linear* ratio cannot see through either.

v2 fits the **exact known nonlinearity** instead. For a candidate `c`, the
delivered effort is computable exactly from the stored joint actions:

    z_i(t; c) = clip( x1_i(t) + c · x2_i(t),  −1, 1 )

and the per-agent/per-channel model is `y = g·z(c) + noise` with closed-form g.
A 1-D grid search over c (joint SSE across agents × {hip, ankle} channels), with a
`lock_gain` accept gate, recovers c where the ratio failed — clipped samples
support the c that makes them clip, unclipped samples give the slope, and the
sum-zero collinearity is broken because z(c) is *nonlinear* in (x1, x2).

`ĉ_now` is used ONLY to tag transitions and steer the localizer/anchor — it is
never a network input (Prohibition 2). The legacy ratio estimator is kept as the
`c_now_ratio` diagnostic column and the T1 regression target.

--------------------------------------------------------------------------------
RECON extensions (RECON spec §2.1) — both default-OFF, ECL configs untouched
--------------------------------------------------------------------------------
``rho_grid``  list or None (default None). When set, the c-grid search is run
              inside an outer loop over candidate leak rates ρ̂, and the (ρ̂, ĉ)
              pair minimising the joint SSE wins. This drops the "ρ is known"
              assumption to "the *template* (L) is known" (RECON A1 / T4). Absent
              => the v2 behaviour exactly: one fit at the configured ``rho``.
``do_scan``   bool (default True). The readout-index scan reads
              ``buffer.obs``/``buffer.next_obs``; RECON's on-policy buffer view
              carries neither (it feeds ``raw_readout`` directly and runs its own
              scan on the fresh rollout), so it turns the scan off.
``nuisance_instant``
              bool (default False). Fit ``y ≈ g·clip(x1+c·x2) + h·S`` instead of
              ``y ≈ g·clip(x1+c·x2)``, where ``S = Σ_{j≠i} u_j`` at the SAME step.
              The Ant's legs are mechanically coupled through the torso, so the
              others' torques move this joint even at c=0; with nowhere to put
              that, the base model charges it to c (measured: ĉ≈0.54 at
              payload≈0, ECL DEBUG-4). The leaked and instantaneous couplings are
              separable by their ~70° of phase, so this recovers c alone. See
              ``_fit_unit``.

All three hooks are additive: with ``rho_grid=None, do_scan=True,
nuisance_instant=False`` every number this file produces is bit-identical to the
pre-RECON version. (``_leak`` was vectorised over the agent axis to keep the
ρ-grid's cost negligible; the recursion is elementwise, so the arithmetic is
unchanged. ``_channel_units`` now carries S as a 4th element, which the base path
ignores.)
"""

import numpy as np

# 1 payload cycle ≈ pcr_period (_P=40000 in ant.py) × ~20 rollout threads of
# collected transitions; used only to set the c_max_seen decay half-life (C1.3).
_CYCLE_TRANSITIONS = 8.0e5


class ECLIdentifier:
    def __init__(self, n_agents, cfg=None):
        cfg = dict(cfg or {})
        self.N = int(n_agents)
        self.W = int(cfg.get("W", 2000))                     # window in per-thread steps
        self.rho = float(cfg.get("rho", 0.8))                # env leak (structural)
        self.lam = float(cfg.get("lambda_reg", 1e-3))        # ridge (ratio path only)
        smooth_hl = float(cfg.get("smooth_windows", 2.0))    # smoothing half-life (windows)
        self.mu_c = 1.0 - 0.5 ** (1.0 / max(1.0, smooth_hl))

        # channels
        self.hip_action_idx = int(cfg.get("hip_action_idx", 0))
        self.ankle_action_idx = int(cfg.get("ankle_action_idx", 1))
        default_ro = [19 + 2 * i for i in range(self.N)]
        self.qvel_idx = np.asarray(cfg.get("readout_qvel_idx", default_ro), dtype=np.int64)
        default_ank = [self.qvel_idx[i] + 1 for i in range(self.N)]
        self.qvel_idx_ankle = np.asarray(
            cfg.get("readout_qvel_idx_ankle", default_ank), dtype=np.int64
        )
        self.use_ankle = bool(cfg.get("use_ankle_channel", True))

        # v2 clipfit knobs (C1)
        self.mode = str(cfg.get("identifier_mode", "clipfit"))   # "clipfit" | "ratio"
        self.grid_dc = float(cfg.get("grid_dc", 0.025))
        self.c_grid_max = float(cfg.get("c_grid_max", 1.2))
        self.lock_min_gain = float(cfg.get("lock_min_gain", 0.01))
        self.margin = int(cfg.get("x2_margin", 25))              # C1.1 warm-start margin
        self.c_clip = float(cfg.get("c_clip", 1.5))             # c physically ≤ 0.9

        # RECON extensions (§2.1) — see the module docstring. Both default-OFF.
        rg = cfg.get("rho_grid", None)
        self.rho_grid = [float(r) for r in rg] if rg else None
        self.rho_hat = self.rho          # the winning ρ̂ (== self.rho when no grid)
        self.do_scan = bool(cfg.get("do_scan", True))
        # fit the plant's own same-step inter-leg coupling as a nuisance term so
        # it is not charged to c (see _fit_unit). Default OFF => ECL unchanged.
        self.nuisance_instant = bool(cfg.get("nuisance_instant", False))
        self.h_med = 0.0

        # ratio-path guards (kept for the diagnostic column)
        self.b1_sig_k = float(cfg.get("b1_sig_k", 3.0))
        self.r2_min = float(cfg.get("r2_min", 0.02))

        # decaying c_max_seen (C1.3): half-life of cmax_halflife_cycles payload cycles
        cyc = float(cfg.get("cmax_halflife_cycles", 2))
        self.c_max_init = float(cfg.get("c_max_init", 0.1))
        self.cmax_lambda = 0.5 ** (self.W / max(1.0, cyc * _CYCLE_TRANSITIONS))

        self.c_now = 0.0
        self.c_max_seen = self.c_max_init

        # diagnostics (refreshed every window, even when held)
        self.cond_number = 0.0
        self.b1_med = 0.0
        self.b2_med = 0.0
        self.r2_med = 0.0
        self.corr_x1y = 0.0
        self.corr_x2y = 0.0
        self.scan_best_corr = 0.0
        self.scan_best_offset = 0        # hip index offset (0 ⇒ index right)
        self.scan_best_offset_ankle = 0  # ankle index offset (0 ⇒ index right)
        self.c_now_ratio = 0.0           # legacy estimator output (A/B diagnostic)
        self.lock_gain = 0.0             # clipfit fit quality (0 ⇒ unidentifiable)
        self.clip_frac = 0.0             # fraction on the rails at ĉ (observability meter)
        self.sumzero_frac = 0.0          # common-mode power fraction (F5.2 meter)
        self.n_valid = 0
        self.n_good = 0
        self.n_refresh = 0

    # ------------------------------------------------------------------ helpers
    def _leak_all(self, S, dones, rho):
        """x2 = leak_ρ(S) for every agent at once, reset at each episode start.

        S: (N, wsteps, nt), dones: (wsteps, nt) -> x2 with S's shape. x2(t) uses S
        up to t-1. Elementwise in the agent axis, so this is exactly ``_leak``
        applied per agent (vectorised only to keep the ρ-grid cheap)."""
        x2 = np.zeros_like(S)
        reset = dones > 0.5                                   # (wsteps, nt)
        for t in range(1, S.shape[1]):
            x2[:, t] = np.where(
                reset[t - 1][None, :], 0.0, rho * x2[:, t - 1] + (1.0 - rho) * S[:, t - 1]
            )
        return x2

    def _leak(self, S, dones):
        """Single-agent/single-channel ``_leak_all`` at the configured ρ. S, dones:
        (wsteps, nt) — returns same shape. Kept as the original entry point."""
        return self._leak_all(S[None], dones, self.rho)[0]

    def _prep(self, buffer):
        """Prep the LATEST window (W+margin per-thread steps). Returns dict or None."""
        cur = int(buffer.cur_size)
        nt = int(buffer.n_rollout_threads)
        wsteps = min(self.W + self.margin, cur // max(1, nt))
        if wsteps <= self.margin + 10:
            return None
        L = wsteps * nt
        bsz = buffer.buffer_size
        inds = (buffer.idx - L + np.arange(L)) % bsz
        return self._prep_from_inds(buffer, inds, wsteps, nt)

    def _prep_from_inds(self, buffer, inds, wsteps, nt):
        """Build per-agent/per-channel (x1, x2, y) and the kept-sample mask
        (non-terminal AND past the x2 warm-up margin, C1.1) for an explicit window."""
        dones = buffer.dones[inds, 0].reshape(wsteps, nt)

        hip = np.stack([buffer.actions[i][inds, self.hip_action_idx].reshape(wsteps, nt)
                        for i in range(self.N)])                     # (N, wsteps, nt)
        ank = np.stack([buffer.actions[i][inds, self.ankle_action_idx].reshape(wsteps, nt)
                        for i in range(self.N)])

        # coupling sums Σ_{j≠i} (per channel)
        S_hip = hip.sum(axis=0)[None] - hip
        S_ank = ank.sum(axis=0)[None] - ank
        x2_hip = self._leak_all(S_hip, dones, self.rho)
        x2_ank = self._leak_all(S_ank, dones, self.rho)

        # readouts: prefer the raw (un-normalized) stash; fall back to obs delta
        raw = getattr(buffer, "raw_readout", None)
        if raw is not None and getattr(raw, "ndim", 0) == 3:        # (bsz, N, 2)
            y_hip = np.stack([raw[inds, i, 0].reshape(wsteps, nt) for i in range(self.N)]).astype(np.float64)
            y_ank = np.stack([raw[inds, i, 1].reshape(wsteps, nt) for i in range(self.N)]).astype(np.float64)
        elif raw is not None:                                       # (bsz, N) v1
            y_hip = np.stack([raw[inds, i].reshape(wsteps, nt) for i in range(self.N)]).astype(np.float64)
            y_ank = None
        else:                                                       # obs-delta fallback (mock)
            qi, qa = self.qvel_idx, self.qvel_idx_ankle
            oh = np.stack([buffer.obs[i][inds, qi[i]].reshape(wsteps, nt) for i in range(self.N)])
            nh = np.stack([buffer.next_obs[i][inds, qi[i]].reshape(wsteps, nt) for i in range(self.N)])
            y_hip = (nh - oh).astype(np.float64)
            if int(np.max(qa)) < buffer.obs[0].shape[1]:
                oa = np.stack([buffer.obs[i][inds, qa[i]].reshape(wsteps, nt) for i in range(self.N)])
                na = np.stack([buffer.next_obs[i][inds, qa[i]].reshape(wsteps, nt) for i in range(self.N)])
                y_ank = (na - oa).astype(np.float64)
            else:
                y_ank = None

        # kept samples: non-terminal AND past the warm-up margin (C1.1 / F7a)
        step_idx = np.broadcast_to(np.arange(wsteps)[:, None], (wsteps, nt))
        keep = ((dones < 0.5) & (step_idx >= self.margin)).ravel()
        if keep.sum() < 200:
            keep = (step_idx >= self.margin).ravel()
        if keep.sum() < 100:
            keep = np.ones(wsteps * nt, dtype=bool)

        return {
            "inds": inds, "wsteps": wsteps, "nt": nt, "dones": dones, "keep": keep,
            "hip": hip, "ank": ank, "S_hip": S_hip, "S_ank": S_ank,
            "x2_hip": x2_hip, "x2_ank": x2_ank,
            "y_hip": y_hip, "y_ank": y_ank,
        }

    def _x2_for_rho(self, d, rho):
        """Recompute both channels' x2 at a candidate ρ (RECON ρ-grid, §2.1)."""
        return (self._leak_all(d["S_hip"], d["dones"], rho),
                self._leak_all(d["S_ank"], d["dones"], rho))

    def _channel_units_x2(self, d, x2_hip, x2_ank):
        """``_channel_units`` against an explicitly supplied x2 pair (ρ-grid)."""
        keep = d["keep"]
        units = []
        for i in range(self.N):
            units.append((d["hip"][i].ravel()[keep], x2_hip[i].ravel()[keep],
                          d["y_hip"][i].ravel()[keep], d["S_hip"][i].ravel()[keep]))
        if self.use_ankle and d["y_ank"] is not None:
            for i in range(self.N):
                units.append((d["ank"][i].ravel()[keep], x2_ank[i].ravel()[keep],
                              d["y_ank"][i].ravel()[keep], d["S_ank"][i].ravel()[keep]))
        return units

    def _channel_units(self, d):
        """Assemble the list of (x1, x2, y, S) raveled+kept arrays over all agents
        and all available channels (hip always; ankle if enabled and present).
        ``S`` is the *instantaneous* coupling Σ_{j≠i} u_j — the nuisance regressor;
        it is ignored unless ``nuisance_instant`` is set."""
        return self._channel_units_x2(d, d["x2_hip"], d["x2_ank"])

    # ------------------------------------------------------------------ clipfit
    def _clipfit(self, units):
        """C1: grid search over c minimising joint SSE of y ≈ g·clip(x1+c·x2, ±1).
        Returns (c_hat, lock_gain, clip_frac)."""
        return self._clipfit_sse(units)[:3]

    def _fit_unit(self, z, zz, yu, ss_u, a22_u, b2_u):
        """Least-squares residual of one channel unit at a candidate c.

        Base model (ECL):        y ≈ g·z            (z = clip(x1 + c·x2), demeaned)
        With ``nuisance_instant``: y ≈ g·z + h·S    (S = Σ_{j≠i} u_j, same step)

        Why the nuisance term: on a physically-coupled plant the *other legs'
        torques move this joint through the torso at the SAME step, even at c=0.
        The base model has nowhere to put that, so it charges it to c — which is
        why this estimator was measured reading c≈0.54 at payload≈0 (ECL DEBUG-4)
        and railing at the grid ceiling. The two effects are separable because
        they live on different timescales: the parasitic liability is the LEAKED
        sum of the others' PAST actions (ρ=0.8 ⇒ ~4-step lag ⇒ ~70° of phase at a
        trotting gait), the torso coupling is instantaneous. Fitting both lets c
        keep only the leaky part — which is the part (L) actually describes.

        ``zz`` is the caller's already-computed ``z @ z`` (> 1e-12, guarded there).
        The base path takes NO ridge, so it stays bit-identical to pre-RECON ECL;
        the ridge only conditions the 2x2 solve.

        Returns (residual, g, h)."""
        b1 = float(z @ yu)
        if ss_u is None:
            g = b1 / zz
            return yu - g * z, g, 0.0
        a11 = zz + self.lam
        a12 = float(z @ ss_u)
        det = a11 * a22_u - a12 * a12
        if abs(det) < 1e-12:                      # collinear: drop the nuisance
            g = b1 / zz
            return yu - g * z, g, 0.0
        g = (a22_u * b1 - a12 * b2_u) / det
        h = (-a12 * b1 + a11 * b2_u) / det
        return yu - g * z - h * ss_u, g, h

    def _clipfit_sse(self, units):
        """``_clipfit`` + the winning joint SSE, which the RECON ρ-grid compares
        across candidate ρ̂. Returns (c_hat, lock_gain, clip_frac, sse_min).

        The SSE is comparable across ρ: the residuals are against the same y's,
        and the c=0 reference (z = clip(x1)) does not involve ρ at all.

        NOTE on lock_gain under ``nuisance_instant``: the c=0 null becomes
        "own torque + instantaneous coupling", i.e. a strictly stronger null, so
        lock_gain now measures the PCR-specific (leaky) signal alone and reads
        lower than before. That is the number the gate should have been reading.
        """
        grid = np.arange(0.0, self.c_grid_max + 1e-9, self.grid_dc)
        ys = [(u[2] - u[2].mean()) for u in units]
        # the nuisance regressor depends on neither c nor ρ -> hoist it
        if self.nuisance_instant:
            ss = [(u[3] - u[3].mean()) for u in units]
            a22 = [float(s @ s) + self.lam for s in ss]
            b2 = [float(s @ y) for s, y in zip(ss, ys)]
        else:
            ss = [None] * len(units)
            a22 = [0.0] * len(units)
            b2 = [0.0] * len(units)
        sse = np.zeros(grid.size)
        for gi, c in enumerate(grid):
            tot = 0.0
            for ui, ((x1u, x2u, _, _), yu) in enumerate(zip(units, ys)):
                z = np.clip(x1u + c * x2u, -1.0, 1.0)
                z = z - z.mean()
                zz = float(z @ z)
                if zz < 1e-12:
                    tot += float(yu @ yu)
                    continue
                r, _, _ = self._fit_unit(z, zz, yu, ss[ui], a22[ui], b2[ui])
                tot += float(r @ r)
            sse[gi] = tot

        k = int(np.argmin(sse))
        c_hat = float(grid[k])
        # parabolic refinement on (k-1, k, k+1)
        if 0 < k < grid.size - 1:
            s0, s1, s2 = sse[k - 1], sse[k], sse[k + 1]
            denom = s0 - 2.0 * s1 + s2
            if abs(denom) > 1e-12:
                delta = 0.5 * (s0 - s2) / denom
                c_hat = float(grid[k] + np.clip(delta, -1.0, 1.0) * self.grid_dc)
        c_hat = float(np.clip(c_hat, 0.0, self.c_clip))
        sse0 = float(sse[0]) if grid[0] == 0.0 else float(sse.max())
        lock_gain = 1.0 - sse[k] / max(sse0, 1e-12)

        # clip_frac: fraction on the rails at ĉ, over HIP units, median over agents
        cfracs = []
        for (x1u, x2u, _, _) in units[: self.N]:
            z = x1u + c_hat * x2u
            cfracs.append(float(np.mean(np.abs(z) > 0.98)))
        clip_frac = float(np.median(cfracs)) if cfracs else 0.0

        # h_med: the fitted instantaneous-coupling gain at ĉ. This is the meter
        # for whether the physical (torso) coupling is real and how much of the
        # old ĉ was actually it. |h| ≈ 0 ⇒ the nuisance term was unnecessary.
        if self.nuisance_instant:
            hs = []
            for ui, ((x1u, x2u, _, _), yu) in enumerate(zip(units, ys)):
                z = np.clip(x1u + c_hat * x2u, -1.0, 1.0)
                z = z - z.mean()
                zz = float(z @ z)
                if zz < 1e-12:
                    continue
                _, _, h = self._fit_unit(z, zz, yu, ss[ui], a22[ui], b2[ui])
                hs.append(h)
            self.h_med = float(np.median(hs)) if hs else 0.0
        return c_hat, float(lock_gain), clip_frac, float(sse[k])

    # ------------------------------------------------------------------ ratio (legacy)
    def _ratio(self, d):
        """v1 two-coefficient ratio on the HIP channel — kept as a diagnostic.
        Populates b1/b2/r2/corr diagnostics and returns (c_ratio, n_valid)."""
        keep = d["keep"]
        c_i, conds, b1s, b2s, r2s, cx1y, cx2y = [], [], [], [], [], [], []
        for i in range(self.N):
            x1i = d["hip"][i].ravel()[keep]; x1i = x1i - x1i.mean()
            x2i = d["x2_hip"][i].ravel()[keep]; x2i = x2i - x2i.mean()
            yi = d["y_hip"][i].ravel()[keep]; yi = yi - yi.mean()
            a11 = float(x1i @ x1i) + self.lam
            a22 = float(x2i @ x2i) + self.lam
            a12 = float(x1i @ x2i)
            b1v = float(x1i @ yi); b2v = float(x2i @ yi)
            det = a11 * a22 - a12 * a12
            if abs(det) < 1e-12:
                continue
            b1 = (a22 * b1v - a12 * b2v) / det
            b2 = (-a12 * b1v + a11 * b2v) / det
            sse = float(((yi - b1 * x1i - b2 * x2i) ** 2).sum())
            sst = float(yi @ yi) + 1e-12
            r2 = 1.0 - sse / sst
            dof = max(1, x1i.shape[0] - 2)
            se_b1 = np.sqrt(max(0.0, sse / dof) * a22 / det)
            tr = a11 + a22
            disc = max(0.0, tr * tr - 4.0 * det)
            l1 = 0.5 * (tr + np.sqrt(disc)); l2 = 0.5 * (tr - np.sqrt(disc))
            conds.append(l1 / max(abs(l2), 1e-12))
            nx1 = np.sqrt(float(x1i @ x1i)) + 1e-12
            nx2 = np.sqrt(float(x2i @ x2i)) + 1e-12
            ny = np.sqrt(float(yi @ yi)) + 1e-12
            cx1y.append(b1v / (nx1 * ny)); cx2y.append(b2v / (nx2 * ny))
            b1s.append(b1); b2s.append(b2); r2s.append(r2)
            if abs(b1) > self.b1_sig_k * se_b1 and r2 > self.r2_min:
                c_i.append(b2 / b1)
        self.cond_number = float(np.median(conds)) if conds else 0.0
        self.b1_med = float(np.median(b1s)) if b1s else 0.0
        self.b2_med = float(np.median(b2s)) if b2s else 0.0
        self.r2_med = float(np.median(r2s)) if r2s else 0.0
        self.corr_x1y = float(np.median(cx1y)) if cx1y else 0.0
        self.corr_x2y = float(np.median(cx2y)) if cx2y else 0.0
        if not c_i:
            return 0.0, 0
        return float(np.clip(np.median(c_i), 0.0, self.c_clip)), len(c_i)

    # ------------------------------------------------------------------ estimate
    def _estimate(self, buffer, d):
        """Run both estimators on prepped data d. Returns (c_est, accept)."""
        # scan (index verification, hip + ankle) and common-mode meter
        if self.do_scan:
            self._scan_from_buffer(buffer, d)
        keep = d["keep"]
        tot_hip = d["hip"].sum(axis=0).ravel()[keep]
        num = float(np.var(tot_hip))
        den = float(sum(np.var(d["hip"][i].ravel()[keep]) for i in range(self.N))) + 1e-12
        self.sumzero_frac = num / den

        # ratio (diagnostic)
        c_ratio, n_valid_ratio = self._ratio(d)
        self.c_now_ratio = c_ratio

        # clipfit (primary). RECON §2.1 ext 1: when a ρ-grid is configured, the
        # c-grid runs inside an outer loop over ρ̂ and the joint-SSE winner takes
        # both parameters — the template, not ρ, is what the learner must know.
        if self.rho_grid:
            best = None
            for r in self.rho_grid:
                x2h, x2a = self._x2_for_rho(d, r)
                cand = self._clipfit_sse(self._channel_units_x2(d, x2h, x2a))
                if best is None or cand[3] < best[0][3]:
                    best = (cand, r)
            (c_clip, lock_gain, clip_frac, _), self.rho_hat = best
        else:
            self.rho_hat = self.rho
            c_clip, lock_gain, clip_frac = self._clipfit(self._channel_units(d))
        self.lock_gain = lock_gain
        self.clip_frac = clip_frac

        if self.mode == "ratio":
            self.n_valid = n_valid_ratio
            return c_ratio, (n_valid_ratio > 0)
        self.n_valid = n_valid_ratio
        return c_clip, (lock_gain > self.lock_min_gain)

    def refresh(self, buffer):
        """Recompute ĉ_now from the buffer's most recent window. Returns ĉ_now."""
        d = self._prep(buffer)
        if d is None:
            return self.c_now
        c_est, accept = self._estimate(buffer, d)
        self.n_refresh += 1

        # decaying c_max_seen (applied every refresh, floored at c_max_init) — C1.3
        if accept:
            self.c_now = (1.0 - self.mu_c) * self.c_now + self.mu_c * float(c_est)
            self.n_good += 1
        # (if not accepted, HOLD c_now — same semantics as v1's guard)
        self.c_max_seen = max(self.c_now, self.c_max_seen * self.cmax_lambda, self.c_max_init)
        return self.c_now

    def fit_window(self, buffer, start, wsteps_read):
        """Clipfit one explicit window of per-thread blocks starting at slot
        ``start`` (length ``wsteps_read`` per-thread steps). Returns
        (c_hat, lock_gain). Used by the runner's retag sweep (C3.1); no smoothing,
        no scan, no state mutation."""
        nt = int(buffer.n_rollout_threads)
        if wsteps_read <= self.margin + 10:
            return 0.0, 0.0
        L = wsteps_read * nt
        bsz = buffer.buffer_size
        inds = (start + np.arange(L)) % bsz
        d = self._prep_from_inds(buffer, inds, wsteps_read, nt)
        if d is None:
            return 0.0, 0.0
        c_hat, lock_gain, _ = self._clipfit(self._channel_units(d))
        return float(c_hat), float(lock_gain)

    # ------------------------------------------------------------------ scan
    def _scan_offset(self, buffer, inds, torque):
        """Modal offset (over agents) between the configured index and the obs
        coordinate whose Δ best correlates with `torque` (per-agent (N,wsteps,nt))."""
        obs_dim = int(buffer.obs[0].shape[1])
        kmax = max(1, obs_dim - 1)
        best_idx = np.zeros(self.N, dtype=np.int64)
        best_cor = np.zeros(self.N)
        for i in range(self.N):
            x1 = torque[i].ravel(); x1 = x1 - x1.mean()
            nx1 = np.sqrt(float(x1 @ x1)) + 1e-12
            o = buffer.obs[i][inds, :kmax].astype(np.float64)
            n = buffer.next_obs[i][inds, :kmax].astype(np.float64)
            dd = n - o
            dd = dd - dd.mean(axis=0, keepdims=True)
            den = np.sqrt((dd * dd).sum(axis=0)) * nx1 + 1e-12
            cor = np.abs((x1 @ dd) / den)
            k = int(np.argmax(cor))
            best_idx[i] = k; best_cor[i] = cor[k]
        return best_idx, best_cor

    def _scan_from_buffer(self, buffer, d):
        b_hip, c_hip = self._scan_offset(buffer, d["inds"], d["hip"])
        off_hip = b_hip - self.qvel_idx[: self.N]
        vals, cnts = np.unique(off_hip, return_counts=True)
        self.scan_best_offset = int(vals[int(np.argmax(cnts))])
        self.scan_best_corr = float(np.median(c_hip))
        if self.use_ankle:
            b_ank, _ = self._scan_offset(buffer, d["inds"], d["ank"])
            off_ank = b_ank - self.qvel_idx_ankle[: self.N]
            vals2, cnts2 = np.unique(off_ank, return_counts=True)
            self.scan_best_offset_ankle = int(vals2[int(np.argmax(cnts2))])

    def diagnostics(self):
        return {
            "c_now": self.c_now,
            "rho_hat": self.rho_hat,
            "h_med": self.h_med,
            "c_max_seen": self.c_max_seen,
            "cond_number": self.cond_number,
            "b1_med": self.b1_med,
            "b2_med": self.b2_med,
            "r2_med": self.r2_med,
            "corr_x1y": self.corr_x1y,
            "corr_x2y": self.corr_x2y,
            "scan_best_corr": self.scan_best_corr,
            "scan_best_offset": self.scan_best_offset,
            "scan_best_offset_ankle": self.scan_best_offset_ankle,
            "c_now_ratio": self.c_now_ratio,
            "lock_gain": self.lock_gain,
            "clip_frac": self.clip_frac,
            "sumzero_frac": self.sumzero_frac,
            "n_valid": self.n_valid,
            "n_good": self.n_good,
            "n_refresh": self.n_refresh,
        }
