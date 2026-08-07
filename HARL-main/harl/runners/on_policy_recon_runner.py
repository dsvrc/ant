"""Runner for RECON on the on-policy (HAPPO) host — the reference arm.

RECON is a plug-in: it touches four host call-sites and nothing else (spec §3.1).

  (1) obs augmentation before space sizing   -> ``ReconMujocoMulti`` declares +k
                                                 / +k·N; this runner writes ℓ̂ into
                                                 those slots (``warmup``/``insert``)
  (2) an action-interface wrapper            -> ``_compensate`` ([CP]) in ``run``
  (3) one supervised training call/iteration -> ``_recon_update`` ([ID]→[RE]→[DI])
  (4) rollout-buffer read access             -> ``_recon_update`` reads obs/masks

``train()`` calls ``_recon_update()`` and then HAPPO's ``train()`` **unchanged**.
The filter has its own Adam; there is no joint objective and no trade-off
coefficient (Prohibition 4). Host hyper-parameters are HAPPO's, verbatim.

Why the filter lives here and not in the env shim: with n_rollout_threads > 1 the
envs are separate processes, so a torch module inside the shim would never see a
gradient. Placement does not touch the theory — the filter's input tensor is
strictly ``(o_i, u_i(t−1))`` per agent, [CP] reads only ``ℓ̂_i``, and the batch
axis never mixes agents; N filters batched in one process is arithmetically N
decentralized filters (spec §2.3/§2.6).

Information hygiene (spec §2.6), enforced by construction:
  * ``ĉ``, ``ρ̂``, ``ℓ̃`` are trainer-side only and never enter a network input.
  * ``ℓ̂`` (local filter output) is the policy input, the [CP] gain's operand,
    and is logged.
  * the env's ``pcr_*`` telemetry is read **only** to write diagnostic columns
    (``*_true_*``, ``c_true``) — never fed back — except in the two labeled arms
    ``filter_oracle`` (ℓ̂ := true d) and ``ANT_PCR_ORACLE`` (env-side obs), both
    of which print an unmissable ``[ORACLE ARM]`` banner.

Debug trace (the point of which is to localize a failure, not to prove success):
``recon_debug.csv`` splits the chain into independently falsifiable links —
  ĉ locked?            -> c_hat / c_true / c_corr / lock_gain / rho_hat
  labels right?        -> label_r2 / label_err     ([ID]+[RE], vs true d)
  filter fits labels?  -> filter_mse / filter_mse_first          ([DI])
  filter right?        -> filter_true_r2 / filter_true_err       ([F], = T3's ε)
  compensation biting? -> sat_frac / u_minus_a / beta_*          ([CP], = A2/T5)
so a failed run says *which* of T1–T5's premises broke. ``recon_debug.log`` is
the same story in prose, with the gate warnings called out.
"""

import csv
import os

import numpy as np
import torch

from harl.envs.mamujoco.ecl.ecl_identifier import ECLIdentifier
from harl.envs.mamujoco.recon.filter import ReconFilter
from harl.envs.mamujoco.recon.relabel import (
    ReconIdWindow,
    calibrate_gain,
    disturbance_target,
    nominal_gain,
    readout_corr_profile,
    relabel,
    scan_readout_offset,
)
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n

_CSV_COLUMNS = [
    # --- where we are ---
    "env_step", "iter",
    # --- [ID]: is the driver identified? (G2: lock corr > 0.95) ---
    "c_hat", "c_floor", "c_pcr", "g0_mean", "rho_hat", "h_med", "lock_gain",
    "clip_frac", "sumzero_frac", "c_now_ratio", "c_true", "c_err", "c_corr",
    "payload", "locked",
    # --- [RE]: are the manufactured labels the real liability? ---
    "label_r2", "label_err", "label_rms",
    # --- [DI]/[F]: does the filter learn them, and is it right about reality? ---
    "filter_mse", "filter_mse_first", "filter_grad_norm",
    "filter_true_r2", "filter_true_err", "lhat_rms", "ltrue_rms",
    # --- [CP]: is compensation biting, and is authority (A2) intact? ---
    "beta_hip", "beta_ank", "u_minus_a", "sat_frac",
    # --- host / sanity ---
    "reward_mean", "scan_best_offset", "scan_best_corr", "scan_corr_at_idx",
    "n_label_valid",
]

_EVAL_CSV_COLUMNS = ["env_step", "thread", "payload_end", "ep_return", "ep_len"]


class OnPolicyReconRunner(OnPolicyHARunner):
    """HAPPO + RECON ([ID] -> [RE] -> [DI] -> [CE] -> [CP])."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyReconRunner, self).__init__(args, algo_args, env_args)

        cfg = dict(algo_args["algo"].get("recon", {}))
        self.recon_cfg = cfg
        self.compensate = bool(cfg.get("compensate", True))
        self.filter_oracle = bool(cfg.get("filter_oracle", False))
        # how the filter's TARGET is produced:
        #   self_supervised — d̃ = (y − ĝ0·a)/ĝ0 from the OWN readout (default). No
        #     central c; not anti-observable; vanishes at the trough by construction.
        #   central — the original ĉ·x2 hindsight label (anti-observable on the
        #     trained gait, kept for the ablation / the paper's negative result).
        self.label_mode = str(cfg.get("label_mode", "self_supervised"))
        self.debug_every = int(cfg.get("debug_every", 1))
        self.scan_every = int(cfg.get("scan_every", 50))

        if self.algo_args["render"]["use_render"]:
            return

        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self.N = self.num_agents

        # Without the shim the dimension arithmetic below still "passes" while
        # silently slicing a raw obs and calling the last 2 coordinates ℓ̂ — so
        # check the flag itself, not just the arithmetic.
        assert bool(self.env_args.get("recon", False)), (
            "RECON requires env_args.recon: true, which selects ReconMujocoMulti "
            "— the shim that declares the augmented obs/share spaces BEFORE HARL "
            "sizes the actor/critic/buffers, and that stashes the raw readout "
            "[ID] regresses on. Use tuned_configs/mamujoco/Ant-v2-4x2/recon/."
        )

        # ---- dims. The env declared obs = raw + k and share = raw + k·N; it
        #      returns the raw vectors and this runner writes the ℓ̂ slots.
        self.k = int(self.envs.action_space[0].shape[0])
        self.obs_dim = int(self.envs.observation_space[0].shape[0])
        self.share_dim = int(self.envs.share_observation_space[0].shape[0])
        self.raw_obs_dim = self.obs_dim - self.k
        self.raw_share_dim = self.share_dim - self.k * self.N
        assert self.raw_obs_dim > 0 and self.raw_share_dim > 0, (
            "RECON: the env must declare the augmented spaces (env_args.recon: true "
            "-> ReconMujocoMulti). Got obs_dim=%d, share_dim=%d, k=%d, N=%d."
            % (self.obs_dim, self.share_dim, self.k, self.N)
        )
        self.act_low = np.asarray(self.envs.action_space[0].low, dtype=np.float64)
        self.act_high = np.asarray(self.envs.action_space[0].high, dtype=np.float64)

        # ---- [F]+[DI] ------------------------------------------------------
        fcfg = dict(cfg.get("filter", {}))
        fcfg.setdefault("beta_mode", cfg.get("beta_mode", "fixed"))
        fcfg.setdefault("beta_fixed", cfg.get("beta_fixed", 1.0))
        self.recon_filter = ReconFilter(
            self.raw_obs_dim, self.k, self.k, fcfg, self.device
        )
        self.beta = self.recon_filter.beta()

        # ---- [ID] ----------------------------------------------------------
        icfg = dict(cfg.get("identifier", {}))
        icfg.setdefault("identifier_mode", icfg.pop("mode", "clipfit"))
        # rho is the benchmark's KNOWN structural constant (0.8). The grid was
        # observed to mis-select 0.6 and, with the nuisance term, to bias itself
        # there (lower ρ ⇒ x2 more collinear with S ⇒ lower SSE by overfitting).
        # Fixed ρ=0.8 by default; the grid stays available as a robustness
        # ablation (`identifier.rho_grid: [...]`).
        icfg.setdefault("rho_grid", None)
        icfg.setdefault("rho", 0.8)
        icfg.setdefault("W", 2000)
        icfg.setdefault("lock_min_gain", 0.01)
        # The instantaneous-coupling nuisance term BACKFIRED (run #3): S and x2 are
        # collinear, so it could not separate the constant torso coupling from the
        # payload-modulated PCR coupling — it left the constant in c AND ate the
        # modulation (c_corr 0.47 without it -> 0.03 with it). OFF now; the
        # constant offset is removed by trough-baseline subtraction instead (below).
        icfg.setdefault("nuisance_instant", False)
        icfg["do_scan"] = False          # RECON scans the fresh rollout instead
        # the readout map is measured, not derived — mirror it into [ID] so its
        # own defaults can never silently disagree with the env shim's
        rcfg = dict(env_args.get("recon_cfg", {}))
        for key in ("readout_qvel_idx", "readout_qvel_idx_ankle"):
            if key in rcfg:
                icfg.setdefault(key, rcfg[key])
        self.identifier = ECLIdentifier(self.N, icfg)
        self.id_W = int(icfg["W"])
        # refresh once every W new per-thread steps — ECL's cadence, and exactly
        # the window over which ĉ is treated as constant by [RE] (T3's Δ_c·W).
        self.refresh_every = max(1, int(round(self.id_W / max(1, self.T))))
        self.id_window = ReconIdWindow(
            self.N, self.k, self.nt, self.id_W + self.identifier.margin
        )

        # ---- per-rollout stashes ------------------------------------------
        z4 = (self.T, self.nt, self.N, self.k)
        self._st_u = np.zeros(z4)                  # executed actions
        self._st_a = np.zeros(z4)                  # policy samples (for u−a)
        self._st_lhat = np.zeros(z4)               # ℓ̂ the policy acted on
        self._st_y = np.zeros((self.T, self.nt, self.N, 2))     # raw readouts
        self._st_dtrue = np.full(z4, np.nan)       # pcr_d_applied  (diagnostic)
        self._st_dnext = np.full(z4, np.nan)       # pcr_d_next     (oracle arm)
        self._st_done = np.zeros((self.T, self.nt))
        self._st_payload = np.full((self.T, self.nt), np.nan)
        self._st_sat = np.full((self.T, self.nt), np.nan)
        self._st_dobs = None                       # (T, nt, D) — scan only
        self._st_rew = np.zeros((self.T, self.nt))

        # ---- carried state (the on-policy rollout is contiguous) -----------
        self._f_state = None            # filter hidden, threads×agents
        self._cur_lhat = np.zeros((self.nt, self.N, self.k))
        self._u_prev_iter_start = np.zeros((self.nt, self.N, self.k))
        self._re_carry = None           # [RE] leak state across iterations
        self._iter = 0
        self._since_refresh = 10 ** 9   # force a refresh on iteration 1
        self._c_hat = 0.0               # raw identifier output
        self._c_pcr = 0.0               # after trough-baseline subtraction (used by [RE])
        self._locked = False
        self._c_pairs = []              # (ĉ, c_true) for the running lock corr
        self._last_warn_iter = 0
        self._scan_bad = 0

        # ---- trough-baseline subtraction (spec-style c_max_seen, mirrored) ----
        # The clipfit reads   c_hat ≈ c_physical(const torso coupling) + c_PCR(t).
        # At the payload trough the true PCR severity is 0, so whatever c_hat reads
        # there IS c_physical. Estimate it as a low percentile of recent c_hat and
        # subtract:  c_PCR = max(0, c_hat − c_floor). This is what makes the labels
        # (and hence the capable filter, which already hits true_r2≈0.8 at the peak)
        # VANISH at the trough, so [CP] stops injecting harm where there is none —
        # run #3's single remaining failure. Needs one full payload cycle
        # (~200 iters at the tuned config) of history to seat the floor.
        self.trough_baseline = bool(cfg.get("trough_baseline", True))
        self._c_floor = 0.0
        self._c_floor_pct = float(cfg.get("trough_floor_pct", 10.0))
        # self-supervised nominal-gain calibration (label_mode: self_supervised)
        self._g0 = None                 # (N, kc) robust global nominal gain (median)
        from collections import deque as _dq
        self._g0_hist = _dq(maxlen=int(cfg.get("gain_hist", 40)))

        # auto-select the readout column per (agent, channel): use whichever raw-obs
        # coordinate its own torque actually drives, locked over the first few
        # (high-excitation) iterations. Removes the fragile fixed-index assumption
        # that crashed runs on the velocity/position swap. Falls back to the
        # configured index if the env doesn't stash the full obs delta.
        self.auto_readout = bool(cfg.get("auto_readout", True))
        self.auto_lock_after = int(cfg.get("auto_lock_after", 5))
        # only consider the joint region (qpos[2:]=13 + qvel=14 = 27) — the readout
        # must be a joint coordinate, never a cfrc_ext contact-force column.
        self.auto_readout_maxcol = int(cfg.get("auto_readout_maxcol", 27))
        self._auto_cols = None          # (N, kc) locked readout columns
        self._scan_prof = None          # accumulated |corr| profile, per channel
        # c_hat is appended once per identifier REFRESH (every W/T ≈ 10 iters, i.e.
        # ~20 appends per ~200-iter payload cycle), so ~2 cycles ≈ 40 appends. The
        # floor then tracks the recent trough and adapts if the gait/coupling drifts.
        from collections import deque
        self._c_hist = deque(maxlen=int(cfg.get("trough_hist", 40)))
        self._scan_corr_at_idx = float("nan")
        # a configured readout column is "valid" if own torque explains this much
        # of it; run #1's wrong column sat near ~0.1, a correct one near ~0.5.
        self._scan_min_corr = float(cfg.get("scan_min_corr", 0.25))

        # ---- diagnostic-only truth (NEVER used by the method) --------------
        # σ is read for the c_true column only: c = A(t)·SEVERITY. It is not an
        # input to [ID]/[RE]/[F]/[CP] and removing it changes no training bit.
        self.severity_diag = float(
            cfg.get("severity_diag", os.environ.get("ANT_PCR_SEVERITY", 0.45))
        )

        if self.filter_oracle:
            print(
                "\n" + "=" * 72 + "\n[ORACLE ARM] filter_oracle=True — ℓ̂ is the env's "
                "TRUE d (info: pcr_d_next), not\n             the filter. This is the "
                "Stage-A ceiling of [F]; never the headline.\n" + "=" * 72, flush=True,
            )
        print(
            f"[RECON] host=HAPPO k={self.k} raw_obs={self.raw_obs_dim} "
            f"obs={self.obs_dim} share={self.share_dim} | label_mode={self.label_mode} "
            f"| filter={self.recon_filter.arch}"
            f"(hidden={self.recon_filter.hidden}, {self.recon_filter.n_params()} params, "
            f"lr={self.recon_filter.lr}, epochs={self.recon_filter.epochs}) | "
            f"compensate={self.compensate} beta_mode={self.recon_filter.beta_mode} | "
            f"identifier=W{self.id_W}/every {self.refresh_every} iters (diagnostic in "
            f"self_supervised mode)",
            flush=True,
        )

        # The base __init__ already ran restore() for actor/critic, before the
        # filter existed — pick the filter up now.
        if self.algo_args["train"]["model_dir"] is not None:
            path = str(self.algo_args["train"]["model_dir"]) + "/recon_filter.pt"
            if os.path.exists(path):
                self.recon_filter.restore(path)
                self.beta = self.recon_filter.beta()
                print(f"[RECON] restored filter from {path}")

        self._open_debug()

    # ==================================================================== debug
    def _open_debug(self):
        self._dbg = self._dbg_w = self._log = self._eval_dbg = self._eval_dbg_w = None
        if not bool(self.recon_cfg.get("debug", True)):
            return
        if getattr(self, "run_dir", None) is None:
            return
        p = os.path.join(str(self.run_dir), "recon_debug.csv")
        self._dbg = open(p, "w", newline="", encoding="utf-8")
        self._dbg_w = csv.writer(self._dbg)
        self._dbg_w.writerow(_CSV_COLUMNS)
        lp = os.path.join(str(self.run_dir), "recon_debug.log")
        self._log = open(lp, "w", encoding="utf-8")
        self._log.write(
            "# RECON debug trace. Chain: [ID] c_hat -> [RE] labels -> [DI] filter "
            "-> [F] lhat -> [CP] compensation.\n"
            "# A failure should be readable as the FIRST link whose column goes "
            "bad; see recon/README.md for the column key and the gates.\n"
            f"# severity(diagnostic only)={self.severity_diag} "
            f"compensate={self.compensate} filter_oracle={self.filter_oracle}\n"
        )
        self._log.flush()
        ep = os.path.join(str(self.run_dir), "recon_eval.csv")
        self._eval_dbg = open(ep, "w", newline="", encoding="utf-8")
        self._eval_dbg_w = csv.writer(self._eval_dbg)
        self._eval_dbg_w.writerow(_EVAL_CSV_COLUMNS)
        print(f"[RECON] debug trace -> {os.path.abspath(p)}")
        print(f"[RECON] debug log   -> {os.path.abspath(lp)}")
        print(f"[RECON] eval trace  -> {os.path.abspath(ep)}")

    def _warn(self, msg):
        if self._log is not None:
            self._log.write(f"[WARN] {msg}\n")
            self._log.flush()

    # ================================================================= helpers
    def _augment_obs(self, obs, lhat):
        """o_i ⊕ ℓ̂_i. obs (nt, N, raw) + lhat (nt, N, k) -> (nt, N, raw+k)."""
        return np.concatenate([obs, lhat], axis=-1).astype(np.float32)

    def _augment_share(self, share_obs, lhat):
        """share_obs ⊕ (ℓ̂_1..ℓ̂_N) — the critic mirror (ECL's precedent for ε).
        The central critic is trainer-side, so seeing every agent's ℓ̂ is standard
        CTDE; execution is unaffected."""
        flat = lhat.reshape(lhat.shape[0], 1, self.N * self.k)
        return np.concatenate(
            [share_obs, np.repeat(flat, share_obs.shape[1], axis=1)], axis=-1
        ).astype(np.float32)

    def _filter_advance(self, raw_obs, u_prev, masks):
        """ℓ̂(t) = f_ψ(o_i(t), u_i(t−1)) for every (thread, agent).

        Batch order is (thread, agent) — identical to every reshape in
        ``_recon_update``, so collection and training see the same layout.
        """
        B = self.nt * self.N
        lhat, self._f_state = self.recon_filter.step_np(
            raw_obs.reshape(B, self.raw_obs_dim),
            u_prev.reshape(B, self.k),
            self._f_state,
            masks.reshape(B, 1),
        )
        return lhat.reshape(self.nt, self.N, self.k).astype(np.float64)

    def _compensate(self, actions):
        """[CP]: u = clip(a − β⊙ℓ̂, low, high) — the inverse-channel layer.

        T2: with ℓ̂ = ℓ and β = 1 the env delivers clip((a − ℓ) + ℓ) = clip(a),
        i.e. the *stationary* actuator, for any driver path. With ℓ̂ = ℓ − e it
        delivers clip(a + e): the residual disturbance IS the estimation error,
        which is what makes T3's bound linear in a quantity we measure.

        Off (``compensate: false``) this is the identity and `recon` is the
        conditioning-only arm.
        """
        if not self.compensate:
            return actions
        u = actions - self.beta[None, None, :] * self._cur_lhat
        return np.clip(u, self.act_low, self.act_high)

    def _read_infos(self, infos):
        """Pull this step's readouts + (diagnostic) env telemetry out of infos."""
        y = np.zeros((self.nt, self.N, 2))
        dtrue = np.full((self.nt, self.N, self.k), np.nan)
        dnext = np.full((self.nt, self.N, self.k), np.nan)
        payload = np.full(self.nt, np.nan)
        sat = np.full(self.nt, np.nan)
        dobs = None
        for i in range(self.nt):
            d = infos[i]
            if isinstance(d, (list, tuple, np.ndarray)):
                d = d[0]
            if not isinstance(d, dict):
                continue
            if "recon_raw_y" in d:
                y[i] = np.asarray(d["recon_raw_y"], dtype=np.float64)
            # d is the env's flat (N*k,) torque vector; agent i owns [k*i : k*i+k]
            # (Ant 4x2: hip=2i, ankle=2i+1). Size-guarded so a non-PCR env leaves
            # these columns NaN instead of crashing a run over a diagnostic.
            for key, dst in (("pcr_d_applied", dtrue), ("pcr_d_next", dnext)):
                if key in d:
                    v = np.asarray(d[key], dtype=np.float64).ravel()
                    if v.size == self.N * self.k:
                        dst[i] = v.reshape(self.N, self.k)
            payload[i] = float(d.get("pcr_payload", np.nan))
            sat[i] = float(d.get("pcr_sat_frac", np.nan))
            if "recon_raw_dobs" in d:
                v = np.asarray(d["recon_raw_dobs"], dtype=np.float64)
                if dobs is None:
                    dobs = np.zeros((self.nt, v.shape[0]))
                dobs[i] = v
        return y, dtrue, dnext, payload, sat, dobs

    def _select_readout(self):
        """Return the readout y (T, nt, N, kc) for the self-supervised target.

        With ``auto_readout`` and the full obs delta available, pick per (agent,
        channel) the raw-obs column its own torque actually drives — accumulated
        over the first ``auto_lock_after`` high-excitation iterations, then locked.
        Otherwise fall back to the env's fixed-index ``recon_raw_y`` (self._st_y).
        """
        if not self.auto_readout or self._st_dobs is None:
            return self._st_y
        # accumulate the |corr| profile per channel until locked
        if self._auto_cols is None:
            kc = self._st_u.shape[-1]
            D = self._st_dobs.shape[-1]
            if self._scan_prof is None:
                self._scan_prof = np.zeros((kc, self.N, D))
            for ch in range(kc):
                self._scan_prof[ch] += readout_corr_profile(
                    self._st_dobs, self._st_u[..., ch]
                )
            if self._iter >= self.auto_lock_after:
                mc = min(self.auto_readout_maxcol, D)          # joint region only
                self._auto_cols = np.stack(
                    [np.argmax(self._scan_prof[ch][:, :mc], axis=1)
                     for ch in range(kc)],
                    axis=1,
                )                                              # (N, kc)
                corrs = np.stack(
                    [self._scan_prof[ch][np.arange(self.N), self._auto_cols[:, ch]]
                     / max(1, self._iter) for ch in range(kc)], axis=1)
                self._warn(
                    f"iter {self._iter}: LOCKED auto readout columns "
                    f"{self._auto_cols.tolist()} (mean |corr| "
                    f"{np.round(corrs, 3).tolist()}); configured was "
                    f"{self.recon_readout_idx().tolist()}."
                )
                print(f"[RECON] auto-selected readout columns {self._auto_cols.tolist()} "
                      f"(own torque best explains these obs coords)", flush=True)
                # drop the warmup gains that used the fixed (maybe weak) readout
                self._g0_hist.clear()
            else:
                return self._st_y                              # not locked yet
        # build y from the locked columns of the full obs delta
        y = np.zeros((self.T, self.nt, self.N, self._auto_cols.shape[1]))
        for i in range(self.N):
            for ch in range(self._auto_cols.shape[1]):
                y[:, :, i, ch] = self._st_dobs[:, :, int(self._auto_cols[i, ch])]
        return y

    # ================================================================== warmup
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        self._f_state = self.recon_filter.init_state(self.nt * self.N)
        zeros_u = np.zeros((self.nt, self.N, self.k))
        # episode start: the filter's state is reset and there is no previous
        # action; the true liability is 0 here (ant.py's reset_model zeroes _d),
        # which is exactly what the zero-init head emits.
        self._cur_lhat = self._filter_advance(
            obs, zeros_u, np.zeros((self.nt, self.N, 1), dtype=np.float32)
        )
        if self.filter_oracle:
            self._cur_lhat = zeros_u.copy()
        self._u_prev_iter_start = zeros_u.copy()
        self._re_carry = None

        aug_obs = self._augment_obs(obs, self._cur_lhat)
        for agent_id in range(self.N):
            self.actor_buffer[agent_id].obs[0] = aug_obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        aug_share = self._augment_share(share_obs, self._cur_lhat)
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = aug_share[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = aug_share.copy()

    # ===================================================================== run
    def run(self):
        """HAPPO's loop with two RECON seams: [CP] at the env boundary, and the
        filter advance that produces the ℓ̂ the next action conditions on."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        self.logger.init(episodes)

        for episode in range(1, episodes + 1):
            if self.algo_args["train"]["use_linear_lr_decay"]:
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.N):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(episode)
            self.prep_rollout()
            for step in range(self.T):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)          # policy sees o ⊕ ℓ̂(t) [CE]

                u = self._compensate(actions)   # [CP]
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(u)

                # stash this step, then advance the filter to ℓ̂(t+1)
                self._post_step(step, actions, u, obs, rewards, dones, infos)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,          # the buffer stores a, NOT u (PPO ratio)
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.logger.per_step(data)
                self.insert(data)

            self.compute()
            self.prep_training()
            actor_train_infos, critic_train_info = self.train()

            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def _post_step(self, step, actions, u, obs, rewards, dones, infos):
        """Record step t, then produce ℓ̂(t+1) for the next action."""
        dones_env = np.all(dones, axis=1)                      # (nt,)
        y, dtrue, dnext, payload, sat, dobs = self._read_infos(infos)

        # The EXECUTED torque is the *clipped* action: ant.py opens with
        # `tau = np.clip(a, -1, 1)` on whatever it is handed. That torque — not
        # the raw Gaussian sample — is what feeds Φ, so it is what [ID]/[RE] must
        # regress on (spec §2.1 ext 2) and what the filter conditions on. This
        # matters most when [CP] is OFF, where u is the unclipped sample and a
        # trained Ant saturates a large fraction of its steps.
        u_exec = np.clip(u, self.act_low, self.act_high)
        self._st_a[step] = np.clip(actions, self.act_low, self.act_high)
        self._st_u[step] = u_exec
        self._st_lhat[step] = self._cur_lhat                   # what acted at t
        self._st_dtrue[step] = dtrue                           # what t actually faced
        self._st_dnext[step] = dnext
        self._st_y[step] = y
        self._st_done[step] = dones_env
        self._st_payload[step] = payload
        self._st_sat[step] = sat
        self._st_rew[step] = np.mean(rewards, axis=1).flatten()
        if dobs is not None:
            if self._st_dobs is None or self._st_dobs.shape[1:] != dobs.shape:
                self._st_dobs = np.zeros((self.T,) + dobs.shape)
            self._st_dobs[step] = dobs

        # masks: 0 => step t+1 begins a new episode (state and u_prev reset)
        masks = (~dones_env).astype(np.float32)[:, None, None]  # (nt, 1, 1)
        u_prev = u_exec * masks
        self._cur_lhat = self._filter_advance(
            obs, u_prev, np.broadcast_to(masks, (self.nt, self.N, 1))
        )
        if self.filter_oracle:
            # ℓ̂(t+1) := the true d the NEXT action will face (info: pcr_d_next,
            # read after step t, so this is still causal), zeroed at an episode
            # start because the env's reset_model zeroes _d. LABELED ARM ONLY.
            self._cur_lhat = np.nan_to_num(dnext) * masks

    # ================================================================== insert
    def insert(self, data):
        """HAPPO's insert with the obs/share_obs augmented by ℓ̂(t+1).

        The *stored* observation is the augmented one — the policy is trained on
        exactly the vector it acted on (PPO consistency), and ``collect(step)``
        reads it straight back out of the buffer.
        """
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.N,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.N,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        masks = np.ones((self.nt, self.N, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.N, 1), dtype=np.float32
        )
        active_masks = np.ones((self.nt, self.N, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.N, 1), dtype=np.float32
        )

        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.N)
                    ]
                    for info in infos
                ]
            )

        aug_obs = self._augment_obs(obs, self._cur_lhat)          # [CE]
        for agent_id in range(self.N):
            self.actor_buffer[agent_id].insert(
                aug_obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        aug_share = self._augment_share(share_obs, self._cur_lhat)
        if self.state_type == "EP":
            self.critic_buffer.insert(
                aug_share[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                aug_share, rnn_states_critic, values, rewards, masks, bad_masks
            )

    # =================================================================== train
    def train(self):
        """[ID] -> [RE] -> [DI], then the host update, untouched."""
        self._recon_update()
        return super(OnPolicyReconRunner, self).train()

    def _recon_update(self):
        self._iter += 1
        self._since_refresh += 1

        # ---------------------------------------------------------- [ID] -----
        self.id_window.push_rollout(self._st_u, self._st_y, self._st_done)
        if self._since_refresh >= self.refresh_every:
            self._c_hat = float(self.identifier.refresh(self.id_window.view()))
            self._locked = self.identifier.lock_gain > self.identifier.lock_min_gain
            self._since_refresh = 0
            if self._locked:
                self._c_hist.append(self._c_hat)
        rho_hat = float(self.identifier.rho_hat)

        # -------------------------------------------- trough-baseline (c_PCR) --
        # c_hat ≈ c_physical (const torso coupling) + c_PCR(t). Subtract the
        # trough floor (a low percentile of recent c_hat = c_physical) so the
        # labels vanish at the payload trough. See __init__.
        if self.trough_baseline and len(self._c_hist) >= 8:
            self._c_floor = float(np.percentile(self._c_hist, self._c_floor_pct))
        self._c_pcr = max(0.0, self._c_hat - self._c_floor) if self.trough_baseline \
            else self._c_hat

        # ---------------------------------------------------------- [RE] -----
        # Two target modes (see __init__.label_mode):
        if self.label_mode == "self_supervised":
            # Local disturbance-observer target d̃ = (y − ĝ0·a)/ĝ0 from the OWN
            # readout — no central c. ĝ0 is the robust GLOBAL nominal gain (median
            # per-window <y,a>/<a,a>), which leaves only a benign, phase-independent
            # own-action leak — unlike the central ĉ, whose error is anti-correlated
            # with the load (measured −0.35). The causal filter's lag then separates
            # the disturbance (leaked history) from that residual own-action term.
            readout = self._select_readout()                  # (T, nt, N, kc)
            g0_raw = nominal_gain(readout, self._st_u)         # (N, kc), kc==k
            self._g0_hist.append(g0_raw)
            self._g0 = (calibrate_gain(self._g0_hist)
                        if len(self._g0_hist) >= 8 else g0_raw)
            labels = disturbance_target(readout, self._st_u, self._g0)
            # always-valid: the readout is present every step (no lock gating)
            valid_flag = 1.0
        else:
            # The leak state carries even when unlocked (it does not depend on ĉ),
            # so labels stay exact the moment the identifier locks. [RE] uses
            # c_PCR (the payload-tracking part), NOT the raw c_hat.
            labels, self._re_carry = relabel(
                self._st_u, self._st_done, rho_hat, self._c_pcr, self._re_carry
            )
            valid_flag = 1.0 if self._locked else 0.0

        # ---------------------------------------------------------- [DI] -----
        B = self.nt * self.N
        raw_obs_seq = np.stack(
            [
                self.actor_buffer[i].obs[: self.T, :, : self.raw_obs_dim]
                for i in range(self.N)
            ],
            axis=2,
        )                                                   # (T, nt, N, raw)
        u_prev_seq = np.concatenate(
            [self._u_prev_iter_start[None], self._st_u[:-1]], axis=0
        )                                                   # (T, nt, N, k)
        masks = self.actor_buffer[0].masks[: self.T]        # (T, nt, 1)
        masks_b = np.repeat(masks[:, :, None, :], self.N, axis=2)
        u_prev_seq = u_prev_seq * masks_b                   # no action before t=0
        valid = np.full((self.T, B, 1), valid_flag, dtype=np.float32)

        finfo = self.recon_filter.update(
            raw_obs_seq.reshape(self.T, B, self.raw_obs_dim),
            u_prev_seq.reshape(self.T, B, self.k),
            labels.reshape(self.T, B, self.k),
            masks_b.reshape(self.T, B, 1),
            valid,
        )
        self.beta = self.recon_filter.beta()
        self._u_prev_iter_start = self._st_u[-1].copy()

        if self._iter % max(1, self.debug_every) == 0:
            self._write_debug(labels, rho_hat, finfo)

    # ============================================================ diagnostics
    @staticmethod
    def _r2_rms(pred, truth):
        """(R², rms error) of pred against truth over the finite entries."""
        m = np.isfinite(truth) & np.isfinite(pred)
        if m.sum() < 10:
            return float("nan"), float("nan")
        e = pred[m] - truth[m]
        mse = float(np.mean(e ** 2))
        var = float(np.var(truth[m]))
        r2 = 1.0 - mse / var if var > 1e-12 else float("nan")
        return r2, float(np.sqrt(mse))

    def _write_debug(self, labels, rho_hat, finfo):
        step = self._iter * self.T * self.nt

        # --- truth columns. DIAGNOSTIC ONLY: nothing below feeds the method. ---
        payload = float(np.nanmean(self._st_payload)) if np.any(
            np.isfinite(self._st_payload)
        ) else float("nan")
        c_true = self.severity_diag * payload
        c_err = abs(self._c_pcr - c_true) if np.isfinite(c_true) else float("nan")
        # c_corr tracks c_PCR (the payload-tracking part after baseline
        # subtraction) against the truth — that is the quantity [RE] actually
        # uses. The raw c_hat's correlation is c_hat_corr below.
        c_corr = float("nan")
        if np.isfinite(c_true):
            self._c_pairs.append((self._c_pcr, self._c_hat, c_true))
            if len(self._c_pairs) >= 5:
                a = np.asarray(self._c_pairs[-500:])
                if np.std(a[:, 0]) > 1e-9 and np.std(a[:, 2]) > 1e-9:
                    c_corr = float(np.corrcoef(a[:, 0], a[:, 2])[0, 1])

        label_r2, label_err = self._r2_rms(labels, self._st_dtrue)
        filt_r2, filt_err = self._r2_rms(self._st_lhat, self._st_dtrue)
        lhat_rms = float(np.sqrt(np.mean(self._st_lhat ** 2)))
        ltrue_rms = (
            float(np.sqrt(np.nanmean(self._st_dtrue ** 2)))
            if np.any(np.isfinite(self._st_dtrue))
            else float("nan")
        )
        label_rms = float(np.sqrt(np.mean(labels ** 2)))
        u_minus_a = float(np.sqrt(np.mean((self._st_u - self._st_a) ** 2)))
        sat = float(np.nanmean(self._st_sat)) if np.any(np.isfinite(self._st_sat)) else float("nan")

        scan_off, scan_corr = float("nan"), float("nan")
        # dense for the first few iterations (a random policy is maximally
        # excited, so these are the MOST reliable scans and they let a wrong map
        # abort in seconds rather than after scan_every*episode_length steps)
        scan_now = self._iter <= 3 or (self._iter - 1) % max(1, self.scan_every) == 0
        if self._st_dobs is not None and scan_now:
            try:
                scan_off, scan_corr, scan_corr_at_idx = scan_readout_offset(
                    self._st_dobs, self._st_u[..., 0], self.recon_readout_idx()
                )
                self._scan_corr_at_idx = scan_corr_at_idx
                self._check_scan(scan_off, scan_corr, scan_corr_at_idx)
            except AssertionError:
                raise
            except Exception as e:  # a diagnostic must never break a run
                self._warn(f"iter {self._iter}: scan failed ({e}).")

        row = [
            step, self._iter,
            round(self._c_hat, 6), round(self._c_floor, 6), round(self._c_pcr, 6),
            round(float(np.mean(self._g0)), 5) if self._g0 is not None else float("nan"),
            round(rho_hat, 4),
            round(float(self.identifier.h_med), 5),
            round(float(self.identifier.lock_gain), 5),
            round(float(self.identifier.clip_frac), 4),
            round(float(self.identifier.sumzero_frac), 4),
            round(float(self.identifier.c_now_ratio), 5),
            round(c_true, 6), round(c_err, 6), round(c_corr, 4),
            round(payload, 5), int(self._locked),
            round(label_r2, 4), round(label_err, 6), round(label_rms, 6),
            round(float(finfo["filter_mse"]), 8),
            round(float(finfo["filter_mse_first"]), 8),
            round(float(finfo["filter_grad_norm"]), 4),
            round(filt_r2, 4), round(filt_err, 6),
            round(lhat_rms, 6), round(ltrue_rms, 6),
            round(float(self.beta[0]), 4),
            round(float(self.beta[min(1, self.k - 1)]), 4),
            round(u_minus_a, 5), round(sat, 5),
            round(float(np.mean(self._st_rew)), 5),
            scan_off, round(scan_corr, 4) if np.isfinite(scan_corr) else scan_corr,
            round(self._scan_corr_at_idx, 4)
            if np.isfinite(self._scan_corr_at_idx) else self._scan_corr_at_idx,
            int(finfo["n_label_valid"]),
        ]
        if self._dbg_w is not None:
            self._dbg_w.writerow(row)
            self._dbg.flush()

        if self.writter is not None:
            for key, val in [
                ("c_hat", self._c_hat), ("c_true", c_true), ("c_corr", c_corr),
                ("lock_gain", self.identifier.lock_gain), ("rho_hat", rho_hat),
                ("label_r2", label_r2), ("filter_true_r2", filt_r2),
                ("filter_true_err", filt_err), ("filter_mse", finfo["filter_mse"]),
                ("sat_frac", sat), ("u_minus_a", u_minus_a),
                ("beta_hip", float(self.beta[0])),
            ]:
                if val is not None and np.isfinite(val):
                    self.writter.add_scalar(f"recon/{key}", float(val), step)

        # ---- the prose trace + the gate warnings (spec Part 5, G2) ----
        if self._log is not None:
            self._log.write(
                f"it={self._iter:5d} step={step:9d} | [ID] c_hat={self._c_hat:.4f} "
                f"c_true={c_true:.4f} c_corr={c_corr:.3f} (diagnostic) | "
                f"[RE:{self.label_mode[:4]}] g0={float(np.mean(self._g0)) if self._g0 is not None else float('nan'):.3f} "
                f"label_r2={label_r2:.3f} err={label_err:.4f} | "
                f"[DI] mse={finfo['filter_mse']:.5f} | [F] true_r2={filt_r2:.3f} "
                f"err={filt_err:.4f} (|l|rms {lhat_rms:.3f} vs {ltrue_rms:.3f}) | "
                f"[CP] beta={np.round(self.beta, 3).tolist()} du={u_minus_a:.3f} "
                f"sat={sat:.3f} | rew={float(np.mean(self._st_rew)):.3f}\n"
            )
            self._log.flush()

        # gate warnings on elapsed iterations, not on a modulus that `debug_every`
        # could step over entirely
        if self._iter - self._last_warn_iter >= 50:
            self._last_warn_iter = self._iter
            if not self._locked:
                self._warn(
                    f"iter {self._iter}: [ID] has NOT locked (lock_gain="
                    f"{self.identifier.lock_gain:.4f} <= {self.identifier.lock_min_gain}). "
                    f"No labels are being produced, so [DI] is idle and ℓ̂ stays ~0 "
                    f"— `recon` is degenerating to plain HAPPO. Suspect T4 "
                    f"(excitation/observability), not [F]."
                )
            if np.isfinite(c_corr) and self._iter > 200 and c_corr < 0.95:
                self._warn(
                    f"iter {self._iter}: ĉ lock corr={c_corr:.3f} < 0.95 — G2's "
                    f"identification gate is failing; ε_id dominates T3's bound."
                )
            # A CONSTANT ĉ is the silent killer: c_corr goes nan (zero variance)
            # so the gate above never fires, while [RE]'s labels stop vanishing at
            # the trough and [CP] injects a payload-independent disturbance
            # exactly where the env is benign (symptom: trough-slice < peak-slice).
            if not np.isfinite(c_corr) and self._locked and self._iter > 100:
                self._warn(
                    f"iter {self._iter}: ĉ has ZERO variance (c_hat={self._c_hat:.4f} "
                    f"constant) — c_corr is undefined, so the lock gate above is "
                    f"vacuous. [RE]'s labels can no longer track the payload."
                )
            if self._locked and abs(self._c_hat - self.identifier.c_grid_max) < \
                    2 * self.identifier.grid_dc:
                self._warn(
                    f"iter {self._iter}: ĉ={self._c_hat:.3f} is RAILED at the "
                    f"c-grid ceiling ({self.identifier.c_grid_max}) — the fit wants "
                    f"c beyond the physical range, so it is not seeing the real "
                    f"driver. Suspect the readout map, or an unmodelled same-step "
                    f"coupling (identifier.nuisance_instant)."
                )
            if np.isfinite(label_r2) and self._locked and label_r2 < 0.9:
                self._warn(
                    f"iter {self._iter}: label_r2={label_r2:.3f} — the manufactured "
                    f"labels are NOT the true liability although [ID] claims a lock. "
                    f"Suspect the Φ/index map or ρ̂ (rho_hat={rho_hat:.2f}), not [F]."
                )
            if np.isfinite(filt_r2) and self._iter > 200 and filt_r2 < 0.5:
                self._warn(
                    f"iter {self._iter}: filter_true_r2={filt_r2:.3f} < 0.5 — A4 "
                    f"(local decodability, E5's κ) is the binding constraint. "
                    f"T3 says the return gap is linear in this error."
                )
            if np.isfinite(sat) and sat > 0.2:
                self._warn(
                    f"iter {self._iter}: sat_frac={sat:.3f} — actuator authority "
                    f"(A2) is being exhausted, which is exactly T5's frontier "
                    f"failure: compensation becomes self-defeating above c*."
                )

    def recon_readout_idx(self):
        """The configured hip readout indices (for the scan diagnostic)."""
        cfg = dict(self.env_args.get("recon_cfg", {}))
        return np.asarray(
            cfg.get("readout_qvel_idx", [17 + 2 * i for i in range(self.N)]),
            dtype=np.int64,
        )

    def _check_scan(self, scan_off, scan_corr, scan_corr_at_idx):
        """Guard the readout map, gating on the RIGHT quantity.

        The expensive failure (run #1) is [ID] regressing own torque on a column
        it does NOT drive — then the fit rails, labels are pure scale error, and
        [CP] injects more disturbance than the NS. The correct test for that is
        ``corr_at_idx``: does the *configured* column respond to own torque?

        It is NOT ``offset == 0``. Own torque moves both a joint's velocity and
        its position, so two valid readouts live one obs-block apart and the
        argmax flips between them by gait — an argmax gate false-fired at 1.6M on
        a perfectly good velocity readout (the position channel had briefly won).

        So: configured column responds well  -> OK (silent), even if some other
        column edges it. Configured column responds POORLY and a clearly better
        one exists -> that is the real "wrong map"; warn, and abort only on a
        repeat (``scan_abort``, default on; set false to force a warning).
        """
        cfg_ok = np.isfinite(scan_corr_at_idx) and scan_corr_at_idx >= self._scan_min_corr
        if cfg_ok:
            self._scan_bad = 0
            if scan_off != 0 and self._log is not None:
                self._log.write(
                    f"[note] iter {self._iter}: readout scan argmax is offset "
                    f"{scan_off} (corr {scan_corr:.3f}) but the CONFIGURED map "
                    f"also responds to own torque (corr_at_idx={scan_corr_at_idx:.3f}"
                    f" >= {self._scan_min_corr}) — both are valid readouts "
                    f"(velocity vs position), keeping the configured one.\n"
                )
                self._log.flush()
            return

        # configured column does NOT respond to own torque -> genuinely wrong map
        suggest = (self.recon_readout_idx() + scan_off).tolist() if scan_off else None
        msg = (
            f"[RECON] the CONFIGURED readout map does not respond to own torque: "
            f"corr_at_idx={scan_corr_at_idx:.3f} < {self._scan_min_corr} "
            f"(configured hips={self.recon_readout_idx().tolist()}, best-column "
            f"corr={scan_corr:.3f} at offset {scan_off}).\n"
            f"        [ID] would be regressing own torque on a column it does not "
            f"drive — the run #1 failure."
            + (f" Try recon_cfg.readout_qvel_idx={suggest} and "
               f"readout_qvel_idx_ankle={[s + 1 for s in suggest]}."
               if suggest else "")
        )
        self._warn(msg)
        self._scan_bad += 1
        # With auto_readout ON, the readout is chosen from the scan itself, so a
        # weak CONFIGURED column is not a failure — it is exactly the case
        # auto-selection handles. Only abort when the user pinned a fixed index
        # (auto_readout: false) AND opted into aborting.
        if (not self.auto_readout and self._scan_bad >= 2
                and bool(self.recon_cfg.get("scan_abort", False))):
            raise AssertionError(msg)

    # ==================================================================== eval
    @torch.no_grad()
    def eval(self):
        """De-aliased eval (C4): the eval envs' payload clocks are stratified
        across the cycle (``pcr_eval_dephase``), so a round is a true
        cycle-average rather than a phase-aliased snapshot. The filter and [CP]
        run exactly as in training — this is the deployable agent, using only
        local quantities.
        """
        self.logger.eval_init()
        eval_episode = 0
        n = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # a filter state private to the eval envs — never touches self._f_state
        f_state = self.recon_filter.init_state(n * self.N)
        cur_lhat = np.zeros((n, self.N, self.k))
        zeros_u = np.zeros((n, self.N, self.k))
        lhat, f_state = self.recon_filter.step_np(
            eval_obs[:, :, : self.raw_obs_dim].reshape(n * self.N, self.raw_obs_dim),
            zeros_u.reshape(n * self.N, self.k),
            f_state,
            np.zeros((n * self.N, 1), dtype=np.float32),
        )
        if not self.filter_oracle:
            cur_lhat = lhat.reshape(n, self.N, self.k).astype(np.float64)

        eval_rnn_states = np.zeros(
            (n, self.N, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32
        )
        eval_masks = np.ones((n, self.N, 1), dtype=np.float32)
        one_ep_rew = [[] for _ in range(n)]
        one_ep_len = np.zeros(n, dtype=np.int64)
        ep_returns, ep_payloads, ep_lens = [], [], []

        while True:
            eval_actions_collector = []
            for agent_id in range(self.N):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id, : self.raw_obs_dim], cur_lhat[:, agent_id]],
                    axis=-1,
                ).astype(np.float32)
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    aug_obs,
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # [CP] at eval, identically to training
            if self.compensate:
                u = np.clip(
                    eval_actions - self.beta[None, None, :] * cur_lhat,
                    self.act_low,
                    self.act_high,
                )
            else:
                u = eval_actions

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(u)

            for i in range(n):
                one_ep_rew[i].append(eval_rewards[i])
            one_ep_len += 1
            eval_dones_env = np.all(eval_dones, axis=1)

            masks_f = (~eval_dones_env).astype(np.float32)[:, None, None]
            # the executed torque, exactly as in _post_step (ant.py clips it)
            u_exec = np.clip(u, self.act_low, self.act_high)
            lhat, f_state = self.recon_filter.step_np(
                eval_obs[:, :, : self.raw_obs_dim].reshape(n * self.N, self.raw_obs_dim),
                (u_exec * masks_f).reshape(n * self.N, self.k),
                f_state,
                np.broadcast_to(masks_f, (n, self.N, 1)).reshape(n * self.N, 1),
            )
            cur_lhat = lhat.reshape(n, self.N, self.k).astype(np.float64)
            if self.filter_oracle:
                dn = np.full((n, self.N, self.k), np.nan)
                for i in range(n):
                    d = eval_infos[i]
                    if isinstance(d, (list, tuple, np.ndarray)):
                        d = d[0]
                    if isinstance(d, dict) and "pcr_d_next" in d:
                        dn[i] = np.asarray(d["pcr_d_next"]).reshape(self.N, self.k)
                cur_lhat = np.nan_to_num(dn) * masks_f

            self.logger.eval_per_step(
                (
                    eval_obs,
                    eval_share_obs,
                    eval_rewards,
                    eval_dones,
                    eval_infos,
                    eval_available_actions,
                )
            )

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.N,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones((n, self.N, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.N, 1), dtype=np.float32
            )

            for i in range(n):
                if eval_dones_env[i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(i)
                    ep_ret = float(np.mean(np.sum(one_ep_rew[i], axis=0)))
                    d = eval_infos[i]
                    if isinstance(d, (list, tuple, np.ndarray)):
                        d = d[0]
                    pay = float(d.get("pcr_payload", np.nan)) if isinstance(d, dict) else np.nan
                    ep_returns.append(ep_ret)
                    ep_payloads.append(pay)
                    ep_lens.append(int(one_ep_len[i]))
                    if self._eval_dbg_w is not None:
                        self._eval_dbg_w.writerow(
                            [
                                self._iter * self.T * self.nt, i, round(pay, 5),
                                round(ep_ret, 3), int(one_ep_len[i]),
                            ]
                        )
                    one_ep_rew[i] = []
                    one_ep_len[i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

        if self._eval_dbg is not None:
            self._eval_dbg.flush()
        self._report_eval_slices(ep_returns, ep_payloads, ep_lens)

    def _report_eval_slices(self, ep_returns, ep_payloads, ep_lens):
        """Per-phase slices alongside the cycle average (Prohibition 5)."""
        rets = np.asarray(ep_returns, dtype=np.float64)
        pays = np.asarray(ep_payloads, dtype=np.float64)
        if rets.size == 0:
            return
        avg = float(np.mean(rets))
        trough = peak = float("nan")
        finite = np.isfinite(pays)
        if finite.sum() >= 5:
            pf, rf = pays[finite], rets[finite]
            lo, hi = np.quantile(pf, 0.1), np.quantile(pf, 0.8)
            if np.any(pf <= lo):
                trough = float(np.mean(rf[pf <= lo]))
            if np.any(pf >= hi):
                peak = float(np.mean(rf[pf >= hi]))
        step = self._iter * self.T * self.nt
        print(
            f"[RECON eval] step={step} cycle-avg={avg:.1f} "
            f"trough-slice={trough:.1f} peak-slice={peak:.1f} "
            f"len={float(np.mean(ep_lens)):.0f}"
        )
        if self._log is not None:
            self._log.write(
                f"[EVAL] step={step} cycle_avg={avg:.2f} trough={trough:.2f} "
                f"peak={peak:.2f} len={float(np.mean(ep_lens)):.1f}\n"
            )
            self._log.flush()
        if self.writter is not None:
            self.writter.add_scalar("recon/eval_cycle_avg", avg, step)
            if np.isfinite(trough):
                self.writter.add_scalar("recon/eval_trough_slice", trough, step)
            if np.isfinite(peak):
                self.writter.add_scalar("recon/eval_peak_slice", peak, step)

    # =========================================================== save/restore
    def save(self):
        super(OnPolicyReconRunner, self).save()
        self.recon_filter.save(str(self.save_dir) + "/recon_filter.pt")

    def restore(self):
        super(OnPolicyReconRunner, self).restore()
        f = getattr(self, "recon_filter", None)
        if f is not None:
            path = str(self.algo_args["train"]["model_dir"]) + "/recon_filter.pt"
            if os.path.exists(path):
                f.restore(path)
                self.beta = f.beta()

    def close(self):
        for fh in (
            getattr(self, "_dbg", None),
            getattr(self, "_log", None),
            getattr(self, "_eval_dbg", None),
        ):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        super(OnPolicyReconRunner, self).close()
