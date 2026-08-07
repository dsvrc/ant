"""Flight recorder for the PCR non-stationary Ant (no intervention).

This wraps ``MujocoMulti`` and changes **nothing** about the dynamics, the
action, the observation, or the reward -- it is plain HAPPO/HASAC on the PCR
ant. Its only job is to record, every step, the complete picture of *why and
where the blind policy fails*, so the failure mechanism can be read off directly:

* the exogenous driver and the disturbance actually applied this step
  (`pcr_payload`, `d_app_mean/max`);
* **actuator saturation** -- the fraction of joints where the commanded torque
  plus the parasitic load leaves the `[-1, 1]` range (`sat_frac`): this is
  "the disturbance exceeds the actuator's authority", the hard-failure mode;
* **termination proximity** -- the torso height vs the env's `[0.2, 1.0]` bound
  (`torso_height`), plus whether the episode ended and whether it was a *fall*
  (`done`, `fall`) and how long it survived (`ep_step`);
* the **reward decomposition** (`r_forward/ctrl/contact/survive`): whether the
  return drops because the ant moves less (forward), fights the load (ctrl),
  or tips over and loses the survival stream (survive going to 0 on termination).

Enable with ``env_args.pcr_diag: true`` and run any algorithm (e.g. plain
``--algo happo``); each parallel env writes its own CSV to ``pcr_diag_dir``.

--------------------------------------------------------------------------------
Diagnosis-campaign extensions (spec §3.1) -- all APPEND-ONLY
--------------------------------------------------------------------------------
* **mode-resolved power** (`d_*_common/diff`, `tau_*_common/diff`): for hips
  ``h = x[0,2,4,6]``, ``common = mean(h)**2`` and ``diff = var(h)`` (population);
  likewise ankles, and identically for the commanded torque. These are the
  observables of the Part-4 loop-geometry analysis: the E2 derivation predicts
  the *difference-mode DC* content is amplified up to x9 into d while
  gait-frequency content sees only ~x0.2, so whether privileged cancellation can
  work at all is read off these columns, not off the return.
* `clip_frac_cmd` -- fraction of joints with ``|a| > 0.999`` (the policy riding
  the rails). Distinct from `sat_frac`, which keeps its meaning: ``|tau+d| > 1``
  (the disturbance exceeding actuator authority).
* `fwd_vel` -- ``(xposafter - xposbefore)/dt`` (equals `reward_forward`).
* `sumzero_resid` -- ``|mean(tau_hip)| + |mean(tau_ank)|``: distance from the
  sum-zero manifold. E4 needs it because the post-projection clip can re-break
  sum-zero, and that must be visible rather than assumed away.
* **per-episode roll-up** ``*_episodes.csv`` -- return, length, termination cause
  (`fall_low`/`fall_high`/`nonfinite`/`timeout`), payload at start/end, and the
  means of every per-step column. This is what E1's deficit decomposition
  (achievement-mediated vs termination-mediated) is computed from.
* **NPZ trajectory dump** (`dump_traj`) -- the raw material for E5.

  .. note:: **Deliberate deviation from spec §3.1.** The spec says "decimated
     1-in-K full records". Step-decimation would make E5 *impossible*: its whole
     object is the last-L **consecutive** history window, which a 1-in-K stream
     no longer contains. The dump therefore decimates at the **episode** level
     (`dump_every_k_episodes`) -- keeping 1 episode in K, contiguous -- which
     honours the size cap for the same reason while preserving the lag structure.
     Records are float32 and ring-buffered against `dump_max_mb` (default 512).
"""

import os
import csv
import threading
from collections import deque

import numpy as np

from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

_N_ACT = 8

# v1 columns (unchanged, in order) ...
_COLS = [
    "step_global", "ep_step", "done", "fall",
    "pcr_payload", "pcr_load", "pcr_loadmax",
    "d_app_mean", "d_app_max",
    "torso_height",
    "sat_frac", "tau_absmean", "delivered_absmean",
    "reward", "r_forward", "r_ctrl", "r_contact", "r_survive",
    # ... campaign additions (spec §3.1), appended so old parsers keep working
    "d_hip_common", "d_hip_diff", "d_ank_common", "d_ank_diff",
    "tau_hip_common", "tau_hip_diff", "tau_ank_common", "tau_ank_diff",
    "clip_frac_cmd", "fwd_vel", "sumzero_resid", "pcr_clock", "term_cause",
]

_EP_COLS = [
    "ep_index", "step_global_end", "ep_return", "ep_len", "term_cause",
    "payload_start", "payload_end",
    "mean_reward", "mean_r_forward", "mean_r_ctrl", "mean_r_contact",
    "mean_r_survive", "mean_fwd_vel", "mean_sat_frac", "mean_clip_frac_cmd",
    "mean_d_app_mean", "mean_d_app_max", "mean_torso_height",
    "mean_d_hip_common", "mean_d_hip_diff", "mean_d_ank_common", "mean_d_ank_diff",
    "mean_tau_hip_common", "mean_tau_hip_diff", "mean_tau_ank_common",
    "mean_tau_ank_diff", "mean_sumzero_resid",
]

# per-step columns that get a mean_* roll-up (name -> episode column)
_ROLLUP = ["reward", "r_forward", "r_ctrl", "r_contact", "r_survive", "fwd_vel",
           "sat_frac", "clip_frac_cmd", "d_app_mean", "d_app_max", "torso_height",
           "d_hip_common", "d_hip_diff", "d_ank_common", "d_ank_diff",
           "tau_hip_common", "tau_hip_diff", "tau_ank_common", "tau_ank_diff",
           "sumzero_resid"]


def _modes(x):
    """(common, diff) power of a 4-vector: ``mean(x)**2`` and population ``var(x)``.

    These are the two eigen-directions of the coupling operator M = 11' - I on 4
    legs (eigenvalues +3 common, -1 difference) — the decomposition the E2
    loop-geometry derivation is written in.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x) ** 2), float(np.var(x))


def build_row(step_global, ep_step, done, fall, term_cause, tau, d_app, delivered,
              height, info, reward):
    """Pure function: per-step record -> the `_COLS` dict.

    Split out so the schema can be unit-tested without a simulator (see
    ``selftest``), and so ``diag_tier0.py`` can log identically.
    """
    d_app = np.asarray(d_app, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if d_app.shape == tau.shape and tau.shape == (_N_ACT,):
        raw = tau + d_app
        sat_frac = float(np.mean(np.abs(raw) > 1.0))
        delivered_absmean = float(np.mean(np.abs(delivered)))
        d_mean, d_max = float(np.mean(np.abs(d_app))), float(np.max(np.abs(d_app)))
        dhc, dhd = _modes(d_app[0::2])
        dac, dad = _modes(d_app[1::2])
        thc, thd = _modes(tau[0::2])
        tac, tad = _modes(tau[1::2])
        sumzero = float(abs(np.mean(tau[0::2])) + abs(np.mean(tau[1::2])))
        clip_cmd = float(np.mean(np.abs(tau) > 0.999))
        tau_absmean = float(np.mean(np.abs(tau)))
    else:                                   # non-PCR ant / shape mismatch
        sat_frac = delivered_absmean = tau_absmean = float("nan")
        d_mean = d_max = float("nan")
        dhc = dhd = dac = dad = thc = thd = tac = tad = float("nan")
        sumzero = clip_cmd = float("nan")

    return {
        "step_global": step_global, "ep_step": ep_step,
        "done": float(done), "fall": float(fall),
        "pcr_payload": float(info.get("pcr_payload", float("nan"))),
        "pcr_load": float(info.get("pcr_load", float("nan"))),
        "pcr_loadmax": float(info.get("pcr_loadmax", float("nan"))),
        "d_app_mean": d_mean, "d_app_max": d_max,
        "torso_height": height,
        "sat_frac": sat_frac, "tau_absmean": tau_absmean,
        "delivered_absmean": delivered_absmean,
        "reward": float(reward),
        "r_forward": float(info.get("reward_forward", float("nan"))),
        "r_ctrl": float(info.get("reward_ctrl", float("nan"))),
        "r_contact": float(info.get("reward_contact", float("nan"))),
        "r_survive": float(info.get("reward_survive", float("nan"))),
        "d_hip_common": dhc, "d_hip_diff": dhd,
        "d_ank_common": dac, "d_ank_diff": dad,
        "tau_hip_common": thc, "tau_hip_diff": thd,
        "tau_ank_common": tac, "tau_ank_diff": tad,
        "clip_frac_cmd": clip_cmd,
        "fwd_vel": float(info.get("reward_forward", float("nan"))),
        "sumzero_resid": sumzero,
        "pcr_clock": int(info.get("pcr_clock", -1)),
        "term_cause": term_cause,
    }


class _PcrDiagLogger:
    """One CSV per env instance (pid + in-process counter -> no contention),
    plus an episode roll-up CSV and an optional NPZ trajectory ring."""

    _counter = 0
    _lock = threading.Lock()

    def __init__(self, debug_dir="./pcr_diag", interval=1, dump_traj=False,
                 dump_every_k_episodes=10, dump_max_mb=512):
        self.interval = max(1, int(interval))
        os.makedirs(debug_dir, exist_ok=True)
        with _PcrDiagLogger._lock:
            inst = _PcrDiagLogger._counter
            _PcrDiagLogger._counter += 1
        self.tag = f"pid{os.getpid()}_inst{inst}"
        self.dir = debug_dir
        self.path = os.path.join(debug_dir, f"pcr_diag_{self.tag}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_COLS)
        self._row = 0

        self.ep_path = os.path.join(debug_dir, f"pcr_diag_{self.tag}_episodes.csv")
        self._ep_file = open(self.ep_path, "w", newline="", encoding="utf-8")
        self._ep_writer = csv.writer(self._ep_file)
        self._ep_writer.writerow(_EP_COLS)

        self.dump_traj = bool(dump_traj)
        self.dump_k = max(1, int(dump_every_k_episodes))
        self.dump_max_bytes = int(float(dump_max_mb) * 1024 * 1024)
        self._dumped = deque()          # (path, bytes) ring
        self._dumped_bytes = 0

        print(f"[PCR-DIAG] failure trace -> {os.path.abspath(self.path)} "
              f"(every {self.interval} step(s))", flush=True)
        print(f"[PCR-DIAG] episode roll-up -> {os.path.abspath(self.ep_path)}",
              flush=True)
        if self.dump_traj:
            print(f"[PCR-DIAG] E5 trajectory dump -> {os.path.abspath(debug_dir)}"
                  f"/traj_{self.tag}_ep*.npz  (1 episode in {self.dump_k}, "
                  f"cap {dump_max_mb} MB, ring)", flush=True)

    # -- per-step ----------------------------------------------------------
    def log(self, row):
        self._row += 1
        # always log the terminal step of an episode; subsample the rest.
        if row["done"] < 0.5 and self._row % self.interval != 0:
            return
        self._writer.writerow([row[c] for c in _COLS])
        self._file.flush()

    # -- per-episode -------------------------------------------------------
    def log_episode(self, ep_index, rows):
        if not rows:
            return
        rec = [ep_index, rows[-1]["step_global"],
               float(np.sum([r["reward"] for r in rows])), len(rows),
               rows[-1]["term_cause"],
               rows[0]["pcr_payload"], rows[-1]["pcr_payload"]]
        for c in _ROLLUP:
            vals = np.asarray([r[c] for r in rows], dtype=np.float64)
            rec.append(float(np.nanmean(vals)) if vals.size else float("nan"))
        self._ep_writer.writerow(rec)
        self._ep_file.flush()

    # -- E5 trajectory dump ------------------------------------------------
    def maybe_dump(self, ep_index, traj):
        """Write one whole episode (contiguous! see the module note) as float32."""
        if not self.dump_traj or (ep_index % self.dump_k) != 0 or not traj["act"]:
            return
        path = os.path.join(self.dir, f"traj_{self.tag}_ep{ep_index:06d}.npz")
        arrs = {k: np.asarray(v, dtype=np.float32) for k, v in traj.items()
                if k != "ep_id"}
        arrs["ep_id"] = np.full(len(traj["act"]), ep_index, dtype=np.int64)
        np.savez_compressed(path, **arrs)
        size = os.path.getsize(path)
        self._dumped.append((path, size))
        self._dumped_bytes += size
        while self._dumped_bytes > self.dump_max_bytes and len(self._dumped) > 1:
            old, sz = self._dumped.popleft()      # ring: drop the oldest episode
            try:
                os.remove(old)
                self._dumped_bytes -= sz
            except OSError:
                break

    def close(self):
        for f in (getattr(self, "_file", None), getattr(self, "_ep_file", None)):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass


class PcrDiagMujocoMulti(MujocoMulti):
    """``MujocoMulti`` + a read-only per-step failure trace. Dynamics untouched."""

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        cfg = dict(kwargs["env_args"].get("pcr_diag_cfg", {}))
        self._diag = _PcrDiagLogger(
            debug_dir=cfg.get("dir", "./pcr_diag"),
            interval=int(cfg.get("interval", 10)),
            dump_traj=bool(cfg.get("dump_traj", False)),
            dump_every_k_episodes=int(cfg.get("dump_every_k_episodes", 10)),
            dump_max_mb=float(cfg.get("dump_max_mb", 512)),
        )
        self._ep_step = 0
        self._step_global = 0
        self._ep_index = 0
        self._ep_rows = []
        self._traj = self._new_traj()
        self._prev_obs = None

    @staticmethod
    def _new_traj():
        return {k: [] for k in ("obs", "act", "d_applied", "d_next", "payload",
                                "rew_terms", "ep_id")}

    def step(self, actions):
        # o_t: the obs the POLICY saw before acting. MujocoMulti.step returns the
        # POST-step obs, so E5's feature/target pairing needs this carried over.
        pre_obs = self._prev_obs

        # fallback for a pre-campaign ant.py that lacks pcr_d_applied: the load
        # about to be applied is env._d before the step.
        d_app_fallback = np.asarray(getattr(self.env, "_d", np.zeros(1)),
                                    dtype=np.float64).copy()

        obs_n, state_n, rewards, dones, infos, avail = super().step(actions)
        self._prev_obs = obs_n

        self._ep_step += 1
        self._step_global += 1
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        done_env = bool(np.all(dones))
        # a "fall" = real termination (height out of bounds), not a time-limit truncation.
        bad = bool(info0.get("bad_transition", False))
        fall = done_env and not bad

        # The COMMANDED torque is what the env actually received: with a Tier-0
        # probe installed that is the probe's output, not `actions`. Never log the
        # nominal action as if it were the command.
        cmd = info0.get("probe_commanded_action")
        if cmd is None:
            cmd = np.concatenate([
                np.asarray(actions[i][: self.true_action_space[i].low.shape[0]],
                           dtype=np.float64) for i in range(self.n_agents)
            ])
        tau = np.clip(np.asarray(cmd, dtype=np.float64).reshape(-1), -1.0, 1.0)
        d_app = np.asarray(info0.get("pcr_d_applied", d_app_fallback),
                           dtype=np.float64).reshape(-1)
        d_next = np.asarray(info0.get("pcr_d_next", np.full(_N_ACT, np.nan)),
                            dtype=np.float64).reshape(-1)
        delivered = (np.clip(tau + d_app, -1.0, 1.0) if d_app.shape == tau.shape
                     else np.full_like(tau, np.nan))

        height = float("nan")
        finite = True
        try:
            sv = self.env.state_vector()
            height = float(sv[2])
            finite = bool(np.isfinite(sv).all())
        except Exception:
            pass

        if not done_env:
            term_cause = ""
        elif bad:
            term_cause = "timeout"
        elif not finite:
            term_cause = "nonfinite"
        elif height < 0.2:
            term_cause = "fall_low"
        elif height > 1.0:
            term_cause = "fall_high"
        else:
            term_cause = "other"

        row = build_row(self._step_global, self._ep_step, done_env, fall,
                        term_cause, tau, d_app, delivered, height, info0,
                        float(np.mean(np.asarray(rewards))))
        self._diag.log(row)
        self._ep_rows.append(row)

        if self._diag.dump_traj and pre_obs is not None:
            self._traj["obs"].append(np.asarray(pre_obs, dtype=np.float32))
            self._traj["act"].append(tau.astype(np.float32))
            self._traj["d_applied"].append(d_app.astype(np.float32))
            self._traj["d_next"].append(d_next.astype(np.float32))
            self._traj["payload"].append(np.float32(row["pcr_payload"]))
            self._traj["rew_terms"].append(np.asarray(
                [row["r_forward"], row["r_ctrl"], row["r_contact"],
                 row["r_survive"]], dtype=np.float32))

        if done_env:
            self._diag.log_episode(self._ep_index, self._ep_rows)
            self._diag.maybe_dump(self._ep_index, self._traj)
            self._ep_index += 1
            self._ep_rows = []
            self._traj = self._new_traj()
            self._ep_step = 0
        return obs_n, state_n, rewards, dones, infos, avail

    def reset(self, **kwargs):
        # A reset that is NOT preceded by done (e.g. the eval driver starting a
        # round) still closes the episode's books, else its rows leak into the next.
        if self._ep_rows:
            self._diag.log_episode(self._ep_index, self._ep_rows)
            self._diag.maybe_dump(self._ep_index, self._traj)
            self._ep_index += 1
            self._ep_rows = []
            self._traj = self._new_traj()
        self._ep_step = 0
        out = super().reset(**kwargs)
        self._prev_obs = out[0]
        return out

    def close(self):
        if getattr(self, "_diag", None) is not None:
            self._diag.close()
        super().close()


# ==========================================================================
#  V0 recorder-schema self-test (no simulator: build_row is a pure function)
# ==========================================================================
def selftest():
    from harl.envs.mamujoco.diag.report_io import DebugReport

    rep = DebugReport(os.path.join("diag_out", "v0", "v0_recorder.md"),
                      title="V0 — flight-recorder schema self-test",
                      subtitle="build_row is pure; no simulator needed")
    ok = True

    rep.h2("T1 — row keys match the declared column list exactly")
    info = dict(pcr_payload=0.7, pcr_load=0.3, pcr_loadmax=0.5, reward_forward=2.0,
                reward_ctrl=-0.4, reward_contact=-0.1, reward_survive=1.0,
                pcr_clock=1234)
    tau = np.array([0.9, -0.2, 0.5, 0.1, -0.7, 0.3, 0.2, -0.4])
    d = np.array([0.3, -0.1, 0.2, 0.05, -0.2, 0.1, 0.15, -0.05])
    row = build_row(10, 3, False, False, "", tau, d, np.clip(tau + d, -1, 1),
                    0.6, info, 2.5)
    missing = [c for c in _COLS if c not in row]
    extra = [k for k in row if k not in _COLS]
    good = not missing and not extra
    ok &= good
    rep.line(f"  columns declared={len(_COLS)}, produced={len(row)}; "
             f"missing={missing}, extra={extra}   {'OK' if good else 'FAIL'}")

    rep.h2("T2 — mode decomposition matches its definition")
    c, v = _modes(d[0::2])
    good = (abs(c - np.mean(d[0::2]) ** 2) < 1e-12
            and abs(v - np.var(d[0::2])) < 1e-12
            and abs(row["d_hip_common"] - c) < 1e-12
            and abs(row["d_hip_diff"] - v) < 1e-12)
    ok &= good
    rep.line(f"  d_hip_common = mean(h)^2 = {c:.6f}; d_hip_diff = var(h) = "
             f"{v:.6f}   {'OK' if good else 'FAIL'}")
    # the identity that makes the decomposition meaningful: total power splits
    tot = float(np.mean(d[0::2] ** 2))
    good = abs(tot - (c + v)) < 1e-12
    ok &= good
    rep.line(f"  mean(h^2) == common + diff : {tot:.6f} == {c + v:.6f}   "
             f"{'OK' if good else 'FAIL'}")

    rep.h2("T3 — sat_frac vs clip_frac_cmd are different things")
    tau2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 2 joints railed
    d2 = np.array([0.5, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])     # 1 exceeds authority
    r2 = build_row(1, 1, False, False, "", tau2, d2, np.clip(tau2 + d2, -1, 1),
                   0.6, info, 1.0)
    good = abs(r2["clip_frac_cmd"] - 2 / 8) < 1e-12 and abs(r2["sat_frac"] - 1 / 8) < 1e-12
    ok &= good
    rep.line(f"  clip_frac_cmd={r2['clip_frac_cmd']:.3f} (|a|>0.999: policy on the "
             f"rails), sat_frac={r2['sat_frac']:.3f} (|tau+d|>1: authority "
             f"exceeded)   {'OK' if good else 'FAIL'}")

    rep.h2("T4 — episode roll-up columns are all producible")
    missing = [c for c in _ROLLUP if c not in _COLS]
    good = not missing and len(_EP_COLS) == 7 + len(_ROLLUP)
    ok &= good
    rep.line(f"  roll-up sources missing from _COLS: {missing}; _EP_COLS="
             f"{len(_EP_COLS)} == 7 + {len(_ROLLUP)}   {'OK' if good else 'FAIL'}")

    rep.h2("SUMMARY")
    rep.verdict("V0 recorder schema", ok)
    rep.close()
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(selftest())
