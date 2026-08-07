"""Runner for O-MAX — training is bit-identical to HAPPO; it ONLY adds logging.

The ceiling ladder needs to answer two questions about why a fully-equipped arm
tops out where it does, and the stock HAPPO runner writes nothing but the eval
return. This subclass reads the per-step diagnostics that ``OmaxMujocoMulti``
stashes into ``info`` and writes ``omax_debug.csv`` (one row per rollout) plus a
peak-vs-trough split, without touching a single training computation.

The two questions, and the columns that answer them:

  Q1 — IS THE DISTURBANCE ACTUALLY REMOVED FROM WHAT THE POLICY EXPERIENCES?
       (i.e. is the privileged info truly delivered?)
    ``d_absmean/absmax``   how big is the disturbance there is to cancel
    ``residual_absmean``   |delivered − clip(a_intended)| — the disturbance the
                           policy STILL feels. ≈0 ⇒ it experiences the stationary
                           env exactly; >0 ⇒ the comp is LEAKING it through.
    ``u_clip_frac``        fraction of joints where the compensated command railed
                           (|a−d|>1) — the only place residual can be nonzero.
    ``a_absmax``           is the policy itself saturating (driving the rails)?

  Q2 — IS THE MECHANISM USING THE INFORMATION CORRECTLY?
    ``timing_err_max``     |d_used − d_applied| — did the comp cancel the RIGHT d?
                           Must be ~0; nonzero ⇒ an off-by-one / reset bug.
    reward decomposition   r_forward / r_ctrl / r_contact / r_survive — WHERE the
                           return is lost vs a stationary walker (bad gait? the
                           control cost of compensating? falling over?).
    ``ep_len``             short episodes ⇒ the Ant is falling.
    ``actor_std``          has exploration collapsed?

Read it as: if ``timing_err≈0`` and ``residual≈0`` then the info IS delivered and
used correctly, and a low return can ONLY be the reward decomposition / optimization
(the ceiling is genuinely ~that number). If ``residual`` is large at the peak, the
comp is saturating and the disturbance is NOT fully cancelled — the info is not
being delivered on the strong gait, and that is the frontier biting.
"""

import csv
import os

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner

_COLS = [
    "env_step", "rollout", "payload_mean",
    # Q1: is the disturbance removed from what the policy sees?
    "d_absmean", "d_absmax", "residual_absmean", "residual_absmax",
    "u_clip_frac", "a_absmax", "u_minus_a",
    # Q2: is the mechanism using the right info + where is the return lost?
    "timing_err_max", "r_total", "r_forward", "r_ctrl", "r_contact", "r_survive",
    "ep_len_mean", "actor_std",
    # peak vs trough (WHERE it breaks): residual + reward split by payload phase
    "residual_peak", "residual_trough", "r_total_peak", "r_total_trough",
    "r_ctrl_peak", "r_ctrl_trough", "r_forward_peak", "r_forward_trough",
]


class OnPolicyOmaxRunner(OnPolicyHARunner):
    """HAPPO + O-MAX diagnostic logging (training untouched)."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyOmaxRunner, self).__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])
        self._omax_step = 0
        self._roll = 0
        self._acc = self._fresh()
        self._ep_lens = []
        self._ep_len_running = np.zeros(self.nt)
        self._dbg = self._dbg_w = None
        if getattr(self, "run_dir", None) is not None:
            p = os.path.join(str(self.run_dir), "omax_debug.csv")
            self._dbg = open(p, "w", newline="", encoding="utf-8")
            self._dbg_w = csv.writer(self._dbg)
            self._dbg_w.writerow(_COLS)
            print(f"[OMAX] diagnostic trace -> {os.path.abspath(p)}", flush=True)

    @staticmethod
    def _fresh():
        keys = [
            "payload", "d_absmean", "d_absmax", "residual", "residual_max",
            "u_clip", "a_absmax", "u_minus_a", "timing_err",
            "r_total", "r_forward", "r_ctrl", "r_contact", "r_survive",
        ]
        acc = {k: [] for k in keys}
        # peak/trough-binned lists
        for ph in ("peak", "trough"):
            for k in ("residual", "r_total", "r_ctrl", "r_forward"):
                acc[f"{k}_{ph}"] = []
        return acc

    def _actor_std(self):
        try:
            ao = self.actor[0].actor.act.action_out
            import torch
            with torch.no_grad():
                std = torch.sigmoid(ao.log_std / ao.std_x_coef) * ao.std_y_coef
                if getattr(ao, "std_floor", 0.0) > 0.0:
                    std = std.clamp(min=ao.std_floor)
                return float(std.mean().cpu())
        except Exception:
            return float("nan")

    def insert(self, data):
        infos = data[4]
        rewards = data[2]
        dones = data[3]
        a = self._acc
        for i in range(len(infos)):
            d = infos[i][0] if isinstance(infos[i], (list, tuple, np.ndarray)) else infos[i]
            if not isinstance(d, dict):
                continue
            pay = float(d.get("pcr_payload", np.nan))
            res = float(d.get("omax_residual_absmean", np.nan))
            rt = float(np.asarray(rewards[i]).reshape(-1)[0])
            rc = float(d.get("reward_ctrl", np.nan))
            rf = float(d.get("reward_forward", np.nan))
            a["payload"].append(pay)
            a["d_absmean"].append(float(d.get("omax_d_absmean", np.nan)))
            a["d_absmax"].append(float(d.get("omax_d_absmax", np.nan)))
            a["residual"].append(res)
            a["residual_max"].append(float(d.get("omax_residual_absmax", np.nan)))
            a["u_clip"].append(float(d.get("omax_u_clip", np.nan)))
            a["a_absmax"].append(float(d.get("omax_a_absmax", np.nan)))
            a["u_minus_a"].append(float(d.get("omax_u_minus_a", np.nan)))
            a["timing_err"].append(float(d.get("omax_timing_err", np.nan)))
            a["r_total"].append(rt)
            a["r_forward"].append(rf)
            a["r_ctrl"].append(rc)
            a["r_contact"].append(float(d.get("reward_contact", np.nan)))
            a["r_survive"].append(float(d.get("reward_survive", np.nan)))
            if np.isfinite(pay):
                ph = "peak" if pay > 0.7 else ("trough" if pay < 0.1 else None)
                if ph is not None:
                    a[f"residual_{ph}"].append(res)
                    a[f"r_total_{ph}"].append(rt)
                    a[f"r_ctrl_{ph}"].append(rc)
                    a[f"r_forward_{ph}"].append(rf)

        # episode-length tracking (falling ⇒ short episodes)
        self._ep_len_running += 1
        dones_env = np.all(dones, axis=1)
        for i in range(self.nt):
            if dones_env[i]:
                self._ep_lens.append(float(self._ep_len_running[i]))
                self._ep_len_running[i] = 0

        super(OnPolicyOmaxRunner, self).insert(data)

        self._omax_step += 1
        if self._omax_step % self.T == 0:
            self._write_row()

    @staticmethod
    def _m(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.mean(x)) if x else float("nan")

    def _write_row(self):
        if self._dbg_w is None:
            self._acc = self._fresh()
            return
        self._roll += 1
        a = self._acc
        m = self._m
        step = self._roll * self.T * self.nt
        row = [
            step, self._roll, round(m(a["payload"]), 4),
            round(m(a["d_absmean"]), 5), round(m(a["d_absmax"]), 5),
            round(m(a["residual"]), 6), round(m(a["residual_max"]), 6),
            round(m(a["u_clip"]), 5), round(m(a["a_absmax"]), 4),
            round(m(a["u_minus_a"]), 5),
            round(m(a["timing_err"]), 8),
            round(m(a["r_total"]), 4), round(m(a["r_forward"]), 4),
            round(m(a["r_ctrl"]), 4), round(m(a["r_contact"]), 4),
            round(m(a["r_survive"]), 4),
            round(self._m(self._ep_lens), 1) if self._ep_lens else float("nan"),
            round(self._actor_std(), 5),
            round(m(a["residual_peak"]), 6), round(m(a["residual_trough"]), 6),
            round(m(a["r_total_peak"]), 4), round(m(a["r_total_trough"]), 4),
            round(m(a["r_ctrl_peak"]), 4), round(m(a["r_ctrl_trough"]), 4),
            round(m(a["r_forward_peak"]), 4), round(m(a["r_forward_trough"]), 4),
        ]
        self._dbg_w.writerow(row)
        self._dbg.flush()
        # a compact human-readable heartbeat every ~10 rollouts
        if self._roll % 10 == 1:
            print(
                f"[OMAX dbg] roll={self._roll} step={step} | Q2 timing_err="
                f"{row[11]:.2e} (want 0) | Q1 residual={row[5]:.4f} "
                f"(peak {row[19]:.4f} / trough {row[20]:.4f}, want ~0) "
                f"u_clip={row[7]:.3f} a_absmax={row[8]:.2f} | rew {row[12]:.2f} "
                f"(fwd {row[13]:.2f} ctrl {row[14]:.2f} surv {row[16]:.2f}) "
                f"ep_len={row[17]} std={row[18]:.3f}",
                flush=True,
            )
        self._acc = self._fresh()
        self._ep_lens = []

    def close(self):
        if self._dbg is not None:
            try:
                self._dbg.close()
            except Exception:
                pass
        super(OnPolicyOmaxRunner, self).close()
