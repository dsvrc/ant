"""Runner for PACT — training is bit-identical to HAPPO; it ONLY adds logging.

PACT's whole mechanism lives in the env wrapper (``PactMujocoMulti``): the extra
β-control action dimension, the exact ``x2`` recursion, and the compensation are
all env-side, so the HAPPO training loop, actor, critic and buffers are untouched
(the extra action dim is learned by PPO like any other continuous action). This
subclass therefore changes no training computation — it reads the per-step
diagnostics the wrapper stashes into ``info`` and writes ``pact_debug.csv`` (one
row per rollout) plus the one hard gate.

Columns answer three questions:

  IS THE ARITHMETIC EXACT?  (the wiring certificate — the ONLY hard gate)
    ``cos_x2_dnext``   mean PER-STEP cosine similarity between the x2 and
                       pcr_d_next 8-vectors, over payload>0.3 steps. Per step
                       d ≈ c·x2 (parallel vectors), so this is ~1 when the peer-
                       action recursion is wired right and ~0 under any index /
                       reset / one-step-timing bug — and, being normalized per
                       step, it is invariant to how c drifts across steps (a
                       Pearson corr POOLED across a cycle would read only ~0.95
                       from the c-fan even when every point is exactly d=c·x2, so
                       cosine, not pooled corr, is the correct gate). It tests
                       ARITHMETIC and is GAIT-INDEPENDENT — hence safe to hard-
                       abort on, unlike the gait-dependent readout scans that
                       burned earlier methods. < 0.999 late ⇒ miswired; fix it.

  IS β TRACKING THE HIDDEN DRIVER?  (the whole ballgame — see spec §1 risk)
    ``beta`` / ``c_true`` / ``beta_minus_c``  β vs the true c = A·σ (diagnostic).
    ``beta_peak`` / ``beta_trough``           β must rise at the peak and → 0 at
                                              the trough (leftover comp at the
                                              trough is self-inflicted harm).
    ``resid_absmean``  mean|pcr_d_next − β·x2| — the disturbance the policy STILL
                       feels. Small ⇒ compensation is biting.

  WHERE IS THE RETURN?  reward decomposition + ep_len + actor_std (as in O-MAX).

If ``corr_x2_dnext≈1`` and β tracks the payload, PACT is doing exactly what O1 did
with the true d — and the return should approach the O1 ceiling.
"""

import csv
import os

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner

_COLS = [
    "env_step", "rollout", "payload_mean",
    # β vs the hidden driver (diagnostic truth)
    "beta", "c_true", "beta_minus_c",
    # the arithmetic exactness gate + compensation state
    "cos_x2_dnext", "x2_absmean", "resid_absmean", "u_clip_frac", "dbeta_absmean",
    # where the return is
    "r_total", "r_forward", "r_ctrl", "r_contact", "r_survive",
    "ep_len_mean", "actor_std",
    # phase split (does β modulate with the cycle?)
    "beta_peak", "beta_trough", "r_total_peak", "r_total_trough",
]


class OnPolicyPactRunner(OnPolicyHARunner):
    """HAPPO + PACT diagnostic logging + the arithmetic exactness gate."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyPactRunner, self).__init__(args, algo_args, env_args)
        self.T = int(self.algo_args["train"]["episode_length"])
        self.nt = int(self.algo_args["train"]["n_rollout_threads"])

        # PACT-1 declares pact1_cfg; PACT declares pact_cfg. Read whichever is
        # present -- without this, a PACT-1 run silently inherits PACT's 0.999 gate,
        # which a noisy PREDICTION (rather than an exact waveform) cannot meet, and
        # the run aborts on a correct implementation.
        cfg = dict(env_args.get("pact1_cfg", env_args.get("pact_cfg", {})))
        # σ is DIAGNOSTIC ONLY (the c_true / beta_minus_c columns); it never enters
        # the method. Read from the env-var the deployed ant.py honours, default 0.45.
        self.sigma_diag = float(
            cfg.get("sigma_diag", os.environ.get("ANT_PCR_SEVERITY", 0.45))
        )
        # The ONE hard gate (arithmetic, gait-independent). Default: abort on a
        # miswire — this is safe here precisely because the quantity is arithmetic
        # (not a gait-dependent readout). Set gate_abort:false to warn-only.
        self._gate_abort = bool(cfg.get("gate_abort", True))
        self._gate_min_corr = float(cfg.get("gate_min_corr", 0.999))
        self._gate_after_steps = int(cfg.get("gate_after_steps", 200000))
        self._gate_min_samples = int(cfg.get("gate_min_samples", 2000))
        self._gate_done = False

        self._roll = 0
        self._acc = self._fresh()
        self._ep_lens = []
        self._ep_len_running = np.zeros(self.nt)
        self._pact_step = 0

        self._dbg = self._dbg_w = None
        if getattr(self, "run_dir", None) is not None:
            p = os.path.join(str(self.run_dir), "pact_debug.csv")
            self._dbg = open(p, "w", newline="", encoding="utf-8")
            self._dbg_w = csv.writer(self._dbg)
            self._dbg_w.writerow(_COLS)
            print(f"[PACT] diagnostic trace -> {os.path.abspath(p)}", flush=True)

    @staticmethod
    def _fresh():
        keys = [
            "payload", "beta", "dbeta", "x2_absmean", "resid", "u_clip",
            "r_total", "r_forward", "r_ctrl", "r_contact", "r_survive",
        ]
        acc = {k: [] for k in keys}
        # phase-binned
        for ph in ("peak", "trough"):
            acc[f"beta_{ph}"] = []
            acc[f"r_total_{ph}"] = []
        # flat arrays for the exactness correlation (payload>0.3 only)
        acc["x2_flat"] = []
        acc["dnext_flat"] = []
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
            beta = float(d.get("pact_beta", np.nan))
            rt = float(np.asarray(rewards[i]).reshape(-1)[0])
            a["payload"].append(pay)
            a["beta"].append(beta)
            a["dbeta"].append(float(d.get("pact_dbeta", np.nan)))
            a["x2_absmean"].append(float(d.get("pact_x2_absmean", np.nan)))
            a["resid"].append(float(d.get("pact_resid_absmean", np.nan)))
            a["u_clip"].append(float(d.get("pact_u_clip", np.nan)))
            a["r_total"].append(rt)
            a["r_forward"].append(float(d.get("reward_forward", np.nan)))
            a["r_ctrl"].append(float(d.get("reward_ctrl", np.nan)))
            a["r_contact"].append(float(d.get("reward_contact", np.nan)))
            a["r_survive"].append(float(d.get("reward_survive", np.nan)))
            if np.isfinite(pay):
                ph = "peak" if pay > 0.7 else ("trough" if pay < 0.1 else None)
                if ph is not None:
                    a[f"beta_{ph}"].append(beta)
                    a[f"r_total_{ph}"].append(rt)
                # exactness correlation: pair x2 with pcr_d_next where the driver
                # is meaningfully on (payload>0.3), so the pair has signal.
                if pay > 0.3:
                    x2 = d.get("pact_x2")
                    dn = d.get("pact_d_next")
                    if x2 is not None and dn is not None:
                        a["x2_flat"].append(np.asarray(x2, dtype=np.float64))
                        a["dnext_flat"].append(np.asarray(dn, dtype=np.float64))

        self._ep_len_running += 1
        dones_env = np.all(dones, axis=1)
        for i in range(self.nt):
            if dones_env[i]:
                self._ep_lens.append(float(self._ep_len_running[i]))
                self._ep_len_running[i] = 0

        super(OnPolicyPactRunner, self).insert(data)

        self._pact_step += 1
        if self._pact_step % self.T == 0:
            self._write_row()

    @staticmethod
    def _m(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.mean(x)) if x else float("nan")

    def _waveform_cos(self, a):
        """Mean per-step cosine(x2, pcr_d_next) over the rollout's payload>0.3
        steps — the c-invariant exactness metric (see the module docstring)."""
        if not a["x2_flat"] or not a["dnext_flat"]:
            return float("nan"), 0
        X2 = np.asarray(a["x2_flat"], dtype=np.float64)      # (n_steps, n_act)
        D = np.asarray(a["dnext_flat"], dtype=np.float64)
        num = np.sum(X2 * D, axis=1)
        nx = np.linalg.norm(X2, axis=1)
        nd = np.linalg.norm(D, axis=1)
        m = (nx > 1e-3) & (nd > 1e-3) & np.isfinite(num)
        if m.sum() < 10:
            return float("nan"), int(m.sum())
        cos = num[m] / (nx[m] * nd[m])
        return float(np.mean(cos)), int(m.sum())

    def _write_row(self):
        self._roll += 1
        a = self._acc
        m = self._m
        step = self._roll * self.T * self.nt
        corr, n_corr = self._waveform_cos(a)
        payload_mean = m(a["payload"])
        beta_mean = m(a["beta"])
        c_true = self.sigma_diag * payload_mean if np.isfinite(payload_mean) else float("nan")
        beta_minus_c = beta_mean - c_true if np.isfinite(c_true) else float("nan")

        if self._dbg_w is not None:
            row = [
                step, self._roll, round(payload_mean, 4),
                round(beta_mean, 5), round(c_true, 5), round(beta_minus_c, 5),
                round(corr, 6) if np.isfinite(corr) else corr,
                round(m(a["x2_absmean"]), 6), round(m(a["resid"]), 6),
                round(m(a["u_clip"]), 5), round(m(a["dbeta"]), 6),
                round(m(a["r_total"]), 4), round(m(a["r_forward"]), 4),
                round(m(a["r_ctrl"]), 4), round(m(a["r_contact"]), 4),
                round(m(a["r_survive"]), 4),
                round(self._m(self._ep_lens), 1) if self._ep_lens else float("nan"),
                round(self._actor_std(), 5),
                round(m(a["beta_peak"]), 5), round(m(a["beta_trough"]), 5),
                round(m(a["r_total_peak"]), 4), round(m(a["r_total_trough"]), 4),
            ]
            self._dbg_w.writerow(row)
            self._dbg.flush()

        if self._roll % 10 == 1:
            print(
                f"[PACT dbg] roll={self._roll} step={step} | GATE cos(x2,d_next)="
                f"{corr:.4f} (want ~1, n={n_corr}) | beta={beta_mean:.3f} "
                f"c_true={c_true:.3f} (peak {m(a['beta_peak']):.3f} / trough "
                f"{m(a['beta_trough']):.3f}) | resid={m(a['resid']):.4f} "
                f"u_clip={m(a['u_clip']):.3f} | rew {m(a['r_total']):.2f} "
                f"ep_len={round(self._m(self._ep_lens), 0) if self._ep_lens else float('nan')} "
                f"std={self._actor_std():.3f}",
                flush=True,
            )

        self._check_gate(step, corr, n_corr)
        self._acc = self._fresh()
        self._ep_lens = []

    def _check_gate(self, step, corr, n_corr):
        """The ONE hard gate: arithmetic exactness of x2 vs the true disturbance.

        Fires once, at the first rollout past ``gate_after_steps`` with enough
        payload>0.3 samples. ``corr`` here is the mean per-step COSINE(x2, d_next)
        — a gait-INDEPENDENT arithmetic fact — so a hard abort is diagnosing a real
        wiring bug, categorically different from the gait-dependent readout-scan
        aborts that cost earlier methods good runs. Set ``pact_cfg.gate_abort:
        false`` to downgrade to a warning.
        """
        if self._gate_done or step < self._gate_after_steps:
            return
        if not np.isfinite(corr) or n_corr < self._gate_min_samples:
            return  # defer until a rollout has enough signal to judge
        self._gate_done = True
        if corr >= self._gate_min_corr:
            print(f"[PACT][GATE PASS] cos(x2, pcr_d_next)={corr:.5f} >= "
                  f"{self._gate_min_corr} at step {step}: the peer-action waveform "
                  f"is exact. The only unknown left is the scalar beta.", flush=True)
            return
        msg = (
            f"[PACT][GATE FAIL] cos(x2, pcr_d_next)={corr:.5f} < {self._gate_min_corr} "
            f"at step {step} (n={n_corr}). x2 does NOT match the env's true "
            f"disturbance waveform — the wrapper is miswired (index order, episode "
            f"reset masking, or the one-step timing). This is an arithmetic fact, "
            f"not a gait effect: FIX the wiring, do not reinterpret."
        )
        if self._gate_abort:
            raise AssertionError(msg)
        print("[PACT][WARN] " + msg + "  (gate_abort=false: continuing anyway.)",
              flush=True)

    def close(self):
        if self._dbg is not None:
            try:
                self._dbg.close()
            except Exception:
                pass
        super(OnPolicyPactRunner, self).close()
