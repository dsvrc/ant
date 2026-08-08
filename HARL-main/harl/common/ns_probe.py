"""NS liveness probe — is the non-stationarity actually biting?

Attaches to EVERY on-policy arm, including plain blind HAPPO, which is the point:
PACT arms write ``pact_debug.csv`` and therefore show their own telemetry, but a
blind baseline writes nothing, so a silently-inert NS is invisible in exactly the
arm you most need to trust. This closes that hole.

It is pure logging. It reads the env's ``pcr_*`` info keys, writes
``ns_debug.csv``, prints a periodic ``[NS]`` line, and shouts if the disturbance is
not there. It never touches the control path, the buffer, or the loss, and it is a
complete no-op on envs that publish no ``pcr_*`` keys (SMAC, football, ...).

WHY THIS EXISTS (a real failure it would have caught in one line of output):
a run reported blind HAPPO scoring ~5.5k under a severity that Phase 1 had measured
as catastrophic (privileged controller: -99 return). The disturbance was not being
applied. Nothing in the training logs said so, because the only quantity anyone was
watching was the return -- and a return that looks *good* is the one failure mode
nobody investigates.

THE THREE FAILURE MODES IT SEPARATES
------------------------------------
  INERT       |d| ~ 0 throughout. The env file is not the deployed one, MASK=off,
              severity is 0, or a stale .pyc is being imported. The NS is absent.
  ASLEEP      |d| is small only because the DRIVER is: A(t) never leaves its trough
              over the whole window. The NS is wired up but the run is sampling one
              phase of a 40k-step cycle. (This is the same aliasing that made an
              eval win-rate read as a square wave.)
  ALIVE       A sweeps its range and |d| tracks it. The NS is doing its job; if the
              return is still high, that is a real result about the task.
"""

import csv
import os

import numpy as np

_KEYS = ("pcr_payload", "pcr_load", "pcr_loadmax", "pcr_sat_frac")


class NSLivenessProbe:
    """Accumulate the env's NS telemetry and report it, loudly when it is wrong."""

    def __init__(self, run_dir=None, report_every=200):
        self.report_every = int(report_every)
        self._n = 0                  # inserts seen since the last report
        self._acc = self._fresh()
        self._total = 0              # inserts ever
        self._active = None          # None until the first info with pcr_ keys
        self._warned_inert = False
        self._warned_asleep = False
        self._w = self._f = None
        if run_dir is not None:
            try:
                p = os.path.join(str(run_dir), "ns_debug.csv")
                self._f = open(p, "w", newline="", encoding="utf-8")
                self._w = csv.writer(self._f)
                self._w.writerow([
                    "insert", "A_mean", "A_min", "A_max", "d_mean", "d_max",
                    "sat_frac", "theta0", "theta1", "theta2", "n_samples",
                ])
            except Exception:
                self._w = self._f = None

    @staticmethod
    def _fresh():
        return {"A": [], "d": [], "dmax": [], "sat": [], "th": []}

    # ------------------------------------------------------------------ read
    def observe(self, infos):
        """``infos`` is the per-thread list the runner already has in insert()."""
        if self._active is False:
            return                                   # env publishes no pcr_* keys
        a = self._acc
        seen = False
        for per_thread in infos:
            d = per_thread[0] if isinstance(per_thread, (list, tuple, np.ndarray)) \
                else per_thread
            if not isinstance(d, dict) or "pcr_load" not in d:
                continue
            seen = True
            a["A"].append(float(d.get("pcr_payload", np.nan)))
            a["d"].append(float(d.get("pcr_load", np.nan)))
            a["dmax"].append(float(d.get("pcr_loadmax", np.nan)))
            a["sat"].append(float(d.get("pcr_sat_frac", np.nan)))
            th = d.get("pcr_theta")
            if th is not None:
                a["th"].append(np.asarray(th, dtype=np.float64).reshape(-1))
        if self._active is None:
            self._active = seen
            if not seen:
                return
        self._n += 1
        self._total += 1

    # ---------------------------------------------------------------- report
    def maybe_report(self, env_step=None):
        if not self._active or self._n < self.report_every:
            return
        a, self._acc, self._n = self._acc, self._fresh(), 0
        if not a["d"]:
            return

        def m(x, f=np.mean):
            x = np.asarray(x, dtype=np.float64)
            x = x[np.isfinite(x)]
            return float(f(x)) if x.size else float("nan")

        A_mean, A_min, A_max = m(a["A"]), m(a["A"], np.min), m(a["A"], np.max)
        d_mean, d_max = m(a["d"]), m(a["dmax"], np.max)
        sat = m(a["sat"])
        th = (np.mean(np.stack(a["th"], 0), axis=0) if a["th"]
              else np.full(3, float("nan")))
        tag = f"step={env_step}" if env_step is not None else f"n={self._total}"

        if self._w is not None:
            self._w.writerow([
                self._total, round(A_mean, 4), round(A_min, 4), round(A_max, 4),
                round(d_mean, 6), round(d_max, 6), round(sat, 5),
                *[round(float(v), 4) for v in (list(th) + [float("nan")] * 3)[:3]],
                len(a["d"]),
            ])
            self._f.flush()

        th_s = "" if not a["th"] else f" theta=[{th[0]:.2f} {th[1]:.2f} {th[2]:.2f}]"
        print(f"[NS] {tag} | A {A_min:.2f}-{A_max:.2f} (mean {A_mean:.2f}) | "
              f"|d| mean {d_mean:.4f} max {d_max:.4f} | sat {sat:.4f}{th_s}",
              flush=True)

        # --- INERT: the disturbance is simply not there ----------------------
        if d_max < 1e-6 and not self._warned_inert:
            self._warned_inert = True
            print(
                "\n" + "!" * 78 +
                "\n[NS][INERT] *** THE NON-STATIONARITY IS NOT BEING APPLIED. ***\n"
                f"  max |d| over {len(a['d'])} samples is {d_max:.2e}. Every number this\n"
                "  run produces is a STATIONARY-TASK number; do not compare it to any\n"
                "  arm that ran with the NS live. Check, in this order:\n"
                "    1. the [DIAG ENV] banner at the top of this log -- does SEVERITY\n"
                "       match what you passed? (a hardcoded literal in the env file\n"
                "       silently ignores ANT_PCR_SEVERITY)\n"
                "    2. ANT_PCR_MASK -- 'off' zeroes the coupling entirely\n"
                "    3. is gym/envs/mujoco/ant.py actually the deployed file, and is\n"
                "       there a stale __pycache__/ant.cpython-*.pyc beside it?\n"
                + "!" * 78 + "\n", flush=True)

        # --- ASLEEP: wired up, but the driver never woke during this window ---
        elif A_max < 0.05 and not self._warned_asleep:
            self._warned_asleep = True
            print(
                f"[NS][ASLEEP] the driver A(t) stayed in [{A_min:.3f}, {A_max:.3f}] for "
                f"this whole window, so |d| is small for a CLOCK reason, not a severity\n"
                f"  one. The driver period is long (40k steps); a short window samples "
                f"one phase of it. Read a full cycle, or freeze the driver to measure "
                f"severity.", flush=True)

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
