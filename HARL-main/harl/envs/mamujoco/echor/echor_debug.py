"""Per-step debug tracer for ECHO-R (opt-in).

Focused on the handful of quantities that actually explain whether the method is
working, chosen from a first diagnostic run where the raw estimate was pure
noise:

* `pcr_payload` -- the ground-truth driver A(t). The estimate must track THIS.
* `pcr_load`    -- the true liability |d|; is the non-stationarity even active
                   yet (it ramps up over the payload period)?
* `c_hat_i`     -- the per-agent estimate actually fed to the policy.
* `H_i`         -- the echo-path gain (the thing being measured); ~0 ⇒ the probe
                   is not reaching the liability.
* `snr_G_i`     -- **direct-channel SNR** = |E[G]|/std[G]. This is the root-cause
                   readout: `< 1` means the agent cannot even detect its OWN
                   probe, so the ratio H/G is noise and `c_hat` is meaningless.
                   Raise `eps` / `lam_halflife` until this climbs above 1.
* `snr_H_i`     -- echo-channel SNR; when it exceeds 1 the estimate is real.
* `reward`,`done` -- outcome / episode boundaries.

Each env instance writes its own file (`echor_debug_pid<PID>_inst<N>.csv`) so the
parallel workers never contend. For a clean focused session run with
`n_rollout_threads=1 n_eval_rollout_threads=1 debug_interval=1` -> two files.

Diagnostic sink only: never read back into the estimator or policy (Part 10).
"""

import os
import csv
import time
import threading

import numpy as np

# Per-agent quantities, expanded to one column per agent (``<key>_<i>``).
_PER_AGENT_KEYS = ["c_hat", "H", "snr_G", "snr_H"]
_SCALAR_KEYS = ["t"]                                   # adapter clock
_EXTRA_KEYS = ["pcr_payload", "pcr_load", "reward", "done"]


class EchoRDebugLogger:
    """CSV tracer; one instance per environment wrapper."""

    _counter = 0
    _lock = threading.Lock()

    def __init__(self, n_agents, debug_dir="./echor_debug", interval=1, tag=""):
        self.n = int(n_agents)
        self.interval = max(1, int(interval))
        os.makedirs(debug_dir, exist_ok=True)
        with EchoRDebugLogger._lock:
            inst = EchoRDebugLogger._counter
            EchoRDebugLogger._counter += 1
        name = f"echor_debug_pid{os.getpid()}_inst{inst}"
        if tag:
            name += f"_{tag}"
        self.path = os.path.join(debug_dir, name + ".csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._wrote_header = False
        self._row = 0
        print(f"[ECHO-R] debug trace -> {os.path.abspath(self.path)} "
              f"(every {self.interval} step(s))")

    def _write_header(self):
        cols = list(_SCALAR_KEYS)
        for k in _PER_AGENT_KEYS:
            cols += [f"{k}_{i}" for i in range(self.n)]
        cols += list(_EXTRA_KEYS)
        self._writer.writerow(cols)
        self._wrote_header = True

    def log(self, snap, extra):
        """Append one row if this step is on the subsample grid.

        Args:
            snap: (dict) EchoRAdapter.debug_snapshot() output.
            extra: (dict) ground-truth / outcome values keyed by ``_EXTRA_KEYS``.
        """
        self._row += 1
        if self._row % self.interval != 0:
            return
        if not self._wrote_header:
            self._write_header()
        row = [int(snap["t"])]
        for k in _PER_AGENT_KEYS:
            row.extend(float(v) for v in np.asarray(snap[k]).ravel())
        row.extend(float(extra.get(k, float("nan"))) for k in _EXTRA_KEYS)
        self._writer.writerow(row)
        self._file.flush()  # flush every write: the trace survives a crash/kill

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
