"""ECL local harm-envelope adapter (agent-side, pure numpy) — spec [E] / §3.5.

The envelope `ε_i` is the *decentralized chart* of the equilibrium path: the
fraction of the agent's own-channel readout power that its **own** effort does
NOT explain. On the equilibrium path that residual is dominated by the
liability `ℓ_i`, so `ε_i` is a stable monotone correlate of `c(t)` — exactly the
kind of context the policy can condition on (inherited from ECHO-R P3: any stable
monotone correlate is as good as `c`). It is computed by a fixed O(1)/step
formula from the agent's own quantities only — no probe, no learned parts, no
access to anything but own obs + own action (Part 1 hygiene, Prohibition 5).

Per agent, all recursions are EMAs and persist across episodes; only the
readout's one-step pairing is reset at episode start (owned by the env shim).
"""

import numpy as np


class EnvelopeAdapter:
    """One instance per env; per-agent state in length-N arrays."""

    def __init__(self, n_agents, cfg=None):
        cfg = dict(cfg or {})
        self.N = int(n_agents)
        self.delta = float(cfg.get("delta", 1e-6))
        # own-effort -> own-readout running gain (fast; half-life ~200 steps)
        gain_hl = float(cfg.get("gain_halflife", 200))
        self.mu = 1.0 - 0.5 ** (1.0 / max(1.0, gain_hl))
        # residual/total power fractions (v2 C2.1: half-life 2000 -> 600 cuts lag)
        pow_hl = float(cfg.get("power_halflife", 600))
        self.mu_e = 1.0 - 0.5 ** (1.0 / max(1.0, pow_hl))
        # slow centering of the raw readout (remove DC/gait mean)
        cen_hl = float(cfg.get("center_halflife", 200))
        self.mu_c = 1.0 - 0.5 ** (1.0 / max(1.0, cen_hl))

        # v2 C2.2: range normalization — fast-attack / slow-release min/max tracker
        self.range_norm = bool(cfg.get("range_norm", True))
        q_fast_hl = float(cfg.get("q_fast_hl", 50))
        q_slow_hl = float(cfg.get("q_slow_hl", 80000))
        self.mu_qfast = 1.0 - 0.5 ** (1.0 / max(1.0, q_fast_hl))
        self.mu_qslow = 1.0 - 0.5 ** (1.0 / max(1.0, q_slow_hl))

        # persistent per-agent state
        self.ema_y = np.zeros(self.N)   # slow center of y
        self.Sxy = np.zeros(self.N)     # E[x1 * y]
        self.Sxx = np.zeros(self.N)     # E[x1^2]
        self.P_r = np.zeros(self.N)     # residual power
        self.P_y = np.zeros(self.N)     # total (centered) readout power
        self.eps_raw = np.zeros(self.N)  # residual-power fraction (raw ε)
        self.eps_norm = np.zeros(self.N) # range-normalized ε̃ (appended to obs)
        self.eps = np.zeros(self.N)      # the exposed value (ε̃ if range_norm else ε)
        self.q_lo = np.zeros(self.N)     # slow running min of ε
        self.q_hi = np.zeros(self.N)     # slow running max of ε
        self._q_init = False
        self.gain_last = np.zeros(self.N)

    def update(self, x1, y):
        """Update the envelope from this step's own effort and readout (§3.5).

        Args:
            x1: (N,) each agent's own effort coordinate (Ant: own hip torque, the
                commanded action at t).
            y:  (N,) each agent's own-channel readout (Ant: own hip qvel one-step
                delta), RAW (centering is done here).
        Returns:
            (N,) the updated envelope ``ε_i`` (residual-power fraction).
        """
        x1 = np.asarray(x1, dtype=np.float64).reshape(self.N)
        y = np.asarray(y, dtype=np.float64).reshape(self.N)

        yc = y - self.ema_y                                   # centered readout
        self.ema_y += self.mu_c * (y - self.ema_y)

        # running least-squares gain of own effort -> own readout
        self.Sxy += self.mu * (x1 * yc - self.Sxy)
        self.Sxx += self.mu * (x1 * x1 - self.Sxx)
        gain = self.Sxy / np.maximum(self.Sxx, self.delta)
        self.gain_last = gain

        r = yc - gain * x1                                    # own-explained residual
        self.P_r += self.mu_e * (r * r - self.P_r)
        self.P_y += self.mu_e * (yc * yc - self.P_y)
        eps = np.sqrt(self.P_r / np.maximum(self.P_y, self.delta))
        self.eps_raw = eps

        # range normalization (C2.2): ε̃ ∈ [0,1] anchored to the cycle envelope.
        # Because the driver revisits both extremes every cycle, (q_lo, q_hi) pin
        # to that envelope and the same c maps to the same ε̃ (time-consistent,
        # unlike a running z-score); the min-max tracker also absorbs the slow
        # attenuation of the physical channel as the gait improves.
        if not self._q_init:
            self.q_lo[:] = eps
            self.q_hi[:] = eps
            self._q_init = True
        lo_mu = np.where(eps < self.q_lo, self.mu_qfast, self.mu_qslow)
        hi_mu = np.where(eps > self.q_hi, self.mu_qfast, self.mu_qslow)
        self.q_lo += lo_mu * (eps - self.q_lo)
        self.q_hi += hi_mu * (eps - self.q_hi)
        self.eps_norm = np.clip(
            (eps - self.q_lo) / np.maximum(self.q_hi - self.q_lo, 0.01), 0.0, 1.0
        )

        self.eps = self.eps_norm if self.range_norm else self.eps_raw
        return self.eps.copy()

    def reset_episode(self):
        """Persistent across episodes (Prohibition 7); only the env shim resets
        the readout pairing. Nothing to do here."""
        return

    def diagnostics(self):
        return {
            "eps": self.eps.copy(),
            "eps_raw": self.eps_raw.copy(),
            "eps_norm": self.eps_norm.copy(),
            "q_lo": self.q_lo.copy(),
            "q_hi": self.q_hi.copy(),
            "P_r": self.P_r.copy(),
            "P_y": self.P_y.copy(),
            "gain": self.gain_last.copy(),
        }
