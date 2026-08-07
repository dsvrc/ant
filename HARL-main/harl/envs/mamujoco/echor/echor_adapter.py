"""ECHO-R adapter -- framework-agnostic, pure-numpy driver estimator.

Implements the *agent side* of ECHO-R exactly as specified in
``ECHO-R_implementation_spec.md`` (Parts 3 and 4): an orthogonal-code active
probe injected into the coupling channel, a dual (own / echo) lag-summed
demodulator, and a ratiometric, anchored estimate ``c_hat`` of the hidden
effective severity ``c(t) = A(t)*sigma``.

Design summary (why every piece is the way it is -- Part 2):
  * P1 -- estimate only the slow scalar ``c``; the fast liability structure is
    left to the host policy, which conditions on ``c_hat``.
  * P2 -- the probe is information-theoretically necessary: ``c`` is
    passively unidentifiable from one agent's marginals.
  * P3 -- only a stable *monotone correlate* of ``c`` is needed; constant gain
    errors / offsets / truncated-lag biases are cosmetic.
  * P4 -- the ratio ``H_hat / G_hat`` cancels the unknown, state-dependent
    plant gain ``G_i`` (including its slow drift over training and HARL's
    per-vector obs normalisation), because both paths traverse the same plant.
  * P5 -- the ``j != i`` exclusion makes the own-code channel a clean
    self-calibration of ``G_i`` and keeps the echo channel uncontaminated.

The class holds per-agent state in arrays indexed by agent; a single instance
serves all ``N`` agents of one environment (they step in lockstep, so the probe
clock is common knowledge without communication).  Nothing here reads hidden
state, the env clock, the liability, other agents' observations, or any ``info``
key (information-hygiene rules, Part 1 / Part 10).
"""

import numpy as np

# Primitive LFSR feedback taps for maximum-length sequences (Appendix A).
# degree -> (period L, taps).  taps are 1-indexed bit positions XOR-ed together.
_MSEQ_TAPS = {
    7: (127, (7, 6)),
    8: (255, (8, 6, 5, 4)),
    9: (511, (9, 5)),
    10: (1023, (10, 7)),
}
_L_TO_DEGREE = {L: d for d, (L, _) in _MSEQ_TAPS.items()}


def m_sequence(degree, taps, seed=1):
    """Fibonacci LFSR over GF(2); output mapped {0,1} -> {+1,-1} (Appendix A).

    Returns an int8 numpy array of period ``L = 2**degree - 1`` with values +/-1,
    near-delta autocorrelation and near-zero cross-correlation between distinct
    cyclic shifts -- the property that makes demodulation reject everything
    except the matched channel.
    """
    n = (1 << degree) - 1
    state, out = seed, []
    for _ in range(n):
        out.append(1 - 2 * (state & 1))  # bit0 -> {+1,-1}
        fb = 0
        for tp in taps:
            fb ^= (state >> (tp - 1)) & 1
        state = (state >> 1) | (fb << (degree - 1))
    return np.array(out, dtype=np.int8)


class EchoRAdapter:
    """Pure-numpy ECHO-R layer (one instance per environment, shared over agents).

    Args:
        n_agents: (int) number of agents ``N``.
        cfg: (dict) configuration; every key has a derived default (Part 7) so
            the only two real knobs are ``eps`` and ``lam_halflife``.
    """

    def __init__(self, n_agents, cfg=None):
        cfg = dict(cfg or {})
        self.N = int(n_agents)

        # ---- knobs & derived constants (Part 7) ----
        self.eps = float(cfg.get("eps", 0.01))                 # knob 1: probe amplitude
        self.T_chip = int(cfg.get("T_chip", 3))                # ~0.5-1 * tau_leak
        self.L = int(cfg.get("L", 127))                        # smallest m-seq with L >= N*delta
        self.delta = int(cfg.get("delta_shift", 16))           # >= K_chips + tau_leak/T_chip + 4
        self.K = int(cfg.get("K", 15))                         # lag window ~ 3*tau_leak
        halflife = float(cfg.get("lam_halflife", 1500))        # knob 2: demod EMA half-life (steps)
        self.lam = 1.0 - 0.5 ** (1.0 / max(1.0, halflife))
        self.rho = float(cfg.get("rho_hat", 0.8))              # structural leak retention (KNOWN_RHO)
        self.W_anchor = int(cfg.get("W_anchor", 100000))       # >= 1 driver period (steps)
        self.anchor_sub = int(cfg.get("anchor_subsample", 100))
        self.anchor_q = float(cfg.get("anchor_q", 0.05))
        self.g_min_frac = float(cfg.get("g_min_frac", 0.1))
        self.c_clip = float(cfg.get("c_clip", 10.0))           # transient guard for obs-norm
        center_hl = float(cfg.get("center_halflife", 200))     # >> chip, << lam half-life
        self.beta_c = 1.0 - 0.5 ** (1.0 / max(1.0, center_hl))
        self.g_rate = float(cfg.get("g_ema_rate", 0.002))      # running-scale EMA for the g_min guard
        # Minimum direct-channel SNR (|E[G]|/std[G]) required to trust the ratio.
        # Below it the estimate is held at 0 so a noise-dominated probe degrades
        # to *blind* (never worse) instead of injecting garbage context; raise
        # eps / lam until the debug trace shows Gsnr climbing above this.
        self.snr_gate = float(cfg.get("snr_gate", 1.0))

        # shift spacing must keep each agent's lag window clear of neighbours,
        # and every agent must get a distinct code within one period.
        assert self.N * self.delta <= self.L, (
            f"ECHO-R: need L >= N*delta ({self.N}*{self.delta}); raise L (255/511/1023)."
        )

        # ---- base code + per-agent cyclic shifts (3.1) ----
        degree = _L_TO_DEGREE.get(self.L)
        if degree is None:
            raise ValueError(f"ECHO-R: unsupported code length L={self.L} (use 127/255/511/1023).")
        _, taps = _MSEQ_TAPS[degree]
        self.B = m_sequence(degree, taps).astype(np.float64)   # period L, +/-1

        # ---- filter constant computed once, never tuned (3.6) ----
        self.KAPPA_F = self._compute_kappa_f()

        # ---- persistent state (never reset across episodes -- 3.7 / Part 10) ----
        self.t = 0                                             # probe clock
        self.phi = np.zeros(self.N)                            # leak-filtered code template phi_j(t)
        self.code_hist = np.zeros((self.K + 1, self.N))        # ring: [C(t), C(t-1), ..., C(t-K)]
        self.psi_hist = np.zeros((self.K + 1, self.N))         # ring: [Psi(t), ..., Psi(t-K)]
        self.R_own = np.zeros((self.N, self.K + 1))            # own-code correlograms
        self.R_agg = np.zeros((self.N, self.K + 1))            # aggregate-of-others correlograms
        self.c_hat = np.zeros(self.N)                          # anchored estimate (exposed to policy)
        self.c_raw = np.zeros(self.N)                          # pre-anchor ratio estimate
        self.ema_slow = np.zeros(self.N)                       # slow-centering EMA for the readout
        self.abs_G = np.zeros(self.N)                          # running |G_hat| scale (g_min guard)
        self.G_last = np.zeros(self.N)
        self.H_last = np.zeros(self.N)
        # slow mean/second-moment of the path gains -> channel SNR (reliability).
        self.G_ema = np.zeros(self.N)
        self.G2_ema = np.zeros(self.N)
        self.H_ema = np.zeros(self.N)
        self.H2_ema = np.zeros(self.N)
        self.snr_G = np.zeros(self.N)
        self.snr_H = np.zeros(self.N)
        # last-step values captured purely for optional debug logging
        self.code_last = np.zeros(self.N)
        self.y_last = np.zeros(self.N)
        self.yc_last = np.zeros(self.N)
        self.frozen_last = np.zeros(self.N, dtype=bool)

        # anchor ring buffer (subsampled): one sample every ``anchor_sub`` steps.
        self.anchor_size = max(1, self.W_anchor // max(1, self.anchor_sub))
        self.anchor_buf = np.zeros((self.N, self.anchor_size))
        self.anchor_idx = 0
        self.anchor_filled = 0
        self.anchor_count = 0
        self.b_hat = np.zeros(self.N)

        # prime the ring buffers at t = 0 (phi(-1) = 0)
        c0 = self._code(0)
        self.phi = (1.0 - self.rho) * c0
        self.code_hist[0] = c0
        self.psi_hist[0] = self.phi.sum() - self.phi

    # ------------------------------------------------------------------ codes
    def _code(self, t):
        """Per-agent probe code ``C_i(t)`` (3.1): cyclic shift by ``i*delta`` chips."""
        m = t // self.T_chip
        idx = (m + np.arange(self.N) * self.delta) % self.L
        return self.B[idx]

    def current_code(self):
        """``C_i(t)`` at the current clock -- used by the injection step."""
        return self._code(self.t)

    # -------------------------------------------------------------- injection
    def modulate_cont(self, actions, mask):
        """Continuous-action injection (3.2): ``a_i = u_i + eps * C_i(t) * e_i``.

        Args:
            actions: (np.ndarray) host policy sample, shape ``(N, act_dim)``.
            mask: (np.ndarray) 0/1 effort-coordinate mask ``e_i``, shape ``(act_dim,)``.
        Returns:
            (np.ndarray) probe-injected actions (a fresh copy; the input is never
            mutated, so the host buffer keeps the pre-probe action -- 5.2).
        """
        code = self._code(self.t)                              # (N,)
        self.code_last = code                                  # for debug logging
        a = np.array(actions, dtype=np.float64, copy=True)
        mask = np.asarray(mask, dtype=np.float64)
        if mask.ndim == 1:                                     # shared coordinate mask
            inc = self.eps * code[:, None] * mask[None, :]
        else:                                                  # per-agent mask (N, act_dim)
            inc = self.eps * code[:, None] * mask
        a = a + inc
        return a

    # ----------------------------------------------------------- demodulation
    def observe(self, y, done=False):
        """Post-step update (3.4-3.5): demodulate the readout, then advance the clock.

        Must be called exactly once per environment step, *after* ``env.step``.
        The demodulation correlates ``y(t+1)`` against the templates at time ``t``
        (the ring currently holds lags ``t .. t-K``); the clock/templates are then
        advanced to ``t+1``.

        Args:
            y: (np.ndarray) raw local readout ``y_i(t+1)``, shape ``(N,)``.
            done: (bool) whether the env terminated this step; if so the
                correlogram update is skipped (avoids the boundary transient),
                but the clock still advances (never reset -- Part 10).
        Returns:
            (np.ndarray) the anchored estimate ``c_hat(t+1)``, shape ``(N,)``.
        """
        y = np.asarray(y, dtype=np.float64).reshape(self.N)
        self.y_last = y                                        # for debug logging
        if not done:
            # high-pass / centering (3.3): rejected by demod anyway, but cuts variance
            yc = y - self.ema_slow
            self.yc_last = yc                                  # for debug logging
            self.ema_slow += self.beta_c * (y - self.ema_slow)

            # EMA correlograms (3.4): R[k] <- (1-lam) R[k] + lam * yc * template(t-k)
            self.R_own += self.lam * (yc[:, None] * self.code_hist.T - self.R_own)
            self.R_agg += self.lam * (yc[:, None] * self.psi_hist.T - self.R_agg)

            # lag-summed path gains (3.4)
            G = self.R_own.sum(axis=1)                         # direct path  -> unknown plant G_i
            H = self.R_agg.sum(axis=1)                         # echo path    -> G_i * c * kappa_F
            self.G_last, self.H_last = G, H

            # channel reliability (SNR = |mean| / std of the path gain): the key
            # health signal -- if the direct-channel SNR is < 1 the probe is
            # undetectable and the ratio is noise.
            self.G_ema += self.lam * (G - self.G_ema)
            self.G2_ema += self.lam * (G * G - self.G2_ema)
            self.H_ema += self.lam * (H - self.H_ema)
            self.H2_ema += self.lam * (H * H - self.H2_ema)
            self.snr_G = np.abs(self.G_ema) / np.sqrt(
                np.maximum(self.G2_ema - self.G_ema ** 2, 1e-18)
            )
            self.snr_H = np.abs(self.H_ema) / np.sqrt(
                np.maximum(self.H2_ema - self.H_ema ** 2, 1e-18)
            )

            # ratio (3.5): cancels eps, code power and the unknown plant gain G_i
            self.abs_G += self.g_rate * (np.abs(G) - self.abs_G)
            g_min = np.maximum(self.g_min_frac * self.abs_G, 1e-8)
            denom = G * self.KAPPA_F
            ok = (np.abs(G) >= g_min) & (np.abs(denom) > 1e-12)
            self.frozen_last = ~ok                             # agents whose ratio was frozen this step
            self.c_raw = np.where(ok, H / np.where(np.abs(denom) > 1e-12, denom, 1.0), self.c_raw)

            # anchoring (3.5): subtract the c-independent physical-coupling floor,
            # estimated as the low quantile over >= 1 driver period.
            self.anchor_count += 1
            if self.anchor_count % self.anchor_sub == 0:
                self.anchor_buf[:, self.anchor_idx] = self.c_raw
                self.anchor_idx = (self.anchor_idx + 1) % self.anchor_size
                self.anchor_filled = min(self.anchor_filled + 1, self.anchor_size)
                self.b_hat = np.quantile(
                    self.anchor_buf[:, : self.anchor_filled], self.anchor_q, axis=1
                )

            c_cand = np.clip(self.c_raw - self.b_hat, 0.0, self.c_clip)
            # SNR gate: only trust the estimate on agents whose direct channel is
            # statistically real; otherwise emit 0 (== blind) rather than noise.
            if self.snr_gate > 0.0:
                c_cand = np.where(self.snr_G >= self.snr_gate, c_cand, 0.0)
            self.c_hat = c_cand
        else:
            self.frozen_last = np.ones(self.N, dtype=bool)     # readout skipped on episode end

        # advance templates and clock (3.7 step 7): t <- t+1
        self.t += 1
        cnew = self._code(self.t)
        self.phi = self.rho * self.phi + (1.0 - self.rho) * cnew
        psi = self.phi.sum() - self.phi                        # Psi_i = sum_{j != i} phi_j
        self.code_hist = np.roll(self.code_hist, 1, axis=0)
        self.code_hist[0] = cnew
        self.psi_hist = np.roll(self.psi_hist, 1, axis=0)
        self.psi_hist[0] = psi
        return self.c_hat.copy()

    def reset_episode(self):
        """Episode boundary hook.

        Per 3.7 / Part 10 the probe clock, correlograms, EMAs and ``c_hat`` all
        persist across episodes; only the readout's one-step pairing is reset,
        which the env shim owns (it drops the cross-boundary readout).  Nothing
        to do here -- kept for API symmetry.
        """
        return

    # --------------------------------------------------------------- kappa_F
    def _compute_kappa_f(self):
        """Numeric truncated-lag filter constant (3.6), computed once at init.

        Runs the production lag-summed demodulator on a synthetic unit-plant
        pair (direct = raw code, echo = leak-filtered code) and returns the
        ratio of the two lag-summed gains.  This folds the leak filter's
        truncated-lag transfer and the code autocorrelation spread into one
        constant so the runtime ratio needs no analytic filter algebra.
        """
        n_chips = 20 * self.L
        chips = np.resize(self.B, n_chips)                     # tile base code
        x = np.repeat(chips, self.T_chip).astype(np.float64)   # expand chips -> steps
        f = np.empty_like(x)
        acc = 0.0
        for i in range(x.shape[0]):
            acc = self.rho * acc + (1.0 - self.rho) * x[i]
            f[i] = acc
        s_dd = 0.0
        s_ee = 0.0
        for k in range(self.K + 1):
            s_dd += float(np.mean(x * np.roll(x, k)))          # sum_k autocorr_x(k)
            s_ee += float(np.mean(f * np.roll(f, k)))          # sum_k autocorr_f(k)
        return s_ee / s_dd if s_dd != 0.0 else 1.0

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self):
        """Per-agent internals for logging / the Part-8 tracking test (never fed back)."""
        return {
            "c_hat": self.c_hat.copy(),
            "c_raw": self.c_raw.copy(),
            "b_hat": self.b_hat.copy(),
            "G": self.G_last.copy(),
            "H": self.H_last.copy(),
            "kappa_f": self.KAPPA_F,
            "t": self.t,
        }

    def debug_snapshot(self):
        """Full internal state for the optional per-step debug trace.

        Every value that helps explain *why the estimate is (not) tracking the
        driver*: the estimate itself (``c_hat``), the pre-anchor ratio
        (``c_raw``) and anchor floor (``b_hat``); the two demodulated path gains
        (``G`` direct, ``H`` echo) whose ratio is the estimate; the running
        ``|G|`` scale and the ``frozen`` guard mask (is the own channel
        observable?); the raw and centered readout (``y``, ``yc`` -- is there any
        signal on the wire?); the injected probe code (``code`` -- is the probe
        live?); and the scalars ``t``, ``kappa_f``, ``eps``, ``lam``.
        """
        return {
            "t": self.t,
            "kappa_f": self.KAPPA_F,
            "eps": self.eps,
            "lam": self.lam,
            "anchor_filled": self.anchor_filled,
            "c_hat": self.c_hat.copy(),
            "c_raw": self.c_raw.copy(),
            "b_hat": self.b_hat.copy(),
            "G": self.G_last.copy(),
            "H": self.H_last.copy(),
            "snr_G": self.snr_G.copy(),
            "snr_H": self.snr_H.copy(),
            "abs_G": self.abs_G.copy(),
            "ema_slow": self.ema_slow.copy(),
            "y": self.y_last.copy(),
            "yc": self.yc_last.copy(),
            "code": self.code_last.copy(),
            "frozen": self.frozen_last.astype(np.float64),
        }
