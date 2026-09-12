# ===========================================================================
#  VERBATIM VENDORED COPY of URB's ``pact1/core.py``.  DO NOT EDIT.
#
#  The porting brief is explicit: "import from pact1/core.py.  Do not
#  reimplement it.  A cross-cell comparison run on two implementations measures
#  the implementations."  URB is a separate checkout that is not on this repo's
#  import path, so the file is vendored rather than imported -- but it is copied
#  byte for byte, and ``smac_ns/selftest.py`` re-checks that it still matches the
#  upstream file if that checkout is present.
#
#  Everything below this banner is upstream code.
# ===========================================================================

"""PACT-1 core arithmetic — pure numpy.

No torch, no routerl, no SUMO. Every function here is exercised by
``pact1/selftest.py`` without a simulator, which is the whole point: the parts
that can be wrong silently are the parts that must be testable for free.

THE REDUCTION, IN URB TERMS
--------------------------------------------------------------------------------
A traveller's excess delay over free flow is, to first order in a link
performance function (BPR: ``t_a = t_a^0 (1 + alpha (v_a/c_a)^beta)``), a sum over
the links of its route of a per-link sensitivity times the peer flow on that link:

    tt_i / fft_i - 1  ~=  beta_0  +  sum_{a in route_i} f'_a * v_a ,
    v_a = sum_{j != i, temporally co-present} 1[a in route_j]

Project the unknown sensitivity field ``f'`` onto ``r`` KNOWN road classes and the
whole thing collapses to

    y_i(t)  =  beta*(t) . psi_i(t) ,     psi_i = [1, x_1, ..., x_r]

with ``x_m`` computable EXACTLY from peers' executed route choices and the network
geometry, and ``beta*`` an (r+1)-vector holding every unknown. That is T2 (the
reduction) instantiated in a domain we did not build: r parameters, independent of
N and of the number of links.

WHAT IS DIFFERENT FROM ANT, STATED PLAINLY
--------------------------------------------------------------------------------
Ant's harm is an additive torque you can subtract off, so compensation restores the
stationary game exactly (T3, conjugacy). Base URB has NO such inverse: you cannot
subtract minutes off a congested road. The only response available is to choose a
different route.

So the channel inverse is replaced by a TRUST-WEIGHTED LOGIT SHIFT (``steer_logits``):
the estimator predicts each candidate route's peer-induced delay, and trust decides
how much that prediction biases the choice. T3 does NOT transfer. What does
transfer, unchanged:

  * T2, the reduction                    -> this file's regressor
  * the honest sensor                    -> realized travel time vs own free-flow
  * per-agent RLS with forgetting        -> AgentRLS
  * the inverted trust prior (III.5)     -> trust_from_w, bias 2.2 -> 0.90
  * covariance-gated reliance (III.5)    -> rls_confidence
  * the floor property (III.4)           -> g = 0 gives the untouched policy, EXACTLY
  * T4, the compensation commons (III.8) -> steering onto a route loads it; this is
                                            Wardrop/Braess and is logged as
                                            ``route_switch_frac`` + ``herd_index``
"""

import numpy as np

__all__ = [
    "AgentRLS",
    "predict_excess",
    "relative_excess",
    "rls_confidence",
    "steer_logits",
    "trust_from_w",
]


# ==========================================================================
#  The estimator
# ==========================================================================
class AgentRLS:
    """Per-agent recursive least squares with a forgetting factor.

    Estimates ``beta* = [intercept, sensitivity_1..r]`` from this traveller's OWN
    one row per day: its realized relative excess delay regressed on the peer-load
    features of the route it actually drove.

    Decentralized by construction. Agent i never sees another agent's residual —
    only peers' executed route choices, which a connected fleet broadcasts anyway
    (the exact analogue of Ant's peer torques).

    The forgetting factor ``mu`` is the bias/variance dial of the tracking floor
    (guide III.9): small forgets fast (tracks drift, amplifies noise), large
    averages hard (rejects noise, lags drift). The floor is Theta(sqrt(noise*drift))
    and cannot be tuned away — only balanced. Declare it, sweep it, never tune it.

    NOTE ON SCALE: this is a numerically identical implementation to the Ant
    version (harl/envs/mamujoco/pact/pact1_mujoco.py::AgentRLS), including the
    symmetrisation of P. Kept deliberately in sync so a bug found in one is a bug
    found in both.
    """

    def __init__(self, r, mu=0.999, p0=10.0, beta0=None):
        self.r = int(r)
        self.mu = float(mu)
        self.p0 = float(p0)
        self.P = np.eye(self.r) * float(p0)
        self.beta = (
            np.zeros(self.r) if beta0 is None
            else np.asarray(beta0, dtype=np.float64).copy()
        )
        self.innov = 0.0            # |prediction error| on the last update
        self.n_updates = 0

    def update(self, Phi, y):
        """Phi: (k, r) regressor rows. y: (k,) targets. Returns the new beta.

        Rows whose regressor is numerically zero are SKIPPED, not processed. A
        zero regressor carries no information about beta but still divides P by mu,
        which inflates the covariance every step and silently tightens the
        effective forgetting factor. (This is exactly the covariance-windup path
        that a dead regressor opens; guarding it here keeps ``mu`` meaning what the
        banner says it means.)
        """
        Phi = np.atleast_2d(np.asarray(Phi, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        tot, used = 0.0, 0
        for j in range(Phi.shape[0]):
            phi = Phi[j]
            if not np.all(np.isfinite(phi)) or not np.isfinite(y[j]):
                continue
            if float(phi @ phi) < 1e-18:
                continue                      # dead row: no information, no windup
            Pphi = self.P @ phi
            denom = self.mu + float(phi @ Pphi)
            if denom < 1e-12:
                continue
            K = Pphi / denom
            e = float(y[j] - phi @ self.beta)
            self.beta = self.beta + K * e
            self.P = (self.P - np.outer(K, Pphi)) / self.mu
            self.P = 0.5 * (self.P + self.P.T)          # keep it symmetric
            tot += abs(e)
            used += 1
        if used:
            self.innov = tot / used
            self.n_updates += used
        return self.beta

    def predict(self, Phi):
        """Phi: (..., r) -> predicted target(s)."""
        return np.asarray(Phi, dtype=np.float64) @ self.beta


# ==========================================================================
#  Sensor, prediction, trust
# ==========================================================================
def relative_excess(travel_time, free_flow_time, clip=10.0):
    """The honest sensor: ``tt / fft - 1``, the traveller's own relative excess delay.

    This is proprioception, not privilege — every connected vehicle knows how long
    its trip took and what the road would cost empty. It is the exact analogue of
    Ant's motor-current reading and of SMAC's "I watched my own shot go astray".

    It is also the natural target of a link performance function: BPR says
    ``t/t0 - 1 = alpha (v/c)^beta``, so relative excess is the quantity that is
    (approximately) linear in load and free of route-length scaling.

    ``clip`` bounds SUMO outliers (a vehicle caught behind an incident can read 50x
    free flow and would otherwise dominate the least squares). Declared constant,
    reported as ``clip_frac`` — never tuned to make a curve look better.
    """
    tt = np.asarray(travel_time, dtype=np.float64)
    fft = np.maximum(np.asarray(free_flow_time, dtype=np.float64), 1e-6)
    return np.clip(tt / fft - 1.0, -1.0, float(clip))


def predict_excess(beta_hat, psi):
    """d_hat = beta_hat . psi, for one agent over its K candidate routes.

    ``psi`` is (K, r): row k is ``[1, x_1(k), ..., x_r(k)]`` for candidate route k,
    built from the FORECAST peer occupancy (see coordinator). Pure arithmetic.
    """
    return np.asarray(psi, dtype=np.float64) @ np.asarray(beta_hat, dtype=np.float64)


def trust_from_w(w, prev, ema, g_max=1.0, bias=2.2):
    """g = g_max * sigmoid(w + bias), EMA-smoothed.

    *** THE BIAS IS THE POINT. *** (guide III.5)

    Old PACT's gain had to TRACK a hidden, phase-dependent c(t), so initialising at
    half was a sensible hedge. PACT-1's estimator already supplies the magnitude, so
    this knob tracks NOTHING — the optimal trust is a CONSTANT whenever the estimate
    is right. Initialising at half therefore starts at half of a known-correct
    answer and asks a weak, noisy policy gradient to walk uphill to it. Measured on
    Ant: over 10M steps trust went 0.485 -> 0.444 while cos(d_hat, d_true) sat at
    0.99 — the entire budget spent at half compensation with a correct waveform.
    Return 3642 against 5444 for the inverted prior.

    So bias=2.2 puts w=0 at sigmoid(2.2) = 0.90 of g_max: trust the estimator unless
    the return says otherwise. The floor property is untouched (w -> -inf still
    gives g=0), it is now a safety net rather than the starting point.
    """
    w = np.asarray(w, dtype=np.float64)
    target = g_max / (1.0 + np.exp(-np.clip(w + bias, -30.0, 30.0)))
    return ema * np.asarray(prev, dtype=np.float64) + (1.0 - ema) * target


def rls_confidence(P, p0, r):
    """TRACE confidence: conf = 1/(1 + tr(P)/p0).

    *** THIS IS THE WRONG GATE ON A NEAR-DEGENERATE REGRESSOR. Kept only for the
    ablation; ``conf_mode: "pred"`` is the default. ***

    It equals 1/(1+r) at the prior and rises to 1 as P shrinks, which is the
    intended behaviour when every direction is excited. But tr(P) is dominated by
    the LEAST excited direction, and RLS with forgetting inflates exactly those
    directions by 1/mu on every update, without bound. So once the fleet converges
    and the regressor stops varying, tr(P) grows even though the PREDICTION stays
    perfect -- and the gate quietly disarms a working compensator.

    Measured on SMAC: conf 0.75 -> 0.44 over 1.8M steps against a 0.5 threshold.
    Measured on URB saint_arnoult, far worse because route choice freezes hard:
    conf 0.20 -> 0.005 over 4200 days while fit_r2 sat at 0.9998, taking applied
    trust to 0.003 and reducing the whole arm to plain IPPO.
    """
    return 1.0 / (1.0 + float(np.trace(P)) / max(1e-9, float(p0)))


def rls_confidence_pred(P, p0, r, psi):
    """PREDICTION confidence -- the default, and the right one.

        conf = 1 / (1 + r * psi^T P psi / (p0 * ||psi||^2))

    The compensator only ever uses ``beta_hat . psi``, so the uncertainty that
    matters is the uncertainty of THAT scalar, not of the whole parameter vector.
    Directions the data never excites are also directions the prediction never
    uses, so their inflated variance is correctly ignored.

    Same range and the same cold value as the trace version -- at the prior
    ``P = p0*I`` gives ``psi^T P psi = p0*||psi||^2`` and hence exactly ``1/(1+r)``
    -- so thresholds carry over unchanged. It converges to 1 as the prediction
    firms up, and, unlike the trace, it does not decay when excitation dies.
    """
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    n2 = float(psi @ psi)
    if not np.isfinite(n2) or n2 < 1e-18:
        return 1.0 / (1.0 + r)
    v = float(psi @ (np.asarray(P, dtype=np.float64) @ psi))
    if not np.isfinite(v) or v < 0.0:
        return 1.0 / (1.0 + r)
    return 1.0 / (1.0 + r * v / max(1e-12, float(p0) * n2))


# ==========================================================================
#  The steering channel (URB's replacement for Ant's channel inverse)
# ==========================================================================
def steer_logits(logits, tt_hat, g, kappa=1.0, valid=None):
    """Trust-weighted model-based prior over routes.

        z_k      = (tt_hat_k - mean tt_hat) / std tt_hat      [over valid routes]
        logits_k = logits_k - g * kappa * z_k

    ``tt_hat_k`` is the predicted travel time of route k (free-flow time inflated by
    the estimated peer-induced excess). Routes the model expects to be loaded get
    pushed down; the z-score makes the shift dimensionless so ``kappa`` is a single
    declared constant rather than a per-network scale.

    *** THE FLOOR PROPERTY. *** At ``g == 0`` this returns ``logits`` UNCHANGED —
    bit-for-bit, not approximately — however wrong ``beta_hat`` is. The estimator
    sits entirely outside the worst-case decision path, so a diverging estimate can
    fail to help but can never drag the arm below plain IPPO. ``selftest.py`` gates
    this exactly.

    The same property holds when the K predicted times are identical (std 0): the
    shift is defined to be zero rather than NaN.

    ``valid``: optional boolean mask over routes (action masking). Statistics are
    taken over valid routes only and invalid entries are left untouched.
    """
    logits = np.asarray(logits, dtype=np.float64)
    g = float(g)
    if g == 0.0:
        return logits.copy()                       # EXACT floor — do not touch it

    tt_hat = np.asarray(tt_hat, dtype=np.float64)
    if valid is None:
        m = np.ones_like(tt_hat, dtype=bool)
    else:
        m = np.asarray(valid, dtype=bool)
    m = m & np.isfinite(tt_hat)
    if m.sum() < 2:
        return logits.copy()

    sub = tt_hat[m]
    sd = float(sub.std())
    if not np.isfinite(sd) or sd < 1e-12:
        return logits.copy()

    z = np.zeros_like(tt_hat)
    z[m] = (sub - sub.mean()) / sd
    return logits - g * float(kappa) * z


def herd_index(counts):
    """Concentration of the fleet's choices, in [0, 1]: 0 = perfectly spread,
    1 = everybody on one option. Herfindahl index over the choice histogram.

    This is the T4 (III.8) commons signature read directly off the fleet: steering
    onto the route the model says is fast is exactly what LOADS that route, so a
    rising herd index alongside a rising trust is the externality becoming visible.
    Logged, never acted on.
    """
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return float("nan")
    p = c / tot
    n = len(p)
    if n < 2:
        return float("nan")
    h = float((p * p).sum())
    return (h - 1.0 / n) / (1.0 - 1.0 / n)          # normalised to [0, 1]
