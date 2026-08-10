"""PACT-1 core for SMAC -- pure numpy, no pysc2, no torch.

Kept separate from ``StarCraft2_Env.py`` so the arithmetic can be certified on a
machine with no StarCraft II installed (``test_pact1.py``), and so the env and the
test can never drift apart.

WHAT PACT-1 CHANGES ON SMAC
---------------------------
Old SMAC-PACT was handed ``x2_i`` -- the env computed the shared bus load and
appended it to the observation. That is even more given away than Ant's version:
the method did not compute anything, it was told the answer.

PACT-1 tells the agent only which INTERFERENCE PATHS EXIST, not how strongly each
one carries. On 3s5z the squad is 3 stalkers + 5 zealots, so the natural basis is

    B_same_i  = (sum of ENGAGED same-type teammates, j != i) / (N-1)
    B_cross_i = (sum of ENGAGED other-type teammates)        / (N-1)

Physically: units sharing a fire-control channel interfere differently from units on
different channels, and how the squad's emissions split across those channels is a
property of the loadout -- unknown, and drifting as units die and re-form.

    B_same + B_cross == the legacy uniform average, EXACTLY

so with theta = (1/2, 1/2) and MIXNORM = 2 the new env reproduces the old one byte
for byte (asserted in test_pact1.py). The hardened env strictly CONTAINS the old.

    x_m,i  <-  RHO * x_m,i  +  (1-RHO) * B_m,i          (per-basis leak)
    psi_i  =   MIXNORM * [x_1,i, ..., x_r,i]            (computable by the agent)
    ell_i  =   c(t) * sum_m theta_m * psi_m,i  =  beta* . psi_i ,   beta* = c(t)*theta

So the unknown is the r-vector ``beta*``, drifting, exactly as on Ant.

THE SENSOR (why this is not privileged)
----------------------------------------
The harm is a target DEFLECTION: the shot lands ``s = round(ell_i*(K-1))`` places
along the unit's list of K attackable enemies. A unit can SEE where its own shot
landed, and it knows what it aimed at, so it observes the net displacement. It also
knows the pre-shift ``s_hat`` it applied itself. Therefore it can reconstruct

    s = s_observed_net + s_hat        =>       ell_meas = s / (K-1)

a direct, quantized reading of its own liability (error <= 0.5/(K-1) from the round).
Nothing about the payload, the driver, or the other units' liabilities is revealed --
this is the unit watching its own shot go astray, the exact analogue of Ant's joint
torque sensor.

Measurements arrive only when the unit fires with K > 1, so the estimator sees
intermittent data. RLS with a forgetting factor handles that natively.
"""

import numpy as np

MIXNORM = 2.0                      # makes theta=(1/2,1/2) reproduce the legacy load
THETA_LEGACY = np.array([0.5, 0.5])
R = 2                              # basis size: same-type, cross-type


# ---------------------------------------------------------------- basis
def type_split(exert, types, denom):
    """The r=2 interference basis, per agent.

    ``exert`` (n,)  each unit's exertion Phi_j (engaged, or firing)
    ``types`` (n,)  unit-type id per agent; -1 for never-seen units
    ``denom``       N-1, matching the legacy normalisation

    Returns (B_same, B_cross), each (n,). Both EXCLUDE agent i's own exertion, so
    the category-C signature holds for every theta: at N=1 both are empty and the
    liability stays 0 no matter how large the driver grows.

    B_same + B_cross == (sum_{j!=i} exert_j)/denom == the legacy uniform load.
    """
    exert = np.asarray(exert, dtype=np.float64)
    types = np.asarray(types)
    n = exert.shape[0]
    tot = float(exert.sum())
    same = np.zeros(n)
    # NB: -1 ("type not yet seen", before init_units has run) is treated as its own
    # group rather than skipped. Skipping it left same[i] = -exert[i]/denom, i.e. a
    # NEGATIVE load, which would corrupt both the liability and the regressor on the
    # very first steps of a run.
    for t in np.unique(types):
        m = types == t
        same[m] = exert[m].sum()          # total of this type, own share removed below
    same = (same - exert) / denom         # sum over SAME type, j != i
    cross = (tot - exert) / denom - same  # everything else, j != i
    return same, cross


def legacy_load(exert, denom):
    """The pre-PACT-1 shared load: (sum_{j!=i} exert_j)/(N-1)."""
    exert = np.asarray(exert, dtype=np.float64)
    return (float(exert.sum()) - exert) / denom


# ---------------------------------------------------------------- theta
def theta_anchors(seed, radius, conc, r=R):
    """The two loadout anchors this run's interference split slides between.

    Pulled toward the legacy point by ``radius``: the harm is NOT constant across
    the simplex (a split that puts everything on one channel is a different task at
    a different effective severity), so an unbounded theta would wander outside the
    certified frontier and measure nothing. radius=0 reproduces the legacy env.
    """
    rng = np.random.RandomState(7000 + int(seed))
    a = rng.dirichlet([conc] * r)
    b = rng.dirichlet([conc] * r)
    leg = np.full(r, 1.0 / r)          # (1/2, 1/2) at r=2 == THETA_LEGACY
    return ((1.0 - radius) * leg + radius * a,
            (1.0 - radius) * leg + radius * b)


def theta_at(clock, period, th_a, th_b):
    """theta(t): the loadout slides between its anchors and back once per ``period``,
    on a symmetric smoothstep. Slower than the driver cycle, so it is a second
    TIMESCALE rather than a second driver."""
    ph = (clock % period) / float(period)
    x = 2.0 * ph if ph < 0.5 else 2.0 * (1.0 - ph)
    w = x * x * (3.0 - 2.0 * x)
    return (1.0 - w) * np.asarray(th_a) + w * np.asarray(th_b)


# ---------------------------------------------------------------- estimator
class AgentRLS:
    """Per-agent recursive least squares with forgetting, on this unit's OWN
    intermittent deflection readings. Decentralized: unit i never sees another
    unit's residual, only the shared engagement that builds psi."""

    def __init__(self, r=R, mu=0.995, p0=1.0):
        self.r = int(r)
        self.mu = float(mu)
        self.p0 = float(p0)
        self.P = np.eye(self.r) * self.p0
        self.beta = np.zeros(self.r)
        self.n_upd = 0
        self.innov = 0.0

    def update(self, psi, y):
        """One scalar observation: y ~= beta . psi."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        Pp = self.P @ psi
        den = self.mu + float(psi @ Pp)
        if den < 1e-12:
            return self.beta
        K = Pp / den
        e = float(y) - float(psi @ self.beta)
        self.beta = self.beta + K * e
        self.P = (self.P - np.outer(K, Pp)) / self.mu
        self.P = 0.5 * (self.P + self.P.T)
        self.beta = np.clip(self.beta, 0.0, 10.0)   # beta* = c*theta >= 0 by construction
        self.innov = abs(e)
        self.n_upd += 1
        return self.beta

    def confidence(self):
        """Self-reported trust in [0,1] from the covariance: 1/(1+r) when cold,
        -> 1 as P shrinks. The estimator's own uncertainty IS the trust prior, so
        early reliance ramps in without a hand-set warmup.

        *** THIS IS THE PARAMETER confidence and it is the WRONG GATE for a
        compensator.  Use confidence_pred() unless you specifically want it. ***
        tr(P) is dominated by the LEAST excited direction, and RLS with forgetting
        inflates every unexcited direction by 1/mu on every update, without bound.
        On SMAC the regressor is near-degenerate by construction -- with Phi=alive
        B_same and B_cross both track squad size, so E[psi psi^T] is ill-conditioned
        (guide III.6: "only the projection beta*.psi is identifiable, not the
        split") -- so tr(P) GROWS over a run even while the prediction beta_hat.psi
        stays good, and any gate keyed to it eventually disarms an estimator that is
        working perfectly.  Measured on the 3s5z PACT-1 run: conf 0.75 -> 0.44 over
        1.8M steps and still falling, i.e. on track to sit far below the 0.5 arming
        threshold by the time the curriculum switched the NS on at 6M."""
        return 1.0 / (1.0 + float(np.trace(self.P)) / max(1e-9, self.p0))

    def confidence_pred(self, psi):
        """Trust in the quantity the compensator actually uses: the PREDICTION
        beta_hat . psi, whose posterior variance is psi^T P psi -- not the parameter
        vector, whose unexcited directions are irrelevant to it.

        Scaled to keep exactly the semantics (and therefore the threshold) of
        confidence(): cold P = p0*I gives 1/(1+r), and it rises to 1 as the variance
        along psi collapses.  Falls back to the trace form when psi ~ 0, where there
        is nothing to predict anyway."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        n2 = float(psi @ psi)
        if n2 <= 1e-12:
            return self.confidence()
        v = float(psi @ (self.P @ psi)) / n2          # variance per unit direction
        return 1.0 / (1.0 + self.r * v / max(1e-9, self.p0))


# ---------------------------------------------------------------- channel
def predict_ell(beta, psi):
    """ell_hat = beta_hat . psi -- the deflection the NEXT shot will suffer."""
    return float(np.dot(np.asarray(beta, dtype=np.float64),
                        np.asarray(psi, dtype=np.float64)))


def shift_from_ell(ell, k):
    """The env's harm channel: how many places along the K attackable enemies the
    delivered target is displaced. Deterministic given ell and K."""
    if k <= 1:
        return 0
    return int(round(float(np.clip(ell, 0.0, 1.0)) * (k - 1)))


def ell_from_shift(s, k):
    """The sensor inverse: what the observed displacement says the liability was.
    Quantisation error <= 0.5/(k-1)."""
    if k <= 1:
        return None
    return float(s) / float(k - 1)
