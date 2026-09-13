"""The steering channel as the policy applies it -- one numpy reference.

``pact1_core.steer_logits`` is vendored verbatim and stays that way.  What this
instance adds lives here, in the function the story test calls and
``check_actor.py`` holds the torch actor to, bit for bit in the gated cases and to
rounding elsewhere.

TWO CHANNELS, BOTH DIMENSIONLESS (P-6.1), BOTH EXACTLY INERT AT g = 0 (P-7.1)
--------------------------------------------------------------------------------
``zscore``    the reference instance's shift, verbatim:
                  logits_k -= g * kappa * zscore_k(cost)
``logratio``  the shift in the units the harm is in:
                  logits_k -= g * kappa * (log(1 + cost_k) - mean_valid log(1 + cost))
              i.e. pi(k) is reweighted by (1 + excess_k) ** (-g * kappa): an option
              expected to waste half its damage keeps 2**-0.9 = 0.54 of its odds.

WHY THE CHOICE MATTERS HERE.  A z-score is scale-free: whatever the spread, the
most-loaded option is pushed ~2 logits away.  In URB that is harmless -- the
z-score runs over predicted TRAVEL TIME, free-flow time inflated by the excess, so
as the excess vanishes the ranking falls back to route length, which the host
already agrees with.  In SMAC the cost IS the excess, and the most-loaded option is
the focus target.  Measured in story.py at sigma = 1: steering on the TRUE excess
through the z-score cut the focus-fire host's peak-phase win rate from 0.63 to
0.39, while the harm it avoided was ~1% of damage.  The log-ratio shift is
proportional to what is actually wasted -- 1% moves a logit by 0.01 -- so it can
only push hard where joining a target really throws damage away.

THE FLOOR.  P-7.1 makes the shift exactly zero when the predictions are identical;
``floor`` extends that to a declared resolution (std of the valid options'
predicted relative excess).  It is essential for ``zscore`` (a residual beta_hat in
the placebo would otherwise be blown up to full size) and nearly moot for
``logratio`` (a residual is already a residual-sized shift).

WHY IN THE POLICY, NOT THE LAYER.  The gate must see the same option set as the
shift.  SMAC masks attack actions to enemies in shooting range, and only the policy
knows that mask at decision time.
"""

import numpy as np

from .pact1_core import steer_logits

CHANNELS = ("zscore", "logratio")
FLAT = 1e-12            # the spread below which predictions count as identical


def spread(cost, valid):
    """std of the predicted costs over the valid options; NaN below two options."""
    c = np.asarray(cost, dtype=np.float64)
    m = np.asarray(valid, dtype=bool) & np.isfinite(c)
    return float(c[m].std()) if m.sum() >= 2 else float("nan")


def logratio_shift(cost, valid):
    """``-(log(1 + c_k) - mean over valid)`` on the valid options, 0 elsewhere.
    Predictions are floored at -0.99 so a fitted value below -1 stays finite."""
    c = np.asarray(cost, dtype=np.float64)
    m = np.asarray(valid, dtype=bool) & np.isfinite(c)
    out = np.zeros_like(c)
    if m.sum() < 2:
        return out
    lc = np.log1p(np.maximum(c[m], -0.99))
    out[m] = -(lc - lc.mean())
    return out


def steer(logits, cost, g, kappa, valid, floor, channel="zscore"):
    """The policy's shift over ``valid``: the logits untouched when g == 0, when
    fewer than two options are valid, or when the spread is below ``floor``."""
    if channel not in CHANNELS:
        raise ValueError("unknown steering channel %r" % (channel,))
    lg = np.asarray(logits, dtype=np.float64)
    s = spread(cost, valid)
    if float(g) == 0.0 or not np.isfinite(s) or s <= FLAT or s < float(floor):
        return lg.copy()
    if channel == "zscore":
        return steer_logits(lg, cost, g, kappa, valid)
    return lg + float(g) * float(kappa) * logratio_shift(cost, valid)
