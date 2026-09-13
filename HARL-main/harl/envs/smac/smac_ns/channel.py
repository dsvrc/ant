"""The steering channel as the policy applies it: ``steer_logits`` plus the floor.

``pact1_core.steer_logits`` is vendored verbatim and stays that way.  The floor is
the one addition this instance makes, so it lives here, in one numpy reference
that the story test calls and ``check_actor.py`` holds the torch actor to.

WHY A FLOOR.  P-6.1's z-score is scale-free: options predicted at 0.00271 and
0.00269 get z = +1 and -1 and a full-size shove.  URB never felt this, because its
z-score runs over predicted TRAVEL TIME -- free-flow time inflated by the excess --
so as the excess vanishes the ranking falls back to route length, which the host
already agrees with.  Here the predicted cost IS the excess, and in the placebo the
true excess is exactly zero: a residual beta_hat there ranks targets by peer load
and pushes agents off focus fire exactly when overkill costs nothing extra.

P-7.1 makes the shift exactly zero when the predictions are identical.  The floor
extends that to predictions identical within a declared resolution: if the VALID
options' predicted relative excess spreads (std) by less than ``floor``, the logits
are returned bit for bit.

WHY IN THE POLICY, NOT THE LAYER.  The gate must see the same option set as the
z-score.  SMAC masks attack actions to enemies in shooting range, and only the
policy knows that mask at decision time; the layer sees every living enemy, one
step earlier.
"""

import numpy as np

from .pact1_core import steer_logits


def spread(cost, valid):
    """std of the predicted costs over the valid options; NaN below two options."""
    c = np.asarray(cost, dtype=np.float64)
    m = np.asarray(valid, dtype=bool) & np.isfinite(c)
    return float(c[m].std()) if m.sum() >= 2 else float("nan")


def steer(logits, cost, g, kappa, valid, floor):
    """``steer_logits`` over ``valid``, or the logits untouched below ``floor``."""
    s = spread(cost, valid)
    if not np.isfinite(s) or s < float(floor):
        return np.asarray(logits, dtype=np.float64).copy()
    return steer_logits(logits, cost, g, kappa, valid)
