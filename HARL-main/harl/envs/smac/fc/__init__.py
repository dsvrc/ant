"""Formation Congestion (the NS) + PACT (the method), for SMAC.

    StarCraft2_Env.py       stock SMAC.  One hook: a per-agent move-stride
                            multiplier, exactly 1.0 unless a wrapper drives it.
    operator.py             the DECLARED coupling operator W, from SC2's own unit
                            data, plus the exertion functional Phi and the geometry.
    driver.py               the exogenous driver A(t) and the severity dial g.
    severity_env.py         the NS.  EVERY arm runs inside this.
    pact_core.py            the method's arithmetic, pure numpy.
    pact_env.py             the compensator, as the outermost wrapper.
    certificates.py         Part-C ceiling + gates G0..G7.  Run BEFORE any method.
    selfcheck.py            the arithmetic self-check.  No simulator needed.
    calibrate.py            pick k_scale / severity on the real environment.
    README.md               the runbook and the command list.
"""

from .driver import Driver, dial, driver, is_placebo, assert_dial
from .operator import Exertion, build_W, composition, enemy_composition
from .pact_core import AgentCompensator, Basis, RLS, compensation_delta
from .pact_env import PactEnv
from .severity_env import FormationCongestionEnv

__all__ = [
    "AgentCompensator", "Basis", "Driver", "Exertion", "FormationCongestionEnv",
    "PactEnv", "RLS", "assert_dial", "build_W", "compensation_delta",
    "composition", "dial", "driver", "enemy_composition", "is_placebo",
]
