"""Knob access on the **deployed** PCR ant  [campaign spec Part 2].

This module exists because of one easy, silent, campaign-invalidating mistake::

    from harl.envs.mamujoco.diag import ant_diag
    ant_diag.set_freeze_a(1.0)          # <-- DOES NOTHING to the running env

``MujocoMulti`` builds its env with ``gym.make("Ant-v2")``, which instantiates
``gym.envs.mujoco.ant.AntEnv``. Deployment copies ``ant_diag.py`` *over* that
file — so the two are the same **source** but different **module objects**, with
different globals. Setting a knob on the repo copy leaves the running env at its
default, and every Tier-0 cell would then quietly measure A=<whatever the env var
said> while the report labels it A=<what you asked for>. That is the E-5 class of
error the whole campaign exists to stop repeating.

So: always go through this module. It resolves the module gym actually uses,
verifies it really is ``ant_diag`` (and not stock Ant, and not ``ant_pcr_v1``),
and forwards.

    from harl.envs.mamujoco.diag import knobs
    knobs.require_deployed()      # loud failure if ant_diag.py is not deployed
    knobs.set_freeze_a(1.0)       # affects envs constructed from now on
"""

_REQUIRED = ("set_freeze_a", "set_severity", "set_mask", "set_dcap",
             "current_knobs", "AntEnv")


def deployed():
    """The ant module ``gym.make('Ant-v2')`` instantiates."""
    import gym.envs.mujoco.ant as m

    return m


def is_deployed():
    try:
        m = deployed()
    except Exception:
        return False
    return all(hasattr(m, a) for a in _REQUIRED)


def require_deployed():
    """Raise unless ``ant_diag.py`` is the deployed ``gym/envs/mujoco/ant.py``."""
    m = deployed()
    missing = [a for a in _REQUIRED if not hasattr(m, a)]
    if missing:
        raise RuntimeError(
            "gym/envs/mujoco/ant.py is NOT ant_diag.py (missing: %s).\n"
            "Deploy it before running any diagnostic stage:\n"
            "    cp harl/envs/mamujoco/diag/ant_diag.py "
            "$(python -c 'import gym.envs.mujoco.ant as m; print(m.__file__)')\n"
            "and run the V0 golden test first (Prohibition 5):\n"
            "    python -m harl.envs.mamujoco.diag.test_ant_diag\n"
            "Its file is currently: %s" % (missing, getattr(m, "__file__", "?"))
        )
    return m


def set_freeze_a(v):
    require_deployed().set_freeze_a(v)


def set_severity(v):
    require_deployed().set_severity(v)


def set_mask(v):
    require_deployed().set_mask(v)


def set_dcap(v):
    require_deployed().set_dcap(v)


def current_knobs():
    return require_deployed().current_knobs()


def apply(freeze_a=None, severity=None, mask=None, dcap=None):
    """Set a whole cell's knobs at once; returns the resulting snapshot.

    ``None`` means "restore the default": freeze_a=None => the payload drifts,
    dcap=None => no cap. ``severity``/``mask`` of None are left untouched, since
    those have no meaningful 'off'.
    """
    m = require_deployed()
    m.set_freeze_a(freeze_a)
    m.set_dcap(dcap)
    if severity is not None:
        m.set_severity(severity)
    if mask is not None:
        m.set_mask(mask)
    return m.current_knobs()
