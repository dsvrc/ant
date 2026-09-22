"""Dump the machine's OWN coupling operator from the model, once.

`PACT_NS_SPEC` NS-1.2 wants the operator "written down from the environment's
structure, never fitted from run data", and names the domain's own sensitivity
matrix as the thing to write down -- incidence-over-capacity in URB, PTDF in
POWER.  The mechanical counterpart is the joint-space **inverse inertia** at the
nominal pose: ``M^-1[p, q]`` is exactly "how much does a unit torque at q
accelerate p".

This script reads it out of the loaded model at the NOMINAL pose (``init_qpos``,
zero velocity -- not whatever pose a reset happened to produce) and writes
``operator.json`` beside ``structure.py``.  From then on ``kernel()`` uses the
machine's own numbers instead of the geometric surrogate, and the layer
re-checks the committed matrix against the live model at startup.

Needs mujoco, so it runs where the simulator is.  Run it ONCE, commit the JSON,
then re-run ``ceiling.py`` (the references change) and ``selftest.py``.

    python -m harl.envs.mamujoco.ant_ns.dump_operator
    python -m harl.envs.mamujoco.ant_ns.dump_operator --show      # no write
"""

import argparse
import json

import numpy as np

from harl.utils.configs_tools import get_defaults_yaml_args

from . import make_ant_ns_env
from .structure import (CLASS_NAMES, JOINT_NAMES, N_JOINTS, OPERATOR_JSON, class_of,
                        model_inverse_inertia, surrogate_kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print only, write nothing")
    ap.add_argument("--out", default=OPERATOR_JSON)
    a = ap.parse_args()

    _, ea = get_defaults_yaml_args("happo", "mamujoco_ns")
    ea = dict(ea)
    #  Build with the dial OFF: this reads the STOCK machine's structure, and it
    #  must not depend on any severity setting.  ns_on=0 also skips the gates
    #  that would otherwise want the very operator we are about to produce.
    ea["ns_on"] = 0
    env = make_ant_ns_env(ea, 0, 1)
    raw = env.env

    #  the NOMINAL pose, not whatever a reset produced: the operator is a
    #  declared linearisation about the robot's own rest configuration
    raw.set_state(raw.init_qpos, np.zeros_like(raw.init_qvel))
    raw.sim.forward()

    Minv = model_inverse_inertia(raw)
    if Minv is None:
        raise SystemExit(
            "the installed mujoco binding exposes no dense mass matrix, so the "
            "operator cannot be dumped.  ant_ns will keep using the geometric "
            "surrogate; say so in the paper.")
    A = np.abs(np.asarray(Minv, dtype=np.float64))
    off = ~np.eye(N_JOINTS, dtype=bool)
    norm = A.copy()
    norm[~off] = 0.0
    norm = norm / max(float(norm[off].mean()), 1e-30)

    print("joint order: %s" % list(JOINT_NAMES))
    print("")
    print("|M^-1| at the nominal pose, normalised to mean off-diagonal 1:")
    for p in range(N_JOINTS):
        print("  %-8s %s" % (JOINT_NAMES[p], "  ".join("%6.3f" % v for v in norm[p])))
    o = norm[off]
    print("")
    print("  off-diagonal: min=%.3f max=%.3f ratio=%.1fx spread=%.3f"
          % (o.min(), o.max(), o.max() / max(o.min(), 1e-12), o.std() / o.mean()))
    print("  self terms (diagonal of M^-1), hips %s ankles %s"
          % (np.round(np.diag(A)[::2], 3).tolist(), np.round(np.diag(A)[1::2], 3).tolist()))

    sur = surrogate_kernel()
    corr = float(np.corrcoef(norm[off], sur[off])[0, 1])
    print("  correlation with the geometric surrogate: %+.3f" % corr)
    print("")
    print("per load path, mean |M^-1| (this is what the r = 3 classes average over):")
    for m, nm in enumerate(CLASS_NAMES):
        sel = np.array([[class_of(p, q) == m and p != q for q in range(N_JOINTS)]
                        for p in range(N_JOINTS)])
        print("  %-14s %.3f  (%d ordered pairs)" % (nm, norm[sel].mean(), int(sel.sum())))

    env.close()
    if a.show:
        print("")
        print("--show: nothing written")
        return
    blob = dict(
        abs_minv=A.tolist(),
        meta=dict(
            dumped_from="%s %s" % (ea.get("scenario"), ea.get("agent_conf")),
            joint_names=list(JOINT_NAMES),
            pose="nominal (init_qpos, zero velocity)",
            note="|M^-1| of the actuated joints; kernel() normalises it to mean "
                 "off-diagonal 1.  Declared structure read from the model, never "
                 "fitted from run data.",
        ),
    )
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    print("")
    print("wrote %s" % a.out)
    print("NOW RE-RUN, in this order -- the references change with the operator:")
    print("  python -m harl.envs.mamujoco.ant_ns.ceiling --out harl/envs/mamujoco/ant_ns/ceiling.json")
    print("  python -m harl.envs.mamujoco.ant_ns.selftest")
    print("  python -m harl.envs.mamujoco.ant_ns.check_plumbing   # commit the new ns_load_norm")


if __name__ == "__main__":
    main()
