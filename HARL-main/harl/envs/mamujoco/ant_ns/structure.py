"""The robot's DECLARED structure: joints, load paths, and the gate that checks
the declared layout against the model the simulator actually loaded.

`PACT_NS_SPEC` NS-1.2: *"A declared coupling operator W... Written down from the
environment's structure, never fitted from run data."*

WHY ANT IS THE EASIEST INSTANCE TO DEFEND
--------------------------------------------------------------------------------
In URB the medium is a road network, in SMAC an enemy line, in GRF a running
lane -- all of them structures we had to NAME before we could use them.  In Ant
the medium is already there and needs no naming at all: **four legs are bolted
to one torso**.  Drive one leg and the torso reacts; every other leg is hanging
off that same torso and feels it at its own hips.  That is not an injected
coupling, it is Newton's third law through a shared rigid body, and it is
present in stock Ant at every severity including zero.

What the dial injects is therefore NOT the coupling.  It is a slow drift in
**how strongly the structure transmits** -- the `theta(t)` of the design guide's
`C_i[theta(t)]` -- and the lone-agent projection stays exactly stationary
because the sum runs over `j != i`.

THE DECLARED OPERATOR
--------------------------------------------------------------------------------
Two public facts about the hardware, and nothing else:

  * WHERE each joint is bolted.  A torque at joint q shakes the trunk, and how
    much of that reaches joint p falls off with the distance between their
    anchors:  ``kappa(p, q) = 1 / (1 + (dist / L_torso)^2)``.  The anchors and
    ``L_torso`` are read straight out of ``ant.xml`` -- they are the robot's
    published geometry, the counterpart of a road's painted lane count.
  * WHAT each joint is.  A hip carries the whole leg; an ankle carries only the
    foot, so the same trunk reaction throws an ankle about further.  That is the
    RECEIVER susceptibility ``recv``, public because it is a property of your
    own drivetrain.

    W[p, q] = recv_p * kappa(p, q) * 1[agent(p) != agent(q)]     W[p, p] = 0

The diagonal is ZERO BY ASSERTION, and so is every entry inside one agent's own
joint block: that is what makes the estimated quantity a coupling rather than a
self-effect, and what makes a lone agent read exactly zero at any severity.

THE r CLASSES (P-1.1, P-1.2)
--------------------------------------------------------------------------------
``r = 3`` load paths, by the joint TYPES at each end:

    0  hip   <- hip      1  ankle <- ankle      2  cross (hip <-> ankle)

Public: which joint is a hip and which an ankle is the hardware.  NOT public:
``beta*_m(t)``, how much a unit of a neighbour's torque on path m actually
costs you today -- which is what drifts as the drivetrain warms, and what the
estimator has to recover.

``r`` is independent of the number of agents AND of the number of joints, so the
identical basis serves ``2x4``, ``4x2`` and ``8x1`` -- which is what makes the
N-scaling experiment a measurement rather than three separate designs.

GATE 4, AND WHY IT IS NOT OPTIONAL HERE
--------------------------------------------------------------------------------
``ant.xml`` does NOT list its actuators in leg order: the first two entries are
``hip_4`` and ``ankle_4``, then legs 1, 2, 3.  A table written from the leg names
instead of the actuator order would aim every channel at the wrong joint while
every diagnostic looked perfectly healthy -- the spec's one silent catastrophe.
``verify_against_model`` therefore reads the actuator-to-joint map and the body
anchors out of the LOADED model and aborts on any disagreement.

numpy only.  No mujoco.
"""

import numpy as np

__all__ = ["JOINT_NAMES", "JOINT_IS_HIP", "ANCHORS", "LEG_OF", "TORSO_RADIUS", "RUNNABLE",
           "N_JOINTS", "N_CLASSES", "CLASS_NAMES", "class_of", "kernel", "recv_vector",
           "partition_of", "agent_of", "verify_against_model", "describe"]

#: ``ant.xml``'s actuator order -- legs 4, 1, 2, 3, each hip before its ankle.
#: Checked against the loaded model by ``verify_against_model`` (gate 4).
JOINT_NAMES = ("hip_4", "ankle_4", "hip_1", "ankle_1",
               "hip_2", "ankle_2", "hip_3", "ankle_3")
N_JOINTS = len(JOINT_NAMES)

#: Published body geometry, torso frame, from ``ant.xml``: each leg's aux body
#: sits at (+-0.2, +-0.2) and carries the hip; the ankle is one leg segment
#: further out at (+-0.4, +-0.4).
_LEG_DIR = {1: (1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0), 4: (1.0, -1.0)}
ANCHORS = np.array(
    [[0.2 * _LEG_DIR[int(n.split("_")[1])][0] * (1 if n.startswith("hip") else 2),
      0.2 * _LEG_DIR[int(n.split("_")[1])][1] * (1 if n.startswith("hip") else 2)]
     for n in JOINT_NAMES],
    dtype=np.float64,
)
JOINT_IS_HIP = np.array([n.startswith("hip") for n in JOINT_NAMES], dtype=bool)
LEG_OF = np.array([int(n.split("_")[1]) for n in JOINT_NAMES], dtype=np.int64)

#: ``ant.xml``: ``<geom name="torso_geom" size="0.25" type="sphere"/>``.  The
#: trunk's own scale, used as the transmission length scale.  A published number
#: from the model file, not a tuned one.
TORSO_RADIUS = 0.25

N_CLASSES = 3
CLASS_NAMES = ("hip<-hip", "ankle<-ankle", "cross")


def class_of(p, q):
    """Load-path class of the ordered pair (receiver p, sender q)."""
    hp, hq = bool(JOINT_IS_HIP[p]), bool(JOINT_IS_HIP[q])
    if hp and hq:
        return 0
    if (not hp) and (not hq):
        return 1
    return 2


def kernel(length_scale=TORSO_RADIUS):
    """``kappa[p, q]`` -- the declared structural transmittance, ``(8, 8)``.

    ``1 / (1 + (d / L)^2)`` on the published anchor geometry: a joint twice the
    trunk's own radius away transmits a fifth as much.  Symmetric by itself --
    the asymmetry of ``W`` comes from ``recv`` (the receiver's own
    susceptibility), which is the honest place for it: a compliant unit is
    compliant to its neighbours' load, not weak in isolation.
    """
    d = np.sqrt(((ANCHORS[None, :, :] - ANCHORS[:, None, :]) ** 2).sum(-1))
    return 1.0 / (1.0 + (d / float(length_scale)) ** 2)


def recv_vector(recv_hip, recv_ankle):
    """``recv[p]`` -- the receiver's own susceptibility, ``(8,)``.

    Declared from the drivetrain: a hip reacts against the whole leg's inertia,
    an ankle against the foot's alone, so the same trunk reaction moves an ankle
    further.  ``verify_against_model`` prints the model's OWN inverse-inertia
    diagonal ratio next to these numbers, so the declaration is checked against
    the simulator's dynamics rather than asserted.
    """
    r = np.where(JOINT_IS_HIP, float(recv_hip), float(recv_ankle))
    return r / r.mean()


# ---------------------------------------------------------------------------
#  partitions -- who owns which joints
# ---------------------------------------------------------------------------
#: Ant's partitions in ACTUATOR index order -- which is to say, in the order the
#: torques physically reach the machine.
#:
#: *** THIS IS NOT THE ORDER MAMuJoCo's OWN TABLE PRINTS. ***  ``obsk`` tags each
#: joint with an ``act_ids`` and declares ``2x4`` as act-ids [(2,3,4,5),
#: (6,7,0,1)].  But ``MujocoMulti.step`` never reorders by ``act_ids``: it
#: concatenates the agents' action slices and hands the result straight to
#: MuJoCo, so agent ``a`` drives ``ctrl[sum(dims[:a]) : sum(dims[:a+1])]``.  The
#: disturbance must land on the actuators the agent actually drives, so this
#: table is POSITIONAL.  Measured against obsk: identical leg groups for 4x2,
#: 2x4d and 8x1, and for 2x4 the same "two adjacent legs per agent" shape rotated
#: by one corner -- equivalent under the ant's 4-fold symmetry.  ``smoke.py``
#: settles it at runtime by measuring which actuators an agent's action reaches.
#:
#: ``4x2`` gives one leg per agent; ``8x1`` splits every leg's hip from its own
#: ankle, which moves the strongest load path in the machine into the peer sum.
PARTITIONS = {
    #  NOT a MAMuJoCo partition -- MAMuJoCo offers no 1-agent Ant.  This is the
    #  N = 1 STRUCTURAL PROJECTION, used by gate 3 and by the ceiling's first
    #  row: one agent owning the whole machine has no peers, so every channel is
    #  exactly zero at any severity.  It is a statement about the operator, and
    #  the runnable counterpart of it in the simulator is "command every peer
    #  zero torque", which ``smoke.py`` checks in the engine.
    "1x8": [(0, 1, 2, 3, 4, 5, 6, 7)],
    "2x4": [(0, 1, 2, 3), (4, 5, 6, 7)],
    "2x4d": [(0, 1, 4, 5), (2, 3, 6, 7)],
    "4x2": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "8x1": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
}


#: the partitions MAMuJoCo will actually build an Ant for
RUNNABLE = ("2x4", "2x4d", "4x2", "8x1")


def partition_of(agent_conf):
    """The joint groups of a partition, as a list of tuples of joint indices."""
    key = str(agent_conf)
    if key not in PARTITIONS:
        raise KeyError(
            "ant_ns declares structure for %s; got agent_conf=%r.  Add it to "
            "structure.PARTITIONS (and check it against MAMuJoCo's own "
            "get_parts_and_edges ordering) before running it."
            % (sorted(PARTITIONS), agent_conf))
    return [tuple(int(x) for x in g) for g in PARTITIONS[key]]


def agent_of(parts):
    """``agent_of[p]`` -- which agent owns joint p.  ``(8,)``."""
    out = np.full(N_JOINTS, -1, dtype=np.int64)
    for a, grp in enumerate(parts):
        for p in grp:
            out[int(p)] = a
    if int((out < 0).sum()):
        raise ValueError("partition does not cover every joint: %s" % parts)
    return out


# ---------------------------------------------------------------------------
#  gate 4 -- the declared layout against the model that was actually loaded
# ---------------------------------------------------------------------------
def verify_against_model(env, tol=1e-6):
    """Check the declared joint order and anchors against the LOADED model.

    ``env`` is the raw gym MuJoCo env (``MujocoMulti.env``).  Returns a one-line
    report.  Raises AssertionError on any disagreement -- this is the gate whose
    violation leaves every downstream diagnostic looking healthy while every
    channel points at the wrong joint.
    """
    model = env.model
    names = []
    for k in range(int(model.nu)):
        jid = int(np.asarray(model.actuator_trnid)[k, 0])
        nm = model.joint_names[jid] if hasattr(model, "joint_names") else \
            model.joint_id2name(jid)
        names.append(str(nm))
    if tuple(names) != JOINT_NAMES:
        raise AssertionError(
            "GATE 4 FAILED: the loaded model drives its actuators in the order %s, "
            "structure.py declares %s.  Every channel would aim at the wrong joint "
            "while every diagnostic looked healthy." % (names, list(JOINT_NAMES)))

    # anchors: the world-frame offset of each joint from the torso, at the
    # nominal pose, projected on the ground plane
    anchors = []
    for k in range(int(model.nu)):
        jid = int(np.asarray(model.actuator_trnid)[k, 0])
        bid = int(np.asarray(model.jnt_bodyid)[jid])
        pos = np.asarray(env.sim.data.xanchor)[jid][:2] if hasattr(env.sim.data, "xanchor") \
            else np.asarray(env.sim.data.body_xpos)[bid][:2]
        torso = np.asarray(env.sim.data.body_xpos)[1][:2]
        anchors.append(pos - torso)
    anchors = np.asarray(anchors, dtype=np.float64)
    err = float(np.abs(anchors - ANCHORS).max())
    if err > 0.05:
        raise AssertionError(
            "GATE 4 FAILED: the loaded model's joint anchors differ from the declared "
            "table by %.3f (declared %s, model %s).  ant_ns's geometry belongs to a "
            "different robot." % (err, ANCHORS.tolist(), np.round(anchors, 3).tolist()))
    return ("structure: actuator order and anchors match the loaded model "
            "(max anchor error %.4f)" % err)


def model_inverse_inertia(env):
    """The model's OWN joint-space inverse inertia at the current pose, ``(8, 8)``.

    This is the simulator's answer to "how much does a unit torque at q
    accelerate p".  It is NOT used to build the operator -- the operator is
    declared from published geometry, which is what keeps it a declaration
    rather than a fit -- but it is the right thing to CHECK the declaration
    against, and ``layer.py`` prints the agreement at startup.

    Returns ``None`` when the installed mujoco binding exposes no dense mass
    matrix, so the check degrades to a warning rather than to a crash.
    """
    sim = env.sim
    nv = int(sim.model.nv)
    try:
        import mujoco_py
        M = np.zeros(nv * nv, dtype=np.float64)
        mujoco_py.cymj._mj_fullM(sim.model, M, sim.data.qM)
        M = M.reshape(nv, nv)
    except Exception:
        return None
    # joint-space rows for the actuated joints only
    rows = []
    for k in range(int(sim.model.nu)):
        jid = int(np.asarray(sim.model.actuator_trnid)[k, 0])
        rows.append(int(np.asarray(sim.model.jnt_dofadr)[jid]))
    rows = np.asarray(rows, dtype=np.int64)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    return Minv[np.ix_(rows, rows)]


def describe(length_scale=TORSO_RADIUS):
    """A human-readable dump of the declared structure, for the banner."""
    K = kernel(length_scale)
    off = K[~np.eye(N_JOINTS, dtype=bool)]
    return ("joints=%s  L_torso=%.2f  kappa: min=%.3f max=%.3f ratio=%.1fx"
            % (list(JOINT_NAMES), length_scale, off.min(), off.max(),
               off.max() / max(off.min(), 1e-12)))
