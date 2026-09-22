"""The robot's DECLARED structure: joints, load paths, and the gates that check
the declaration against the model the simulator actually loaded.

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

THE DECLARED OPERATOR, AND WHERE IT COMES FROM
--------------------------------------------------------------------------------
    W[p, q] = recv_p * kappa(p, q) * 1[agent(p) != agent(q)]     W[p, p] = 0

``kappa`` is the transmission structure, and it comes from one of two places.
The banner always says which is in force:

  * COMMITTED (preferred).  The robot's OWN joint-space inverse inertia at the
    nominal pose, ``|M^-1|`` off-diagonal, dumped once from the model by
    ``dump_operator.py`` into ``operator.json``.  This is literally "how much
    does a unit torque at q accelerate p" -- the mechanical counterpart of
    POWER's PTDF and of URB's incidence-over-capacity -- so the injected
    disturbance amplifies the machine's OWN coupling rather than laying a
    differently-shaped one on top.  Read from the model, never fitted from run
    data, and re-checked against the live model at startup.
  * SURROGATE (fallback).  A distance falloff on the published anchor geometry,
    ``1 / (1 + (d / L_torso)^2)``.  Structure, and honest -- but MEASURED
    against the real thing on this repo's Ant it correlates only ~0.19: same
    support and ordering, different shape.  So it is a declared transmission
    MODEL, not the machine's own sensitivity, and a run on it must say so.

``recv`` is the receiver's own susceptibility, and it is what makes ``W``
ASYMMETRIC -- ``kappa`` is symmetric either way (``M^-1`` is symmetric by
construction), so without it NS-1.2's asymmetry requirement is not met.  It is a
DECLARED per-actuator heterogeneity: no two drivetrains on a real machine are
identical, and the unit that has done more work is more easily thrown about by
the same trunk reaction.

It is deliberately NOT claimed to be measurable in the simulator.  An earlier
version of this file declared a hip/ankle susceptibility ratio of 1.86, on the
argument that an ankle carries only the foot.  The model was asked, and answered
1.01 -- Ant's four legs are identical and its hips and ankles are equally easy
to accelerate.  The argument was wrong and has been removed rather than
defended; what remains is an injected property of the hardware, stated as one,
exactly as ``simple_ns`` states ``recv_spread`` and ``smac_ns`` its per-enemy
sensitivity.

The diagonal is ZERO BY ASSERTION, and so is every entry inside one agent's own
joint block: that is what makes the estimated quantity a coupling rather than a
self-effect, and what makes a lone agent read exactly zero at any severity.

THE r CLASSES (P-1.1, P-1.2)
--------------------------------------------------------------------------------
``r = 3`` load paths, by the joint TYPES at each end:

    0  hip   <- hip      1  ankle <- ankle      2  cross (hip <-> ankle)

Public: which joint is a hip and which an ankle is the hardware.  NOT public:
``beta*_m(t)``, how much a unit of a neighbour's torque on path m actually costs
you today -- which drifts as the drivetrain warms, and is what the estimator has
to recover.

``r`` is independent of the number of agents AND of the number of joints, so the
identical basis serves ``2x4``, ``4x2`` and ``8x1`` -- which is what makes the
N-scaling experiment a measurement rather than three separate designs.

THE GATES, AND WHY NEITHER IS OPTIONAL
--------------------------------------------------------------------------------
gate 0  ``verify_host_is_stock`` -- is the HOST already disturbed?  A patched
        gym ``ant.py`` (this repo has shipped one, announcing itself with a
        ``[DIAG ENV] SEVERITY=...`` banner) would make ``ns_severity: 0``
        something other than the stock task, so B0 and every ladder row would be
        measured against an already-disturbed baseline.
gate 4  ``verify_against_model`` -- ``ant.xml`` does NOT list its actuators in
        leg order: the first two entries are ``hip_4`` and ``ankle_4``.  A table
        written from the leg names would aim every channel at the wrong joint
        while every diagnostic looked perfectly healthy.

numpy only.  No mujoco.
"""

import json
import os

import numpy as np

__all__ = ["JOINT_NAMES", "JOINT_IS_HIP", "ANCHORS", "LEG_OF", "TORSO_RADIUS", "RUNNABLE",
           "N_JOINTS", "N_CLASSES", "CLASS_NAMES", "GEOMETRY_TOL", "OPERATOR_JSON",
           "PARTITIONS", "REFERENCE_PARTITION", "class_of", "kernel", "surrogate_kernel",
           "kernel_source",
           "committed_operator", "recv_vector", "partition_of", "agent_of",
           "joint_anchors", "actuated_joint_names", "verify_against_model",
           "verify_host_is_stock", "verify_operator_against_model",
           "model_inverse_inertia", "describe"]

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
#: trunk's own scale, used as the surrogate's transmission length scale.
TORSO_RADIUS = 0.25

N_CLASSES = 3
CLASS_NAMES = ("hip<-hip", "ankle<-ankle", "cross")

#: Max relative deviation allowed between the declared pairwise-distance matrix
#: and the model's own.  Ant's reset perturbs every joint by +-0.1 rad AND the
#: free joint's quaternion, so the body sits at an arbitrary small yaw and the
#: legs are never exactly nominal; measured at fresh resets, 3.6% worst case.
GEOMETRY_TOL = 0.25

#: Where ``dump_operator.py`` writes the machine's own nominal-pose operator.
OPERATOR_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator.json")


def class_of(p, q):
    """Load-path class of the ordered pair (receiver p, sender q)."""
    hp, hq = bool(JOINT_IS_HIP[p]), bool(JOINT_IS_HIP[q])
    if hp and hq:
        return 0
    if (not hp) and (not hq):
        return 1
    return 2


# ---------------------------------------------------------------------------
#  the transmission structure
# ---------------------------------------------------------------------------
def committed_operator():
    """The machine's own nominal-pose ``|M^-1|``, if it has been dumped.

    Returns ``(matrix, meta)`` or ``(None, None)``.  Normalised so its mean
    off-diagonal entry is 1, which keeps ``kappa`` dimensionless and leaves the
    meaning of the severity dial entirely to ``load_norm``.
    """
    if not os.path.exists(OPERATOR_JSON):
        return None, None
    with open(OPERATOR_JSON, "r", encoding="utf-8") as f:
        blob = json.load(f)
    M = np.asarray(blob["abs_minv"], dtype=np.float64)
    if M.shape != (N_JOINTS, N_JOINTS):
        raise ValueError("operator.json holds a %s matrix, expected (%d, %d)"
                         % (M.shape, N_JOINTS, N_JOINTS))
    off = ~np.eye(N_JOINTS, dtype=bool)
    M = M / max(float(M[off].mean()), 1e-30)
    M[~off] = 0.0
    return M, blob.get("meta", {})


def surrogate_kernel(length_scale=TORSO_RADIUS):
    """The distance falloff on the published anchor geometry, ``(8, 8)``.

    ``1 / (1 + (d / L)^2)``: a joint twice the trunk's own radius away transmits
    a fifth as much.  Structure, and honest -- but see the module docstring: it
    correlates only ~0.19 with the machine's own cross-inertia, so it is a
    declared transmission MODEL rather than the machine's own sensitivity.
    """
    d = np.sqrt(((ANCHORS[None, :, :] - ANCHORS[:, None, :]) ** 2).sum(-1))
    k = 1.0 / (1.0 + (d / float(length_scale)) ** 2)
    k[np.eye(N_JOINTS, dtype=bool)] = 0.0
    return k


def kernel(length_scale=TORSO_RADIUS):
    """``kappa[p, q]`` -- the transmission structure in force, ``(8, 8)``.

    The machine's own nominal-pose inverse inertia when ``operator.json`` has
    been committed; the geometric surrogate otherwise.  ``kernel_source()`` says
    which, and the layer prints it at startup.
    """
    M, _ = committed_operator()
    return surrogate_kernel(length_scale) if M is None else M


def kernel_source():
    """``(which, one-line description)`` -- "committed" or "surrogate"."""
    M, meta = committed_operator()
    if M is None:
        return ("surrogate",
                "geometric distance falloff on the published anchors -- no "
                "operator.json; run dump_operator.py where mujoco is installed")
    return ("committed",
            "the model's OWN nominal-pose |M^-1| (%s)"
            % ((meta or {}).get("dumped_from", "unknown model")))


def recv_vector(recv_spread):
    """``recv[p]`` -- the receiver's own susceptibility, ``(8,)``, mean exactly 1.

    A DECLARED per-actuator heterogeneity: no two drivetrains on a real machine
    are identical, and a unit that has done more work is more easily thrown
    about by the same trunk reaction.  Deterministic in the joint index alone --
    no RNG, no run data -- so it is identical across arms, seeds and severities.

    It is what makes ``W`` asymmetric, and it is NOT claimed to be measurable in
    this simulator: Ant's four legs are identical and the model's own
    inverse-inertia diagonal ratio measures 1.01.  Injected, and stated as such.
    """
    idx = np.arange(N_JOINTS, dtype=np.float64)
    fan = np.cos(np.pi * idx / (N_JOINTS - 1))          # +1 .. -1, deterministic
    r = 1.0 + float(recv_spread) * fan
    if np.any(r <= 0):
        raise ValueError("recv_spread must be < 1 to keep every susceptibility positive")
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
#: settled it at runtime by driving agent 0 and reading ``sim.data.ctrl``: it
#: touched [0, 1] = (hip_4, ankle_4), which is this table.
PARTITIONS = {
    #  NOT a MAMuJoCo partition -- MAMuJoCo offers no 1-agent Ant.  This is the
    #  N = 1 STRUCTURAL PROJECTION, used by gate 3 and by the ceiling's first
    #  row: one agent owning the whole machine has no peers, so every channel is
    #  exactly zero at any severity.
    "1x8": [(0, 1, 2, 3, 4, 5, 6, 7)],
    "2x4": [(0, 1, 2, 3), (4, 5, 6, 7)],
    "2x4d": [(0, 1, 4, 5), (2, 3, 6, 7)],
    "4x2": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "8x1": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
}

#: the partitions MAMuJoCo will actually build an Ant for
RUNNABLE = ("2x4", "2x4d", "4x2", "8x1")

#: The partition the committed ``ns_load_norm`` belongs to.  A mismatch on THIS
#: partition means the operator changed and the committed value is stale (the
#: layer aborts); a mismatch on any other partition is the N-scaling and is
#: expected (the layer warns).
REFERENCE_PARTITION = "4x2"


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
#  gate 0 -- is the host already disturbed?
# ---------------------------------------------------------------------------
def verify_host_is_stock(env):
    """Refuse to stack this dial on a host that is ALREADY disturbed.

    A patched ``ant.py`` is not hypothetical: this repo's earlier PACT work
    shipped a diagnostic Ant that applies its own peer coupling and announces
    itself with a ``[DIAG ENV] SEVERITY=...`` banner.  Running ANT-NS on top of
    that would mean

      * ``sigma = 0`` is NOT the stock task -- the B0 reference is already
        disturbed, so every ladder is measured against the wrong baseline;
      * two couplings are live at once, and the estimator is shown a target it
        has no basis for.

    Both produce entirely plausible numbers, which is why this aborts rather
    than warns.  Returns a one-line report when the host is clean.
    """
    import inspect

    cls = type(env)
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):  # pragma: no cover -- no source available
        return "host: source unavailable, could not check for a patched ant.py"
    markers = [m for m in ("DIAG ENV", "SEVERITY", "_coupling", "pcr_", "PCR_",
                           "FREEZE_A", "CORACLE")
               if m in src]
    if markers:
        raise AssertionError(
            "THE HOST IS ALREADY DISTURBED.  %s.%s contains %s, which means the "
            "installed gym Ant is a patched/diagnostic env applying its own "
            "coupling -- so `ns_severity: 0` would NOT be the stock task and every "
            "arm would be measured against a disturbed baseline.  FIX: restore the "
            "stock `gym/envs/mujoco/ant.py`, or set that env's own severity to 0 "
            "through whatever switch it reads, and re-run.  To proceed anyway -- "
            "which makes the run UNREPORTABLE -- set `ns_allow_patched_host: 1` in "
            "the task config."
            % (getattr(cls, "__module__", "?"), cls.__name__, markers))
    return "host: %s.%s is stock (no coupling markers in its source)" % (
        getattr(cls, "__module__", "?"), cls.__name__)


# ---------------------------------------------------------------------------
#  gate 4 -- the declared layout against the model that was actually loaded
# ---------------------------------------------------------------------------
def actuated_joint_names(env):
    """The joint each actuator drives, in actuator order."""
    model = env.model
    names = []
    for k in range(int(model.nu)):
        jid = int(np.asarray(model.actuator_trnid)[k, 0])
        nm = (model.joint_names[jid] if hasattr(model, "joint_names")
              else model.joint_id2name(jid))
        names.append(str(nm))
    return names


def joint_anchors(env):
    """The actuated joints' anchors in the model's own frame, ``(8, 2)``."""
    model, data = env.model, env.sim.data
    out = []
    torso = np.asarray(data.body_xpos)[1][:2]
    for k in range(int(model.nu)):
        jid = int(np.asarray(model.actuator_trnid)[k, 0])
        if hasattr(data, "xanchor"):
            pos = np.asarray(data.xanchor)[jid][:2]
        else:  # pragma: no cover
            bid = int(np.asarray(model.jnt_bodyid)[jid])
            pos = np.asarray(data.body_xpos)[bid][:2]
        out.append(pos - torso)
    return np.asarray(out, dtype=np.float64)


def verify_against_model(env, tol=GEOMETRY_TOL):
    """Check the declared joint order and geometry against the LOADED model.

    ``env`` is the raw gym MuJoCo env (``MujocoMulti.env``).  Returns a one-line
    report.  Raises AssertionError on disagreement.

    THE ORDER is checked exactly: a permuted actuator table is the spec's one
    silent catastrophe, and ``ant.xml`` really does list ``hip_4`` first.

    THE GEOMETRY is checked through the PAIRWISE DISTANCE MATRIX, which is
    rotation- and translation-invariant.  Raw anchor coordinates are in the
    WORLD frame and Ant's ``reset_model`` perturbs the free joint's quaternion,
    so the whole body sits at a small arbitrary yaw -- measured on two fresh
    resets, every joint rotated by the same amount, once -10 deg and once +9
    deg, with radii correct to 2% and pairwise distances to 3.6%.  Comparing
    coordinates fails on a robot that is entirely correct, and did.
    """
    names = actuated_joint_names(env)
    if tuple(names) != JOINT_NAMES:
        raise AssertionError(
            "GATE 4 FAILED: the loaded model drives its actuators in the order %s, "
            "structure.py declares %s.  Every channel would aim at the wrong joint "
            "while every diagnostic looked healthy." % (names, list(JOINT_NAMES)))

    obs = joint_anchors(env)
    dd = np.sqrt(((ANCHORS[None] - ANCHORS[:, None]) ** 2).sum(-1))
    dm = np.sqrt(((obs[None] - obs[:, None]) ** 2).sum(-1))
    off = ~np.eye(N_JOINTS, dtype=bool)
    rel = float((np.abs(dm - dd)[off] / np.maximum(dd[off], 1e-12)).max())
    if rel > tol:
        raise AssertionError(
            "GATE 4 FAILED: the loaded model's joint-to-joint distances differ from "
            "the declared geometry by %.1f%% (tolerance %.0f%%).  Declared radii %s, "
            "model radii %s.  ant_ns's geometry belongs to a different robot."
            % (100 * rel, 100 * tol,
               np.round(np.linalg.norm(ANCHORS, axis=1), 3).tolist(),
               np.round(np.linalg.norm(obs, axis=1), 3).tolist()))
    return ("structure: actuator order matches; joint-to-joint distances agree with "
            "the declared geometry to %.1f%%" % (100 * rel))


def model_inverse_inertia(env):
    """The model's OWN joint-space inverse inertia at the current pose, ``(8, 8)``.

    The simulator's answer to "how much does a unit torque at q accelerate p".
    Returns ``None`` when the installed binding exposes no dense mass matrix, so
    the check degrades to a warning rather than to a crash.
    """
    sim = env.sim
    nv = int(sim.model.nv)
    M = None
    try:
        import mujoco_py
        buf = np.zeros(nv * nv, dtype=np.float64)
        mujoco_py.cymj._mj_fullM(sim.model, buf, sim.data.qM)
        M = buf.reshape(nv, nv)
    except Exception:
        try:
            import mujoco
            M = np.zeros((nv, nv), dtype=np.float64)
            mujoco.mj_fullM(sim.model, M, sim.data.qM)
        except Exception:
            return None
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


def verify_operator_against_model(env, tol=0.35):
    """When a COMMITTED operator is in force, re-check it against the live model.

    The committed matrix was dumped at the nominal pose and the running model is
    at whatever pose it is in, so this is a shape check rather than an equality.
    Returns a one-line report, ``None`` when no operator is committed, and
    raises when the two describe different machines.
    """
    M, meta = committed_operator()
    if M is None:
        return None
    live = model_inverse_inertia(env)
    if live is None:
        return ("operator: committed, but the installed binding exposes no dense "
                "mass matrix, so it could not be re-checked against the model")
    off = ~np.eye(N_JOINTS, dtype=bool)
    lv = np.abs(live)[off]
    lv = lv / max(lv.mean(), 1e-30)
    corr = float(np.corrcoef(M[off], lv)[0, 1])
    if corr < 1.0 - tol:
        raise AssertionError(
            "GATE FAILED: the committed operator correlates only %+.3f with the "
            "model's own inverse inertia at the running pose.  operator.json was "
            "dumped from %s -- if that is a different robot or a different mujoco, "
            "re-run dump_operator.py and then ceiling.py."
            % (corr, (meta or {}).get("dumped_from", "an unknown model")))
    return ("operator: committed |M^-1| re-checked against the live model, corr "
            "%+.3f" % corr)


def describe(length_scale=TORSO_RADIUS):
    """A human-readable dump of the transmission structure, for the banner."""
    src, note = kernel_source()
    K = kernel(length_scale)
    off = K[~np.eye(N_JOINTS, dtype=bool)]
    nz = off[off > 0]
    return ("joints=%s  kappa=%s: %d/%d live links, min=%.3f max=%.3f ratio=%.1fx  (%s)"
            % (list(JOINT_NAMES), src, nz.size, off.size, nz.min(), nz.max(),
               nz.max() / max(nz.min(), 1e-12), note))
