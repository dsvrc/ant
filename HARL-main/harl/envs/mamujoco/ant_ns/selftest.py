"""The conformance suite.  No mujoco, no torch, no learning framework.

`PACT_NS_SPEC` I.7: *"Fourteen offline checks... Port all of them; each
corresponds to a requirement above and each has failed at least once during
development."*  Plus the channel identities that make THIS cell the invertible
one, and the (B)-vs-(C) and (A)-vs-(C) decision procedures as MEASUREMENTS.

Run::

    python -m harl.envs.mamujoco.ant_ns.selftest

Ends with ``ALL CHECKS PASSED`` or lists the failures and exits 1.  ~40 s.
"""

import sys
import time

import numpy as np

from harl.envs.smac.smac_ns.pact1_core import AgentRLS, rls_confidence_pred

from .channel import BoundedRLS, PactConfig, TrunkChannel
from .coupling import Coupling, _LoneCoupling
from .driver import DialParams, ThermalDriver
from .structure import (ANCHORS, CLASS_NAMES, JOINT_IS_HIP, LEG_OF, N_JOINTS,
                        PARTITIONS, RUNNABLE, class_of, kernel, partition_of,
                        recv_vector)

FAILS = []
P = DialParams()
PERIOD = P.period
PEAK = int(P.period * P.warm_fraction / 2)        # the driver's maximum
COLD = int(P.period * P.warm_fraction)            # first step of the placebo half


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


# ---------------------------------------------------------------------------
#  a synthetic gait: torque streams without a simulator
# ---------------------------------------------------------------------------
class Gait(object):
    """A smooth, phase-offset torque pattern plus noise -- a stand-in for a
    walking controller.  Deterministic given the seed, so two arms driven by the
    same Gait see the identical commanded stream."""

    def __init__(self, seed, amp=0.7, noise=0.25, w=0.25):
        self.rng = np.random.RandomState(seed)
        self.phase = self.rng.uniform(0, 2 * np.pi, N_JOINTS)
        self.amp, self.noise, self.w = amp, noise, w

    def __call__(self, t):
        a = self.amp * np.sin(self.w * t + self.phase)
        a = a + self.noise * self.rng.randn(N_JOINTS)
        return np.clip(a, -1.0, 1.0)


def build(agent_conf="4x2", sigma=1.0, pact=None, clock0=0, **dial):
    p = DialParams(severity=sigma, **dial)
    drv = ThermalDriver(p)
    c = Coupling(agent_conf, p, drv.send)
    ch = TrunkChannel(p, drv, c, pact or PactConfig(), clock0=clock0, ctrl_range=1.0)
    ref, scale = c.geometric_reference(samples=512)
    #  a lone agent has NO peer load, so there is nothing to normalise by and
    #  `load_norm` refuses (correctly -- an inert coupling means sigma has no
    #  meaning).  The disturbance is identically zero there, so any positive
    #  constant serves; the layer does the same for a one-agent partition.
    ln = 1.0 if c.n == 1 else c.load_norm(samples=512)
    ch.set_references(ln, ref, scale)
    return ch


def run(ch, steps, seed=0, gait=None):
    """Drive a channel with a synthetic gait; returns the commanded and executed
    torque streams plus the per-step disturbance and sensor."""
    g = gait or Gait(seed)
    A, U, D, Y = [], [], [], []
    for t in range(steps):
        a = g(t)
        u = ch.step(a)
        A.append(a.copy())
        U.append(u.copy())
        D.append(ch.d.copy())
        Y.append(ch.y.copy())
    return (np.array(A), np.array(U), np.array(D), np.array(Y)), g


# ===================================================================== operator
def t_operator():
    print("operator  (NS-1.2, P-3.1, P-3.2, P-3.3, gate 4)")
    p = DialParams()
    drv = ThermalDriver(p)
    c = Coupling("4x2", p, drv.send)
    st = c.operator_stats()
    check("zero_diagonal", st["diag_max"] == 0.0)
    check("asymmetric_and_spread",
          st["asymmetry"] > 0.05 and st["spread"] > 0.05 and st["ratio"] > 1.5,
          "asym=%.3f spread=%.3f ratio=%.1fx links=%d"
          % (st["asymmetry"], st["spread"], st["ratio"], st["n_links"]))
    off = c.W[c.W > 0]
    proxy = np.ones_like(off)
    check("a_flat_proxy_really_is_flat", float(proxy.std()) == 0.0,
          "the proxy NS-1.2 forbids has spread 0 against the real %.3f" % st["spread"])
    try:
        check("vectorised_basis_equals_brute_force", True, c.verify())
    except AssertionError as ex:
        check("vectorised_basis_equals_brute_force", False, str(ex))

    lone = _LoneCoupling(p, drv.send)
    _, x1 = lone.project(lone._settle(np.ones(N_JOINTS)))
    check("n1_gives_exactly_zero_peer_load", float(np.abs(x1).max()) == 0.0)

    #  the declared table's own internal consistency: gate 4 checks it against
    #  the loaded model, and this checks it against itself (a hip and its own
    #  ankle belong to one leg, the ankle sits further out, and the actuator
    #  order really is legs 4, 1, 2, 3 in hip/ankle pairs).
    pairs_ok = all(LEG_OF[2 * k] == LEG_OF[2 * k + 1] and JOINT_IS_HIP[2 * k]
                   and not JOINT_IS_HIP[2 * k + 1] for k in range(4))
    radii = np.linalg.norm(ANCHORS, axis=1)
    check("declared_joint_table_is_self_consistent",
          pairs_ok and list(LEG_OF[::2]) == [4, 1, 2, 3]
          and all(radii[2 * k + 1] > radii[2 * k] for k in range(4)),
          "ctrl pairs are (hip, ankle) of legs %s, ankles further out"
          % list(LEG_OF[::2]))
    K = kernel(p.length_scale)
    same_leg = K[0, 1]
    adjacent = K[0, 2]
    diagonal = K[0, 4]
    check("kernel_orders_the_load_paths_by_geometry",
          same_leg > adjacent > diagonal,
          "own ankle %.3f > adjacent leg's hip %.3f > diagonal leg's hip %.3f"
          % (same_leg, adjacent, diagonal))
    check("classes_are_independent_of_the_partition",
          len({Coupling(k, p, drv.send).r for k in RUNNABLE}) == 1
          and Coupling("8x1", p, drv.send).r == len(CLASS_NAMES),
          "r = %d for every one of %s (P-1.1)" % (c.r, list(RUNNABLE)))
    check("class_of_is_a_partition_of_every_ordered_pair",
          all(class_of(a, b) in (0, 1, 2) for a in range(N_JOINTS) for b in range(N_JOINTS))
          and class_of(0, 2) == 0 and class_of(1, 3) == 1 and class_of(0, 3) == 2)
    check("front_paths_cost_more_than_cross_by_declaration",
          drv.send[0] > drv.send[1] > drv.send[2],
          "send = %s for %s (mean 1)" % (np.round(drv.send, 3).tolist(), list(CLASS_NAMES)))
    rv = recv_vector(p.recv_spread)
    check("recv_is_normalised_and_spread",
          abs(float(rv.mean()) - 1.0) < 1e-12 and float(rv.max() / rv.min()) > 1.5,
          "recv in [%.2f, %.2f], ratio %.1fx -- a DECLARED per-actuator "
          "heterogeneity, and the only source of the operator's asymmetry"
          % (rv.min(), rv.max(), rv.max() / rv.min()))

    #  P-3.3 / trap 4: the centred regressor must be O(1).  Getting the
    #  reference wrong gave psi = 11.8 on a previous instance, which wrecked the
    #  design matrix while fit_gain still read 0.86.
    ref, scale = c.geometric_reference(samples=512)
    rng = np.random.RandomState(3)
    zs = []
    for _ in range(200):
        _, x = c.project(c._settle(rng.uniform(-1, 1, N_JOINTS)))
        zs.append(c.design(x, ref, scale)[:, 1:])
    z = np.concatenate(zs)
    check("centred_regressor_is_order_one",
          float(np.abs(z).mean()) < 2.0 and float(np.abs(z).max()) < 15.0,
          "mean|z|=%.2f max|z|=%.2f ref=%s scale=%s"
          % (np.abs(z).mean(), np.abs(z).max(), np.round(ref, 3).tolist(),
             np.round(scale, 3).tolist()))
    ln = c.load_norm(samples=512)
    check("load_norm_is_positive", ln > 0.05, "load_norm=%.4f" % ln)
    #  every partition must cover every joint exactly once
    cover = all(sorted(j for g in partition_of(k) for j in g) == list(range(N_JOINTS))
                for k in PARTITIONS)
    check("every_partition_covers_every_joint_once", cover,
          "%s" % sorted(PARTITIONS))


# ======================================================================== dial
def t_dial():
    print("dial  (NS-2.1, NS-2.2, NS-2.3, NS-2.4, NS-2.5, NS-5.1)")
    drv = ThermalDriver(P)
    try:
        facts = drv.certify()
        check("certify_passes_over_the_whole_domain", True,
              "placebo %d/%d, %s" % (facts["placebo_steps"], facts["period"],
                                     facts["anchor"]))
    except AssertionError as ex:
        check("certify_passes_over_the_whole_domain", False, str(ex))
    t = np.arange(PERIOD)
    check("identity_at_zero_is_exact", np.all(drv.amplitude(t, 0.0) == 0.0))
    a1, a2 = drv.amplitude(t, 1.0), drv.amplitude(t, 2.0)
    check("monotone_and_never_generous", np.all(a2 >= a1) and np.all(a1 >= 0.0))
    pl = drv.is_placebo(t)
    check("placebo_regime_is_exactly_inert",
          np.all(drv.amplitude(t, 3.0)[pl] == 0.0) and int(pl.sum()) == PERIOD // 2)
    pk = float(drv.amplitude(PEAK, 1.0))
    check("anchor_is_a_stated_calibration_not_a_published_constant", abs(pk - 0.14) < 1e-12,
          "peak amplitude at sigma=1 is %.3f of the torque range; write 'stated "
          "calibration' in the paper" % pk)
    mp = ThermalDriver(DialParams(mean_preserving=True))
    check("mean_preserving_normalises_the_cycle_mean",
          abs(float(mp.amplitude(t, 1.0).mean()) - 0.14) < 1e-9
          and np.all(mp.amplitude(t, 1.0)[pl] == 0.0))
    #  the driver must be slow against an episode (Ant's limit is 1000 steps),
    #  or the surface would drift within one episode and the agent would be
    #  chasing it rather than living with it
    Aa = drv.A(np.arange(PERIOD))
    per_step = float(np.abs(np.diff(Aa)).max())
    per_mem = float(np.abs(Aa[100:] - Aa[:-100]).max())      # the RLS memory, mu=0.99
    per_ep = float(np.abs(Aa[1000:] - Aa[:-1000]).max())     # Ant's episode limit
    check("driver_is_slow_against_the_estimator's_own_memory", per_mem < 0.05,
          "A moves at most %.4f per step, %.3f over the 100-step RLS memory and "
          "%.3f over a full 1000-step episode -- slow against the control loop and "
          "the estimator, NOT frozen within an episode (the README says so)"
          % (per_step, per_mem, per_ep))


# ===================================================================== channel
def t_channel():
    print("channel  (I.2, NS-1.4, NS-3.3, P-2.1, P-7.1, II.6 -- the invertible cell)")
    (A, U, _, _), _ = run(build(sigma=0.0, clock0=PEAK), 400)
    check("identity_when_the_dial_is_off", np.array_equal(A, U),
          "400 steps at the driver peak: executed torque == commanded, bit for bit")
    (A, U, _, _), _ = run(build(sigma=3.0, clock0=COLD), 400)
    check("placebo_produces_no_disturbance", np.array_equal(A, U))

    #  I.2 / gate 3: the lone-agent projection is untouched at any severity
    ch1 = build("1x8", sigma=3.0, clock0=PEAK)
    (A, U, D, _), _ = run(ch1, 300)
    check("n1_reads_exactly_zero_at_sigma_3",
          np.array_equal(A, U) and float(np.abs(D).max()) == 0.0
          and ch1.n_dial_live == 0)
    #  and its RUNNABLE counterpart: peers that DELIVERED nothing load nobody.
    #  (Commanding them zero is not the same thing -- their own actuators are
    #  disturbed too, so they deliver a little, and that little is real load.
    #  What the claim is about is DELIVERED torque, so that is what is zeroed.)
    ch = build(sigma=3.0, clock0=PEAK)
    a = np.zeros(N_JOINTS)
    a[0:2] = [0.8, -0.6]                           # agent 0 drives, nobody else
    for t in range(200):
        ch.tau_prev[:] = 0.0
        u = ch.step(a)
    check("peers_that_delivered_nothing_give_exactly_zero_at_sigma_3",
          float(np.abs(ch.d).max()) == 0.0 and np.array_equal(u, a)
          and ch.n_dial_live == 0,
          "one agent pushing hard, every peer's delivered torque zero: |d| = 0.0")

    #  SELF-EXCLUSION, measured: what I DELIVERED must not change what I feel.
    #  Stated instantaneously, which is the only way it is true and the only way
    #  it is claimed: ``d_i(t)`` is a function of the OTHER joints' delivered
    #  torques at t-1.  Over several steps my own torque does come back to me --
    #  it changes what my peers deliver, and they load me -- and that loop is the
    #  compensation commons (I.5), not a leak in ``j != i``.
    base = np.array([0.3, -0.4, 0.6, 0.2, -0.5, 0.35, 0.45, -0.25])
    da = []
    for own in ([0.9, -0.9], [-0.2, 0.4]):
        ch = build(sigma=2.0, clock0=PEAK)
        tau = base.copy()
        tau[0:2] = own                             # MY OWN delivered torque
        ch.tau_prev = tau
        ch.step(np.zeros(N_JOINTS))
        da.append(ch.d[0:2].copy())
    check("my_own_delivered_torque_does_not_load_me_(j_!=_i)",
          float(np.abs(da[0] - da[1]).max()) == 0.0,
          "identical peers, own delivered torque %s vs %s: d identical to %.1e"
          % ([0.9, -0.9], [-0.2, 0.4], float(np.abs(da[0] - da[1]).max())))

    #  THE (B)-vs-(C) DECISION PROCEDURE, as a measurement
    ch1b = build("1x8", sigma=3.0, clock0=PEAK, direct=True)
    (A, U, D, _), _ = run(ch1b, 300)
    check("direct_B_control_makes_a_lone_agent_feel_it",
          not np.array_equal(A, U) and float(np.abs(D).max()) > 0
          and ch1b.n_dial_live > 0,
          "the dial reached the lone agent on %d of 300 steps" % ch1b.n_dial_live)

    #  THE (A)-vs-(C) DECISION PROCEDURE: partners acting but FROZEN, the drift
    #  persists -- so this is not learning-induced non-stationarity
    ch = build(sigma=1.0)
    ds, As = [], []
    frozen = np.array([0.0, 0.0, 0.6, -0.4, 0.5, 0.3, -0.7, 0.2])
    for t in range(0, PERIOD, 50):
        ch.step(frozen, clock=t)
        ds.append(abs(ch.mag[0]))
        As.append(ch.A)
    ds, As = np.array(ds), np.array(As)
    live = As > 0
    corr = float(np.corrcoef(ds[live], As[live])[0, 1])
    check("frozen_partners_do_not_remove_the_drift_(not_A)",
          corr > 0.99 and ds[live].max() > 0 and np.all(ds[~live] == 0.0),
          "corr(|d|, A) = %.4f with every peer torque held constant" % corr)

    #  P-2.1: the sensor measures the disturbance, unbiased, with no clipping
    #  Away from the actuator's own limit the sensor IS the disturbance.  The
    #  gait is kept clear of the limit on purpose: where the actuator saturates
    #  the drive really did deliver something else, and the sensor is right to
    #  report that -- so a test that mixed the two would be testing nothing.
    ch = build(sigma=1.0, clock0=PEAK)
    (_, _, D, Y), _ = run(ch, 600, gait=Gait(4, amp=0.35, noise=0.10))
    mags = np.array([np.linalg.norm(D[t, 0:2]) for t in range(D.shape[0])])
    fin = np.isfinite(Y[:, 0])
    err = float(np.abs(np.abs(Y[fin, 0]) - mags[fin]).max())
    check("sensor_is_the_disturbance_along_the_public_direction",
          err < 1e-9 and fin.mean() > 0.95 and ch.n_clipped == 0,
          "max |  |y| - |d|  | = %.2e on the %.0f%% of steps where a direction "
          "exists, with 0 actuator saturations (y is NaN before any peer load "
          "has arrived, which is correct)" % (err, 100 * fin.mean()))

    #  P-7.1 the floor: trust 0 with a GARBAGE estimate is the blind stream, bit
    #  for bit
    blind = build(sigma=2.0, clock0=PEAK)
    off = build(sigma=2.0, clock0=PEAK, pact=PactConfig(enabled=True, trust="off"))
    for r in off.rls:
        r.beta[:] = 1e6
    (A1, U1, _, _), _ = run(blind, 500, seed=11)
    (A2, U2, _, _), _ = run(off, 500, seed=11)
    check("floor_property_is_exact_(pactoff_==_blind)",
          np.array_equal(U1, U2) and np.array_equal(A1, A2) and not np.array_equal(A1, U1),
          "500 steps identical, and the dial really was live in both")

    #  II.6 the inverse: the oracle cancels EXACTLY, so its trajectory is the
    #  sigma = 0 trajectory.  corr_clip is disabled so the identity is the
    #  channel's and not the relief valve's.
    orc = build(sigma=2.0, clock0=PEAK, corr_clip=0.0,
                pact=PactConfig(enabled=True, trust="fixed", g_fixed=1.0, oracle=True))
    (A, U, D, _), _ = run(orc, 500, seed=11)
    check("oracle_cancels_exactly_(channel_inverse)",
          np.array_equal(A, U) and D.max() > 0 and orc.n_dial_live > 0
          and orc.n_harmed == 0,
          "dial live on %d steps, executed == commanded on all 500, max |d| = %.3f"
          % (orc.n_dial_live, np.abs(D).max()))
    #  ... and the relief valve: past it the oracle can no longer cancel, which
    #  is what bounds sigma* by physics rather than by a number we chose
    sat = build(sigma=30.0, clock0=PEAK,
                pact=PactConfig(enabled=True, trust="fixed", g_fixed=1.0, oracle=True))
    (A, U, _, _), _ = run(sat, 300, seed=11)
    check("oracle_saturates_at_corr_clip_(sigma_star_is_declared_physics)",
          not np.array_equal(A, U) and sat.n_corr_clipped > 0,
          "the correction hit its bound %d times at sigma=30" % sat.n_corr_clipped)

    ch = build(sigma=1.0, clock0=PEAK)
    run(ch, 400)
    check("layer_counts_what_it_touched",
          ch.n_dial_live > 0 and ch.n_harmed > 0,
          "dial_live=%d harmed=%d actuator_clipped=%d"
          % (ch.n_dial_live, ch.n_harmed, ch.n_clipped))
    check("close_report_refuses_an_inert_severity_arm",
          build(sigma=1.0).close_report(tag="  [selftest]") is False)
    check("close_report_accepts_the_oracle_(dial_live_but_nothing_enacted)",
          orc.close_report(tag="  [selftest]") is True)

    bad = build(sigma=1.0, clock0=PEAK,
                pact=PactConfig(enabled=True, trust="fixed", warmup=0))
    for r in bad.rls:
        r.beta[:] = np.nan
    (_, U, _, _), _ = run(bad, 200)
    check("non_finite_estimate_is_treated_as_no_information",
          bad.n_diverged > 0 and np.all(np.isfinite(U))
          and float(np.abs(bad.c).max()) == 0.0)
    huge = build(sigma=1.0, clock0=PEAK,
                 pact=PactConfig(enabled=True, trust="fixed", warmup=0))
    g = Gait(5)
    cs = []
    for t in range(200):
        for r in huge.rls:
            r.beta[:] = 50.0
        huge.step(g(t))
        cs.append(float(np.abs(huge.c).max()))
    check("corr_clip_bounds_the_correction",
          max(cs) <= huge.p.corr_clip + 1e-12 and max(cs) > 0,
          "max|c| = %.3f with corr_clip = %.1f of the torque range"
          % (max(cs), huge.p.corr_clip))
    check("executed_torque_never_leaves_the_actuator_range",
          float(np.abs(U).max()) <= 1.0 + 1e-12)


# =================================================================== estimator
def t_estimator():
    print("estimator  (II.4, II.5, P-4.2, P-5.2 -- URB's core, vendored)")
    rng = np.random.RandomState(0)
    beta = np.array([0.1, 0.4, -0.2, 0.05])
    r = AgentRLS(4, mu=0.999, p0=10.0)
    for _ in range(800):
        psi = np.concatenate([[1.0], rng.randn(3)])
        r.update(psi[None], np.array([psi @ beta + 0.01 * rng.randn()]))
    check("rls_recovers_known_beta", float(np.abs(r.beta - beta).max()) < 0.02,
          "max|err| = %.4f" % float(np.abs(r.beta - beta).max()))
    r = AgentRLS(4, mu=0.99, p0=10.0)
    errs = []
    for t in range(10000):
        b = np.array([0.1, 0.4 * np.sin(np.pi * t / 5000.0) ** 2, -0.2, 0.05])
        psi = np.concatenate([[1.0], rng.randn(3)])
        r.update(psi[None], np.array([psi @ b + 0.02 * rng.randn()]))
        if t > 500:
            errs.append(abs(r.beta[1] - b[1]))
    check("rls_tracks_drift", float(np.mean(errs)) < 0.03,
          "mean |beta_1 err| = %.4f while beta_1 cycles 0..0.4 over 10000 steps"
          % float(np.mean(errs)))
    r = AgentRLS(4, mu=0.99, p0=10.0)
    P0 = r.P.copy()
    r.update(np.zeros((1, 4)), np.array([1.0]))
    check("rls_dead_row_does_not_inflate_covariance",
          np.array_equal(r.P, P0) and r.n_updates == 0)
    b = BoundedRLS(4, mu=0.9, p0=10.0, p_trace_max=10.0)
    for _ in range(500):
        b.update(np.array([[1.0, 1.0, 0.0, 0.0]]), np.array([0.3]))
    check("covariance_bound_engages_on_dead_excitation",
          float(np.trace(b.P)) <= 10.0 * 10.0 * 4 + 1e-6 and b.n_bounded > 0,
          "trace = %.1f, cap = %.0f, bounded %d times"
          % (float(np.trace(b.P)), 400.0, b.n_bounded))
    r = AgentRLS(4, mu=0.99, p0=10.0)
    psi = np.array([1.0, 0.5, -0.3, 0.2])
    cold = rls_confidence_pred(r.P, 10.0, 4, psi)
    for _ in range(200):
        r.update(psi[None] + 0.1 * rng.randn(1, 4), np.array([0.1]))
    warm = rls_confidence_pred(r.P, 10.0, 4, psi)
    check("confidence_cold_to_warm", abs(cold - 0.2) < 1e-9 and warm > 0.9,
          "cold = %.3f, warm = %.3f" % (cold, warm))
    for _ in range(3000):
        r.update(psi[None], np.array([0.1]))
    check("pred_confidence_survives_dead_excitation",
          rls_confidence_pred(r.P, 10.0, 4, psi) > 0.9,
          "conf = %.3f after 3000 identical rows (the trace gate would have "
          "disarmed here)" % rls_confidence_pred(r.P, 10.0, 4, psi))


# ================================================================== end to end
def t_end_to_end():
    print("end to end  (II.10 -- does the reduction hold, and does the inverse recover?)")
    t0 = time.time()
    lead, window = 2500, 2500

    def arm(pact=None, clock0=PEAK - lead, **dial):
        ch = build(sigma=2.0, pact=pact, clock0=clock0, **dial)
        _, g = run(ch, (PEAK - clock0) % PERIOD, seed=21)
        fg = float(np.mean(ch.fit_gain))
        bc = float(np.mean(ch.beta_cos))
        be = float(np.mean(ch.beta_err))
        panel = ch.info(0)
        (A, U, D, _), _ = run(ch, window, seed=21, gait=g)
        resid = float(np.abs(U - A).mean())
        return ch, fg, bc, be, resid, float(np.abs(D).mean()), panel

    blind, _, _, _, rb, db, _ = arm()
    full, fg, bc, be, rf, _, panel = arm(PactConfig(enabled=True, trust="fixed"))
    ic, fgi, _, _, ri, _, _ = arm(PactConfig(enabled=True, trust="fixed",
                                             intercept_only=True))
    check("end_to_end_identifies_the_trunk_law", fg > 0.3 and bc > 0.8,
          "at the warm peak: fit_gain=%.3f beta_cos=%.3f beta_relerr=%.3f cond_psi=%.1f"
          % (fg, bc, be, full.cond_psi))
    check("compensation_removes_most_of_the_disturbance", rf < 0.35 * rb,
          "mean |executed - commanded| over the peak window: blind=%.4f pact=%.4f "
          "intercept=%.4f  (the disturbance itself is %.4f)" % (rb, rf, ri, db))
    check("intercept_arm_cannot_identify_on_C", fgi < 0.15 and ri > rf,
          "intercept fit_gain=%.3f -- an intercept-only model against an "
          "intercept-only null" % fgi)

    blindB, _, _, _, rbB, _, _ = arm(direct=True)
    fullB, fgB, _, _, rfB, _, _ = arm(PactConfig(enabled=True, trust="fixed"), direct=True)
    icB, _, _, _, riB, _, _ = arm(PactConfig(enabled=True, trust="fixed",
                                             intercept_only=True), direct=True)
    #  THE MEASUREMENT THE CLASSIFICATION RESTS ON.
    #
    #  It is NOT "fit_gain collapses on B" -- it does not, and asserting that
    #  would have been wrong: a per-agent adaptive INTERCEPT tracks a level shift
    #  perfectly well, so prediction is fine in both cells and fit_gain is high
    #  in both.  What separates the cells is whether DELETING THE PEER CHANNELS
    #  costs anything:
    #
    #     (C)  intercept is much worse than full   -> the coupling is the problem
    #     (B)  intercept is as good as full        -> nothing here is multi-agent
    ratio_C = ri / max(rf, 1e-12)
    ratio_B = riB / max(rfB, 1e-12)
    check("deleting_the_peer_channels_costs_on_C_and_not_on_B",
          ratio_C > 2.0 and ratio_B < 1.5,
          "residual(intercept)/residual(full): C = %.2fx (peer channels are "
          "load-bearing), B = %.2fx (they carry nothing).  B fit_gain = %.3f is "
          "HIGH, and that is the honest result." % (ratio_C, ratio_B, fgB))
    check("cond_psi_is_reported_and_finite",
          np.isfinite(full.cond_psi) and full.cond_psi > 1.0,
          "cond_psi=%.1f (gate 7 warns above 1e3, never aborts)" % full.cond_psi)

    #  the realistic regime: trained through a whole placebo half, then the next
    #  warm phase arrives -- does the estimator re-acquire the law?
    re, fg2, bc2, be2, _, _, _ = arm(PactConfig(enabled=True, trust="fixed"),
                                     clock0=COLD - 500)
    check("estimator_reacquires_the_law_after_a_placebo_half",
          fg2 > 0.3 and bc2 > 0.8,
          "second warm peak after %d cold steps: fit_gain=%.3f beta_cos=%.3f "
          "relerr=%.3f" % (PERIOD // 2, fg2, bc2, be2))

    #  the N-scaling, through the channel rather than the structure
    per_joint = {}
    for conf in ("2x4", "4x2", "8x1"):
        ch = build(conf, sigma=2.0, clock0=PEAK)
        (A, U, D, _), _ = run(ch, 600, seed=7)
        per_joint[conf] = float(np.abs(D).mean())
    check("per_actuator_disturbance_rises_with_N",
          per_joint["2x4"] < per_joint["4x2"] < per_joint["8x1"],
          "mean |d| per joint: 2x4=%.4f 4x2=%.4f 8x1=%.4f (NS-4.2, and all three "
          "are runnable)" % (per_joint["2x4"], per_joint["4x2"], per_joint["8x1"]))

    print("  panel at the warm peak (full arm, agent 0): %s"
          % {k: round(v, 4) for k, v in panel.items()
             if k in ("ns_fit_gain", "ns_beta_cos", "ns_trust", "ns_conf", "ns_pred_err",
                      "ns_clip_frac", "ns_x_std", "ns_cond_psi")})
    print("  (%.1f s)" % (time.time() - t0))


def main():
    t_operator()
    t_dial()
    t_channel()
    t_estimator()
    t_end_to_end()
    if FAILS:
        print("")
        print("FAILED %d check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
