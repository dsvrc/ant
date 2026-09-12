"""The conformance suite.  No gfootball, no torch, no learning framework.

`PACT_NS_SPEC` I.7: *"Fourteen offline checks... Port all of them; each
corresponds to a requirement above and each has failed at least once during
development."*  Plus the channel identities that make THIS cell the invertible
one, and the (B)-vs-(C) and (A)-vs-(C) decision procedures as MEASUREMENTS.

Run::

    python -m harl.envs.football.grf_ns.selftest

Ends with ``ALL CHECKS PASSED`` or lists the failures and exits 1.  ~30 s.
"""

import sys
import time

import numpy as np

from .actions import (DIR_FIRST, DIR_VEC, IDLE, N_DIRS, RELEASE_DIRECTION, SHORT_PASS,
                      SPRINT, rotate, verify_action_table)
from .channel import BoundedRLS, PactConfig, SwerveChannel
from .coupling import Coupling
from .driver import DialParams, PitchDriver
from harl.envs.smac.smac_ns.pact1_core import AgentRLS, rls_confidence_pred

FAILS = []
SPAWN = np.array([[0.60, 0.00], [0.70, 0.20], [0.70, -0.20]])
P = DialParams()
PERIOD = P.period
PEAK = int(P.period * P.wet_fraction / 2)         # the driver's maximum
DRY = int(P.period * P.wet_fraction)              # first step of the placebo half


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-60s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


# ---------------------------------------------------------------------------
#  a synthetic squad: GRF's action semantics without GRF
# ---------------------------------------------------------------------------
class Squad(object):
    """Positions integrate the EXECUTED heading, as the engine would."""

    def __init__(self, n, seed, speed=0.012, box=(0.3, 1.0, -0.4, 0.4)):
        self.n = n
        self.rng = np.random.RandomState(seed)
        self.pos = SPAWN[:n].copy() if n <= 3 else np.column_stack(
            [self.rng.uniform(0.4, 0.9, n), self.rng.uniform(-0.3, 0.3, n)])
        self.speed = speed
        self.box = box
        self.head = np.full(n, -1)
        self.owner = 0

    def policy(self, t):
        """A movement-heavy random policy in GRF's own action indices."""
        a = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            u = self.rng.rand()
            if u < 0.65:
                a[i] = IDLE
            elif u < 0.90:
                a[i] = DIR_FIRST + self.rng.randint(0, N_DIRS)
            elif u < 0.94:
                a[i] = RELEASE_DIRECTION
            elif u < 0.97:
                a[i] = SPRINT
            else:
                a[i] = SHORT_PASS
        if t % 50 == 0:
            self.owner = int(self.rng.randint(0, self.n))
        return a

    def advance(self, exec_a):
        for i in range(self.n):
            if DIR_FIRST <= exec_a[i] < DIR_FIRST + N_DIRS:
                self.head[i] = exec_a[i] - DIR_FIRST
            elif exec_a[i] == RELEASE_DIRECTION:
                self.head[i] = -1
            if self.head[i] >= 0:
                self.pos[i] += self.speed * DIR_VEC[self.head[i]]
        x0, x1, y0, y1 = self.box
        self.pos[:, 0] = np.clip(self.pos[:, 0], x0, x1)
        self.pos[:, 1] = np.clip(self.pos[:, 1], y0, y1)


def build(n=3, sigma=1.0, pact=None, clock0=0, **kw):
    p = DialParams(severity=sigma, **kw)
    drv = PitchDriver(p)
    c = Coupling(n, p, drv.send)
    ch = SwerveChannel(n, p, drv, c, pact or PactConfig(), clock0=clock0)
    pos_ref = SPAWN[:n][None] if n <= 3 else np.column_stack(
        [np.linspace(0.5, 0.8, n), np.linspace(-0.3, 0.3, n)])[None]
    ref, scale = c.geometric_reference(pos_ref)
    ln = c.load_norm(pos_ref) if n > 1 else 0.25
    ch.set_references(ln, ref, scale)
    return ch


def run(ch, steps, seed=0, squad=None):
    """Drive a channel with a synthetic squad; returns the commanded / executed
    action streams and the per-step k, d, y."""
    sq = squad or Squad(ch.n, seed)
    A, E, K, D, Y = [], [], [], [], []
    for t in range(steps):
        a = sq.policy(t)
        e = ch.step(a, sq.pos, sq.owner)
        sq.advance(e)
        A.append(a.copy())
        E.append(e.copy())
        K.append(ch.k.copy())
        D.append(ch.d.copy())
        Y.append(ch.y.copy())
    return (np.array(A), np.array(E), np.array(K), np.array(D), np.array(Y)), sq


# ===================================================================== operator
def t_operator():
    print("operator  (NS-1.2, P-3.1, P-3.2, P-3.3, gate 4)")
    p = DialParams()
    drv = PitchDriver(p)
    c = Coupling(3, p, drv.send)
    head = np.array([4, 4, 4])                    # all heading right
    recv = np.array([p.recv_ball, 1.0, 1.0])
    W = c.W(SPAWN, head, recv)
    st = c.operator_stats(SPAWN, head, recv)
    check("zero_diagonal", np.all(np.diag(W) == 0.0))
    check("asymmetric_and_spread", st["asymmetry"] > 0.05 and st["spread"] > 0.05
          and st["ratio"] > 1.5,
          "asym=%.3f spread=%.3f ratio=%.1fx" % (st["asymmetry"], st["spread"], st["ratio"]))
    off = W[~np.eye(3, dtype=bool)]
    proxy = np.where(off > 0, 1.0, 0.0)
    check("a_flat_proxy_really_is_flat", float(proxy[proxy > 0].std()) == 0.0,
          "proxy spread 0 against the real %.3f" % st["spread"])
    try:
        msg = c.verify(SPAWN[None])
        check("vectorised_basis_equals_brute_force", True, msg)
    except AssertionError as ex:
        check("vectorised_basis_equals_brute_force", False, str(ex))
    c1 = Coupling(1, p, drv.send)
    x1, s1 = c1.channels(SPAWN[:1], np.array([4]), np.array([p.recv_ball]))
    check("n1_gives_exactly_zero_peer_load", float(np.abs(x1).max()) == 0.0
          and float(np.abs(s1).max()) == 0.0)
    x, s = c.channels(np.array([[0.5, 0.0], [0.4, 0.0], [0.1, 0.05]]), np.array([4, 4, 4]))
    check("rear_teammates_are_not_in_the_lane", float(x[0].sum()) == 0.0 and s[0] == 0.0,
          "two peers behind a right-heading player read x=%s" % np.round(x[0], 4).tolist())
    x, _ = c.channels(np.array([[0.5, 0.0], [0.6, 0.0], [0.5, 0.1]]), np.array([4, 4, 4]))
    check("front_and_flank_are_classified_by_geometry", x[0, 0] > 0 and x[0, 1] > 0
          and x[0, 0] > x[0, 1], "x_front=%.3f x_flank=%.3f" % (x[0, 0], x[0, 1]))
    check("front_costs_more_than_flank_by_declaration", drv.send[0] > drv.send[1],
          "send=%s (mean 1)" % np.round(drv.send, 3).tolist())
    try:
        msg = verify_action_table()
        ok = all(rotate(rotate(h, k), -k) == h for h in range(8) for k in range(-9, 10))
        check("action_table_is_the_compass_group", ok, msg)
    except AssertionError as ex:
        check("action_table_is_the_compass_group", False, str(ex))
    ref, scale = c.geometric_reference(SPAWN[None])
    rng = np.random.RandomState(3)
    zs = []
    for _ in range(200):
        pos = SPAWN + rng.randn(3, 2) * 0.1
        xx, _ = c.channels(pos, rng.randint(0, 8, 3))
        zs.append(c.design(xx, ref, scale)[:, 1:])
    z = np.concatenate(zs)
    check("centred_regressor_is_order_one", float(np.abs(z).mean()) < 2.0
          and float(np.abs(z).max()) < 15.0,
          "mean|z|=%.2f max|z|=%.2f  ref=%s scale=%s" % (np.abs(z).mean(), np.abs(z).max(),
                                                          np.round(ref, 3).tolist(),
                                                          np.round(scale, 3).tolist()))
    ln = c.load_norm(SPAWN[None])
    check("load_norm_is_positive_at_spawn", ln > 0.05, "load_norm=%.4f" % ln)


# ======================================================================== dial
def t_dial():
    print("dial  (NS-2.1, NS-2.2, NS-2.3, NS-2.4, NS-2.5, NS-5.1)")
    drv = PitchDriver(P)
    try:
        facts = drv.certify()
        check("certify_passes_over_the_whole_domain", True,
              "placebo %d/%d, %s" % (facts["placebo_steps"], facts["period"], facts["anchor"]))
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
          "peak amplitude at sigma=1 is %.3f step/step; write 'stated calibration'" % pk)
    mp = PitchDriver(DialParams(mean_preserving=True))
    check("mean_preserving_normalises_the_cycle_mean",
          abs(float(mp.amplitude(t, 1.0).mean()) - 0.14) < 1e-9
          and np.all(mp.amplitude(t, 1.0)[pl] == 0.0))
    check("driver_is_slow_relative_to_an_episode",
          float(drv.A(PEAK + 400) - drv.A(PEAK)) > -0.05,
          "A changes by %.3f over a full 400-step episode at the peak"
          % float(drv.A(PEAK + 400) - drv.A(PEAK)))


# ===================================================================== channel
def t_channel():
    print("channel  (I.2, NS-1.4, NS-3.3, P-2.1, P-7.1, II.6 -- the invertible cell)")
    (A, E, K, _, _), _ = run(build(sigma=0.0, clock0=PEAK), 600)
    check("identity_when_the_dial_is_off", np.array_equal(A, E) and not np.any(K),
          "600 steps at the driver peak: executed == commanded")
    (A, E, K, _, _), _ = run(build(sigma=3.0, clock0=DRY), 600)
    check("placebo_produces_no_rotation", np.array_equal(A, E) and not np.any(K))
    (A, E, K, D, _), _ = run(build(n=1, sigma=3.0, clock0=PEAK), 400)
    check("n1_reads_exactly_zero_at_sigma_3", np.array_equal(A, E) and not np.any(K)
          and float(np.abs(D).max()) == 0.0)
    # THE (B)-vs-(C) DECISION PROCEDURE, as a measurement
    (A, E, K, D, _), _ = run(build(n=1, sigma=3.0, clock0=PEAK, direct=True), 400)
    check("direct_B_control_makes_a_lone_player_feel_it", np.any(K != 0) and D.max() > 0,
          "rotations enacted on %d of 400 steps" % int(np.any(K != 0, axis=1).sum()))
    # THE (A)-vs-(C) DECISION PROCEDURE: partners FROZEN, the drift persists
    ch = build(sigma=1.0)
    pos = SPAWN.copy() + np.array([[0, 0], [0.05, -0.05], [0.05, 0.05]])
    ds, As = [], []
    for t in range(0, PERIOD, 40):
        ch.step(np.array([DIR_FIRST + 4, IDLE, IDLE]), pos, -1, clock=t)     # peers never move
        ds.append(ch.d[0])
        As.append(ch.A)
    ds, As = np.array(ds), np.array(As)
    live = As > 0
    corr = float(np.corrcoef(ds[live], As[live])[0, 1])
    check("frozen_partners_do_not_remove_the_drift_(not_A)", corr > 0.99 and ds[live].max() > 0
          and np.all(ds[~live] == 0.0), "corr(d, A)=%.4f with peers frozen" % corr)
    # the sigma-delta: the duty cycle tracks the disturbance, the sensor is unbiased
    ch = build(sigma=1.0)
    ks, ds, ys = [], [], []
    for t in range(2000):
        ch.step(np.array([DIR_FIRST + 4, IDLE, IDLE]), pos, -1, clock=PEAK)
        ks.append(ch.k[0])
        ds.append(ch.d[0])
        ys.append(ch.y[0])
    ks, ds, ys = np.array(ks, dtype=float), np.array(ds), np.array(ys)
    check("duty_cycle_tracks_the_disturbance", abs(ks.mean() - ds.mean()) <= 0.5 / 2000 + 1e-12
          and 0.05 < ds.mean() < 1.0,
          "mean k=%.4f mean d=%.4f (bounded by 1/2T)" % (ks.mean(), ds.mean()))
    check("sensor_is_unbiased_for_the_disturbance", abs(np.nanmean(ys) - ds.mean()) < 1e-3,
          "mean y=%.4f" % np.nanmean(ys))
    # the swerve is AWAY from the traffic: a teammate at -y of a right-heading player
    ch = build(sigma=3.0)
    for t in range(60):
        ch.step(np.array([DIR_FIRST + 4, IDLE, IDLE]),
                np.array([[0.5, 0.0], [0.55, -0.04], [0.9, 0.4]]), -1, clock=PEAK)
    check("swerve_is_away_from_the_traffic", ch.e[0] == 1.0 and ch.applied_rot[0] >= 0
          and DIR_VEC[rotate(4, 1)][1] > 0,
          "e=%+.0f: rotates toward +y, away from the teammate at -y" % ch.e[0])
    # P-7.1 the floor: trust 0 with a GARBAGE estimator is the blind stream bit for bit
    blind = build(sigma=2.0, clock0=PEAK)
    off = build(sigma=2.0, clock0=PEAK, pact=PactConfig(enabled=True, trust="off"))
    for r in off.rls:
        r.beta[:] = 1e6
    (A1, E1, K1, _, _), _ = run(blind, 800, seed=11)
    (A2, E2, K2, _, _), _ = run(off, 800, seed=11)
    check("floor_property_is_exact_(pactoff_==_blind)", np.array_equal(E1, E2)
          and np.array_equal(K1, K2) and np.array_equal(A1, A2) and np.any(K1 != 0),
          "800 steps, %d rotations enacted in both" % int(np.sum(K1 != 0)))
    # II.6 the inverse: the oracle cancels EXACTLY -- the executed stream is the
    # commanded stream, i.e. the sigma = 0 trajectory.  corr_clip off so that the
    # identity is the channel's and not the relief valve's.
    orc = build(sigma=2.0, clock0=PEAK, corr_clip=0.0,
                pact=PactConfig(enabled=True, trust="fixed", g_fixed=1.0, oracle=True))
    (A, E, K, D, _), _ = run(orc, 800, seed=11)
    check("oracle_cancels_exactly_(channel_inverse)", np.array_equal(A, E) and not np.any(K)
          and D.max() > 0 and orc.n_dial_live > 0,
          "dial live on %d steps, rotations enacted 0, max d=%.3f" % (orc.n_dial_live, D.max()))
    sat = build(sigma=40.0, clock0=PEAK,
                pact=PactConfig(enabled=True, trust="fixed", g_fixed=1.0, oracle=True))
    (A, E, K, D, _), _ = run(sat, 400, seed=11)
    check("oracle_saturates_at_corr_clip_(sigma_star_is_declared_physics)",
          np.any(K != 0) and sat.n_corr_clipped > 0,
          "corr clipped %d times at sigma=40" % sat.n_corr_clipped)
    ch = build(sigma=1.0, clock0=PEAK)
    run(ch, 600)
    check("layer_counts_what_it_touched", ch.n_dial_live > 0 and ch.n_harmed > 0
          and ch.n_rot_sent > 0,
          "dial_live=%d harmed=%d rotated=%d" % (ch.n_dial_live, ch.n_harmed, ch.n_rot_sent))
    check("close_report_refuses_an_inert_severity_arm",
          build(sigma=1.0).close_report(tag="  [selftest]") is False)
    check("close_report_accepts_the_oracle_(dial_live_but_nothing_enacted)",
          orc.close_report(tag="  [selftest]") is True)
    bad = build(sigma=1.0, clock0=PEAK, pact=PactConfig(enabled=True, trust="fixed", warmup=0))
    for r in bad.rls:
        r.beta[:] = np.nan
    (A, E, K, _, _), _ = run(bad, 200)
    check("non_finite_estimate_is_treated_as_no_information",
          bad.n_diverged > 0 and np.all(np.isfinite(E)) and float(np.abs(bad.c).max()) == 0.0)
    huge = build(sigma=1.0, clock0=PEAK, pact=PactConfig(enabled=True, trust="fixed", warmup=0))
    sq = Squad(3, 5)
    cs = []
    for t in range(100):
        for r in huge.rls:
            r.beta[:] = 50.0
        e = huge.step(sq.policy(t), sq.pos, sq.owner)
        sq.advance(e)
        cs.append(float(np.abs(huge.c).max()))
    check("corr_clip_bounds_the_correction", max(cs) <= huge.p.corr_clip + 1e-12 and max(cs) > 0,
          "max|c|=%.3f with corr_clip=%.1f" % (max(cs), huge.p.corr_clip))


# =================================================================== estimator
def t_estimator():
    print("estimator  (II.4, II.5, P-4.2, P-5.2 -- URB's core, vendored)")
    rng = np.random.RandomState(0)
    beta = np.array([0.1, 0.4, -0.2])
    r = AgentRLS(3, mu=0.999, p0=10.0)
    for _ in range(600):
        psi = np.array([1.0, rng.randn(), rng.randn()])
        r.update(psi[None], np.array([psi @ beta + 0.01 * rng.randn()]))
    check("rls_recovers_known_beta", float(np.abs(r.beta - beta).max()) < 0.02,
          "max|err|=%.4f" % float(np.abs(r.beta - beta).max()))
    r = AgentRLS(3, mu=0.99, p0=10.0)
    errs = []
    for t in range(6000):
        b = np.array([0.1, 0.4 * np.sin(np.pi * t / 3000.0) ** 2, -0.2])
        psi = np.array([1.0, rng.randn(), rng.randn()])
        r.update(psi[None], np.array([psi @ b + 0.02 * rng.randn()]))
        if t > 500:
            errs.append(abs(r.beta[1] - b[1]))
    check("rls_tracks_drift", float(np.mean(errs)) < 0.03,
          "mean |beta_1 err| = %.4f while beta_1 cycles 0..0.4 over 6000 steps"
          % float(np.mean(errs)))
    r = AgentRLS(3, mu=0.99, p0=10.0)
    P0 = r.P.copy()
    r.update(np.zeros((1, 3)), np.array([1.0]))
    check("rls_dead_row_does_not_inflate_covariance", np.array_equal(r.P, P0) and r.n_updates == 0)
    b = BoundedRLS(3, mu=0.9, p0=10.0, p_trace_max=10.0)
    for _ in range(500):
        b.update(np.array([[1.0, 1.0, 0.0]]), np.array([0.3]))
    check("covariance_bound_engages_on_dead_excitation",
          float(np.trace(b.P)) <= 10.0 * 10.0 * 3 + 1e-6 and b.n_bounded > 0,
          "trace=%.1f cap=%.0f bounded=%d" % (float(np.trace(b.P)), 300.0, b.n_bounded))
    r = AgentRLS(3, mu=0.99, p0=10.0)
    psi = np.array([1.0, 0.5, -0.3])
    cold = rls_confidence_pred(r.P, 10.0, 3, psi)
    for _ in range(200):
        r.update(psi[None] + 0.1 * rng.randn(1, 3), np.array([0.1]))
    warm = rls_confidence_pred(r.P, 10.0, 3, psi)
    check("confidence_cold_to_warm", abs(cold - 0.25) < 1e-9 and warm > 0.9,
          "cold=%.3f warm=%.3f" % (cold, warm))
    for _ in range(3000):
        r.update(psi[None], np.array([0.1]))
    check("pred_confidence_survives_dead_excitation",
          rls_confidence_pred(r.P, 10.0, 3, psi) > 0.9,
          "conf=%.3f after 3000 identical rows" % rls_confidence_pred(r.P, 10.0, 3, psi))

# ================================================================== end to end
def t_end_to_end():
    print("end to end  (II.10 -- does the reduction hold, and does the inverse recover?)")
    t0 = time.time()
    lead, window = 1500, 3000            # ramp into the peak, then read across the peak

    def arm(pact=None, clock0=PEAK - lead, **kw):
        ch = build(sigma=2.0, pact=pact, clock0=clock0, **kw)
        _, sq = run(ch, (PEAK - clock0) % PERIOD, seed=21)        # to the NEXT peak
        fg, bc, be = float(np.mean(ch.fit_gain)), float(np.mean(ch.beta_cos)), \
            float(np.mean(ch.beta_err))
        panel = ch.info(0)
        (_, _, K, _, _), _ = run(ch, window, seed=21, squad=sq)
        return ch, fg, bc, be, float(np.mean(np.abs(K) > 0)), panel

    blind, _, _, _, rb, _ = arm()
    full, fg, bc, be, rf, panel = arm(PactConfig(enabled=True, trust="fixed"))
    ic, fgi, _, _, ri, _ = arm(PactConfig(enabled=True, trust="fixed", intercept_only=True))
    check("end_to_end_identifies_the_lane_law", fg > 0.3 and bc > 0.8,
          "at the wet peak: fit_gain=%.3f beta_cos=%.3f beta_relerr=%.3f cond_psi=%.1f"
          % (fg, bc, be, full.cond_psi))
    check("compensation_removes_most_of_the_enacted_swerve", rf < 0.5 * rb,
          "rotation rate over the peak window: blind=%.3f pact=%.3f intercept=%.3f"
          % (rb, rf, ri))
    check("intercept_arm_cannot_identify_on_C", fgi < 0.15 and ri > rf,
          "intercept fit_gain=%.3f (an intercept-only model against an intercept-only null)"
          % fgi)
    blindB, _, _, _, rbB, _ = arm(direct=True)
    fullB, fgB, _, _, rfB, _ = arm(PactConfig(enabled=True, trust="fixed"), direct=True)
    icB, _, _, _, riB, _ = arm(PactConfig(enabled=True, trust="fixed", intercept_only=True),
                               direct=True)
    check("direct_B_puts_everything_in_the_intercept", fgB < 0.25 and riB < 0.5 * rbB,
          "B: full fit_gain=%.3f | rotation rate blind=%.3f intercept=%.3f full=%.3f"
          % (fgB, rbB, riB, rfB))
    check("cond_psi_is_reported_and_finite", np.isfinite(full.cond_psi) and full.cond_psi > 1.0,
          "cond_psi=%.1f (gate 7 warns above 1e3, never aborts)" % full.cond_psi)
    # the realistic regime: trained through a whole placebo half, then the next
    # wet phase arrives -- does the estimator re-acquire the law?
    re, fg2, bc2, be2, _, _ = arm(PactConfig(enabled=True, trust="fixed"), clock0=DRY - 500)
    check("estimator_reacquires_the_law_after_a_placebo_half", fg2 > 0.3 and bc2 > 0.8,
          "second wet peak after 6000 dry steps: fit_gain=%.3f beta_cos=%.3f relerr=%.3f"
          % (fg2, bc2, be2))
    print("  panel at the wet peak (full arm, agent 0): %s"
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
        print("\nFAILED %d check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
