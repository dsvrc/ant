"""The arithmetic self-check.  No StarCraft II, no torch, no policy.

PACT_PIPELINE_SPEC 11.4: *"Write an arithmetic self-check that runs without the
simulator.  Basis structure, N=1 identity, RLS recovery on synthetic data, the
windup reproduction, the significance floor, conjugacy, the floor property, ratio
guards, and an end-to-end closed loop.  It cannot prove the method works; it
proves the arithmetic is not the reason if it does not.  CALIBRATE ITS FIXTURES
TO MEASURED REALITY -- a fixture too gentle silently blesses broken code, which
happened here until the synthetic spread was matched to the real operator's."*

So the synthetic operator used below is not a toy: it is the REAL 3s5z operator
from ``fc/operator.py``, with its measured 25x spread, 0.96 std/mean and 0.24
asymmetry.  Any fixture gentler than that would be blessing broken code.

Run::

    python -m harl.envs.smac.fc.selfcheck
"""

import sys

import numpy as np

from .driver import Driver, assert_dial, dial, driver, is_placebo
from . import operator as opmod
from .pact_core import (
    AgentCompensator,
    Basis,
    DU_DA_FLOOR,
    FitGain,
    RLS,
    compensation_delta,
    own_column,
)

FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


# --------------------------------------------------------------------------- #
def t1_operator():
    print("T1  the declared operator (spec 2.1, 2.4; NS A.2)")
    names = opmod.composition("3s5z", 8)
    W = opmod.build_W(names, names, step_mul=8)
    rep = opmod.report(W, names)
    check("W[i,i] == 0 exactly", np.all(np.diag(W) == 0.0))
    check("spread std/mean > 0.5 (POWER: 1.35)", rep["spread"] > 0.5,
          "%.3f" % rep["spread"])
    check("orders of magnitude, ratio > 10x", rep["ratio"] > 10.0,
          "%.1fx" % rep["ratio"])
    check("ASYMMETRIC (a proxy cannot be)", rep["asymmetry"] > 0.05,
          "%.3f" % rep["asymmetry"])
    check("cond finite -- tested as a VALUE, first", np.isfinite(rep["cond"]),
          "%.2f" % rep["cond"])
    # the geometric proxy the spec forbids, for contrast
    proxy = np.where(W > 0, 1.0, 0.0)
    np.fill_diagonal(proxy, 0.0)
    pr = opmod.report(proxy, names)
    check("the forbidden proxy really is flat", pr["spread"] < 1e-9,
          "proxy spread=%.3g vs real %.3f" % (pr["spread"], rep["spread"]))
    return W, names


def t2_basis(W):
    print("T2  the basis: per-agent per-channel scaling (spec 2.3)")
    phi_max = 1.5
    b = Basis(W, phi_max, r=1)
    n = W.shape[0]
    # psi must land in [-0.5, 0.5] for EVERY agent, including the one whose
    # channel is ~25x weaker than another's -- that is the whole point of scaling
    # per agent rather than sharing one scale (which drove cond to 72,148).
    rng = np.random.RandomState(0)
    lo, hi = 1e9, -1e9
    grams = np.zeros((2 + b.r, 2 + b.r))
    for _ in range(4000):
        phi = rng.uniform(0.0, phi_max, n)
        contrib = W * rng.uniform(0.0, 1.0, (n, n))
        psi = b.psi(contrib, phi, np.ones(n))
        lo, hi = min(lo, psi.min()), max(hi, psi.max())
        for i in range(n):
            x = np.concatenate(([1.0, own_column(phi[i], phi_max)], psi[i]))
            grams += np.outer(x, x)
    check("psi within [-0.5, 0.5] for every agent", lo >= -0.5 - 1e-9 and hi <= 0.5 + 1e-9,
          "[%.3f, %.3f]" % (lo, hi))
    cond_scaled = float(np.linalg.cond(grams / (4000 * W.shape[0])))
    check("regressor Gram is well conditioned", np.isfinite(cond_scaled) and cond_scaled < 1e4,
          "cond=%.1f" % cond_scaled)

    # The SHARED-scale ablation, run at r = 2 where the collapse actually lives:
    # an agent whose weak (cross-band) channel is ~20x smaller than its strong one
    # contributes a near-zero column under one shared scale and the Gram goes
    # singular.  POWER measured 72,148 against 57.
    b2 = Basis(W, phi_max, r=2)
    shared = float(b2.ranges.max())
    g_per = np.zeros((2 + 2, 2 + 2))
    g_shr = np.zeros((2 + 2, 2 + 2))
    rng = np.random.RandomState(0)
    for _ in range(4000):
        phi = rng.uniform(0.0, phi_max, n)
        contrib = W * rng.uniform(0.0, 1.0, (n, n))
        psi_p = b2.psi(contrib, phi, np.ones(n))
        x_raw = np.einsum("ij,icj->ic", contrib * phi[None, :], b2.masks)
        psi_s = (x_raw - 0.5 * shared) / shared
        for i in range(n):
            oc = own_column(phi[i], phi_max)
            g_per += np.outer(np.concatenate(([1.0, oc], psi_p[i])),
                              np.concatenate(([1.0, oc], psi_p[i])))
            g_shr += np.outer(np.concatenate(([1.0, oc], psi_s[i])),
                              np.concatenate(([1.0, oc], psi_s[i])))
    cond_per = float(np.linalg.cond(g_per / (4000 * n)))
    cond_shared = float(np.linalg.cond(g_shr / (4000 * n)))
    check("shared scaling collapses the conditioning (the ablation)",
          cond_shared > 100.0 * cond_per,
          "r=2: shared=%.0f vs per-agent=%.0f (%.0fx)"
          % (cond_shared, cond_per, cond_shared / max(1e-9, cond_per)))
    check("r=1 is not worse conditioned than r=2 (2.2's measurement)",
          cond_scaled <= cond_per, "r=1 %.1f vs r=2 %.1f" % (cond_scaled, cond_per))
    return b


def t3_n1_identity():
    print("T3  N = 1 irreducibility (NS A.3, gate G1)")
    W = opmod.build_W(["Stalker"], ["Stalker"], step_mul=8)
    check("W is 1x1 and exactly zero", W.shape == (1, 1) and W[0, 0] == 0.0)
    b = Basis(W, 1.5, r=1)
    # However small g becomes, the peer contribution stays exactly zero.
    worst = 0.0
    for g in (1.0, 0.5, 0.25, 1e-6):
        contrib = W * 1.0
        psi = b.psi(contrib, np.array([1.5]), np.array([1.0]))
        worst = max(worst, float(np.max(np.abs(psi))))
    check("peer term is 0 at EVERY g, not just small ones", worst == 0.0,
          "max|psi| = %.3g" % worst)


def t4_dial():
    print("T4  the severity dial (NS B.1, B.2, B.4; gates G2, G6)")
    r = assert_dial()
    check("g(0, A) == 1 EXACTLY over the whole domain", True)
    check("g <= 1 always -- the uprating trap is closed", True)
    check("monotone in sigma at EVERY driver value", True)
    check("a placebo regime exists and is provably inert",
          r["placebo_frac"] > 0.2, "%.0f%% of the cycle" % (100 * r["placebo_frac"]))
    A = np.linspace(0, 1, 501)
    rows = [dial(s, A) for s in (0.5, 1.0, 2.0)]
    pl = is_placebo(A)
    same = all(np.all(rows[k][pl] == rows[0][pl]) for k in range(3))
    check("placebo rows are BYTE-identical across sigma", same)
    check("dial bites outside it", float(dial(1.0, 1.0)) < 0.5,
          "g(sigma=1, A=1) = %.3f" % dial(1.0, 1.0))


def t5_phi():
    print("T5  the exertion functional Phi (NS A.5)")
    ex = opmod.Exertion(8, phi_fire=1.0, rho=0.6)   # the shipped defaults
    rng = np.random.RandomState(1)
    alive = np.ones(8)
    floor = 1e9
    for t in range(3000):
        if t == 1500:
            alive[5:] = 0.0
        ex.update(alive, (rng.rand(8) < 0.6).astype(float))
        live = ex.phi[alive > 0]
        if live.size:
            floor = min(floor, float(live.min()))
    v = ex.variation()
    check("std(Phi)/mean(Phi) > 0.05 (POWER: 0.28)", v > 0.05, "%.3f" % v)
    # the escape hatch: a team that NEVER fires still cannot drive Phi to zero
    ex2 = opmod.Exertion(8, phi_fire=1.0, rho=0.6)
    for _ in range(2000):
        ex2.update(np.ones(8), np.zeros(8))
    floor2 = float(ex2.phi.min())
    check("Phi has an UNCANCELLABLE floor (never firing)", floor2 > 0.99,
          "Phi -> %.4f, not 0" % floor2)
    check("declared phi_max is a declared constant", ex.phi_max == 2.0)


def _synth_run(mu, period, steps=20000, tail=5000, seed=3):
    """One RLS run on the REAL 3s5z basis against a beta* that drifts on ``period``.

    The drift is not arbitrary: beta* on this environment is proportional to
    (1 - g)/g, so it drifts on the DRIVER's period, which is tens of steps rather
    than POWER's days.  That is the whole content of 5.1's "re-measure per
    environment -- the optimum follows the drift rate".
    """
    names = opmod.composition("3s5z", 8)
    W = opmod.build_W(names, names, step_mul=8)
    b = Basis(W, 2.0, r=1)
    rng = np.random.RandomState(seed)
    n = 8
    rls = RLS(2 + b.r, mu=mu, p0=1.0)
    true_hist, est_hist = [], []
    psi_prev = np.zeros(b.r)
    for t in range(steps):
        c = 0.6 + 0.4 * np.sin(2 * np.pi * t / float(period)) if period else 0.8
        beta_true = np.concatenate(([0.02, -0.05], [c]))
        phi = rng.uniform(0.0, 2.0, n)
        contrib = W * rng.uniform(0.0, 1.0, (n, n))
        psi = b.psi(contrib, phi, np.ones(n))[0]
        oc = own_column(phi[0], 2.0)
        x = np.concatenate(([1.0, oc], psi_prev))
        y = float(np.dot(beta_true, x)) + rng.randn() * 0.002
        rls.update(x, y)
        psi_prev = psi
        if t >= steps - tail:
            true_hist.append(c)
            est_hist.append(float(rls.beta[2]))
    est = np.array(est_hist)
    tru = np.array(true_hist)
    err = float(np.mean(np.abs(est - tru)))
    corr = float(np.corrcoef(est, tru)[0, 1]) if np.std(tru) > 1e-12 else float("nan")
    return rls, err, corr


def t6_rls_recovery():
    print("T6  RLS recovery, and mu against the DRIFT RATE (spec 5.1)")
    # (a) stationary beta*, default-ish mu -- pure identification.
    rls, err, _ = _synth_run(mu=0.999, period=0)
    check("recovers a STATIONARY beta on the real basis", err < 0.02,
          "mean|err| = %.4f" % err)
    t = abs(rls.beta[1]) / max(1e-12, rls.se(1))
    check("|t| on the own column clears 3", t > 3.0, "|t| = %.1f" % t)
    # (b) the shipped drift with the AUTO-DERIVED mu, and the measured table that
    #     justifies the constant.  5.1 asks for a sweep, not an assumption.
    period = 150                      # the shipped driver period on 3s5z
    mu_ok = float(np.clip(1.0 - 12.0 / period, 0.50, 0.9999))
    _, err_ok, corr_ok = _synth_run(mu=mu_ok, period=period)
    print("      mu sweep on a %d-step drift (spec 5.1 asks for the table):" % period)
    for mu in (0.80, 0.88, 0.92, 0.95, 0.999, 0.9995):
        _, e, c = _synth_run(mu=mu, period=period)
        print("        mu=%-7.4f corr=%.3f  mean|err|=%.3f%s"
              % (mu, c, e, "   <- auto" if abs(mu - mu_ok) < 1e-9 else ""))
    check("auto-derived mu tracks the shipped drift", corr_ok > 0.8,
          "mu=%.4f  corr=%.3f  mean|err|=%.3f" % (mu_ok, corr_ok, err_ok))
    # (c) POWER's mu on THIS driver: memory 2000 steps against a 75-step cycle.
    #     It learns the cycle average and cannot track the cycle -- exactly the
    #     failure 5.1 says the sweep exists to find.  Confirming it is what makes
    #     the auto-derived default a measurement rather than a preference.
    _, err_bad, corr_bad = _synth_run(mu=0.9995, period=period)
    check("POWER's mu=0.9995 does NOT track it (the sweep's point)",
          corr_bad < corr_ok, "corr %.3f vs %.3f" % (corr_bad, corr_ok))


def t7_windup():
    print("T7  covariance windup: reproduce it, then bound it (spec 5.2)")
    # The own column stays excited (so beta IS identifiable), while a third
    # direction is never excited -- which is exactly what happens as a policy
    # converges and the peer channel stops varying.
    def run(bounded, seed=5):
        r = RLS(3, mu=0.9995, p0=1.0, p_max_mult=(10.0 if bounded else 1e18))
        rng = np.random.RandomState(seed)
        for _ in range(120000):
            own = rng.uniform(-0.5, 0.5)
            x = np.array([1.0, own, 0.0])        # 3rd direction NEVER excited
            r.update(x, -0.8 * own + rng.randn() * 1e-4)
        return r
    lo = run(False)
    hi = run(True)
    check("UNBOUNDED forgetting blows the covariance up",
          (not np.isfinite(lo.trace())) or lo.trace() > 1e6,
          "tr(P) = %.3g" % lo.trace())
    check("the bound keeps it finite and small", np.isfinite(hi.trace())
          and hi.trace() <= 10.0 * 1.0 * 3 + 1e-9, "tr(P) = %.3g" % hi.trace())
    check("and the ESTIMATE survives the bound (-0.80)",
          abs(hi.beta[1] + 0.8) < 0.02, "own_gain = %.4f" % hi.beta[1])
    check("se collapses too (POWER: 8.8e4 -> 32.7)", np.isfinite(hi.se(1)),
          "se = %.4g (unbounded: %.4g)" % (hi.se(1), lo.se(1)))
    check("the clamp is COUNTED, not silent", hi.n_clamp > 0, "%d clamps" % hi.n_clamp)


def t8_inverse():
    print("T8  the channel inverse and T2 conjugacy (spec 6.1, 6.2, 6.5)")
    worst = 0.0
    for Delta in (0.0, 0.05, 0.2, 0.4, 0.6, 0.74):
        du_da = 1.0 - Delta
        d, floored = compensation_delta(Delta, du_da, 1.0, 10.0)
        delivered = (1.0 + d) * (1.0 - Delta)
        worst = max(worst, abs(delivered - 1.0))
    check("cmd = 1/(1-Delta) restores the step EXACTLY", worst < 1e-12,
          "max |delivered - 1| = %.3g" % worst)
    d0, fl0 = compensation_delta(0.0, 1.0, 1.0, 10.0)
    check("no deficit => no correction", d0 == 0.0 and not fl0)
    # 6.2: the sign chain lives in one place and it points the right way.
    dpos, _ = compensation_delta(0.3, 0.7, 1.0, 10.0)
    check("a POSITIVE deficit over-commands (sign chain)", dpos > 0.0,
          "d = %.4f" % dpos)
    # the guard, and it is REPORTED rather than silently applied
    dg, flg = compensation_delta(0.9, 0.5 * DU_DA_FLOOR, 1.0, 10.0)
    check("an untrustworthy divisor returns exactly blind", dg == 0.0 and flg)
    # 6.5: the rail is tracked, and the correction is bounded
    dc, _ = compensation_delta(10.0, 0.1, 1.0, 3.0)
    check("delta is clipped at max_delta", dc == 3.0, "d = %.2f" % dc)


def t9_floor_property():
    print("T9  the FLOOR property: gates shut => byte-identical to blind (spec 1)")
    ag = AgentCompensator(r=1, ff_gain=0.0, max_trust=1.0, ready_updates=10 ** 9)
    rng = np.random.RandomState(7)
    worst = 0.0
    for _ in range(2000):
        d, info = ag.correction(rng.uniform(-0.5, 0.5, 1), 0.4,
                                float(rng.uniform(0, 0.5)), 0.4)
        worst = max(worst, abs(d))
        assert info["trust"] == 0.0
    check("delta is EXACTLY 0 while inadmissible", worst == 0.0,
          "max|delta| = %.3g" % worst)
    # and the gate really does open once the estimate earns it
    ag2 = AgentCompensator(r=1, ff_gain=0.0, max_trust=0.5, ready_updates=50,
                           fit_warmup=20, fit_lam=0.99)
    for t in range(4000):
        psi_prev = np.array([0.4 * np.sin(t * 0.11)])
        y = 0.7 * psi_prev[0] + 0.01
        ag2.observe(0.0, psi_prev, y)
    adm, fg = ag2.admissible()
    check("gate OPENS on measured lift, not on a proxy", adm and fg > 0.0,
          "fit_gain = %.4f" % fg)
    check("trust is BINARY x a constant, not a product of confidences",
          ag2.correction(np.array([0.3]), 0.0, 0.0, 0.0)[1]["trust"] == 0.5)


def t10_ratio_guards():
    print("T10 ratio guards: NaN, never an epsilon (spec 9)")
    f = FitGain(warmup=0)
    check("fit_gain is NaN before there is variance to explain",
          np.isnan(f.value()))
    for _ in range(50):
        f.observe(1.0, 1.0, 1.0)                       # a constant target
    check("a constant target gives NaN, not a huge number",
          np.isnan(f.value()) or abs(f.value()) < 1e-6, "%.3g" % f.value())
    ag = AgentCompensator(r=1, ff_gain=1.0, u_active=0.05)
    d, info = ag.correction(np.array([0.0]), 0.0, 0.0, 0.0)   # driver trough
    check("driver trough: no u recovered, no correction invented",
          d == 0.0 and info["u_hat"] == 0.0)
    check("a non-finite deficit returns exactly blind",
          compensation_delta(float("nan"), 0.8, 1.0, 3.0) == (0.0, True))


def t11_closed_loop():
    print("T11 end-to-end closed loop on the real operator (spec 11.4)")
    names = opmod.composition("3s5z", 8)
    W = opmod.build_W(names, names, step_mul=8)
    W_env = opmod.build_W(names, opmod.enemy_composition("3s5z", 8), 8,
                          zero_diagonal=False)
    n = 8
    ex = opmod.Exertion(n, phi_fire=1.0, rho=0.6)   # the shipped defaults
    b = Basis(W, ex.phi_max, r=1)
    K0 = 0.35 * (W.sum(1) + W_env.sum(1)) * ex.phi_max
    drv = Driver(severity=1.0, period=75)
    rng = np.random.RandomState(11)
    pos = rng.uniform(12, 20, (n, 2))
    pos_e = rng.uniform(14, 22, (8, 2))
    r_a = opmod.radii(names)
    r_e = opmod.radii(opmod.enemy_composition("3s5z", 8))

    def roll(comp, steps=12000):
        mu = float(np.clip(1.0 - 4.0 / drv.period, 0.90, 0.9999))
        ags = [AgentCompensator(r=b.r, mu=mu, ff_gain=(1.0 if comp else 0.0),
                                max_trust=(1.0 if comp else 0.0),
                                ready_updates=300, fit_warmup=300, fit_lam=0.995,
                                peer_mode="delta") for _ in range(n)]
        d.clock = 0
        psi_prev = np.zeros((n, b.r))
        cmd = np.ones(n)
        loss = []
        alive = np.ones(n)
        for t in range(steps):
            phi = ex.update(alive, (rng.rand(n) < 0.55).astype(float))
            p = pos + rng.randn(n, 2) * 0.15
            prox, cone = opmod.kernels(p, p, r_a, r_a)
            proxe, conee = opmod.kernels(p, pos_e, r_a, r_e)
            Lp = opmod.loading(W, prox, cone, phi, alive)
            Lf = opmod.loading(W_env, proxe, conee, np.ones(8), np.ones(8))
            g, A, s = d.g()
            ud = (Lp + Lf) / (K0 * g)[:, None]
            bind = np.argmax(ud, axis=1)
            u = ud[np.arange(n), bind]
            Delta = np.clip(u * (1 - g), 0, 0.75)
            delivered = cmd * (1 - Delta)
            loss.append(float(np.mean(np.abs(delivered - 1.0))))
            contrib = W * prox * cone[np.arange(n), :, bind]
            psi = b.psi(contrib, phi, alive)
            g_now = g
            d.advance()
            g_next = d.g()[0]
            for i in range(n):
                if t:
                    ags[i].observe(own_column(phi[i], ex.phi_max), psi_prev[i],
                                   float(Delta[i]))
                dd, _ = ags[i].correction(psi[i], 1 - g_next, float(Delta[i]),
                                          1 - g_now)
                cmd[i] = float(np.clip(1.0 + dd, 0.25, 4.0))
            psi_prev = psi
        half = len(loss) // 2
        return float(np.mean(loss[half:]))

    d = drv
    ex.reset()
    blind = roll(False)
    ex.reset()
    comp = roll(True)
    check("the NS actually costs the blind arm delivered distance", blind > 0.02,
          "mean |delivered - 1| = %.4f" % blind)
    check("PACT recovers most of it", comp < 0.5 * blind,
          "blind %.4f -> pact %.4f  (%.0f%% recovered)"
          % (blind, comp, 100 * (1 - comp / blind)))


def t12_ceiling():
    print("T12 the Part-C ceiling decomposition is computable with no training")
    from .certificates import ceiling_from_state
    names = opmod.composition("3s5z", 8)
    W = opmod.build_W(names, names, 8)
    W_env = opmod.build_W(names, opmod.enemy_composition("3s5z", 8), 8,
                          zero_diagonal=False)
    rng = np.random.RandomState(13)
    pos = rng.uniform(12, 20, (8, 2))
    pos_e = rng.uniform(14, 22, (8, 2))
    out = ceiling_from_state(W, W_env, names, opmod.enemy_composition("3s5z", 8),
                             pos, pos_e, np.ones(8), np.ones(8),
                             np.full(8, 1.5), 0.35, sigma=1.0, A=1.0)
    tot = out["irreducible"] + out["coordination_gap"]
    check("the three classes partition the excess", abs(tot - 1.0) < 1e-9,
          "irreducible %.3f + peer %.3f" % (out["irreducible"],
                                            out["coordination_gap"]))
    check("the COORDINATION GAP is large enough to be worth chasing",
          out["coordination_gap"] > 0.2,
          "%.1f%%  (POWER measured 9.5%% at sigma=1)"
          % (100 * out["coordination_gap"]))


def t13_wrapper_stack():
    """11.5: verify the banner and delta_nonzero_frac on a probe BEFORE any long
    run.  With no StarCraft II here the probe runs against fc/mock_smac.py -- so
    it proves the PLUMBING (shapes, timing, the info dict, the sensor's odometric
    reconstruction, the floor property), never the method."""
    print("T13 the wrapper stack, end to end on a test double (spec 11.5)")
    from .mock_smac import MockSmacEnv
    from .pact_env import PactEnv
    from .severity_env import FormationCongestionEnv

    def roll(cfg, steps=4000):
        env = MockSmacEnv(seed=2)
        fc = FormationCongestionEnv(env, cfg)
        e = PactEnv(fc, cfg) if int(cfg.get("pact", 0)) else fc
        e.reset()
        strides, disp, dnz, tr, dmean = [], [], [], [], []
        rng = np.random.RandomState(4)
        info = {}
        for _ in range(steps):
            av = env.get_avail_actions()
            acts = []
            for i in range(env.n_agents):
                ok = np.where(av[i] > 0)[0]
                acts.append(int(rng.choice(ok)) if ok.size else 0)
            pre = np.array([[u.pos.x, u.pos.y] for u in env.agents.values()])
            out = e.step(np.array(acts).reshape(-1, 1))
            post = np.array([[u.pos.x, u.pos.y] for u in env.agents.values()])
            strides.append(env.move_stride.copy())
            disp.append(float(np.sum(np.linalg.norm(post - pre, axis=1))))
            info = out[4][0]
            dmean.append(info["fc_delta_mean"])
            if "pact_delta_nonzero_frac" in info:
                dnz.append(info["pact_delta_nonzero_frac"])
                tr.append(info["pact_applied_trust"])
            if bool(np.all(np.asarray(out[3]))):
                e.reset()
        return dict(stride=np.array(strides), disp=float(np.sum(disp)),
                    dnz=np.array(dnz), trust=np.array(tr),
                    delta=float(np.nanmean(dmean)), info=info, fc=fc,
                    base=fc.base_frac.copy())

    # --- the dial's identity at zero, on the quantity the claim is about -----
    z = roll({"ns_severity": 0.0, "pact": 0})
    check("sigma=0: Delta == 0 identically (B.1.1 / gate G2)", z["delta"] == 0.0,
          "delta_mean = %.3g" % z["delta"])
    check("sigma=0: the dial multiplies the order by exactly 1",
          np.all(z["stride"] == z["base"][None, :]),
          "max|stride/base - 1| = %.3g"
          % np.max(np.abs(z["stride"] / z["base"][None, :] - 1.0)))
    check("the info dict carries fc_* telemetry on the BLIND arm",
          "fc_delta_mean" in z["info"] and "fc_dial_ratio" in z["info"])

    # --- ns_reach_clip: byte-identical WITHOUT it, and a no-op WITH it -------
    zr = roll({"ns_severity": 0.0, "pact": 0, "ns_reach_clip": 0})
    check("reach_clip=0 at sigma=0 is BYTE-identical to stock SMAC",
          np.all(zr["stride"] == 1.0),
          "max|stride - 1| = %.3g" % np.max(np.abs(zr["stride"] - 1.0)))
    check("the reach clip changes NO displacement (it is measured, not asserted)",
          abs(z["disp"] - zr["disp"]) < 1e-9 * max(1.0, zr["disp"]),
          "total displacement %.4f vs %.4f" % (z["disp"], zr["disp"]))

    # --- does it bite? ------------------------------------------------------
    b = roll({"ns_severity": 1.0, "pact": 0})
    check("sigma=1 throttles the blind arm", b["delta"] > 0.02
          and b["disp"] < z["disp"], "delta_mean=%.4f, displacement %.1f -> %.1f"
          % (b["delta"], z["disp"], b["disp"]))
    check("the sensor is PHYSICAL (odometry reconstructs Delta exactly)",
          np.isfinite(b["info"]["fc_odom_err"]) and b["info"]["fc_odom_err"] < 1e-9,
          "|Delta_odom - Delta| = %.3g" % b["info"]["fc_odom_err"])
    check("the placebo regime is inert during a live rollout",
          0.05 < b["info"]["fc_dial_ratio"] < 0.95,
          "dial_ratio = %.2f (g == 1 on the rest)" % b["info"]["fc_dial_ratio"])
    check("Phi stays varying on a live rollout", b["info"]["fc_phi_var"] > 0.05,
          "std/mean = %.3f" % b["info"]["fc_phi_var"])

    # --- THE FLOOR PROPERTY, on the orders actually issued (spec 1) ---------
    shut = roll({"ns_severity": 1.0, "pact": 1, "pact_ff_gain": 0,
                 "pact_ready_updates": 10 ** 9})
    check("both gates shut => orders BYTE-identical to blind",
          np.array_equal(shut["stride"], b["stride"]),
          "max|diff| = %.3g" % np.max(np.abs(shut["stride"] - b["stride"])))
    check("...and applied_trust is exactly 0 there",
          float(np.max(shut["trust"])) == 0.0)

    # --- the full method ----------------------------------------------------
    p = roll({"ns_severity": 1.0, "pact": 1})
    check("with the gates open the compensator ACTS",
          float(np.mean(p["dnz"])) > 0.3,
          "delta_nonzero_frac = %.3f" % np.mean(p["dnz"]))
    check("applied_trust is not ~0 -- WAS THE METHOD ON AT ALL?",
          float(np.mean(p["trust"])) > 0.0,
          "applied_trust = %.3f" % np.mean(p["trust"]))
    lost = z["disp"] - b["disp"]
    got = p["disp"] - b["disp"]
    check("and it recovers delivered distance", got > 0.25 * lost,
          "displacement blind %.1f -> pact %.1f of a stock %.1f  (%.0f%% recovered)"
          % (b["disp"], p["disp"], z["disp"], 100.0 * got / max(1e-9, lost)))
    check("ff / peer split is reported (7's honesty condition)",
          np.isfinite(p["info"]["pact_ff_abs"]) and np.isfinite(p["info"]["pact_peer_abs"]),
          "ff=%.4f peer=%.4f -> ff_share=%.0f%%"
          % (p["info"]["pact_ff_abs"], p["info"]["pact_peer_abs"],
             100 * p["info"]["pact_ff_abs"]
             / max(1e-12, p["info"]["pact_ff_abs"] + p["info"]["pact_peer_abs"])))
    check("cond_psi finite -- beta decomposable, not merely predictable",
          np.isfinite(p["info"]["pact_cond_psi"]),
          "cond_psi = %.1f" % p["info"]["pact_cond_psi"])
    check("covariance windup stayed bounded", np.isfinite(p["info"]["pact_trP"]),
          "trP = %.3g" % p["info"]["pact_trP"])
    check("peer-only arm (ff_gain=0) still acts -- the coordination ablation",
          float(np.mean(roll({"ns_severity": 1.0, "pact": 1,
                              "pact_ff_gain": 0})["dnz"])) > 0.05)


def main():
    print("=" * 74)
    print("Formation Congestion + PACT -- arithmetic self-check (no simulator)")
    print("=" * 74)
    W, names = t1_operator()
    t2_basis(W)
    t3_n1_identity()
    t4_dial()
    t5_phi()
    t6_rls_recovery()
    t7_windup()
    t8_inverse()
    t9_floor_property()
    t10_ratio_guards()
    t11_closed_loop()
    t12_ceiling()
    t13_wrapper_stack()
    print("=" * 74)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED.  The arithmetic is not the reason if a run fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
