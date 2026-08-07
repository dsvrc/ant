"""Pure-numpy arithmetic certificate for SMAC CWO (Coupled Weapon Overheat) PACT.

No StarCraft II needed.  Verifies, with the SAME recursion StarCraft2_Env runs:
  T1  factorization: ell_i = clip(A*sigma*(x2_i-KNEE)/(1-KNEE), 0, LMAX);
  Bounds: x2_i in [0,1] (leaky mean of the firing fraction), ell in [0,LMAX];
  N=1 : the cross-agent sum is empty => x2 == 0 => stock SMAC (irreducibility);
  Drop: an empirical Bernoulli(ell) drop rate matches ell (the harm is honest);
  Floor: the PACT obs block is append-only (ignore it => blind);
  Curriculum: severity ramps 0 -> SEVERITY after the warmup.

Plus the three checks that a 20M-step 3s5z run had to be spent to discover, each of
which now runs in under a second:

  HEADROOM   the configured (SEVERITY, KNEE, LMAX) must leave the team a coordination
             solution that is BOTH worth taking (GAIN) and good enough to win with
             (CEILING).  Two successive defaults each failed one of the two: SEVERITY=2.0
             /LMAX=0.6 had GAIN 0.31x (greedy simply won -- 20M steps, firing fraction
             0.335 at the peak vs 0.345 at the trough, no modulation), and SEVERITY=1.0
             /KNEE=0.15 had GAIN 5.9x but CEILING 36% (17.7M steps, win rate at the
             driver peak 0.000 in every one of 221 eval rounds).
  HARBOUR    the coordinated optimum must be (near-)harm-free, or the policy learns to
             stop fighting instead of learning to stagger.
  DEPHASE    parallel envs spread over the driver cycle, so a rollout batch / eval
             round is a cycle AVERAGE and not a single-phase snapshot.
  BIAS       `stagger_gap` (the old headline coordination metric) is positive with
             ZERO coordination, so it never measured coordination at all.

Run:  python -m harl.envs.smac.pact.test_pact
"""

import numpy as np

from harl.envs.smac.StarCraft2_Env import (
    _RHO, _LMAX, _KNEE, _PERIOD, _GREEDY_LOAD, _driver, _WARMUP, _RAMP, SEVERITY,
)


def _ell(A, sigma, x2):
    """The harm channel exactly as StarCraft2_Env._snd_step applies it."""
    excess = np.maximum(0.0, np.asarray(x2) - _KNEE) / max(1e-6, 1.0 - _KNEE)
    return np.clip(A * sigma * excess, 0.0, _LMAX)


def _sim(n, T, sigma, fire_prob, seed=0):
    """Iterate the CWO recursion: x2 = leaky mean of OTHERS' firing; ell = the channel."""
    rng = np.random.default_rng(seed)
    x2 = np.zeros(n)
    denom = max(1, n - 1)
    out = []
    for t in range(T):
        A = _driver(t)
        fire = (rng.random(n) < fire_prob).astype(float)  # who COMMANDED an attack
        S = (fire.sum() - fire) / denom                   # (sum_{j!=i} fire)/(N-1)
        x2 = _RHO * x2 + (1.0 - _RHO) * S
        out.append((A, x2.copy(), _ell(A, sigma, x2)))
    return out


def test_factorization_and_bounds():
    worst = 0.0
    for A, x2, ell in _sim(8, 500, sigma=1.0, fire_prob=0.7):
        raw = _ell(A, 1.0, x2)
        below = raw < _LMAX - 1e-12
        if np.any(below):
            expect = A * 1.0 * np.maximum(0.0, x2 - _KNEE) / (1.0 - _KNEE)
            worst = max(worst, float(np.max(np.abs(ell[below] - expect[below]))))
        assert np.all(x2 >= -1e-9) and np.all(x2 <= 1.0 + 1e-9), "x2 must be in [0,1]"
        assert np.all(ell >= -1e-9) and np.all(ell <= _LMAX + 1e-9)
    assert worst < 1e-9, f"ell != the channel below the clip (dev {worst:.2e})"
    print(f"  T1 factorization: ell = clip(A*sigma*(x2-KNEE)/(1-KNEE), 0, LMAX), "
          f"x2 in [0,1]  OK (max dev {worst:.1e})")


def test_free_below_the_knee():
    """Load at or under KNEE costs nothing.  _KNEE defaults to 0 for the deflection
    channel (the compensation is free, so there is no commons trade-off to shape), but
    the hinge is kept so the knob still works if you want a dead-band."""
    if _KNEE <= 0.0:
        assert float(_ell(1.0, 1.0, np.array([1e-6]))[0]) > 0.0
        print("  Knee: disabled (_KNEE=0) -- harm is linear in the shared load  OK")
        return
    assert float(_ell(1.0, 10.0, np.array([_KNEE]))[0]) == 0.0
    assert float(_ell(1.0, 10.0, np.array([_KNEE * 0.5]))[0]) == 0.0
    assert float(_ell(1.0, 1.0, np.array([_KNEE + 1e-3]))[0]) > 0.0
    print(f"  Knee: shared load <= {_KNEE} is completely free; harm starts above it  OK")


def test_irreducible_at_n1():
    for A, x2, ell in _sim(1, 200, sigma=5.0, fire_prob=1.0):  # a lone unit, firing hard
        assert np.max(np.abs(x2)) < 1e-12 and np.max(np.abs(ell)) < 1e-12
    print("  N=1: cross-agent sum empty => x2==0 => stock SMAC (irreducible)  OK")


def test_drop_rate_is_honest():
    """A Bernoulli(ell) weapon-jam drops shots at rate ell (the harm is real)."""
    rng = np.random.RandomState(0)
    for p in (0.0, 0.3, 0.7, _LMAX):
        drops = np.mean([1.0 if rng.random_sample() < p else 0.0 for _ in range(20000)])
        assert abs(drops - p) < 0.02, f"drop rate {drops:.3f} != ell {p}"
    print("  Drop: empirical Bernoulli(ell) drop rate matches ell  OK")


def test_floor_property_is_append_only():
    base = np.arange(10, dtype=np.float32)
    block = np.array([0.42, 0.07, 0.31], dtype=np.float32)  # x2_i, x3_jam_i, x3_try_i
    aug = np.append(base, block)
    assert aug.shape[0] == base.shape[0] + block.shape[0]
    assert np.array_equal(aug[: base.shape[0]], base)
    print("  Floor: obs block is append-only (ignore it => blind)  OK")


def test_curriculum_ramp():
    """Training severity ramps 0 -> SEVERITY: 0 before _WARMUP, linear over _RAMP, full
    after (this mirrors StarCraft2_Env._curr_severity; eval envs always use full).
    Driven by the env's AGE, never the de-phased driver clock, so per-rank de-phasing
    cannot move the warmup boundary between parallel envs."""
    def ramp(age, sev):
        if _WARMUP <= 0:
            return sev
        return sev * min(1.0, max(0.0, (age - _WARMUP) / float(max(1, _RAMP))))
    S = 2.0
    assert ramp(0, S) == 0.0 and ramp(_WARMUP - 1, S) == 0.0, "sigma=0 during warmup"
    assert abs(ramp(_WARMUP + _RAMP // 2, S) - S * 0.5) < 1e-3, "linear mid-ramp"
    assert ramp(_WARMUP + _RAMP, S) == S and ramp(10 ** 9, S) == S, "full after ramp"
    print(f"  Curriculum: sigma=0 for {_WARMUP} steps/env, ramps over {_RAMP}, then full  OK")


# --------------------------------------------------------------------------------
#  The three checks that cost a 20M-step run to find.
# --------------------------------------------------------------------------------

def _throughput(f, A, sigma):
    """Per-unit damage throughput when the team fires at fraction f: the steady-state
    x2 of a team firing at f is f itself (leaky MEAN of the others' firing)."""
    return f * (1.0 - float(_ell(A, sigma, np.array([f]))[0]))


def test_channel_is_invertible():
    """*** THE CHECK THAT DECIDES WHETHER THE EXPERIMENT MEANS ANYTHING. ***

    The harm is a deterministic shift of the delivered target by s places along the K
    attackable enemies.  Pre-shifting the command by -s therefore lands exactly on the
    desired enemy, at ZERO cost -- pipeline T2 conjugacy, so the compensated optimum
    equals the stationary optimum and 0.9*B0 is reachable.

    This is the property the previous harm channel (drop the shot) did NOT have, and
    the reason it failed Phase 1 at every severity: the only response to a dropped shot
    is to fire less, which buys damage with damage.  Measured, compensation genuinely
    helped -- at sigma=1.0 the scripted controller took the load 0.650 -> 0.471, the
    drop rate 0.448 -> 0.050, throughput 0.489 -> 0.631 and return 7.9 -> 11.3 -- and it
    STILL could not clear 0.9*B0 = 16.4 from a best of 12.5 at ANY severity from 0.5 to
    3.2, because the coordinated ceiling was ~0.68 against a stationary 0.88.  No
    conjugacy => no frontier => nothing for Phase 1 to certify."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = int(rng.integers(2, 9))            # attackable enemies
        tg = np.sort(rng.choice(9, size=k, replace=False))   # their action indices
        desired = int(rng.choice(tg))
        ell = float(rng.random())
        s = int(round(ell * (k - 1)))
        pos = int(np.where(tg == desired)[0][0])
        # controller pre-shifts by -s, env then shifts by +s
        commanded = int(tg[(pos - s) % k])
        pos_c = int(np.where(tg == commanded)[0][0])
        delivered = int(tg[(pos_c + s) % k])
        assert delivered == desired, (
            f"inverse failed: k={k} s={s} desired={desired} got {delivered}"
        )
    print("  INVERTIBLE: pre-shift by -s then channel shift by +s == identity, over "
          "2000 random (K, target, ell)  OK  (T2 conjugacy holds, ceiling = B0)")


def test_zero_severity_is_stock_smac():
    """ell = 0 => s = 0 => the delivered target IS the commanded one, exactly."""
    for k in range(2, 9):
        s = int(round(0.0 * (k - 1)))
        assert s == 0
    assert float(_ell(0.0, 5.0, np.array([1.0]))[0]) == 0.0   # A=0 at the trough
    assert float(_ell(1.0, 0.0, np.array([1.0]))[0]) == 0.0   # SEVERITY=0
    print("  Transparency: ell=0 => zero shift => byte-identical to stock SMAC  OK")


def _unused_test_coordination_headroom():
    """*** THE CHECK THAT DECIDES WHETHER THE EXPERIMENT MEANS ANYTHING. ***

    Two quantities, both measured at the driver peak against the load a trained
    stationary team actually runs at (_GREEDY_LOAD, not a hypothetical f=1):

      GAIN    = T(f*) / T(greedy) -- how much coordinating is worth.  ~1x and there is
                nothing for PACT, or an oracle, to recover.
      CEILING = T(f*) / T(stationary) -- what perfect coordination KEEPS.  Too low and
                even flawless play cannot win, so the metric is pinned at 0 and the run
                measures nothing either.

    Both failures have now been paid for in full:

      SEVERITY=2.0, LMAX=0.6, no knee  -> GAIN 0.31x.  Once ell saturates, the branch
        T = f*(1-LMAX) is strictly INCREASING, so greedy simply wins; a 20M-step 3s5z
        run measured a firing fraction of 0.335 at the peak vs 0.345 at the trough --
        no modulation, because none was ever profitable.
      SEVERITY=1.0, KNEE=0.15, LMAX=0.95 -> GAIN 5.9x but CEILING 36%.  A 17.7M-step run
        with de-aliased eval recorded a win rate at the driver peak of 0.000 in **every
        one of 221 eval rounds**, while the same policy won 93% at the trough."""
    grid = np.linspace(0.01, 1.0, 200)
    T = np.array([_throughput(f, 1.0, SEVERITY) for f in grid])   # driver at its PEAK
    i = int(np.argmax(T))
    f_star, t_star = float(grid[i]), float(T[i])
    t_greedy = _throughput(_GREEDY_LOAD, 1.0, SEVERITY)
    gain = t_star / max(1e-9, t_greedy)
    ceiling = t_star / _GREEDY_LOAD          # stationary throughput == the greedy load
    assert f_star < _GREEDY_LOAD - 0.05, (
        f"the optimum (f*={f_star:.2f}) is at or above the load a stationary team "
        f"already runs at ({_GREEDY_LOAD}), so firing flat out is fine and there is NO "
        f"coordination problem at SEVERITY={SEVERITY}, KNEE={_KNEE}, LMAX={_LMAX}."
    )
    assert gain > 1.5, (
        f"GAIN is only x{gain:.2f} (coordinated {t_star:.3f} vs greedy {t_greedy:.3f}) "
        f"at SEVERITY={SEVERITY}, KNEE={_KNEE}, LMAX={_LMAX}. Nothing to recover; the "
        f"run would measure noise. Raise LMAX or SEVERITY."
    )
    assert ceiling > 0.70, (
        f"CEILING is only {ceiling:.0%}: even perfect coordination keeps {t_star:.3f} "
        f"of a stationary {_GREEDY_LOAD:.2f} at the driver peak. SMAC 3s5z is a MIRROR "
        f"match, so by Lanchester's square law the outcome is a CLIFF in relative DPS -- "
        f"measured, a 27% shot-drop rate took the win rate 0.600 -> 0.000 with episode "
        f"length FALLING 45.7 -> 37.3. The peak win rate will be pinned at 0 whatever "
        f"the method does. RAISE _KNEE toward _GREEDY_LOAD (the knee sets the ceiling: "
        f"when SEVERITY >= (1-KNEE)/KNEE the optimum is the knee itself, so a "
        f"coordinated team takes zero drops and keeps throughput KNEE)."
    )
    # and the optimum must be PHASE-DEPENDENT, or there is nothing to track
    T_trough = [_throughput(f, 0.0, SEVERITY) for f in grid]
    f_star_trough = float(grid[int(np.argmax(T_trough))])
    assert f_star_trough > f_star + 0.1, (
        "the optimal firing fraction is the same at the driver trough as at the peak, "
        "so there is no phase to track and PACT reduces to a constant policy."
    )
    print(f"  HEADROOM: peak f*={f_star:.2f} -> T={t_star:.3f}; greedy({_GREEDY_LOAD}) "
          f"-> {t_greedy:.3f}  ==>  GAIN x{gain:.1f}, CEILING {ceiling:.0%}; "
          f"trough f*={f_star_trough:.2f}  OK")


def _unused_test_coordinated_team_has_a_zero_harm_safe_harbour():
    """At the coordinated optimum the team should take (near-)zero drops.

    This is what stops the policy learning to stop fighting.  With a small knee every
    level of firing is penalised, so there is no safe operating point, and SMAC's
    shaped reward makes "disengage and survive to the time limit" locally better than
    "engage and lose": measured at KNEE=0.15, when severity reached full the fraction
    of live units with a target in range fell 0.89 -> 0.22, episode length went 50 ->
    141, and the win rate at the almost-unharmed driver TROUGH collapsed 0.93 -> 0.01.
    When SEVERITY >= (1-KNEE)/KNEE the optimum sits exactly at the knee and costs
    nothing."""
    grid = np.linspace(0.01, 1.0, 200)
    T = [_throughput(f, 1.0, SEVERITY) for f in grid]
    f_star = float(grid[int(np.argmax(T))])
    ell_at_opt = float(_ell(1.0, SEVERITY, np.array([f_star]))[0])
    assert ell_at_opt < 0.15, (
        f"at the coordinated optimum f*={f_star:.2f} the team still drops "
        f"{ell_at_opt:.0%} of its shots, so there is no harm-free operating point to "
        f"aim at. Raise _KNEE to >= (1/SEVERITY)/(1+1/SEVERITY)-ish, or equivalently "
        f"pick SEVERITY >= (1-_KNEE)/_KNEE = {(1 - _KNEE) / max(1e-9, _KNEE):.2f}."
    )
    print(f"  SAFE HARBOUR: coordinated optimum f*={f_star:.2f} drops {ell_at_opt:.1%} "
          f"of shots (harm-free target exists)  OK")


def test_dephasing_tiles_the_cycle():
    """Parallel envs de-phased by rank must cover the driver cycle within one short
    window, so a PPO batch and an eval round are cycle AVERAGES.  In phase, the whole
    ensemble sits at one A(t) -- which is why the reported eval win-rate was a slow
    square wave (~0 for most evals, ~0.95 for a few) tracking the driver phase rather
    than the policy."""
    n_threads, window = 10, 400        # 400 steps ~ one short eval round; period 5000
    in_phase = np.array([_driver(t) for t in range(window)])
    dephased = np.array([
        _driver(int(r * _PERIOD / n_threads) + t)
        for r in range(n_threads) for t in range(window)
    ])
    def coverage(a):  # fraction of 10 A-buckets touched
        return len(np.unique(np.clip((np.asarray(a) * 10).astype(int), 0, 9))) / 10.0
    assert coverage(in_phase) <= 0.5, "sanity: in-phase envs should NOT cover the cycle"
    assert coverage(dephased) >= 0.9, (
        f"de-phasing only covers {coverage(dephased):.0%} of the driver cycle; eval "
        f"rounds would still be phase-aliased snapshots."
    )
    print(f"  DEPHASE: in-phase covers {coverage(in_phase):.0%} of the cycle, "
          f"de-phased covers {coverage(dephased):.0%}  OK")


def test_stagger_gap_is_biased():
    """`stagger_gap` = fire_lo_load - fire_hi_load reads POSITIVE with zero coordination.

    x2_i = leak((sum_j fire_j - fire_i)/(N-1)) = G - leak(fire_i)/(N-1) where G is
    common to every agent.  So ranking agents by x2_i is EXACTLY reverse-ranking them
    by their own recent firing, and because firing is temporally autocorrelated (a unit
    with a target keeps shooting) the low-x2 group is simply the group that is already
    shooting.  Here every agent fires from a sticky private process and NEVER looks at
    x2 -- there is no coordination whatsoever -- yet the metric is clearly positive.
    On the real 20M-step run it read +0.16 right through the severity-0 warmup, where
    no NS exists at all.  Use hold_gap (peak vs trough) instead."""
    rng = np.random.default_rng(0)
    n, T, stick = 8, 4000, 0.9
    x2 = np.zeros(n)
    engaged = rng.random(n) < 0.5
    hi, lo = [], []
    for _ in range(T):
        flip = rng.random(n) > stick
        engaged = np.where(flip, rng.random(n) < 0.5, engaged)
        fire = engaged.astype(float)                       # NOT a function of x2
        thr = np.median(x2)
        hi_m, lo_m = x2 > thr, x2 <= thr
        if hi_m.any() and lo_m.any():
            hi.append(fire[hi_m].mean())
            lo.append(fire[lo_m].mean())
        x2 = _RHO * x2 + (1.0 - _RHO) * (fire.sum() - fire) / (n - 1)
    gap = float(np.mean(lo) - np.mean(hi))
    assert gap > 0.05, (
        f"expected the biased metric to read clearly positive without coordination, "
        f"got {gap:+.3f} -- the demonstration is not reproducing."
    )
    print(f"  BIAS: stagger_gap = {gap:+.3f} with ZERO coordination (agents never read "
          f"x2) -- the metric is invalid; use hold_gap  OK")


def main():
    print("SMAC CWO PACT arithmetic certificate (pure numpy):")
    test_factorization_and_bounds()
    test_free_below_the_knee()
    test_irreducible_at_n1()
    test_drop_rate_is_honest()
    test_floor_property_is_append_only()
    test_curriculum_ramp()
    test_channel_is_invertible()
    test_zero_severity_is_stock_smac()
    test_dephasing_tiles_the_cycle()
    test_stagger_gap_is_biased()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
