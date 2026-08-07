# Interaction-Mediated Non-Stationarity — the complete pipeline
## Certify the frontier (σ\*) → build the method (PACT). One document, one env at a time.

This is the end-to-end guide for taking a **category-C non-stationarity** on a new
environment from scratch to a working, decentralized method. It merges the two
halves of the program:

- **Phase 1 — Certify (find σ\*).** Turn off learning, hand a scripted controller
  the true hidden driver, sweep severity, and measure the largest severity at
  which compensation still recovers the baseline. This tells you *whether the
  problem is solvable at all*, and at what severity to run it.
- **Phase 2 — Build (PACT).** At a certified severity, deploy Peer-Action
  Compensation with a Trained gain: compute the disturbance waveform exactly from
  shared peer actions, and learn the single hidden scalar that scales it.

The two phases are one idea seen twice. **The scripted law that certifies σ\* in
Phase 1 is exactly the law PACT learns in Phase 2** — Phase 1 is PACT with the
driver handed over and the gain hand-set; Phase 2 is Phase 1 with the driver
*computed* and the gain *learned*. So Phase 1 is not just a gate: it hands you the
method's target and its ceiling.

> **Validated on MAMuJoCo Ant-PCR.** Phase 1 → σ\* = 0.5 (so run at σ=0.45).
> Phase 2 → blind 5000, PACT decentralized ≈ 5500, PACT + CTDE critic ≈ 6000 ≈
> the compensation ceiling (6104). Ant is the worked reference throughout; every
> section ends with "for a new env."

---

# PART I — The problem class (read once)

## I.1 Category-C non-stationarity

A cooperative Markov game is **category-C non-stationary** when each agent carries
a hidden **liability** fed *only by the other agents*, gated by a slow exogenous
driver, harming the agent through the **dynamics** (never the reward):

```
(driver)      c(t) = A(t)·σ,   A(t) ∈ [0,1] slow, exogenous, persists across episodes;  σ = severity
(liability)   ℓ_i(t+1) = ρ·ℓ_i(t) + (1−ρ)·c(t)·Φ_i(x_{−i}(t)),   ℓ_i(0)=0
(harm)        the agent's realized effect is g(a_i, ℓ_i), not a_i, inside the transition;
              reward = the original task reward, byte for byte
```

- **`Φ_i`** — the **exertion functional**: a fixed function of the *other* agents
  (never i's own). This `Σ_{j≠i}` structure is the category-C signature.
- **`ρ`** — a known structural leak constant (short memory, e.g. 0.8 ≈ 5 steps).
- **`g`** — the **harm channel**: additive (`a+ℓ`), multiplicative (`(1−ℓ)a`),
  transform (`T(a)`), or target-drift. This is the one thing that varies per env.

**Litmus tests (the design contract):** at N=1 the effect vanishes (`Φ` has no
others → `ℓ≡0`); with frozen optimal teammates it still drifts (exogenous driver).
Irreducibly multi-agent, genuinely non-stationary.

## I.2 The two facts everything rests on

1. **The liability factorizes:** `ℓ_i(t) = c(t)·x2_i(t)`, where
   `x2_i(t+1) = ρ·x2_i(t) + (1−ρ)·Φ_i(x_{−i}(t))` is the leaky accumulator of the
   **others' exertion** — the same recursion the env runs, minus the scalar `c`.
   `x2_i` is *known arithmetic* if agents share what `Φ` reads; `c(t)` is the one
   slow hidden scalar. **An 8-D estimation problem collapses to 1-D tracking.**
2. **Compensation exists up to a severity limit.** Inverting the harm channel with
   a gain `β≈c` cancels the liability — but only while the inverse stays inside the
   action set. Above a computable severity `c*`, the inverse exceeds the actuator
   (or leaves the feasible set) and *no* controller recovers, with any
   information. **This is σ\*.** Phase 1 measures it; Phase 2 lives below it.

## I.3 Why naïve estimation fails (the motivation for both phases)

Trying to *estimate* the full vector `ℓ_i` from observations hits two walls on a
trained cooperative gait, plus a deep obstacle — all measured, not assumed:

- **Wall A (collinearity):** on a coordinated gait, "the payload loaded me" and
  "my own actuator got stronger" have the same proprioceptive signature; central
  identification of `c` comes out *anti-correlated* with the truth.
- **Wall B (dynamical plant):** a local disturbance observer needs a plant model;
  a scalar-gain approximation over-scales the target by orders of magnitude.
- **Local unobservability:** once you compensate well, the residual vanishes, so
  `c` becomes unobservable *exactly when you are tracking it*.

Phase 1 sidesteps all three by handing over the true driver (no estimation).
Phase 2 sidesteps them by *computing* the waveform and learning only the scalar
the walls don't touch.

---

# PART II — PHASE 1: Certify solvability and find σ\*

**σ\* = the largest severity at which a *privileged, scripted* controller holds the
NS peak to ≥ 90% of the undisturbed baseline B0.** Privileged = handed the true
driver (no estimation error). Scripted = a hand-written law, no learning, runs in
minutes. Peak = the driver frozen at its worst point.

> A learner can at best match a controller with perfect knowledge. So if the
> privileged scripted controller fails at severity σ, **no method can succeed at
> σ** — the bottleneck is the environment, not information or optimization. Certify
> existence *before* spending compute on a method.

## II.1 The five ingredients (build once per env)

1. **Baseline B0.** Train the base algorithm with the NS **off** (driver = 0).
   B0 = its undisturbed return. It is *severity-independent*, so you reuse the same
   checkpoint at every point in the sweep — you never retrain during Phase 1.
2. **A freeze knob** holding the driver at a constant value → a stationary game at
   effective severity `c`.
3. **The privileged signal** — the exact hidden driver, in the **native units** the
   compensation law consumes, read straight from `info` (never piggybacked inside a
   normalized obs vector — see pitfall P3).
4. **A probe shim** that intercepts the action at the env boundary, rewrites it via
   the law, and steps. No gradients. Its per-episode state must reset when the env
   resets (verify the reset path reaches it — pitfall P2).
5. **A severity sweep with per-severity gain re-optimization** (§II.3 — this is
   where the naïve version goes wrong).

## II.2 The compensation law — the one env-specific choice (shared with Phase 2)

**What "compensate for the driver" means depends on how the NS harms the agent.**
Read the env's `step()`, find the exact line where the driver alters the command
before it reaches the dynamics; the law is whatever *inverts that line*; the
bounded resource is whatever that inverse runs into (usually action bounds). **This
same table is declaration #3 of the Phase-2 method — pick it once, use it twice.**

| NS mechanism | how it harms | compensation law (Phase 1 scripted / Phase 2 learned) | privileged signal | bounded resource → what caps σ\* |
|---|---|---|---|---|
| **Additive** `delivered = a + ℓ` | uncommanded push | `u = a − β·ℓ` (Ant: subtract) | `ℓ` | action limits: `‖a−βℓ‖` inside bounds |
| **Multiplicative** `delivered = (1−ℓ)·a` | weak actuators | `u = a/(1−β·ℓ)` (clipped) | `ℓ` | action limits: blows past the rail as ℓ→1 |
| **Transform** `delivered = T(a)` (e.g. R(θ)) | mis-aimed commands | `u = T⁻¹(a)` | `θ`/`T` | invertibility + bounds |
| **Target / goal drift** | you aim at a stale goal | re-aim at the true drifted goal | goal offset | reachability / velocity |
| **Dynamics / parameter shift** | wrong internal model | model-based feed-forward | shifted params | control authority |

**The loop-gain twist (category-C).** Because the driver is *fed by the agents'
own actions*, compensating changes the quantity you compensate — a feedback loop.
Cancelling can *amplify* the disturbance. The tell: **more compensation makes it
worse** at high severity — the best gain drops from "full" toward "none" as you
cross σ\*. If you see that, σ\* is lower than an open-loop estimate suggests.

## II.3 The sweep protocol

```
GIVEN baseline π (return B0), freeze knob, privileged signal, probe shim, law L(driver; β).
1. BAR = 0.90 · B0.                                   # the existence threshold
2. Severity grid bracketing the frontier, e.g. {0.4,0.5,0.6,...}.
3. Gain grid, e.g. β ∈ {0.25, 0.5, 0.75, 1.0}.
4. FOR each severity σ:  freeze the NS at its PEAK, severity σ
       FOR each gain β:  roll π through the shim with L(driver; β), ~40 episodes
                         record return, bounded-resource usage, termination cause
       R(σ) = max_β returns ;  best_β(σ) = argmax        # RE-OPTIMIZE β PER σ
5. σ* = the LARGEST σ with R(σ) ≥ BAR.
6. Redesign target (if needed): σ* − margin (≈0.05).
```

**Re-optimize β per severity, never fix it.** The optimal gain changes with
severity: β=1 at low σ, collapsing toward 0 near σ\* (over-cancelling saturates the
resource). Fixing β at the value best for your *starting* σ measures a frontier
that is wrong — usually far too low. *(This was a real bug in the Ant run.)*

## II.4 Two confounding checks (before believing any result)

1. **Transparency at zero severity.** The probe with driver = 0 must reproduce B0
   *exactly* (CIs overlapping identity). Else the shim corrupts the action
   independent of the NS — fix the probe.
2. **It-works-when-it-should.** At *low* severity, compensation must actually
   *recover* performance. If it never helps anywhere, your law or signal is wrong.

Only when both pass is a high-severity failure a property of the *environment*.

## II.5 Reading the result — four things, not just σ\*

- **σ\*** — the frontier (return-vs-σ curve crossing the bar).
- **Bounded-resource usage vs σ** — should switch on exactly at σ\*. If it never
  saturates yet return collapses, your law isn't the binding constraint.
- **best_β vs σ** — a crossover from "full" to "near-zero" marks the wall and is
  the loop-gain fingerprint; often a cleaner σ\* signal than the return curve.
- **Graceful vs catastrophic collapse** (deficit decomposition) — split the loss
  into achievement-mediated (does less per step) vs termination-mediated (dies
  early). Want ≤ 50% termination-mediated; catastrophic death also starves any
  future learner.
- **Channel attribution** — if the NS has separable components, turn each on alone.
  Often one dominates the harm — a direct redesign lever.

## II.6 The decision Phase 1 forces

```
σ_target ≤ σ* ?
├── YES → WELL-POSED. Build PACT (Phase 2). The scripted law you certified IS the
│         method's target and its ceiling.
└── NO  → ILL-POSED. Redesign, then re-run Phase 1. Dials, in order of ease:
            (a) lower σ to σ*−margin            (one config value)
            (b) attenuate the harmful channel   (keep σ high; needs the attribution probe)
            (c) cap the driver so it never exceeds what the law can undo
```

---

# PART III — PHASE 2: Build PACT at σ ≤ σ\*

Now that a compensation law is certified to exist, deploy it as a learned,
decentralized method. **Phase 2 = Phase 1 with the driver computed (not handed
over) and the gain learned (not hand-set).**

## III.1 The mechanism (all env-side; host RL untouched)

```
                per agent i, every env step
  obs_i ─► policy π (host, UNCHANGED) ─► (a_i , w_i)
                                            │     └─► β_i = β_max·sigmoid(w_i)   [the ONE scalar]
                                            ▼
  share peers' executed actions ─► x2_i = leak_ρ(Σ_{j≠i} Φ)    [EXACT arithmetic]
                                            ▼
                        u_i = channel_inverse(a_i ; β_i·x2_i)   [the Part-II law, β·x2 in place of ℓ]
                                            ▼
                        env.step(u) ; x2_i ← leak_ρ(...) from executed u   (cache for t+1)
  obs augmentation: o_i ⊕ [ x2_i , β_i , ⟨|x2_i|⟩ ]   (post-normalization, native units)
```

1. **Peer-action sharing (declared communication).** Each agent's executed actions
   (what `Φ` reads) are visible to teammates, one step delayed: O(N) scalars/step —
   the minimal message that makes `x2_i` exact. Not learned.
2. **Exact waveform `x2_i`.** Iterate the env's leak recursion on the shared
   actions. Cache it; the value compensating step t is the one updated after t−1
   (one-step timing contract — verified by the gate in §V).
3. **Compensation = the certified channel inverse**, driven by the learned gain:
   `u_i = clip(a_i − β_i·x2_i)` (additive), `a_i/(1−β_i·x2_i)` (throttle), etc.
4. **The one learned scalar β.** One extra bounded action dim `w_i` →
   `β_i = β_max·sigmoid(w_i)`, lightly EMA-smoothed (**direct mode** — starts at a
   partial-compensation value, phase-dependent instantly). Perfect play: `β_i ≈ c`.
5. **Observation block** appended *after* the host normalization (native units).

## III.2 The floor property (why it cannot crater)

With `β ≡ 0`, `u_i = clip(a_i)` — exactly the blind policy. No estimator in the
control path to be wrong, so no configuration performs below blind. Every failure
mode of the estimation-based predecessors is structurally excluded (no teacher, no
identifier, no plant model, no readout).

## III.3 Two design choices that matter (learned empirically)

- **Recurrent policy (required for β to modulate).** A memoryless policy can't
  sense the hidden phase and leaves β at a constant compromise. Recurrence gives
  the memory to estimate the within-episode-≈constant `c` and modulate β.
- **CTDE privileged critic (the last stretch).** Give the *centralized critic only*
  the true driver `A(t)` (training-only; execution stays fully decentralized).
  Standard CTDE; it sharpens the weak β-control gradient. On Ant this lifted 5.5k
  (fully decentralized) → ≈6k (the ceiling).

## III.4 The empirical journey (keep for the paper's method section)

On Ant-PCR (σ=0.45; blind ≈5000; ceiling 6104):

1. **Integrator β + memoryless → 4442 (below blind).** A single global integrated
   gain *collapses to 0*: optimal β is phase-dependent (≈0.44 peak, 0 trough where
   `x2≈0.2` but `c=0`), a phase-blind policy can't make one shared gain be both,
   and the trough penalty wins. Gate perfect throughout — mechanism right, control
   scheme wrong.
2. **Direct per-agent β + recurrent → 5500 (≈50% of the gap).** Direct sigmoid β
   removes the collapse; recurrence gives phase memory. Result: a good *constant*
   β≈0.30 (halves the peak residual). β still sits at its init — the control dim's
   return effect is small/noisy and the phase is locally hard to sense.
3. **+ CTDE critic (+ more steps) → ≈6000 (≈the ceiling).** The sharper value
   moves β and closes the peak gap. Execution stays decentralized.

**Reading for a new env:** expect *constant-β* to be an easy robust win (~half the
gap, zero estimation), and full phase-tracking to be the frontier that benefits
from recurrence + a CTDE critic. The arithmetic transfers with certainty; β-tracking
is the shared research frontier.

---

# PART IV — Theory (both phases in one frame)

Let the stationary game (`c≡0`) have optimal value `V*₀ = B0` in the limit.

- **T1 (exact waveform).** Under action-sharing, `x2_i` equals the env's internal
  accumulator up to the scalar `c`: `ℓ_i = c·x2_i` exactly when `c` is constant over
  the leak window, and to `O((1−ρ)|dc/dt|·window)` otherwise. *Certified by the
  per-step cosine gate.*
- **T2 (conjugacy/reduction).** With `β=c`, the additive channel delivers
  `clip(a_i)` — **byte-identical to the stationary game** for any driver path — so
  the compensated optimum equals `V*₀`. The reward fall is covered by construction.
- **T3 (certainty-equivalence bound).** With `β=c−e`, residual = `e·x2`, and the
  return gap is **linear in the single-scalar tracking error**:
  `V*₀ − V(π_β) ≤ (γ L_P L_V/(1−γ))·sup_t E‖e·x2‖ + sat`.
- **T4 (the σ\* frontier).** T2/T3 require the channel invertible-within-bounds with
  margin. That fails at a computable `c* = σ*` — **the exact quantity Phase 1
  measures.** Phase 1's return-vs-σ crossing is the empirical graph of T4; Phase 2
  runs below it.
- **Local-unobservability (the honest limit).** At `β=c` the residual → 0, so `c` is
  locally unobservable — β must be estimated when the residual is informative and
  *held*, or supplied via the periodic structure / CTDE. This bounds the last
  stretch of β-tracking; it is a structural property of the class, not a tuning
  failure.

---

# PART V — Per-env instantiation: the three declarations (serve both phases)

To port the whole pipeline to a new env, make **three declarations**; everything
else is mechanical and identical across envs.

| # | Declaration | Question | Used in | Ant-PCR |
|---|---|---|---|---|
| 1 | **Exertion `Φ`** | What of the *others* feeds the liability? (the shared message / the privileged signal's source) | Phase 1 signal + Phase 2 `x2` | `Σ_{j≠i} τ_j`, per joint-type |
| 2 | **Leak / coupling** | How does others' exertion accumulate? | Phase 1 freeze-slice + Phase 2 `x2` recursion | leaky sum, ρ=0.8, hips↔hips, ankles↔ankles |
| 3 | **Harm channel `g` + inverse** | How does it strike, and how to undo it? | Phase 1 scripted law + Phase 2 learned law (Part-II table) | additive; inverse = subtract `β·x2`, clip |

**Worked sketches (confirm each against the env's NS code):**

- **SMAC / SMACv2 (move-target-drift).** (1) `Φ` = others' exertion the NS
  aggregates (attack/damage or move-effort); (2) the env's leak; (3) channel =
  target displacement → inverse re-aims by `−β·x2`. Discrete actions where a
  continuous re-aim isn't expressible: deliver `β·x2` (and `x2`) as obs features and
  let the policy pick the corrected discrete action (the soft variant — keeps the
  1-scalar reduction). Tag/window at episode granularity if the buffer is episodic.
- **MAMuJoCo throttle-type.** (1) `Φ` = others' effort; (2) leak; (3) multiplicative
  → inverse `u = a/(1−β·x2)` on the set where the throttle stays above its floor —
  and *that floor is σ\* (T4); certify it in Phase 1*.
- **General rule.** The compensation is always "the channel inverse driven by the
  one learned scalar." Where the inverse isn't a clean reparameterization
  (discrete/constrained), fall back to the obs-feature soft variant.

---

# PART VI — Diagnostics & gates (both phases)

**Phase 1:**
- Transparency (driver=0 → B0), works-when-it-should (low-σ recovery). Non-negotiable.
- Log return, bounded-resource usage, best_β, deficit decomposition, channel
  attribution — read all four (§II.5), not just σ\*.

**Phase 2:**
- **The one hard gate (arithmetic, gait-independent):** mean **per-step cosine**
  between `x2` and the true liability > 0.999. Certifies only the leak wiring
  (index order, reset masking, one-step timing). Use per-step cosine, **not** a
  Pearson correlation pooled across the driver's range — a pooled corr reads only
  ~0.95 from the varying-`c` fan even when every point is exactly `ℓ=c·x2`.
- **β-tracking (the ballgame):** log β, true `c`, their difference, split by phase.
  Success = `β_peak→c`, `β_trough→0`. Flat β at init = phase-blind (needs
  recurrence / CTDE critic).
- **Residual felt:** `mean‖ℓ − β·x2‖` → 0 as β tracks.
- **Floor/safety:** eval < blind is impossible if wired right — suspect a wiring or
  normalization bug, not the method.

---

# PART VII — Pitfalls (merged; each cost real time)

1. **Fixing the gain across the σ sweep** → understated σ\*. Re-optimize β per σ.
   *(The actual Ant Phase-1 bug.)*
2. **Probe/wrapper state leaking across episodes.** Delay buffers, filters, the
   `x2`/"previous driver" slot must reset exactly when the env resets. Some vec-env
   wrappers reset the inner env through a path that *bypasses* your shim — verify
   the reset reaches it, in **both** phases.
3. **Privileged signal arrives mangled.** The law needs the driver in *native
   units*. If the env normalizes the whole obs vector and you piggyback the driver
   inside it, the controller gets a per-step-rescaled driver — an "oracle" that
   isn't. Read it straight from `info` (Phase 1); append obs features *after*
   normalization (Phase 2). *(This bit the training oracle arms on Ant.)*
4. **Same-file-different-module deploys.** A drop-in env file copied over a
   library's module: set knobs on the *deployed* module, not your repo copy.
5. **Confusing "RL can't learn it" with "it can't be done."** Never conclude
   "unsolvable" from a failed *training* run — only from a failed *scripted-
   privileged* run (Phase 1). This discipline is the whole point.
6. **The `c = A·σ` equivalence.** If the driver enters as a product with σ, a frozen
   slice at effective severity `c` is identical whether reached via high-driver×low-σ
   or vice versa. Your baseline collapse profile across the *driver* range already
   gives the profile across *σ* — half the frontier for free, and the repaired
   (lower-σ) env *is* the low-driver slices you already measured.
7. **Per-step cosine, not pooled correlation, for the Phase-2 gate** (§VI).
8. **CI discipline.** ~40 eps/cell, bootstrap CIs; a σ\* that flickers with the seed
   isn't one; a frontier crossing must be a clear gap (hundreds of points).

---

# PART VIII — Honest framing & novelty (for the paper)

- **PACT is a communication method** — agents share executed actions (O(N)
  scalars/step, one-step delay). *Declared*, and *justified by a measured
  impossibility* (Walls A/B: passive identification fails on the trained gait). The
  message is the provably minimal one that makes the waveform exact. Where
  communication is disallowed, PACT degrades to the obs-feature soft variant and the
  constant-β win.
- **Decentralized execution is real.** The actor uses only its own obs, its own
  action history, and locally-computed `x2`, `β`. The CTDE critic (optionally on the
  driver) is training-only.
- **Two honest tiers:** fully-decentralized PACT (no privileged info anywhere)
  recovers ~half the gap; PACT + CTDE critic reaches the compensation ceiling.
  Report both.
- **The frontier is a contribution.** σ\* (T4) certifies solvable-by-construction;
  the local-unobservability bound explains the residual gap. Impossibility
  measurement + certificate + method + ceiling is a coherent narrative.
- **Positioning.** vs **RMA/teacher-student**: no simulator-exposed latent — the
  waveform is *computed*; the learned latent is one scalar. vs **VariBAD/meta-RL**:
  exact waveform, single-scalar belief with a linear value bound. vs **opponent
  modeling**: models the *physical medium* of teammates' actions, not their minds.
  vs **disturbance observers**: the disturbance's own recursion *is* the observer,
  exactly — no plant model. vs **ECHO-R/ECL/RECON**: those *estimate* the vector and
  hit the walls; PACT *computes* it and learns only the scalar — they are its
  motivation and its ablations.

---

# PART IX — End-to-end checklist for a new environment

```
--- PHASE 1: CERTIFY (minutes of eval once infra is built) ---
[ ] 1. Read step(): the exact line the driver alters the command.
[ ] 2. Pick the compensation law that inverts it (Part-II table = declaration #3).
[ ] 3. Identify the bounded resource the law spends (usually action bounds).
[ ] 4. Train B0: base algo with the NS OFF. Record B0. (Reuse this checkpoint.)
[ ] 5. Add a freeze knob (driver held constant at severity σ).
[ ] 6. Expose the privileged driver in info, NATIVE units.
[ ] 7. Build the probe shim: intercept → apply law → step. Verify reset reaches it.
[ ] 8. CHECK A: probe at driver=0 reproduces B0 exactly.
[ ] 9. CHECK B: probe at low severity recovers performance.
[ ]10. Sweep σ × β; max_β R per σ; log return, resource usage, best_β, falls.
[ ]11. σ* = largest σ with max_β R ≥ 0.90·B0. Read best_β crossover + resource onset.
[ ]12. Decide: σ_target ≤ σ* → PHASE 2; else redesign (lower σ / attenuate / cap) → re-run.

--- PHASE 2: BUILD PACT (at σ ≤ σ*) ---
[ ]13. Env wrapper: share peers' executed actions; compute x2 by the exact leak;
       add the w_i action dim and β_i = β_max·sigmoid(w_i); apply the channel
       inverse u_i; append [x2_i, β_i, ⟨|x2_i|⟩] post-normalization to obs and the
       global versions to the critic state; stash the true liability for the
       gate/log ONLY (never the control path).
[ ]14. Runner: log the per-step-cosine gate, β-vs-driver by phase, residual, reward
       decomposition. Hard-abort only on the (gait-independent) cosine gate.
[ ]15. Config: beta_mode=direct, β_max ≈ 1.3× peak c, recurrent policy ON;
       optionally critic_payload=true (CTDE). Host hyperparameters UNCHANGED.
[ ]16. Unit-test the arithmetic (leak exactness, β=0 reduces to blind, per-agent β)
       in pure numpy — no simulator.
[ ]17. Run: blind baseline, PACT (decentralized), PACT + CTDE critic, and the
       Phase-1 scripted-cancellation ceiling. The gap between decentralized PACT
       and the ceiling is the phase-tracking frontier for that env.
```

Infra (steps 4–7, 13) is built once per env. The genuinely env-specific creativity
is **step 2 = declaration #3 = the compensation law**, and the Part-II table covers
the common cases. Everything downstream is mechanical.

---

# APPENDIX — Ant-PCR reference numbers (σ=0.45, 4×2, seed 1)

**Phase 1 (σ\* sweep, scripted cancellation, peak, bar = 0.9·B0 = 4795):**

| σ | best β | max_β R | sat_frac | ≥ bar? |
|---|---|---|---|---|
| 0.4 | 1.0 | 5490 | 0% | ✅ |
| **0.5** | **1.0** | **4857** | **0%** | ✅ **← σ\*** |
| 0.6 | 1.0 | 4089 | 0% | ❌ |
| 0.7 | 1.0 | 2030 | 0% | ❌ |
| 0.8 | 0.75 | 414 | 1.6% | ❌ |
| 0.9 | 0.25 | 222 | 3.9% | ❌ |
| 1.0 | 0.25 | −99 | 5.8% | ❌ |

B0 = 5328. Transparency ✅ (driver-0 → 5328). Works-when-it-should ✅ (c=0.45, β=1:
blind 2080 → 5040 = 95%). best_β crossover 1.0→0.25 = loop-gain fingerprint. Channel
attribution: ankle-only does ~all harm (443 vs hip-only 3641). Decision: σ_target=0.9
> σ\*=0.5 → ill-posed → redesign dial (a) → run at σ=0.45.

**Phase 2 (PACT ladder at σ=0.45):**

| Arm | What it is | Cycle-avg |
|---|---|---|
| stationary (no NS) | upper reference, not the target | ≈ 7000 |
| **O1 = compensation ceiling** | hardwired cancel with the *true* disturbance | **6104** |
| **PACT + CTDE critic** | learned β, driver in critic only; execution decentralized | **≈ 6000** |
| **PACT (fully decentralized)** | learned β, no privileged info anywhere | **≈ 5500** |
| blind HAPPO | no compensation | ≈ 5000 |
| integrator β (negative result) | global gain collapses to 0 | 4442 |

*End — the complete pipeline. Certify the frontier, then build below it.*
