# Finding σ\* — the feasibility frontier of a non-stationarity

*A transferable protocol. General method first; the Ant PCR implementation is the
worked example in §9.*

---

## 0. What σ\* is, and why it is the number that matters

**σ\* is the largest severity at which a *privileged, scripted* controller can
still hold the peak of the non-stationarity to ≥ 90 % of the undisturbed
baseline.**

- **Privileged** = it is handed the hidden driver of the NS (the thing a blind
  agent has to infer). No estimation error.
- **Scripted** = a hand-written control law, no learning, no gradients. Runs in
  minutes.
- **Peak** = the NS held frozen at its worst point.

Why this one number decides everything:

> A learner can, at best, match what a controller *with perfect knowledge* can
> do. So if the privileged scripted controller already fails at severity σ, **no
> method — no estimator, memory, conditioning, or architecture — can succeed at
> σ.** The bottleneck there is not information or optimization; it is the
> environment itself.

σ\* therefore splits the severity axis cleanly:

| region | meaning | what to do |
|---|---|---|
| σ ≤ σ\* | **well-posed** — a compensator exists, so learning has a reachable target | build the method |
| σ > σ\* | **ill-posed** — even perfect knowledge loses | redesign the env |

This is the "certify-then-build" discipline: measure existence *before* spending
compute on a method. Every method that fails above σ\* fails for the same reason,
and no amount of cleverness changes it.

---

## 1. The core idea, in one line

> Turn off learning. Hand a hand-written controller the exact hidden driver.
> Sweep severity. The severity where that controller can no longer recover the
> baseline is σ\*.

Everything else is making that measurement trustworthy and interpretable.

---

## 2. The five ingredients (the same for every env)

1. **A competent baseline policy on the *undisturbed* env, and its return B0.**
   Train your base algorithm with the NS **off** (driver frozen at its trough /
   severity 0). B0 = that policy's return with no disturbance. This is the
   ceiling everything is measured against, and — crucially — the policy is
   **severity-independent**, so you reuse the *same* checkpoint at every point in
   the sweep. You never retrain during frontier-finding.

2. **A "freeze" knob** that holds the NS driver at a chosen constant value. The
   NS drifts; to measure existence *at* severity c you must hold it there. This
   turns the drifting env into a stationary game at effective severity c.

3. **The privileged signal** — the exact hidden driver the compensation law
   needs, exposed in the env's `info` (or obs, on a labeled arm). It must reach
   the controller in the **native units** the law consumes (see pitfall §7.3).

4. **A probe shim** that intercepts the agent's action at the env boundary,
   rewrites it using the privileged signal, and steps the env. No gradients.

5. **A severity sweep with per-severity gain re-optimization** (see §4 — this is
   where the naive version goes wrong).

Ingredients 1–4 are infrastructure you build once per env. Ingredient 5 is the
measurement.

---

## 3. Choose the compensation law — *the only part that changes per env*

This is the heart of transferring the method. **What "compensate for the driver"
means depends entirely on how the NS harms the agent.** Identify your NS's
mechanism, then pick the matching law and the matching *bounded resource* the law
spends. σ\* is where the law's demand exceeds that resource.

| NS mechanism | how it harms | compensation law | privileged signal | bounded resource → what caps σ\* |
|---|---|---|---|---|
| **Additive disturbance** `delivered = a + d` | an uncommanded push | `a' = a − β·d` | `d` | action limits — cancelling needs `‖a−βd‖` inside bounds |
| **Multiplicative / effectiveness** `delivered = η·a`, η<1 | weak actuators | `a' = a / η` (clipped) | `η` | action limits — `a/η` blows past the rail as η→0 |
| **Action-space transform** `delivered = T(a)` (e.g. rotation R(θ)) | mis-aimed commands | `a' = T⁻¹(a)` | `θ` / `T` | invertibility + bounds — T⁻¹ may leave the feasible set |
| **Target / goal drift** (nav, move-target) | you aim at a stale goal | re-aim at the true drifted goal | the goal offset | reachability / velocity — can you still get there in time |
| **Dynamics / parameter shift** (mass, friction, wind) | wrong internal model | model-based feed-forward correction | the shifted params | control authority — force/torque needed to counter it |

**How to find your env's law if it isn't in the table:** read the env's
`step()`. Find the exact line where the driver alters what the agent commanded
before it reaches the simulator/dynamics. The compensation law is whatever
*inverts that line*. The bounded resource is whatever constraint that inverse
runs into (almost always the action bounds, sometimes reachability or
invertibility).

**Watch for the loop-gain twist (category-C / interaction-coupled NS).** If the
driver is *fed by the agents' own actions* (teammates' torques, joint effort),
then compensating changes the very quantity you are compensating — a feedback
loop. Cancelling can *amplify* the disturbance. The tell (see §6) is that **more
compensation makes things worse** at high severity: the best gain drops from
"full" toward "none" as you cross σ\*. If you see that, your law is fighting a
loop, not just a static offset, and σ\* will be lower than a naive open-loop
estimate suggests.

---

## 4. The sweep protocol (general, step by step)

```
GIVEN: baseline policy π (return B0 undisturbed), freeze knob, privileged signal,
       probe shim, compensation law L(driver; β) with a scalar gain β.

1.  Pick the recovery bar:  BAR = 0.90 · B0.            # the existence threshold
2.  Pick a severity grid that brackets the frontier, e.g. {0.4, 0.5, 0.6, ...}.
3.  Pick a gain grid, e.g. β ∈ {0.25, 0.5, 0.75, 1.0}.
4.  FOR each severity σ in the grid:
        freeze the NS at its PEAK, severity = σ
        FOR each gain β:
            roll π through the shim with law L(driver; β), N≈40 episodes
            record: return, the bounded-resource usage, the fall/termination cause
        R(σ) = max over β of the returns        # ← re-optimize β PER severity
        best_β(σ) = the argmax
5.  σ*  =  the LARGEST σ with  R(σ) ≥ BAR.
6.  Redesign target (if you need to lower severity):  σ* − margin  (≈ 0.05).
```

**Why step 4 re-optimizes β and never fixes it:** the optimal gain *changes with
severity*. At low severity full cancellation (β=1) is best; near/above σ\* the
best gain collapses toward zero because over-cancelling saturates the resource. If
you fix β at whatever was best at your *starting* severity, you measure a
frontier that is wrong — usually far too low, because a gain tuned where
cancellation is failing is a near-useless gain everywhere. (This was a real bug in
the Ant run; see §9.)

**Episodes / statistics:** ~40 deterministic episodes per (σ, β) cell. Report
bootstrap CIs. The frontier crossing should be a clear gap (hundreds of points),
not a coin-flip between adjacent σ.

---

## 5. The two confounding checks (do these before believing any result)

A frontier is only meaningful if the probe itself is sound. Two checks, both
non-negotiable:

1. **Transparency at zero severity.** Run the compensation probe with the driver
   at **0**. It must reproduce B0 *exactly* (CIs fully overlapping identity). If
   it doesn't, your shim is corrupting the action independent of the NS — fix the
   probe, do not interpret the frontier.

2. **It-works-when-it-should.** At a *low* severity, compensation must actually
   *recover* performance (lift the blind return back toward B0). If compensation
   never helps anywhere, your compensation law or your privileged signal is wrong
   for this env — you are not measuring existence, you are measuring a broken law.

Only when both pass is a failure at high severity a property of the *environment*
rather than the probe. In the Ant run these two checks are exactly what
distinguished "the benchmark is genuinely ill-posed" from "I have a sign error."

---

## 6. Reading the result — four things, not just σ\*

The sweep gives you more than a single number. Log and read all four:

- **σ\* itself** — the frontier (from the return-vs-σ curve).
- **The bounded-resource usage vs σ** (saturation fraction, out-of-bounds
  fraction, whatever your law spends). σ\* should align with where this *switches
  on*. If the resource never saturates yet return still collapses, your law isn't
  the binding constraint — reconsider the mechanism.
- **best_β vs σ.** A crossover from "full" (β=1 best) to "near-zero" (small β
  best) marks the onset of the wall and is the fingerprint of the loop-gain twist
  (§3). It is often a cleaner signal of σ\* than the return curve.
- **Graceful vs catastrophic collapse** — the *deficit decomposition*. Split the
  return loss into "achievement-mediated" (the agent does less per step) vs
  "termination-mediated" (the agent dies early). A soft, achievement-mediated
  wall means the frontier is a smooth capability limit; a hard,
  termination-mediated one means absorbing-state death, which also starves any
  future learner. (Rule of thumb: want ≤ 50 % termination-mediated at the peak.)

Also worth a probe: **channel attribution.** If the NS has separable components
(here: hip vs ankle coupling), turn each on alone. Often one component drives most
of the harm — a direct lever for redesign (weaken *that* component, keep severity
high elsewhere).

---

## 7. Pitfalls (each cost real time to learn)

1. **Fixing the gain across the sweep** → understated σ\*. Re-optimize β per
   severity (§4). *This was the actual bug in the Ant frontier run.*
2. **Probe state leaking across episodes.** If your shim holds per-episode state
   (delay buffers, filters, a "previous driver" slot), it must reset exactly when
   the env resets. Some vec-env wrappers reset the inner env through a path that
   *bypasses* your shim — verify the reset actually reaches it, or every number is
   quietly corrupted.
3. **The privileged signal arrives mangled.** The compensation law needs the
   driver in its *native units*. If the env normalizes the observation vector and
   you piggyback the driver inside that vector, the controller receives a
   per-step-rescaled driver, not the real one — an "oracle" that isn't. Feed the
   probe the driver straight from `info`, unnormalized. (On Ant this bit the
   *training* oracle arms, not the probes, precisely because the probes read raw
   `info` — a good reason to keep the scripted probe's signal path separate from
   any obs plumbing.)
4. **Same-file-different-module deploys.** If the env is a drop-in file copied
   over a library's module, setting a knob on your repo copy does nothing to the
   running env (different module object). Resolve and set knobs on the *deployed*
   module.
5. **Confusing "RL can't learn it" with "it can't be done."** The entire reason
   to script the privileged controller first. Never conclude "unsolvable" from a
   failed training run — only from a failed *scripted-privileged* run.
6. **The `effective-severity = driver × σ` equivalence.** If the driver enters as
   a product with σ, then a frozen slice at effective severity c is identical
   whether you reach it via a high driver × low σ or vice versa. Exploit it: your
   baseline collapse profile across the *driver* range already tells you the
   collapse profile across *σ* — you may have half the frontier for free, and the
   repaired (lower-σ) env is literally the low-driver slices you already measured.
7. **CI discipline.** 40 eps/cell, bootstrap CIs, and treat a frontier crossing
   as real only on a clear gap. A "σ\*" that flickers with the seed isn't one.

---

## 8. The decision the frontier forces

```
Is your intended severity σ_target ≤ σ* ?
├── YES → the NS is WELL-POSED at σ_target. Build the method.
│         The scripted law you used IS the method's target: a learner that
│         estimates the driver and applies that law will work (its ceiling is
│         the frontier you just measured, and it's above the bar).
│
└── NO  → the NS is ILL-POSED at σ_target. Redesign, then re-run this protocol.
          Three dials, in order of ease:
            (a) lower σ to σ* − margin              (one config value)
            (b) attenuate the harmful channel only  (keep σ high; needs the
                channel-attribution probe first)
            (c) cap the driver so it can never exceed what the law can undo
```

Redesign always re-runs the frontier protocol on the repaired env — the fix is
only certified when the scripted controller passes.

---

## 9. Worked example — the Ant PCR env

The concrete choices that instantiate §2–§7 for the payload-coupled Ant. Use this
as the template to fill in for a new env.

**The NS (read from `ant.py step()`):** additive, teammate-fed disturbance.
`delivered_i = clip(τ_i + d_i, −1, 1)`, where
`d_i ← ρ·d_i + (1−ρ)·A(t)·σ·Σ_{j≠i} τ_j`. So it is **category C** (fed by
teammates' torques → the loop-gain twist applies) and **additive** (→ subtractive
compensation).

| ingredient (§2) | Ant choice |
|---|---|
| **B0 baseline** | HASAC trained with `ANT_PCR_MASK=off` (coupling zeroed → stock Ant), 5 M steps. B0 = **5328**. Reused at every σ. |
| **freeze knob** | `ANT_PCR_FREEZE_A=a` holds the payload constant → effective severity **c = a·σ**. |
| **privileged signal** | `info["pcr_d_next"]` — the exact 8-dim per-joint disturbance the *next* action will face, in torque units, straight from `info`. |
| **probe shim** | `ProbeShim` intercepts the flat 8-dim action at the gym boundary, applies the law, steps. Resets its "previous d" slot on env reset (via `install_probe`, which hooks the reset path the vec-env otherwise bypasses — pitfall §7.2). |
| **compensation law** | additive → `a' = clip(a − β·d, −1, 1)`. Feed-forward is exact because `d_applied(t) == d_next(t−1)`, so the shim carries last step's `pcr_d_next` as this step's estimate. |
| **bounded resource** | actuator torque; usage metric = `sat_frac` = fraction of joints with `|τ+d| > 1`. |

**The sweep (E2b):** severities `{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3}`,
gains `β ∈ {0.25, 0.5, 0.75, 1.0}`, `max_β R` per severity, bar = 0.9·B0 = **4795**.

Result:

| σ | best β | max_β R | sat_frac | ≥ bar? |
|---|---|---|---|---|
| 0.4 | 1.0 | 5490 | 0 % | ✅ |
| **0.5** | **1.0** | **4857** | **0 %** | ✅ **← σ\*** |
| 0.6 | 1.0 | 4089 | 0 % | ❌ |
| 0.7 | 1.0 | 2030 | 0 % | ❌ |
| 0.8 | 0.75 | 414 | 1.6 % | ❌ |
| 0.9 | 0.25 | 222 | 3.9 % | ❌ |
| 1.0 | 0.25 | −99 | 5.8 % | ❌ |

**σ\* = 0.5.** Everything §5–§6 predicted shows up:

- **Transparency check:** cancel-at-driver-0 = 5328 = identity. ✅
- **Works-when-it-should:** at c=0.45, β=1 lifts blind 2080 → 5040 (95 %). ✅
- **best_β crossover:** 1.0 for σ ≤ 0.7, dropping to 0.75 / 0.25 for σ ≥ 0.8 —
  the loop-gain fingerprint (§3, §6). Beyond σ ≈ 0.8, *more cancellation is
  worse than none*.
- **sat_frac** switches on exactly at σ\*'s far side — the resource wall.
- **graceful collapse:** 95 % achievement-mediated at the peak (soft wall).
- **channel attribution (E6):** ankle-only coupling does ~all the harm
  (443 vs hip-only 3641) — the redesign lever.
- **the bug we hit (§7.1):** the first E2b fixed β at 0.25 (the best gain at the
  *failing* σ=0.9) and swept σ with it, reporting σ\* < 0.6 — wrong. Re-optimizing
  β per σ gave the true σ\* = 0.5.
- **the equivalence (§7.6):** because c = A·σ, the σ=0.45 repaired env *is* the
  A≈0.5 slices of the σ=0.9 data — the collapse profile (2080, a 61 % drop) and
  the recovery (5040) were already in hand before the repaired env ever ran.

**Decision:** σ_target = 0.9 > σ\* = 0.5 → ill-posed. Redesign dial (a): set
σ = 0.45. Then the method is a learned disturbance observer applying the same
`a = π − β·d̂` the scripted probe just certified.

---

## 10. Copy-paste checklist for a new env

```
[ ] 1. Read step(): find the exact line the driver alters the command.
[ ] 2. Pick the compensation law that inverts that line (§3 table).
[ ] 3. Identify the bounded resource the law spends (usually action bounds).
[ ] 4. Train B0: base algo with the NS OFF. Record B0. (Reuse this checkpoint.)
[ ] 5. Add a freeze knob (hold the driver constant at severity σ).
[ ] 6. Expose the privileged driver in info, in NATIVE units.
[ ] 7. Build the shim: intercept action → apply law → step. Verify reset reaches it.
[ ] 8. CHECK A: probe at driver=0 reproduces B0 exactly.
[ ] 9. CHECK B: probe at low severity recovers performance.
[ ]10. Sweep σ × β; take max_β R per σ; log return, resource-usage, best_β, falls.
[ ]11. σ* = largest σ with max_β R ≥ 0.90·B0.
[ ]12. Read best_β crossover + resource onset to confirm the wall is the resource.
[ ]13. Decompose collapse (graceful vs catastrophic) + attribute the channel.
[ ]14. Decide: σ_target ≤ σ* → build; else redesign (lower σ / attenuate / cap) → re-run.
```

The infrastructure (steps 4–7) is built once per env. The measurement (steps
8–12) is minutes of eval. The only genuinely env-specific creativity is step 2 —
the compensation law — and the §3 table covers the common cases.
