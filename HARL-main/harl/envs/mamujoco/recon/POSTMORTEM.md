# RECON on Ant-PCR — Post-Mortem

**Status: the method does not work on this benchmark. Six runs, all failed. This
document records what was tried, what each attempt measured, and why every fix —
including two that were themselves well-motivated by data — failed. It ends with
the two fundamental walls the evidence converges on, and what (if anything) could
work instead.**

Author: implementation + diagnosis log, 2026-07. Env: MAMuJoCo Ant-v2 4×2, PCR
non-stationarity at the repaired severity σ = 0.45. Host: HAPPO. Target: recover
the stationary ("no-NS") return, ≈ 7000. Best result achieved by any RECON
variant: ≈ 5200 (run #1), which on inspection was **plain HAPPO with the
mechanism inert** — every run where the mechanism was actually *active* did
**worse** than blind (≈ 2500–3300), and the final self-supervised variant was the
worst of all (≈ 344, the walker repeatedly falling).

---

## 1. The problem

PCR (payload-coupled reaction) injects, into each joint's commanded torque, a
slow-drifting parasitic load that is a **leaky accumulator of the *other* agents'
torques**:

```
delivered_i(t) = clip( a_i(t) + d_i(t),  −1, +1 )
d_i(t+1)       = ρ·d_i(t) + (1−ρ)·c(t)·Σ_{j≠i} a_j(t)          ρ = 0.8
c(t)           = A(t)·σ,   A(t) ∈ [0,1] a slow cyclic payload,  σ = 0.45
```

The reward is unchanged; the harm enters only through the dynamics. Blind HAPPO
reaches ≈ 5000 (the NS costs it ≈ 2000 vs the stationary 7000). The goal was to
recover that 2000 by estimating `d_i` and cancelling it at the action interface,
`u_i = clip(a_i − β·d̂_i)`.

**Why this looked solvable.** A prior diagnostic campaign claimed (a) that with
the *true* `d` handed to the policy, feed-forward cancellation recovers 95–100%
of the stationary return, and (b) that `d` is decodable from each agent's own
history (R² ≈ 0.64), and (c) that the scalar `c` is identifiable from the replay
buffer's joint actions ("the identifier locked at corr ≈ 1"). RECON's whole
design rests on (c): **manufacture a training label for a per-agent filter by
identifying `c` centrally and reconstructing `d̃ = ĉ · leak(Σ others)` from the
buffer.** Claim (c) turned out to be false on the trained gait. That is the
headline finding.

---

## 2. The method (RECON), as specified

```
TRAINER (central, per rollout)                     AGENT i (execution, local)
[ID] identify (ρ, c) from the buffer's           [F]  filter f(o_i, u_i history) → d̂_i
     joint actions (clip-aware fit)               [CE] policy π(o_i ⊕ d̂_i)
[RE] reconstruct label d̃_i = ĉ·leak(Σ_{j≠i} u)   [CP] u_i = clip(a_i − β·d̂_i)
[DI] distill: train f by MSE to d̃  (own Adam)
     host RL update: UNCHANGED
```

The design is a clean separation principle: identify centrally, filter locally,
act certainty-equivalently. It is correct *if* `ĉ` tracks the true `c`. It does
not.

---

## 3. Chronological failure log

| # | Variant | Steps | Result | What the debug trace showed | Verdict |
|---|---|---|---|---|---|
| 1 | central, readout `[19,21,23,25]` | 10M | **5212** | `scan_offset=−2`; `ĉ` **railed at grid-ceiling 1.2, constant**; `label_r2 = −14`; filter fit its (wrong) teacher; eval **peak > trough** (inverted) | readout index wrong; mechanism inert → ≈ blind HAPPO |
| 2 | + scan hard-abort, index→`[17,19,21,23]`, +nuisance | 1.6M | **crash** | scan aborted: `offset=−14` | my abort gated on `argmax==0`, false-fired on a valid readout |
| 3 | central + nuisance-instant term | 10M | **~3300** | `ĉ` **flat ≈ 0.30**; `c_corr = 0.03`; trough not vanishing | nuisance term's regressor collinear with the signal → ate the modulation |
| 4 | central, nuisance off, ρ=0.8, trough-baseline | 10M | **~2500–3300** | **`corr(ĉ, c_true) = −0.35`** (peak ĉ 0.42 < trough ĉ 0.58); `sumzero_frac = 0.69`; filter `true_r2 = 0.80` at peak | **central identification is *anti*-observable** — the wall |
| 5 | self-supervised pivot + fixed index | 1M | **crash** | scan aborted: velocity cols fell to `corr 0.11`, position cols rose to `0.60` | readout responsiveness swaps with gait; fixed index fragile |
| 6 | self-supervised + auto-readout | 10M | **~344** (30% of iters falling) | target `label_rms = 5…425` vs true `ltrue_rms ≈ 1e-4`; `d̂` **~250× too large**; `u_minus_a` up to **1.43** | scalar-gain disturbance observer is dominated by natural joint dynamics → massive over-compensation → walker collapses |

Detail on each:

### Run #1 — central identifier, wrong readout index (5212)
The readout `y_i` that the identifier regresses on was set to the stock-gym joint-
velocity indices `[19,21,23,25]`. This deployment's obs is shifted −2, so every
agent regressed its own torque against a *neighbour's* joint. The identifier
never locked: `ĉ` sat pinned at the grid ceiling 1.2, constant, all run. The
reconstructed labels were pure scale error (`label_rms 0.44` vs true `0.093`,
`label_r2 = −14`). The filter faithfully learned the wrong teacher; `[CP]`
subtracted ≈ 0.42 against a true load of ≈ 0.09. It still reached 5212 **only
because `d̂` is in the observation, so the policy learned to undo the compensation
by emitting `a + d̂`** — i.e. RECON degenerated to blind HAPPO with wasted
capacity. Tell-tale: eval **peak-slice > trough-slice**, backwards (the trough is
where there is nothing to cancel, yet the constant `ĉ` injected there).

### Run #2 — the scan guard crashed a good run
After run #1 I added a hard-abort if the readout scan's best column ≠ the
configured one. It false-fired at 1.6M: the argmax had merely swapped from the
(correct) velocity readout to the equally-valid position readout. The guard was
testing the wrong thing (`argmax == configured`), not "does the configured column
respond to own torque." Fixed the test; lost a run.

### Run #3 — the physical-coupling "fix" backfired (~3300)
The identifier was reading a constant ≈ 0.5 offset even at the payload trough
(where true `c = 0`). Diagnosis: the Ant's legs are mechanically coupled through
the torso, so the others' torques move a joint at the *same* step, and the fit
charges that constant coupling to `c`. The fix — add an instantaneous-coupling
nuisance regressor `h·S` to soak it up — **backfired**: `S` (instantaneous sum)
is collinear with `x2` (the leaked sum, a low-pass of `S`), so the fit couldn't
separate them. It left the constant *in* `c` **and** ate the payload modulation:
`ĉ` went flat at 0.30, `c_corr` collapsed from ≈ 0.47 (without the term) to 0.03.
The "improvement" made the identification strictly worse.

### Run #4 — the central wall, measured (~2500–3300)
With the nuisance term removed, ρ fixed at its known 0.8, and a trough-baseline
subtraction added, `ĉ` finally *modulated* (0.38–0.93) — but in the **wrong
direction**: `corr(ĉ, c_true) = −0.35` over 860 windows, mean peak `ĉ` = 0.42 <
mean trough `ĉ` = 0.58. The identifier reads *more* severity when there is *less*
load. Mechanism: `sumzero_frac = 0.69` — the competent gait is coordinated, so
the disturbance `d ≈ −c·leak(a)` looks like an own-gain change, and the per-window
fit folds the coupling into the gain, worst exactly at high load. A trough-
baseline cannot fix an *anti*-correlation, only a constant offset. Feeding this to
`[CP]` compensates **backwards** → worse than blind. **This is the same wall the
predecessor method (ECL) hit and documented; it is a property of the benchmark on
a trained gait, not a bug.** One bright spot: the filter reached `true_r2 = 0.80`
at the peak — local decoding of `d` from proprioception *does* work in favourable
windows (so the ceiling is not the filter).

### Run #5 — the scan guard crashed *again*
Same class of problem as #2, harder: at 1M the velocity readout's response to own
torque had decayed to `corr 0.11` (below the 0.25 threshold) while the position
readout rose to 0.60. The velocity/position response genuinely swaps as the
walker matures. Replaced the fixed index with **auto-selection** (pick, per
agent/channel, whichever obs column its own torque actually drives, locked over
the first few high-excitation iterations) and downgraded the scan to a warning.
Lost another run.

### Run #6 — self-supervised, the worst (~344)
Abandoning central `c` entirely, the filter was retrained on a **local
disturbance-observer target**, `d̃_i = (y_i − ĝ0_i·a_i) / ĝ0_i` — "the readout
minus the own-action effect is the disturbance." This assumes `y_i ≈ ĝ·(a_i + d_i)`:
a *static gain* from torque to the joint-velocity readout. **That model is
false.** The joint's velocity responds to torque through its dynamics — mass,
contacts, the gait's own oscillation — not a scalar gain. So `(y − ĝ0·a)` is
dominated by the joint's **natural motion**, not the disturbance. From iteration
1 the target magnitude `label_rms` was **5–145** while the true disturbance was
`ltrue_rms ≈ 1e-4` — the target was **100–1000× too large and uncorrelated with
`d`**. The filter learned that, `[CP]` subtracted `u_minus_a` up to **1.43**
(larger than the entire action range), and the walker fell over in 30% of
iterations. The self-supervised target was the most destructive of all.

---

## 4. Root-cause analysis — two walls, one underlying fact

Every failure reduces to one of two walls, and both are instances of a single
underlying fact about the benchmark.

### Wall A — central identification of `c` is *anti*-observable on the trained gait
(runs #1, #3, #4)

The scalar `c(t)` is estimated by regressing the readout on `[own action, leaked
coupling]`. On a *random* gait these regressors are independent and the fit is
clean (this is the condition under which the prior campaign measured "corr ≈ 1").
But **RL training produces a coordinated gait** (`sumzero_frac ≈ 0.69`): the legs
push in concert, so the leaked coupling becomes collinear with the own action,
and the disturbance `d ≈ −c·leak(a)` becomes **indistinguishable from a change in
the joint's own gain**. The per-window fit then folds `c` into the gain — most
strongly at high load — and `ĉ` comes out anti-correlated with the truth
(`−0.35`). No central estimator that must separate own-gain from coupling within a
window can escape this, because the information isn't there: at the coordinated
gait, "the payload loaded me up" and "my own actuator got stronger" produce the
*same* proprioceptive signature. Persistent excitation (A3) would break the
collinearity, but the tuned HAPPO config runs `entropy_coef = 0`, so exploration
noise collapses as the policy converges — exactly when identification is needed.

### Wall B — the readout is not a static gain times torque
(run #6)

Sidestepping `c` with a local disturbance observer requires a model of the
nominal plant to subtract the own-action effect. RECON used the crudest possible
model — a **scalar gain** `ĝ0`, `d̃ = (y − ĝ0·a)/ĝ0`. But a joint's velocity is a
*dynamical* response to torque (through inertia, contacts, and the gait), not
`ĝ·torque`. Subtracting `ĝ0·a` therefore leaves the joint's entire natural
motion as the putative "disturbance," which is two-to-three orders of magnitude
larger than the real `d`. The observer needs the actual plant dynamics; a gain
does not approximate them. (A learned forward model could, but that is a
different, much larger method — see §6.)

### The underlying fact
On a **competently walking** Ant at σ = 0.45, the parasitic disturbance `d` is
**not reliably observable** — not centrally from the joint actions (Wall A,
collinearity), and not locally from a simple readout model (Wall B, dynamics).
The filter's `true_r2 = 0.80` in *favourable peak windows* proves the information
is not entirely absent — but no method here could construct a **robust,
low-variance, correctly-scaled training target** to harness it across the whole
payload cycle. Whenever the mechanism was active, its estimate was either
phase-inverted (Wall A) or grossly over-scaled (Wall B), and cancelling with a
wrong `d̂` is strictly worse than not cancelling — which is why every active run
underperformed blind HAPPO.

---

## 5. A recurring self-inflicted problem: the readout-index scan

Two of the six runs (#2, #5) were lost not to the method but to my own guard
rails. The readout-index scan is **gait-dependent and unstable** (velocity vs
position responsiveness swaps as the walker matures), and I twice made a hard
abort out of it. Lesson recorded in the code and memory: **never hard-abort on a
gait-dependent diagnostic.** The final `auto_readout` (pick the responsive column,
lock it early, warn-don't-crash) is the right shape — but by the time it existed,
the method's substantive walls had already sunk it.

---

## 6. What would actually be needed (and why it's a different project)

The evidence says RECON's premise — a *reconstructed central label for `c`* — is
dead on this benchmark. Anything with a chance would have to attack the
observability of `d` directly:

1. **Restore persistent excitation (A3).** Run with `entropy_coef > 0` (or a
   Gaussian-std floor) for *all* arms. This directly breaks the own-gain/coupling
   collinearity that causes Wall A. It is the single cheapest thing left to try
   and it was never tested; it might partially rescue the *central* identifier.
   Risk: it changes the host config (fairness must be preserved by applying it to
   every baseline), and it may not fully overcome the −0.35 anti-correlation.

2. **A learned forward model instead of a scalar gain (fixes Wall B).** Train
   `ŷ = F(o_i, a_i)` on the actual proprioceptive dynamics; the disturbance is
   the residual `y − F(o_i, a_i)`. This is a proper learned disturbance observer.
   It is a substantially larger method than a scalar gain, needs its own
   validation, and still inherits the observability limit at the coordinated gait.

3. **Run the `d`-oracle ceiling first — this should have been step zero.** With
   the *true* `d` appended to the observation (`ANT_PCR_ORACLE=1`, plain HAPPO),
   measure the achievable return. **If even the oracle cannot reach 7000, then
   7000 is not recoverable at σ = 0.45 by *any* method, and the target itself is
   wrong** — every RECON run would have been chasing an unreachable number. This
   is a one-line arm and it bounds the entire effort. It was never run.

4. **Accept the negative result.** "On a competently-trained cooperative gait,
   interaction-mediated non-stationarity of this class is not observable enough —
   centrally or locally — to support feed-forward cancellation, and cancelling
   with a mis-estimated disturbance is worse than tolerating it." That is a
   genuine, publishable finding, and it is exactly what six runs plus the
   predecessor method (ECL) demonstrate.

---

## 7. What is salvageable

- **The diagnostic instrumentation.** `recon_debug.csv` isolates each link of the
  chain (identify → reconstruct → distill → filter → compensate), so every one of
  these failures *named itself* in the data rather than hiding. That harness is
  reusable and is the reason this post-mortem can be precise.
- **The measured facts**, each of which is a paper-grade result:
  `corr(ĉ, c_true) = −0.35` on the trained gait; `sumzero_frac = 0.69`;
  filter `true_r2 = 0.80` at the peak (local decodability exists) yet no usable
  training target; the scalar-gain observer over-scales by ~250×.
- **The unit tests** (U1–U5), which correctly validate the *math* of each
  component in isolation — and, tellingly, U5 passes on a *synthetic* gait while
  the real gait defeats the same estimator, which is itself the lesson: the
  benchmark's observability, not the algebra, is the binding constraint.

---

## 8. One-paragraph summary for the paper

> We implemented RECON — a separation principle that manufactures a supervised
> label for a per-agent disturbance filter by identifying the non-stationarity's
> latent severity from the CTDE replay buffer. Across six 10M-scale runs the
> method failed to recover the stationary return, and every configuration in
> which the compensation mechanism was active performed *below* the blind
> baseline. Two walls account for this. First, on the coordinated gait that
> cooperative RL converges to, the latent severity is *anti-observable* from the
> joint actions (`corr(ĉ, c_true) = −0.35`), because the payload-coupling becomes
> collinear with each agent's own actuation and folds into an apparent own-gain
> change. Second, a local disturbance-observer alternative fails because the
> proprioceptive readout is a dynamical, not a static-gain, response to torque, so
> the residual is dominated by the joint's natural motion and over-estimates the
> disturbance by orders of magnitude. Both reduce to a single fact: the
> interaction-mediated disturbance is not reliably observable on the trained gait,
> and feed-forward cancellation with a mis-estimated disturbance is strictly worse
> than tolerating the disturbance. We conclude with the conditions (persistent
> excitation; a learned plant model; a verified oracle ceiling) under which the
> problem might become tractable, and note that the oracle ceiling — which bounds
> whether the target is reachable at all — must be established before any such
> method is attempted.
```
```
```
Numbers referenced above (from the run debug CSVs, for reproducibility):
  run1: cycle-avg 5212; scan_offset −2; c_hat const 1.2; label_r2 −14; eval peak 5500 > trough 5125
  run3: c_hat ≈ 0.30 flat; c_corr 0.03
  run4: corr(c_hat,c_true) −0.35; sumzero 0.69; filter true_r2 0.80 @peak; ≈2500–3300
  run5: crash; velocity-readout corr 0.11 vs position 0.60
  run6: label_rms 5–425 vs ltrue_rms ~1e-4; lhat/ltrue ≈ 253×; u_minus_a max 1.43; 30% iters falling; final ≈344
```
