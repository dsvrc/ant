# SMAC — CTI (Coupled Targeting Interference) + PACT

> **The harm channel changed.** It used to DROP the overheated shot (CWO). That channel
> is **not invertible** — the only response is to fire less, which buys damage with
> damage — so pipeline T2 conjugacy never held and Phase 1 failed at *every* severity
> from 0.5 to 3.2 even though compensation demonstrably worked (σ=1.0: load 0.650 →
> 0.471, drop 0.448 → 0.050, throughput 0.489 → 0.631, return 7.9 → 11.3; but 0.9·B0 =
> 16.4 against a best reachable 12.5). It now **deflects the shot onto a different
> enemy**: a deterministic permutation of the attack action, so re-aiming cancels it at
> **zero cost** and the compensation ceiling is B0 itself. Everything below the
> mechanism section (de-phasing, diagnostics, the biased `stagger_gap`, the run
> protocol) is unchanged; the ⚠ GAIN/CEILING section applies only to the old channel and
> is kept for the record.


The clean category-C NS for SMAC v1, replacing the move-target-drift (which either
shrugs off or traps in a "spray damage, survive to timeout" local optimum — SMAC's
shaped reward pays for damage, not consolidation). **CWO throttles the reward-
generating action itself**, so the metric collapses gracefully *and* learnably.

## The mechanism (in `StarCraft2_Env.py` — the whole NS is CWO now)

Units share a weapon power/cooling bus; the more the **others** fire, the more **your**
weapon overheats and cuts out. Dynamics-only — an overheated shot is **dropped**
(the unit does `stop` that step, deals no damage). Reward untouched.

```
fire_j   = 1 if unit j COMMANDED an attack (action >= n_no_attack)     [pure peer-action]
x2_i     <- ρ·x2_i + (1-ρ)·(Σ_{j≠i} fire_j)/(N-1)     ρ=0.85           [shared load, PACT-computable, ∈[0,1]]
ell_i    = clip( A(t)·σ·(x2_i − KNEE)/(1 − KNEE), 0, LMAX )            [deflection, KNEE=0 LMAX=1]
harm: the delivered target is shifted s = round(ell_i·(K−1)) places along unit i's
      K currently-attackable enemies.   inverse: command `desired − s`  →  exact, free.
```

The shot still fires and still does full damage — it lands on the wrong enemy. Blind
squads are hurt because SMAC rewards **focus fire**: a scattered volley kills nothing
and dies to a full-strength enemy line. Nothing is physically removed.

- **Category C:** `A(t)` multiplies `Σ_{j≠i} fire`; N=1 → empty sum → `x2≡0` → stock SMAC
  (irreducible); frozen-but-firing partners still overheat you; can't overheat yourself.
- **No trap:** dropped shots cut *damage dealt* → cut the shaped reward directly, so
  "spray + survive" is no hideout. Movement untouched → consolidation fine → learnable.
- **Only PACT solves it (tragedy of the commons):** firing is individually good but
  collectively overheats the bus. Optimum = **stagger** (fire at fraction f\*<1, and
  f\* depends on the hidden phase `A(t)`). A **blind** agent can't see the shared load →
  fires greedily → everyone's shots drop → low DPS → falls. A **PACT** agent sees `x2_i`
  → holds when the bus is hot → the team staggers → high effective DPS → recovers.

## ⚠ Read this before configuring severity — GAIN and CEILING

The whole method is contingent on **two** arithmetic facts, and each has already cost
a full run. With per-unit throughput `T(f) = f·(1 − ell(f))` at shared load `f`,
measured at the driver peak against the load a trained *stationary* team actually runs
at (`_GREEDY_LOAD`, 0.65 on 3s5z — **measure it, don't assume f=1**):

| | what it is | failure if too low |
|---|---|---|
| **GAIN** = `T(f*)/T(greedy)` | how much coordinating is *worth* | nothing for PACT — or an oracle — to recover |
| **CEILING** = `T(f*)/T(stationary)` | what perfect coordination *keeps* | even flawless play can't win, so the metric is pinned at 0 |

The env prints both at startup and warns on either:

```
[NS] @ driver peak, vs a stationary team's measured load 0.65:  optimal load f*=0.40
     -> throughput 0.400 (ceiling 62% of stationary)  |  greedy -> 0.054  |  GAIN x7.4
```

**Both previous settings failed one of the two:**

- `SEVERITY=2.0, LMAX=0.6` → **GAIN 0.31×**. Past `f = LMAX/(A·σ) = 0.3` the cap turns
  the curve into `0.4·f`, *increasing*, so greedy beat the "stagger" optimum by 3.2×.
  The safety cap meant to stop the team being disarmed is precisely what deleted the
  incentive to stagger. 20M steps confirmed it: firing fraction at the driver peak vs
  trough **0.335 vs 0.345** — no modulation, because none was ever profitable.
- `SEVERITY=1.0, KNEE=0.15, LMAX=0.95` → GAIN 5.9× but **CEILING 36%**. 17.7M steps with
  de-aliased eval: win rate at the driver peak was **0.000 in every one of 221 eval
  rounds**, while the same policy won 93% at the trough. 3s5z is not winnable at 36% of
  its damage output, so no method could ever have moved that number.

- `SEVERITY=2.2, KNEE=0.40` → GAIN 11×, **CEILING 57%**. Phase 1 (scripted, *privileged*)
  returned win 0.000 in every cell from σ=0.5 to σ=2.8.

`KNEE` is the lever for the ceiling. When `SEVERITY ≥ (1−KNEE)/KNEE` the optimum sits
*exactly at the knee*, so a coordinated team takes **zero drops** and keeps throughput
`KNEE`. Size it **just below the measured greedy load** — the current 0.55 against a
measured 0.70 gives a 79% ceiling.

### Why the ceiling has to be that high: 3s5z is a mirror match

3s5z is 3s5z *vs 3s5z*. By Lanchester's square law the win/loss outcome is a **cliff**
in relative DPS, not a gradient — losing damage costs you units, which costs you more
damage. Measured on a B0 checkpoint: a **27% shot-drop rate took the win rate 0.600 →
0.000**, with episode length *falling* 45.7 → 37.3 (the team dies faster; it does not
time out). There is no partial band.

Two consequences:

1. **A coordinated ceiling much below ~80% of stationary damage cannot win at the peak**,
   whatever the method does. That is why `KNEE` sits just under the greedy load.
2. **Read the frontier on return, not win rate.** A 0.9·B0 *win* bar can only be met by
   a controller that restores essentially all the lost damage, and holding fire buys
   damage with damage — structurally it cannot. Win rate reports "ILL-POSED at every
   severity" for a property of the map. Return degrades gracefully (18.2 → 11.5 → 3.4
   over the same sweep) and has resolution. `pact.phase1` now defaults to `--metric ret`.

### The knee is also the team's safe harbour

At `KNEE=0.15` every level of firing above 0.15 was penalised, so there was no
harm-free operating point — and SMAC's shaped reward pays for damage dealt, which makes
"disengage and survive to the time limit" locally better than "engage and lose". That
is an absorbing state: once the team stops firing, `x2 → 0`, no jams are observed, and
there is no gradient back. Measured as severity reached full: `fire_avail` **0.89 →
0.22**, `ep_len` **50 → 141**, and the win rate at the almost-unharmed driver *trough*
collapsed **0.93 → 0.01**. The runner now prints `[PACT][DISENGAGED]` when it sees this,
because it looks like a method failure and isn't.

### What the failing run showed about the dilemma itself

During the ramp the policy moved the *wrong* way: `hold_gap` reached **−0.074**, i.e.
it fired **more** at the peak than at the trough (`fire_peak` held ~0.79 while
`fire_trough` fell 0.81 → 0.71). That is individually rational — if your shots are being
dropped, attempt more — and collectively catastrophic. It is the commons dilemma
behaving exactly as designed; the problem was that there was no reachable good outcome
to move toward.

## Two more things that were silently wrong, and are now fixed

**1. Phase aliasing in eval (and in the PPO batch).** The driver period is 5000 env
steps — ~30× a rollout and ~40× an episode. With every parallel env on the same clock,
40 eval episodes over 10 threads advance the eval clock by only ~4·ep_len ≈ 250–600 of
those 5000 steps, so **each eval round is a single-phase snapshot** and consecutive
evals crawl around the cycle. The reported win rate becomes a slow square wave — 0.0
for most evals, ~0.95 for a few — that measures *the driver phase*, not the policy. Its
beat period even shrinks from ~20 evals to ~10 as episodes lengthen: the fingerprint of
exactly this aliasing. Same problem on the training side: a whole PPO batch sees one
phase, so the critic chases a moving target. `harl/utils/envs_tools.py::_snd_dephase`
now spreads rank *r* to phase *r/n_threads* for both train and eval envs, and
`progress.txt` gains `win_peak, win_trough, phase_coverage` columns. Set
`SMAC_SND_DEPHASE=0` to reproduce the old behaviour.

**2. `stagger_gap` is a biased statistic — do not read it as coordination.** `x2_i`
excludes agent *i*'s own fire, so ranking agents by `x2_i` is very nearly *reverse*-
ranking them by their own recent firing, which is autocorrelated with firing now. So
`fire_lo_load > fire_hi_load` comes out positive with **zero** coordination — measured
at **+0.16 right through the severity-0 warmup**, where no NS exists at all. Relatedly,
`x2_i` differs across agents only by that excluded term (≤ (1−ρ)/(N−1) = 0.021 on
3s5z), so it tells an agent *how hot the bus is*, never *who should hold fire* — the
team has to break the symmetry itself. Use **`hold_gap`** (hold-fire rate at the driver
peak minus at the trough) instead; it is 0 by construction at severity 0 and the runner
now checks that automatically.

## Configs

`env_args.snd_pact` / `snd_oracle` select the mode. `snd_pact_feedback` (default 1)
appends two raw leaky counters of the agent's own weapon alongside `x2_i` — `x3_jam`
(shots of mine that jammed lately) and `x3_try` (shots I attempted lately). Their ratio
is what makes the hidden driver `A(t)` **locally estimable at all**
(`E[jam | I fired] = ell_i`), so `(x2_i, x3_jam_i, x3_try_i)` is the decentralized
residual β-tracking runs on. Two counters rather than one derived rate, so that "I have
no evidence" (`x3_try ≈ 0`) is distinguishable from "I fired and nothing jammed" — a
single held rate froze stale the moment a unit stopped firing, and read 0 ("driver is
low") for the first ~10 steps of every ~50-step episode. Not privileged: it's the
unit's own weapon cutting out. Set to 0 to ablate.

Severity is the one knob `SEVERITY` at the top of `StarCraft2_Env.py` (or
`SMAC_SND_SEVERITY=…` per run); `SMAC_SND_KNEE` / `SMAC_SND_LMAX` shape the channel and
`SMAC_SND_GREEDY_LOAD` tells the banner what to compare against.

`data_chunk_length` is **40**, not the HARL default 10. The driver is essentially
constant within an episode (period 5000 steps vs episodes of ~50), so the phase is an
episode-level latent that has to be integrated out of ~40 firing-steps of Bernoulli jam
evidence; a 10-step BPTT window cannot carry that. It is set identically in every arm's
config, so it does not privilege PACT.

## Run (server)

```bash
# 0) arithmetic certificate (no SC2):
python -m harl.envs.smac.pact.test_pact                            # ends "ALL TESTS PASSED"

# 1) train B0 = the stationary baseline, with the NS OFF, one per obs shape.
#    *** GATE: both B0s must reach a comparable stationary win rate. ***
#    A blind arm that never learned to win stock 3s5z is not a baseline — its 0.0
#    under the NS proves nothing, and "PACT > blind" then just means "one warmup
#    succeeded and the other didn't". This actually happened: on the 20M-step run the
#    blind arm sat at win rate 0.000 for the entire 10M severity-0 warmup (ep_len ~135,
#    ep_reward ~18 — the farm-damage/timeout basin) while PACT, identical dynamics and
#    hyperparameters, hit 0.95 by 4M. Re-seed until the blind B0 learns, and report
#    seeds.
SMAC_SND_SEVERITY=0 python examples/train.py --load_config tuned_configs/smac/3s5z/happo/config.json --exp_name B0_blind --seed 1
SMAC_SND_SEVERITY=0 python examples/train.py --load_config tuned_configs/smac/3s5z/pact/config.json  --exp_name B0_pact  --seed 1

# 2) PHASE 1 — certify that a compensation solution EXISTS and find sigma*.
#    Do this BEFORE spending compute on the arms (pipeline Pitfall 5: never conclude
#    "unsolvable" from a failed TRAINING run — only from a failed scripted-privileged one).
python -m harl.envs.smac.pact.phase1 \
    --load_config results/.../B0_blind/config.json \
    --model_dir   results/.../B0_blind/models \
    --sigmas 0.3,0.5,0.75,1.0 --betas 0,0.5,1.0 --episodes 40
#    beta=1 IS the exact inverse, so it should reproduce B0. Read the RECOVERY column
#    (fraction of the blind->B0 gap the inverse closes) — for a permutation channel it
#    should be ~100%; anything less is the coarseness of an integer shift over K
#    attackable enemies. Pick the LARGEST sigma where blind is clearly hurt AND
#    recovery stays high. Use --bar to set a target other than 0.9*B0.
#    Defaults matter here: --metric ret (win rate has no partial band on a mirror map),
#    --law target (a regulator; `prop` oscillates), --betas up to 5 (the target law's
#    fixed point sits above f* for small beta). Every cell is printed, and each sigma
#    block reports the lowest load the shim ACHIEVED against the target f* — if it
#    never got there, that cell measured a broken controller, not the frontier.
#    -> set SEVERITY to sigma* - 0.05. If NO sigma clears the bar, first rule out those
#       two measurement traps, then redesign (raise KNEE toward the measured greedy
#       load / lower sigma) and re-run. Do not proceed to Phase 2.

# 3) the arms, warm-started from the matching B0 so the comparison is about the NS
#    and not about warmup luck (set model_dir in the config, and SMAC_SND_WARMUP=0):
SMAC_SND_WARMUP=0 python examples/train.py --load_config tuned_configs/smac/3s5z/happo/config.json     --exp_name blind     --seed 1
SMAC_SND_WARMUP=0 python examples/train.py --load_config tuned_configs/smac/3s5z/pact/config.json      --exp_name PACT      --seed 1
SMAC_SND_WARMUP=0 python examples/train.py --load_config tuned_configs/smac/3s5z/pact/config_ctde.json --exp_name PACT_CTDE --seed 1
SMAC_SND_WARMUP=0 python examples/train.py --load_config tuned_configs/smac/3s5z/happo/snd_oracle.json --exp_name oracle    --seed 1
```

Each log opens with `[NS] SMAC CWO severity=…` **and the headroom line — check it
first**. PACT writes a detailed **`pact_debug.csv`** (below). Expected ladder:
`blind < PACT ≤ PACT+CTDE ≲ oracle ≈ stationary`.

## Reading the results

**In `progress.txt`:** read `win_peak` and `win_trough`, not the aggregate. The
aggregate is only meaningful when `phase_coverage` is high (the logger prints
`ALIASED` when it is not). PACT's claim is specifically that `win_peak` rises while
`win_trough` stays at the stationary level.

**In `pact_debug.csv`** (per rollout):

- **Is the NS biting?** `A_mean` (driver), `ell_mean`/`ell_max` (drop prob),
  `drop_frac` (shots actually jammed), `x2_mean`, `x2_spread`, `x3_mean`/`x3try_mean`.
- **Is the team still fighting?** `fire_avail` and `ep_len_mean`. `fire_avail` falling
  while `ep_len_mean` pins at the limit means the team walked away — see the safe-harbour
  section above. Check this *before* concluding anything about coordination.
- **Is PACT coordinating?** `hold_frac` (of the units that *could* attack, the fraction
  that held fire — the real decision variable; raw `fire_frac` is diluted by units with
  nothing in range), `throughput` = `fire_frac·(1−drop_frac)` (the fraction of units
  actually *landing* a shot — a correctly coordinating team **raises** this while
  **lowering** `fire_frac`), and the headline **`hold_gap`** = `hold_frac(peak) −
  hold_frac(trough)`: does the team hold fire more when the bus is hot? `~0` = phase-
  blind. The runner prints `[PACT][CONTROL PASS/FAIL]` after the warmup — `hold_gap`
  must read ~0 at severity 0 or the statistic is biased.
- **`stagger_gap` / `fire_hi_load` / `fire_lo_load`** are kept only for continuity with
  old runs. **They are biased** (see above).

## Notes

- **Learnable, not a trap:** CWO reduces damage → reduces return AND win together, so
  there's no high-return-timeout hideout; blind lands at a harmed but nonzero plateau.
- Phase 1 for CWO is `phase1.py` (the compensation is a *scripted hold-fire law*, not a
  re-aim, but it is still a scripted privileged controller and still gates everything).
  Do not skip it — the whole point of the pipeline is to certify solvability before
  spending compute, and skipping it is what let a configuration with **negative**
  coordination headroom consume a 20M-step run.
