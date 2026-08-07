# Concussion-Coupled Wake Displacement (CWD)
### A category-C, dynamics-only, oracle-separable non-stationarity for SMACv2

---

## 0. One-paragraph summary

In a heavy firefight, sustained weapons discharge and impacts throw off **concussive
overpressure and debris**. A unit maneuvering near where its squadmates are trading
fire gets physically **buffeted** — shoved *away* from the locus of the firefight
(overpressure radiates outward; closer blasts hit harder). An exogenous
**bombardment-tempo** driver `A(t)` (munitions expenditure ramping over the
engagement, dropping abruptly at each resupply/lull) gates how hard the buffeting is.
The more the **other** units are *firing* near you, the stronger the shove. Each unit
therefore carries a hidden, slowly-varying **displacement vector** `d_i`, fed only by
the *other* units' combat. The shove is **added** to the unit's commanded move target
inside the environment; the reward function is left exactly as in stock SMACv2. A
controller that *knows* `d_i` can brace and counter-step; a blind one is knocked off
its micro, scattered out of position, and cut down. The entire effect is governed by
**one severity dial**.

This is the SMACv2 sibling of PCR (Ant) and SND (SMAC): the **same family idea**, a
**genuinely different instance** — the channel is peers' *combat*, not their motion;
the geometry is a range-weighted *radial* push from relative positions; the
accumulator is *impulsive*; and the driver is an *asymmetric* ramp.

---

## 1. Design intent and the seven constraints

| # | Constraint | How CWD satisfies it |
|---|---|---|
| 1 | **Natural / reviewer-plausible** | Concussive overpressure, recoil wash and debris from a nearby sustained firefight physically buffeting an adjacent trooper is textbook battlefield physics; staying on-course through it is a real control problem. |
| 2 | **Strictly category C** | `A(t)` only ever *multiplies* a sum over the **other**, *firing* units. Empty at N=1 ⇒ exact stock SMACv2; persists with frozen (fixed-but-acting) partners; a unit cannot shove itself. (§4) |
| 3 | **Dynamics-only — no reward shaping** | The reward is byte-for-byte the original SMACv2 reward. The effect is entirely in the *delivered move target*. A shoved, mis-positioned unit earns less **only because it physically achieved less**. (§3.6) |
| 4 | **Guaranteed performance collapse** | An additive displacement drags units off their optimised micro (formation, focus fire, spacing); they take unanswered damage and die → the unchanged reward collapses. (§6) |
| 5 | **One unifying idea, fresh instance** | Same template as PCR/SND, but *new* fed quantity (peers' firing), *new* geometry (range-weighted radial push), *new* accumulator (impulsive), *new* driver (asymmetric bombardment ramp). |
| 6 | **Minimal knobs** | Exactly **one** dial: `SMACV2_CWD_SEVERITY`. Everything else (`_CWD_P`, `_CWD_RHO`, `_CWD_R`, `_CWD_DCAP`) is a fixed internal constant. (§8) |
| 7 | **Information-recoverable (oracle-separable)** | The harm is a *mis-direction* (additive translation of the move target), not a loss of capability: knowing `d_i` lets a controller aim out of it. `SMACV2_CWD_ORACLE=1` makes this runnable. (§5) |

Target: **category C** — an exogenous driver whose effect on a unit is channelled
**entirely through the other units**; not a co-learning artefact (survives frozen
partners), not a single-agent disturbance (vanishes at N=1).

---

## 2. The physical story (why a domain expert accepts it)

A firefight is a violent mechanical environment. Sustained automatic fire, weapon
recoil, and rounds impacting nearby cover generate **overpressure waves** and kick up
dust and fragments. A soldier or vehicle **maneuvering** near that exchange is
physically jostled — pushed and disoriented, most strongly *away* from the seat of the
blast, and more so the closer and more intense the exchange.

- The buffeting a unit feels comes from **its squadmates' fighting**, not its own: it
  is the *others'* massed fire nearby that shoves it. A lone unit far from any exchange
  feels nothing.
- The intensity is gated by an **exogenous** factor — the **bombardment tempo** of the
  wider engagement (how much ordnance is being expended right now), which escalates
  through an assault and drops abruptly at resupply or a lull. This is agent-
  independent and drifts slowly over the mission.
- The effect shows up as an **unmodelled displacement** on the moves a unit commands:
  it orders "reposition to here" and, shoved mid-stride, ends up **there**.

Holding a commanded path through concussive disturbance you cannot directly sense is a
real control problem, which is what makes CWD reviewer-plausible rather than an
arbitrary gremlin.

---

## 3. Formal mechanism

### 3.1 Notation
- Units `i ∈ {0,…,N−1}`; discrete actions `{no-op, stop, move N/S/E/W, attack/heal(≥6)}`.
- A move action compiles to a continuous world-space target `p_i ± move_amount·ê`
  before reaching the engine — the lever CWD perturbs. Action *selection* stays
  discrete and unblinded.
- `p_i = (x_i, y_i)` is unit `i`'s world position; `T` is a **global clock**
  (`self._cwd_clock`) incrementing once per env step and **persisting across
  episodes**.

### 3.2 The exogenous driver `A(t)` — the bombardment tempo
```
phase φ = (T mod _P) / _P                          ∈ [0,1)
A(T)    = (φ/c)²                if φ < c            (accelerating, convex escalation)
        = 1 − (φ−c)/(1−c)       if φ ≥ c            (abrupt linear collapse)     ∈ [0,1]
```
with `c = 0.85`. **Continuous** in value (no value shocks), **asymmetric** — a slow
convex build then an abrupt resupply drop — and deliberately a different closed form
from PCR's smoothstep and SND's raised cosine. Period `_P = 3000` steps (also
distinct from SND's `5000`), so many bombardment cycles elapse per run and the metric
repeatedly collapses-and-recovers, robust to run length. Persisting `T` (never reset
on episode) keeps every parallel env phase-synchronised ⇒ a clean aggregate
oscillation. `A(t)` is **hidden** from the agents.

### 3.3 The category-C channel — the other units' fire
A unit `j` is a **concussion source** this step iff it is alive and its action is an
attack/heal (`action_j ≥ n_actions_no_attack`). The shove on unit `i` is the
range-weighted radial push away from the firing others:
```
for each firing peer j ≠ i:
    u_ij = (p_i − p_j) / (‖p_i − p_j‖ + ε)     unit vector pointing AWAY from j
    w_ij = 1 / (1 + ‖p_i − p_j‖ / _R)          closer firefights buffet harder (_R = falloff)
S_i = Σ_{j ≠ i, firing} w_ij · u_ij
```
The sum excludes `i` and requires *firing* peers — a unit's own action never feeds its
own shove, and a quiet neighbourhood produces none. Note the geometry is genuinely
different from SND's position-independent move-flow: here direction and magnitude both
come from the **relative positions** of the *shooting* teammates.

### 3.4 The hidden liability `d_i` — an impulsive accumulator
```
d_i ← _RHO · d_i + (1 − _RHO) · A(T) · SEVERITY · S_i         (2-vector, per unit)
d_i ← clip(d_i, −_DCAP, +_DCAP)                               (per axis)
```
- `_RHO = 0.5` is the **concussive memory**: buffeting is *impulsive*, not slowly-
  building, so the time constant is short (`≈ 1/(1−_RHO) ≈ 2` steps) — deliberately
  snappier than SND's `_RHO = 0.85`. This makes `d` a fast-moving target a reactive
  policy cannot track.
- Steady-state `d* = A(T)·SEVERITY·S_i`; the leak sets speed, not steady-state
  magnitude.
- `_DCAP = 2.0` (= one `move_amount`) caps the per-step displacement (no teleport;
  keeps the oracle inside the discrete-action cancellation budget).
- `d_i` is **reset to zero at the start of every episode**; the clock `T` is **not**.

### 3.5 The throttle — an additive disturbance in the delivered target
For a move action the delivered world target is
```
target_delivered = (p_i ± move_amount·ê) + d_i
```
the **only** change to the dynamics — an **additive offset**, not a multiplicative
scaling (§5). Stop/attack carry no world target, so a unit that plants and fires is
undrifted; only maneuvering through the concussion is corrupted.

### 3.6 The reward — untouched
The stock SMACv2 reward (damage / deaths / win) reads the **realised** unit healths
and deaths produced by the *delivered* moves. A unit shoved out of position takes
unanswered fire and dies, so the team earns less — **only because it physically
achieved less**. No term proportional to `d`, `A`, or any liability appears in the
reward.

### 3.7 Signal-flow
```
   global clock T ──► A(T)  (bombardment tempo, exogenous, hidden, persists across episodes)
                        │
                        ▼
   firing peers  Σ_{j≠i} w_ij·u_ij ──►  × A(T) × SEVERITY
                 (range-weighted radial, the ONLY channel)   │
                                                             ▼
                        d_i ← ρ·d_i + (1−ρ)·(…), clipped   (hidden impulsive shove)
                                                             │
   unit i's move (p_i ± move·ê) ────────────────────────────►(+)──► delivered target ──► engine
                                                             │
                          reward = ORIGINAL SMACv2 (reads realised healths/deaths)
```

---

## 4. Why this is strictly category C

**(i) The driver only multiplies the cross-agent sum.** `A(T)` is solely a factor on
`S_i = Σ_{j≠i, firing} w_ij·u_ij`; never additive on its own. Empty sum ⇒ no channel.

**(ii) Litmus "not B" — vanishes at N=1.** One allied unit ⇒ `{j≠i}` empty ⇒ `S_i=0`
⇒ `d_i ← ρ·d_i → 0`. Exact stock SMACv2. (SMACv2 ships no 1-ally scenario, so this is
the *structural* certificate; even on real maps, a unit whose teammates are all dead or
not firing momentarily feels nothing — the channel is genuinely off.)

**(iii) Litmus "not A" — survives frozen partners.** `A(T)` depends on the global clock
only; frozen-but-acting teammates still fire, so `S_i ≠ 0` and the difficulty drifts. A
task property, not co-learning.

**(iv) Individually exogenous, collectively endogenous.** The sum excludes `i`, so no
unit controls its own shove, yet the team creates it — the harder everyone fights, the
more everyone is buffeted. A **tragedy of the commons** whose optimal sharing keeps
moving as the bombardment drifts.

---

## 5. Why this NS is *solvable* (oracle-separable, constraint 7)

### 5.1 Full-rank, invertible harm
`target_delivered = target_commanded + d_i` is a **translation** — full-rank, exactly
invertible. The unit keeps all of its movement authority; it is merely aimed wrong. A
multiplicative speed loss would be rank-deficient and unrecoverable; CWD avoids that.

### 5.2 The solution: feed-forward rejection
Knowing `d_i`, a controller biases its choice so the *delivered* position matches
intent — pick the cardinal move that best cancels the shove, step against the
accumulated `d`, and plant-and-fire rather than maneuvering into a buffet. The solving
idea:

> **Estimate the hidden bombardment driver and the cross-unit concussion coupling, and
> feed-forward compensate the resulting displacement.**

A driver-conditioned policy / disturbance observer / centralized critic exploiting the
cross-agent structure *can* recover `d_i`; a blind independent learner cannot (it sees
neither which peers are firing where, nor the bombardment phase) and is always a step
behind a fast, drifting shove. **That gap is the experimental result.**

### 5.3 The oracle ablation (existence proof)
`SMACV2_CWD_ORACLE=1` appends the exact `d_i` (2 numbers) to that unit's observation
and the full stacked `d` (`2N`) to the centralized state. Oracle should recover ≈
stock win-rate at any calibrated severity while blind collapses: **oracle ≈ baseline ≫
blind**.

### 5.4 The recoverability regime (a real caveat)
Discrete actions cannot cancel a continuous shove exactly; the oracle chooses the
least-bad cardinal move and times its steps. Within the `_DCAP` budget this coarse,
multi-step compensation is a large learnable advantage; `_DCAP` caps the harm so the
re-aim never saturates. (Same spirit as PCR's actuator-saturation boundary.)

---

## 6. Why the blind policy reliably collapses (constraint 4)

- **Not a regulariser.** An additive displacement injects motion the unit did not
  choose, in a direction set by the *others'* fire, pushing it **off** its micro.
- **It breaks SMACv2 micro.** Holding a ranged line, concentrating fire, and spacing
  are positional; a shoved unit scatters out of arc, out of concentration, and into
  fire.
- **It is a fast-moving target.** With `_RHO = 0.5` the shove reacts within ~2 steps to
  the shifting firefight and drifts with the bombardment — faster than a blind reactive
  policy can observe-and-respond, and impossible to pre-empt without the hidden driver.
- **The escape is costly.** The only blind way to shrink `d` is for the team to stop
  shooting near each other (smaller `S`) — i.e. disperse and hold fire, sacrificing the
  massed fire that wins fights. Both horns (scattered-and-buffeted vs. dispersed-and-
  toothless) lose to the oracle, which fights and maneuvers freely because it cancels
  the shove.

Over training the **periodic, asymmetric** bombardment yields a **collapse-and-recover**
win-rate: high near troughs (`A≈0`, stock env), dropping through each escalation
(`A→1`), envelope suppressed below baseline; the oracle stays high throughout.

---

## 7. Implementation and wiring into HARL

The NS lives **entirely in `harl/envs/smacv2/smacv2_env.py`** (the HARL SMACv2
wrapper); the installed `smacv2` package and the rest of HARL are untouched.

- **Action path (monkeypatch, version-robust).** In `seed()`/`reset()` the wrapper
  patches the underlying `StarCraft2Env.get_agent_action` by *wrapping* the original:
  it calls the original, and only for move commands (guarded by
  `cmd.HasField("target_world_space_pos")`) adds `self._cwd_d[a_id]` to the world
  target. It never reimplements the action pipeline, so it is robust across smacv2
  versions. Action availability is unchanged — the corruption is purely in the
  delivered dynamics.
- **Timing (exact oracle cancellation).** The drift applied on step `t` is the value
  set by `_cwd_advance` at the end of step `t−1` (using step `t−1`'s actions and the
  post-step positions) and exposed in the obs returned there — what the oracle saw
  equals what is applied.
- **Observation/state path.** With `SMACV2_CWD_ORACLE=1`, `_cwd_obs_state()` appends
  `d_i` (2) to each unit's obs and the flat `d` (`2N`) to the (repeated) state; the
  observation/share spaces are grown to match in `seed()`. HARL's `MLPBase` consumes
  the flat vector, so no structural bookkeeping is needed.
- **Episode vs campaign state.** `self._cwd_clock` is **not** reset on `reset()`;
  `self._cwd_d` **is** reset to zero each episode.
- **`info` is safe.** The `cwd_payload` / `cwd_load` / `cwd_loadmax` keys are ignored by
  training and available to TensorBoard.

---

## 8. The single knob and the fixed internals

```
SMACV2_CWD_SEVERITY  (the ONE dial)  shove gain — how hard the blind problem is. (default 0.5)
SMACV2_CWD_ORACLE    0/1             expose d in obs+state (recoverability proof). (default 0)
_CWD_P    = 3000    bombardment period (env-steps). Short ⇒ many cycles ⇒ robust to run length.
_CWD_RHO  = 0.5     concussive memory (impulsive; time const ≈ 2 steps).
_CWD_R    = 5.0     blast falloff radius (world units).
_CWD_DCAP = 2.0     per-axis shove cap (= one move step): no teleport; keeps the oracle recoverable.
```

### 8.1 Calibration procedure
`|d*| ≈ A·SEVERITY·|S|`, and `|S|` depends on how much and how closely the learned
policy fights. Calibrate to the real operating scale:

1. Launch a run; log `cwd_load` (mean `|d|`) and `cwd_loadmax`.
2. Read off the **peak** `cwd_load`.
3. Adjust `SEVERITY` so **peak `cwd_load` ≈ 0.5–1.0** world units (fraction of
   `move_amount = 2`): too small ⇒ shrugged off; too large ⇒ pins at `_DCAP`.
4. Start near `SEVERITY = 0.5`.
5. **Verify the sign:** confirm blind win-rate *drops* as `cwd_payload` rises. Because
   the shove is *outward*, verify it does not accidentally *help* by spreading units off
   an AoE; if the oracle ever beats the stationary baseline, raise `SEVERITY` (and, if
   dispersion helps on that map, prefer a map where concentration is decisive) and
   re-check.

---

## 9. How to run and what to expect

Deploy: this file already lives in the repo (the installed `smacv2` package needs no
edits). Set the dial / oracle via environment variables (read once at import).

```bash
# Stationary baseline (SEVERITY 0 ⇒ exact stock SMACv2):
SMACV2_CWD_SEVERITY=0 python examples/train.py --algo mappo --env smacv2 \
    --exp_name cwd_baseline --map_name protoss_5_vs_5

# Blind run (the real, hard task):
SMACV2_CWD_SEVERITY=0.5 python examples/train.py --algo mappo --env smacv2 \
    --exp_name cwd_blind --map_name protoss_5_vs_5

# Oracle run (the recoverability proof):
SMACV2_CWD_SEVERITY=0.5 SMACV2_CWD_ORACLE=1 python examples/train.py --algo mappo --env smacv2 \
    --exp_name cwd_oracle --map_name protoss_5_vs_5
```

- Any HARL algorithm works (swap `--algo`: `happo`, `hatrpo`, `mappo`, `hasac`, …).
- **Maps:** use a *solved-ish* 5v5 (HARL ships tuned configs for `protoss_5_vs_5`,
  `terran_5_vs_5`, `zerg_5_vs_5`), where a high baseline makes the fall clear.
  `protoss_5_vs_5` is the primary; `terran_5_vs_5` / `zerg_5_vs_5` are alternates.
- **PowerShell:** `$env:SMACV2_CWD_SEVERITY=0.5; $env:SMACV2_CWD_ORACLE=1; python examples/train.py ...`

**Expected plots:** blind win-rate oscillates with the bombardment cycle, envelope
suppressed below baseline; oracle stays high and flat near the stationary baseline;
`cwd_load` tracks `cwd_payload` and its peak is the calibration readout.

---

## 10. Limitations / things to verify on the run machine

- **Confirm harm, not help (§8.1 step 5)** — the mandatory check for any additive push,
  and doubly so for an *outward* shove that could aid dispersion on AoE-heavy maps.
- **Discrete-action recoverability (§5.4)** — keep peak `cwd_load` in 0.5–1.0; `_DCAP`
  guards the top end.
- **`smacv2` API assumptions** — the wrapper reads `env.get_unit_by_id`, `unit.pos`,
  `unit.health`, and `n_actions_no_attack` from the underlying `StarCraft2Env` (all
  standard SMAC API; `n_actions_no_attack` defaults to 6 if absent), and mutates the
  raw command's `target_world_space_pos`. Verify these hold for the installed smacv2
  version (they do for the standard release).
- **Observation normalisation** — appended `d` channels are normalised with the rest;
  another reason not to over-crank `SEVERITY`.

---

## 11. Symbol and hyperparameter reference

| Symbol / name | Meaning | Value |
|---|---|---|
| `SMACV2_CWD_SEVERITY` | shove gain (**the one dial**) | calibrate ≈ 0.4–0.8 (peak `cwd_load` 0.5–1.0) |
| `A(t)` / `cwd_payload` | exogenous bombardment driver | `∈ [0,1]`, asymmetric ramp |
| `S_i` | range-weighted radial push from firing peers | the category-C channel |
| `d_i` / `self._cwd_d` | hidden per-unit shove | `∈ ℝ²`, impulsive, capped; reset each episode |
| `T` / `self._cwd_clock` | global bombardment clock | persists across episodes |
| `_CWD_P` | bombardment period | `3000` steps |
| `_CWD_RHO` | concussive memory | `0.5` (τ ≈ 2 steps) |
| `_CWD_R` | blast falloff radius | `5.0` world units |
| `_CWD_DCAP` | per-axis shove cap | `2.0` (= one move step) |
| `cwd_load` | mean `|d|` (calibration target) | aim peak 0.5–1.0 |
| `cwd_loadmax` | max `|d|` (saturation watch) | keep ≲ `_DCAP` |

---

## 12. How CWD differs from SND (same idea, fresh instance)

| Axis | SND (SMAC) | CWD (SMACv2) |
|---|---|---|
| Physical medium | shared nav relay (spoofing) | concussive overpressure (firefight) |
| Cross-agent channel | peers' **movement** flow | peers' **firing** (combat) |
| Coupling geometry | position-independent net move-flow | range-weighted **radial** push from relative positions |
| Accumulator memory | `ρ = 0.85` (slow, τ≈7) | `ρ = 0.5` (impulsive, τ≈2) |
| Driver shape | raised cosine (symmetric) | convex ramp + abrupt drop (asymmetric) |
| Driver period | 5000 | 3000 |

Both share the one family template — a hidden leaky liability fed by the *others'*
exertion, gated by an exogenous drift, additively mis-directing the agent's own move —
so a single idea (estimate the driver + cross-agent coupling, feed-forward cancel)
solves the whole family, while the encoding functions differ enough that neither is a
reskin of the other.

---

*End of design document.*
