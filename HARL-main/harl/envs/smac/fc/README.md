# Formation Congestion + PACT, on SMAC

The non-stationarity of `NS_FORM_SPEC.md`, instantiated in the form most natural
to StarCraft II, and the method of `PACT_PIPELINE_SPEC.md` built to solve it.

---

## 1. The NS, in one paragraph

StarCraft II ground units are rigid collision bodies. A squad packed into a
contact line jams itself: a unit's move order is only partly realised because
allies are in the way. That is not an invented mechanic — it is why StarCraft
players talk about *surface area*, *concaves* and *unit clumping*, and it is the
oldest coordination problem in the game. Formation Congestion makes it explicit
and dial-able:

| object (NS_FORM_SPEC A.2) | on SMAC |
|---|---|
| medium | the ground unit *i* has to move through |
| loading `u_i = L_i / K_i` | congestion in *i*'s tightest sector |
| operator `W[i,j]` | Minkowski corridor obstruction from SC2's own unit radii, speeds and weapon-range bands. Zero-diagonal, **asymmetric**, **25× spread**, `std/mean = 0.957` (POWER: 1.35) |
| `Phi_j` | `alive_j * (1 + fired_j)` — an **uncancellable floor** plus the part that makes it vary; `std/mean = 0.155` measured (POWER: 0.28) |
| `L^fixed` | enemy bodies: nobody's to move, the irreducible class |
| driver `A(t)` | the enemy line's push / consolidate cycle, one cycle per engagement |
| `g(sigma, A)` | shrinks the squad's usable frontage — **capacity only, never added load** |
| harm channel | `delivered_i = base_i * cmd_i * (1 - Delta_i)`, `Delta_i = u_i (1 - g)` |
| inverse | `cmd = 1/(1 - Delta)` — **continuous and exact until the rail** |
| sensor | `1 - realized/commanded` — the unit's own odometry, reporting the past |
| loop? | **no** — compensating moves the unit, which changes neither liveness nor firing. `ns_phi_move > 0` creates the loop as a contrast ablation |

Two properties fall out **structurally**, not by verification afterwards:

* `sigma = 0` gives `g == 1` and `Delta == 0` **identically**, at every driver
  value — the stock task, byte for byte.
* at `N = 1` the peer sum is empty, so the cross-agent term is **exactly zero
  however small `g` becomes**. Category C, not category B in disguise.

### Why this channel and not target deflection

`NS_FORM_SPEC` A.7 measured the alternative on SMAC directly: on a permutation
(re-aim) channel, partial compensation lands the shot on a *different wrong
target* — `beta=0.5` scored **12.5** against **13.0** for no compensation at all,
while `beta=1.0` scored 17.5. Trust has to be a threshold there, and a ramping
confidence spends its whole warmup in the harmful regime. A throttled stride is
**multiplicative**, so trust *scales* the correction and a partly-right estimate
buys a partly-right outcome — exactly the property that makes the method work.

### The one hook in stock SMAC

`StarCraft2_Env.py` gained a single field, `self.move_stride` (a per-agent
multiplier on the move-order distance), and nothing else. It is exactly `1.0`
unless a wrapper drives it. Everything else in that file is stock.

### The reach clip — read this before questioning `sigma = 0`

Stock SMAC sends every move order to a point `move_amount = 2` away, but a
Stalker covers only 1.47 world units in one step and a Zealot 1.12. So the
order **never binds**, and scaling it down would do nothing until the scaled
distance fell below the reach — a dead zone of exactly the kind A.7 and PACT §6.3
warn about. The wrapper therefore clips the *base* order to the unit's own
one-step reach. That changes **no displacement at all** (the unit was never going
to get further), and the self-check *measures* it rather than asserting it:

```
[PASS] reach_clip=0 at sigma=0 is BYTE-identical to stock SMAC
[PASS] the reach clip changes NO displacement          10994.9750 vs 10994.9750
```

`ns_reach_clip: 0` restores the raw form as an ablation.

---

## 2. What PACT does

Everything lives in an environment wrapper. **The host RL is never modified** —
no loss terms, no critic changes, no extra action dimensions, no observation
augmentation — so every arm shares hyperparameters *and* observation/action
spaces, and an arm difference cannot be an algorithm difference.

```
        obs_i --> pi (host RL, UNCHANGED) --> a_i
                                              |
      peers' EXECUTED exertion (t-1) --> psi --+   exact, from the declared operator
      own deficit Delta_i(t) ----------> y ----+   odometry, reporting the past
                                              v
         RLS(psi, y) --> beta_hat            tracks c(t) = (1-g)/(K g)
         ell_hat = beta_hat[peer part] . psi  minus its standing level
         g_trust = max_trust if admissible else 0
         d_peer  = g_trust * ell_ctrl / (1 - Delta)
         d_ff    = ff_gain * u_hat (1 - g_dial) / (1 - Delta)
         cmd_i   = clip(1 + d_peer + d_ff, 0.25, 4.0)
```

**The floor property is byte-exact.** With the gates shut, `cmd = 1.0` and the
orders issued are *identical* to the blind arm's — asserted, not promised:

```
[PASS] both gates shut => orders BYTE-identical to blind   max|diff| = 0
```

---

## 3. Files

| file | what it is |
|---|---|
| `operator.py` | the declared `W`, `Phi`, the geometry kernels. SC2 unit data, never fitted |
| `driver.py` | `A(t)`, `g(sigma, A)`, and `assert_dial()` — B.1's four requirements, certified at construction on every process |
| `severity_env.py` | **the NS.** Every arm runs inside this |
| `pact_core.py` | the method's arithmetic, pure numpy: basis, RLS, the channel inverse, the gates |
| `pact_env.py` | the compensator, the outermost wrapper |
| `certificates.py` | Part-C ceiling + gates G0–G7. **Run before any method** |
| `selfcheck.py` | 55 arithmetic checks, no simulator needed |
| `calibrate.py` | the sweeps that freeze the task constants |
| `mock_smac.py` | a test double, so the wrapper stack can be probed with no StarCraft II |

---

## 4. Choosing the map — MEASURED, and it decided the project

The harm channel is a stride throttle, so it can only reach reward on a map where
movement decides something, and PACT's "declared operator, not a proxy" claim
only means anything on a map whose squad is heterogeneous enough to give the
operator structure. Both were measured before any training:

```
map         headroom  kite s=0  dial cost/2se   G4b     mv_frac  W_spread  W_asym
3s_vs_5z      3.06      11.1     +2.90/2.54 *   1.258    0.70     0.000     0.000  <- degenerate W
1c3s5z        1.77      19.5     +0.79/1.19     1.062    0.40     0.992     0.333  <- best operator
2s3z          0.84      13.1     -0.91/2.29     0.972    0.55     1.073     0.242
3s5z          0.83      10.9     +0.62/0.72     1.054    0.50     0.957     0.242
MMM2          0.78       3.8     +0.00/0.21     1.085    0.65     0.217     0.379
```

`headroom` = return of a movement-competent control over a focus-fire one at
sigma=0. Below ~1.15 the map cannot transmit a movement channel to reward at
all, and **five of eight maps in the first scan failed for that reason alone**,
including 3s5z, where the fight is a stand-and-shoot slugfest.

Two facts that cost a scan each and are worth stating plainly:

* **A control with no movement skill makes the gates unreadable.** The first
  reference focus-fired and walked at the enemy; against it a PERFECT oracle
  gained 1.03 on 3s5z and *lost* on 2c_vs_64zg, where the dial IMPROVED the
  reference by 1.73 (2se 0.86) — degrading a harmful behaviour is not a harm.
  D.1's warning, arriving from the other direction. `pick_controller` now
  measures both controls at sigma=0 and takes the stronger, printing both.
* **A Delta band is the wrong calibration target.** At the same delivery loss
  (deliv 0.853 vs 0.860) the return cost was **4% on 1c3s5z and 26% on
  3s_vs_5z** — a 6x difference in transmission that no Delta band can see. The
  k_scale sweep therefore runs the oracle on every row and calibrates on **G4b**.

## 5. The runbook

### Step 1 — the arithmetic (seconds, no StarCraft II)

```bash
python -m harl.envs.smac.fc.selfcheck
```

Ends with `ALL CHECKS PASSED`. It cannot prove the method works; it proves the
arithmetic is not the reason if it does not.

### Step 2 — the ceiling and the offline gates (seconds, no StarCraft II)

```bash
python -m harl.envs.smac.fc.certificates --offline --out fc_ceiling.json
```

### Step 3 — is movement worth return on this map?

```bash
python -m harl.envs.smac.fc.certificates --headroom 1c3s5z,3s_vs_5z --episodes 20 --out fc_headroom.json
```

If `headroom` is not comfortably above 1.15, stop: no stride channel can matter
there, whatever the dial is set to.

### Step 4 — calibrate the dial ON THAT MAP, against the control that won

```bash
python -m harl.envs.smac.fc.calibrate --sweep k_scale --map 1c3s5z --steps 8000 --out k.json
```

Each row runs the privileged oracle too, so the sweep reports **G4b** directly.
Take the *weakest* dial that clears 1.30 while the team still fights, write it
into `smac.yaml`, and **commit this output before any method run** — E.2 pitfall
10: *"retuning after seeing a method fail plants the problem; the history is the
evidence that you did not."*

### Step 5 — the full gates, with enough episodes to resolve G4b

```bash
python -m harl.envs.smac.fc.certificates --map 1c3s5z --episodes 60 --out fc_gates.json
```

20 episodes leaves G4b with a standard error of ~0.15, which straddles the 1.30
bar. Commit the output.

### Step 6 — only then train.

## 5. Commands

Everything below is run from the repo root. Any key in
`harl/configs/envs_cfgs/smac.yaml` is a command-line flag of the same name.

### The reference run: B0, stock SMAC

```bash
python examples/train.py --algo mappo --env smac --exp_name b0 --map_name 3s5z --fc 0 --seed 1
```

### The four arms of the experiment (PACT_PIPELINE_SPEC §10)

```bash
python examples/train.py --algo happo_fc --env smac --exp_name blind --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name ff_only --map_name 3s5z --ns_severity 1.0 --pact_max_trust 0 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name pact_full --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name peer_only --map_name 3s5z --ns_severity 1.0 --pact_ff_gain 0 --seed 1
```

`ff_only` is the arm that makes the coordination claim honest: it has the same
actuator and the same domain model as PACT, and only the peer channels are
missing. `peer_only` isolates the coordination term. **Report the `ff_share`
column from `fc_debug.csv` in every table** — POWER's split was 79% local / 21%
coordination and the self-check measures 83% / 17% here.

### The same four on the MAPPO host

```bash
python examples/train.py --algo mappo_fc  --env smac --exp_name blind_mappo --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo pact_mappo --env smac --exp_name pact_mappo --map_name 3s5z --ns_severity 1.0 --seed 1
```

Compare **host to host**: `happo_fc` vs `pact`, `mappo_fc` vs `pact_mappo`.
Comparing `pact` (HAPPO) against `mappo` is a confound.

### Every other HARL algorithm, on the identical NS

The dial is task physics and lives in the env config, so any algorithm in the
repo runs inside it unchanged:

```bash
python examples/train.py --algo mappo --env smac --exp_name mappo_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo happo --env smac --exp_name happo_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo hatrpo --env smac --exp_name hatrpo_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo haa2c --env smac --exp_name haa2c_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo corep --env smac --exp_name corep_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

```bash
python examples/train.py --algo hasac --env smac --exp_name hasac_ns --map_name 3s5z --ns_severity 1.0 --seed 1
```

Use `mappo_fc` / `happo_fc` rather than `mappo` / `happo` when you want the
`fc_debug.csv` telemetry on the baseline — they train bit-identically and only
add the log file.

### The severity sweep

```bash
for S in 0.0 0.5 1.0 1.5 2.0; do python examples/train.py --algo pact --env smac --exp_name sweep_sigma_$S --map_name 3s5z --ns_severity $S --seed 1; done
```

`sigma > 1` is a **beyond-physical stress test** and must be labelled as such in
every table it appears in.

### The `max_trust` sweep — Phase-1 calibration (§8.4)

```bash
for G in 0.0 0.25 0.5 0.75 1.0; do python examples/train.py --algo pact --env smac --exp_name trust_$G --map_name 3s5z --ns_severity 1.0 --pact_max_trust $G --seed 1; done
```

**Calibrate on seed 1 and validate on held-out seeds.** Calibrating and
reporting on the same seed is fitting to the test set. An inverted U here is T4
appearing in a real environment and is *evidence*, not tuning — report the whole
sweep, not the peak.

### The ablations (§10)

```bash
python examples/train.py --algo pact --env smac --exp_name abl_r2 --map_name 3s5z --pact_r 2 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_mu_power --map_name 3s5z --pact_mu 0.9995 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_no_windup_bound --map_name 3s5z --pact_p_max_mult 1e18 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_level_ema --map_name 3s5z --pact_peer_mode ema --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_directional --map_name 3s5z --pact_directional 1 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_loop --map_name 3s5z --ns_phi_move 0.5 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_no_reach_clip --map_name 3s5z --ns_reach_clip 0 --seed 1
```

```bash
python examples/train.py --algo pact --env smac --exp_name abl_mean_preserving --map_name 3s5z --ns_mean_preserving 1 --seed 1
```

`abl_loop` is the T4 contrast: with `ns_phi_move > 0` the compensator's own
motion feeds the medium it compensates against, so the environment becomes
loop-coupled and an interior optimum in `max_trust` should appear. **With the
default `ns_phi_move: 0` it should not** — that is the prediction the spec's
E-table makes for SMAC, and confirming a prediction of *no effect* is worth more
than a fifth environment where everything works.

### The trivial ablation — run it early (§10, last row)

```bash
python examples/train.py --algo happo_fc --env smac --exp_name abl_recurrent --map_name 3s5z --ns_severity 1.0 --use_recurrent_policy True --seed 1
```

If a recurrent blind host matches PACT, the estimator is decoration. Find that
out before spending a campaign on it.

### The placebo run — the strongest single defence (B.4)

```bash
python -m harl.envs.smac.fc.certificates --map 3s5z --episodes 8
```

The `G6` line freezes the driver inside its inert regime and checks that every
severity row is **byte-identical**. A reviewer alleging a rigged knob then has to
explain why the rig switches itself off during the consolidate phase.

### Seeds

Everything above is one seed. Run at least five per arm:

```bash
for SEED in 1 2 3 4 5; do python examples/train.py --algo pact --env smac --exp_name pact_full --map_name 3s5z --ns_severity 1.0 --seed $SEED; done
```

---

## 6. Reading `fc_debug.csv`

Written to the run directory by `pact`, `pact_mappo`, `happo_fc` and `mappo_fc`
— **including the blind arms**, because a silently inert NS is invisible in
exactly the arm you most need to trust.

**Read `applied_trust` and `delta_nonzero_frac` before any other number.**

| column | question | healthy |
|---|---|---|
| `applied_trust` | **was the method ON AT ALL?** | not ~0 |
| `delta_nonzero_frac` | is it acting? | POWER measured 0.42–0.69 |
| `delta_clip_frac` | rail-pinned = constant bias, not compensation | near 0 |
| `ff_abs` / `peer_abs` / `ff_share` | local vs coordination split | report all three |
| `fit_gain_now` | do the peer channels beat the null **now**? | positive, rising |
| `cond_psi` | can `beta` be *decomposed*, not just predicted? | finite, low |
| `trP` / `clamp_frac` | covariance windup | flat / rising late is fine |
| `own_gain_se` | is the own column determined? | `|t| > 3` |
| `state` | INERT / ASLEEP / ALIVE | ALIVE |
| `sigma` / `g` / `dial_ratio` | is the severity live? | `dial_ratio` in (0, 1) |
| `delta_mean` / `u_mean` | is the NS biting? | not ~0 |
| `phi_var` | **is there an escape hatch?** | flat over training, > 0.05 |
| `move_frac` | the channel only bites on move orders | not ~0 |
| `odom_err` | the sensor really is physical | ~0 away from the rail; it rises where `sat_frac` does, because a railed command stops responding |
| `sat_frac` | correction lost to the reach rail | low |

The runner prints the first few to the console every log interval and raises a
named warning — `[PACT][INERT]`, `[PACT][ASLEEP]`, `[PACT][RAIL]`,
`[PACT][ESCAPE]`, `[PACT][COND]` — for each failure the spec paid for.

The file is **never appended across a schema change**: a header mismatch rolls
the old file aside instead.

---

## 7. Honest limits — state all of these

1. **The compensator is classical.** Project onto a known basis, run RLS, invert.
   An adaptive-control reviewer will say so and be right about the *mechanism*.
   The contributions are the reduction that makes it applicable at N agents, the
   commons theory of what happens when everyone compensates, and the
   recoverable-fraction ceiling. Lead with those.
2. **The feedforward is local.** The self-check measures the split at **83%
   local / 17% coordination** on synthetic geometry (POWER: 79/21). If the
   coordination term does not stand on its own in the `pact_ff_gain 0` arm, the
   coordination claim is thin *in this environment* and you must say so.
3. **The operator is declared.** Not privileged — every SC2 player has the unit
   stats — but the information-matched baseline (`ff_only`) is mandatory, or the
   gap is data rather than mechanism.
4. **`Delta_own` is structurally zero here.** `W` is zero-diagonal *and* a unit's
   own body does not obstruct its own corridor, so unlike POWER (76.9% own) the
   whole controllable excess is a peer term. That flatters the Part-C *source*
   split, so `certificates.py` also reports the **temporal** split — how much of
   `Delta(t+1)` the stale local sensor plus the known driver model already
   explains. That residual, not the source share, is what peer anticipation can
   actually buy, and it is what `fit_gain` measures online.
5. **The harm only reaches a unit that is moving, and only matters where movement
   decides the fight.** Measured: five of eight SMAC maps have a movement
   headroom at or below 1.05, i.e. a movement-competent control buys no return
   over a focus-fire one, and on those maps a perfect oracle on the true deficit
   gains nothing. That is a fact about SMAC, not about the method, and it is why
   the map is chosen on the gates (section 4) rather than by preference.
   `move_frac` and `headroom` are first-class diagnostics.
6. **First-order geometry.** The obstruction model is a Minkowski corridor with a
   proximity and a direction kernel; it ignores pathing, terrain and SC2's own
   collision resolution. Compensation acts on the commanded distance, so the
   inverse is exact in the channel we define and approximate in the engine.
7. **Trust is not learned**, only well-initialised and gated.
8. **`beta` may be predictable without being decomposable.** `cond_psi` is
   reported; do not claim to identify `beta` without reading it.
9. **`sigma = 1` has no textbook anchor.** SMAC has no derating datasheet, so
   `ns_depth` is anchored to a *measured* property of the stock game instead and
   `certificates.py` reports it. Anything above `sigma = 1` is labelled a
   beyond-physical stress test.
10. **No warmup curriculum by default.** At `sigma = 0` the compensator is
    provably inert, so arm comparisons inside a warmup window measure basin luck.
    `ns_warmup` / `ns_ramp` exist; using them means never comparing arms inside
    the window.
