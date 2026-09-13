# SMAC-NS — Coupling Under Drift, instantiated in SMAC

The cell this instance fills, stated first because the spec says conflating the
cells is the one thing that is not publishable:

|  | invertible channel | **no inverse** |
|---|---|---|
| **(C) interaction-mediated** | `simple_ns` (VMAS) | **SMAC-NS ← this instance**, `road_ns` (URB) |
| (B) exogenous | the control | — |

> **(C, no inverse) — identification and steering only.** The margin is capped by
> the coordination gap. SMAC's action space is `no-op / stop / move×4 / attack[e]`
> and there is no continuous magnitude to invert, so the channel is a shift over
> the attack options — `pact1_core.steer_logits`, verbatim, plus a declared floor
> (`channel.py`).

## The story, and every clause is a requirement

> *A squad shares one enemy line. Damage poured into an enemy beyond what it has
> left is overkill — wasted. Under the enemy's guard phase each target absorbs
> usefully less, so the same pile-on wastes more. **A lone unit cannot overkill**:
> it fires, the target dies, it retargets. Nothing is wasted at any guard
> intensity.*

Every player has said "stop overkilling that zealot". It is also what a 3s5z
policy at a 90 % win rate has learned to live with — focus fire — which is why the
dial has something to take away from the existing algorithms.

| object | SMAC | check |
|---|---|---|
| medium | the enemy line's absorbable damage | — |
| element `e`, `cap_e` | an enemy unit; its **current** life + shields, read off the game each step | — |
| loading | `u_i = Σ_{j≠i} dmg_j→target(i) / cap`, **max** over the agent's own elements | NS-1.1 |
| operator `W[i,j]` | incidence ÷ capacity × published per-unit damage. **zero-diag, asym 0.545, spread 0.470, ratio 3.6×** | NS-1.2 |
| `r` classes | enemy unit **types** (2 on 3s5z). Independent of N and of the element count | P-1.1 |
| driver `A(t)` | the guard cycle; shrinks absorption, never adds to the loss; **period 15000 steps = 100 episode limits, exactly 0 for half of it** | NS-1.3, NS-2.5 |
| anchor | one SC2 **armour point** against a Stalker's 10-damage weapon = exactly **10 %** | NS-2.4 |
| harm | `excess = f(u/g)/f(u) − 1 = (1/g − 1)·s(u)`, `f(u)=1+αu`, `s(u)=αu/(1+αu)` | NS-1.4 |
| delivered as | the wasted fraction of the damage that **landed**, restored onto the post-tick bar | NS-1.4, NS-3.2 |
| sensor | `realized/nominal − 1` → `relative_excess`, verbatim | P-2.1 |
| channel | the peer-induced overkill share `s(u)` on the option, so `y = β·ψ` is **exact** | P-1.2, P-3.x |

Two identities hold **exactly**, and the suite asserts them rather than hoping:

```
sigma = 0   ->  excess == 0        the stock task, byte for byte
u_i   = 0   ->  harm  == 1.000000  at g = 1e-3, i.e. the harshest dial
```

The second is category C obtained *structurally*, from the `j != i`, not from a
number that happens to be small.

## What the first 3.2M-step run taught — read before changing anything

The first `pact` run (`pact_debug.csv`, 801 rollouts) looked like "weak harm, noisy
steering". It was four implementation faults, each now fixed and each guarded by a
check that reproduces it:

| fault | what it did | fix | guarded by |
|---|---|---|---|
| **restore wrote pre-tick values** | the hook runs after `observe()` and before `update_units()`, so `self.enemies` is the step *before*. Setting shields to *(pre-tick shields + waste)* undid the whole step's shield damage on every shared target in the guard phase. Commands went out on 24 % of all steps while logging 0.085 hp each — the real harm was invisible | restore `w_e × landed_e` onto the **post-tick** bar, written into the snapshot and the engine | `restore_adds_to_the_post_tick_bar_not_the_pre_tick_one`, `engine_write_equals_the_snapshot_the_reward_reads` |
| **healing could be paid as damage** | stock `reward_battle` takes `abs()` of the enemy's net hp change; a restore larger than what landed turns into reward | restore ≤ damage that landed, same step | `net_hit_point_change_of_every_enemy_stays_non_negative` |
| **the shift z-scored moves against attacks** | stop/move carry cost 0 (no prediction); with attack costs ~0.003 every move gained ~0.8 logits and every attack lost ~1.0 — a **1.7-logit anti-attack prior** on a focus-fire map | `pact_steer_from = 6`, required | `unmasked_zscore_pushes_attacks_below_moves`, `missing_pact_steer_from_refuses_to_build` |
| **driver period 150 vs estimator memory ~2–3k steps** | β̂ averaged the cycle, never read zero in the placebo, and the scale-free z-score steered agents off focus fire where overkill cost nothing extra | period = 100 episode limits (URB: 100 days; GRF: 12000); floor on the predicted spread | `default_period_is_100_episode_limits`, `first_run_period_is_flagged_not_trackable`, `below_the_floor_the_host_logits_are_bit_for_bit` |

Two design corrections came out of the same review. **Capacity is the remaining
bar**, not the published maximum: against a full bar one step of peer fire is a
sliver (the first run read `u = 0.027`), while overkill is by definition about what
is *left*. And because remaining capacity makes `u` span 0…5, a regressor linear in
`u` flattens every predicted difference between targets; in **share units**
`s(u) = αu/(1+αu)` the reduction is exact and β identifies `1/g − 1` itself
(`reduction_is_exact_in_share_units`, `rls_on_share_channels_recovers_one_over_g_minus_one`).

## Files

| file | what |
|---|---|
| `driver.py` | `A(t)`, the dial, `certify()` — the four NS-2 gates over the whole domain |
| `coupling.py` | the declared `W`, the `r` classes, the share basis, the geometric centring, the one-pass option basis |
| `layer.py` | the severity **mixin**: below the method, above the host |
| `channel.py` | the policy's shift: `steer_logits` + the floor, as one numpy reference |
| `ceiling.py` | Part C — the coordination gap, with **no training** |
| `selftest.py` | the conformance suite, **no StarCraft II, no torch** |
| `check_actor.py` | the torch `StochasticPolicy` itself against `channel.py` |
| `toy.py` | a stand-in engine that reproduces `StarCraft2_Env.step`'s order of operations |
| `story.py` | a Lanchester battle through the real layer: what the dial costs a focus-fire host, what steering returns |
| `pact1_core.py` | **verbatim vendored** `pact1/core.py`. Do not edit |

The host gained exactly **one** no-op hook, `_ns_hook`, called between the engine
tick and `update_units` — the only point where harm can touch reward, observation,
termination and record together.

The trust head is **one scalar** in `StochasticPolicy`, applied as a logit bias
inside the softmax, so the host's clipped surrogate, entropy bonus and optimiser
are byte-for-byte unchanged (P-6.2). The per-action predicted cost rides the
**tail of the observation** and the base network never sees it, so every
PACT-family arm has the identical host network. The floor is applied in the policy
because only the policy knows which attacks are in range.

## Runbook

**1. Offline, before any cluster time (minutes, no StarCraft II).**

```bash
python -m harl.envs.smac.smac_ns.selftest
```
```bash
python -m harl.envs.smac.smac_ns.check_actor
```
```bash
python -m harl.envs.smac.smac_ns.story
```

**2. Part C, committed before any method run (NS-4.1).**

```bash
python -m harl.envs.smac.smac_ns.ceiling --map 3s5z --out ceiling.json
```

**3. The arms, host to host.** `pactoff` is bit-identical to MAPPO and writes the
same `pact_debug.csv`, so it is the baseline to read against `pact`.

```bash
python examples/train.py --algo pactoff --env smac --exp_name blind --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name pact --seed 1
```
```bash
python examples/train.py --algo pact_oracle --env smac --exp_name oracle --seed 1
```
```bash
python examples/train.py --algo pact_fixed --env smac --exp_name fixed --seed 1
```

`intercept` reduces **exactly** to `pactoff` in this port (with the peer channels
deleted every option predicts the same cost, so the shift is zero); do not spend a
run on it.

**4. The existing algorithms, inside the identical physics.**

```bash
python examples/train.py --algo mappo --env smac --exp_name mappo_ns --seed 1
```
```bash
python examples/train.py --algo happo --env smac --exp_name happo_ns --seed 1
```

**5. B0 reference — stock SMAC, dial off.**

```bash
python examples/train.py --algo mappo --env smac --exp_name b0 --ns_on 0 --seed 1
```

**6. The severity ladder.** `σ > 1` is a **beyond-physical stress test** and must
be labelled as such in every table and figure.

```bash
python examples/train.py --algo pactoff --env smac --exp_name blind_s3 --ns_severity 3.0 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name pact_s3 --ns_severity 3.0 --seed 1
```

**7. Ablations.** Declared constants are swept, never tuned.

```bash
python examples/train.py --algo pact --env smac --exp_name abl_floor0 --ns_steer_floor 0.0 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name abl_mu --ns_mu 0.99 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name abl_cap_max --ns_cap_mode max --seed 1
```

Run 5 seeds per arm.

## Reading `pact_debug.csv`

II.10's principle: **"is the method working" and "is it winning" are separate
columns.** Workers are de-phased, so every rollout mixes all phases of the cycle;
the `*_peak` / `*_dry` columns split the same sums by the phase each step or
episode was played in — that is where the story is visible inside one run.

| column | question | healthy |
|---|---|---|
| `ns_harmed`, `ns_restored` | **did the dial fire at all?** | non-zero for σ>0 |
| `waste_frac_peak` / `waste_frac_dry` | how much landed damage the guard takes, at its peak / in the placebo | peak > 0; **dry exactly 0** (NS-2.5) |
| `ns_harm_delivery_cum` | delivered ÷ designed harm (killing blows cannot be restored) | stable |
| `overkill_frac` | stock overkill — III.1's observable for calibrating α | — |
| `ns_mem_frac` | estimator memory ÷ driver period, from the observed row rate | < 0.25 |
| `gated_frac_peak` / `gated_frac_dry` | how often the floor binds (over all living enemies) | low at peak, **≈1 in the placebo** |
| `ns_fit_gain` / `ns_pred_gain` | does the reduction hold / is it predictable | rise, stay up |
| `ns_cond_psi` | can β be decomposed, or only used? | finite; warn only |
| `trust_pol` | the reliance the policy has set | stays up if steering pays |
| `ns_herd_index` | commons signature | **logged, never acted on** |
| `win_rate_peak` / `win_rate_dry` | **the separation**: blind's gap should exceed pact's | — |

The console prints the same as a `PHASE` line, and warns `[GATED]`,
`[PLACEBO-STEER]`, `[PLACEBO-HARM]` and `[NOT-TRACKING]` when a phase column says
the mechanism is not doing what it should.

## Honest limits

1. **Steering only.** No inverse exists, so the claim is identification and
   steering, and the margin is capped by the coordination gap. Do not write
   "compensation".
2. **The driver's shape is injected.** The *amplitude* is published (SC2 armour),
   but StarCraft has no weather, and NS-1.3's "a practitioner names it unprompted"
   is met only in part. Say so in those words.
3. **The Part-C endpoint carries no information** (100 % by construction). The
   N-scaling curve is the result.
4. **A killing blow is never harmed** — a death cannot be undone through the debug
   channel — so the delivered harm is below the designed harm;
   `ns_harm_delivery_cum` reports by how much.
5. **α = 2.28 is borrowed from URB.** Calibrate it once from `overkill_frac` by a
   stated procedure, then freeze it.
6. **The story test is a toy** — a fixed host, no space, no cooldowns. It chooses a
   severity worth a cluster run; it is not a result.
