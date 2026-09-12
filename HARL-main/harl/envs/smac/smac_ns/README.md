# SMAC-NS — Coupling Under Drift, instantiated in SMAC

The cell this instance fills, stated first because the spec says conflating the
cells is the one thing that is not publishable:

|  | invertible channel | **no inverse** |
|---|---|---|
| **(C) interaction-mediated** | `simple_ns` (VMAS) | **SMAC-NS ← this instance**, `road_ns` (URB) |
| (B) exogenous | the control | — |

> **(C, no inverse) — identification and steering only.** The margin is capped by
> the coordination gap. SMAC's action space is `no-op / stop / move×4 / attack[e]`
> and its only continuous magnitude is the move stride, which was *measured* to be
> inert on 3s5z (a movement-competent control scores 0.83× a focus-fire one there,
> and a perfect oracle on a stride channel gained 1.03). So there is no inverse to
> claim, and the channel is a shift over the K attack options — `steer_logits`
> from `pact1/core.py`, verbatim.

## The story, and every clause is a requirement

> *A squad shares one enemy line. Damage poured into an enemy beyond what it can
> absorb is overkill — wasted. Under the enemy's guard phase each target absorbs
> usefully less, so the same volume of fire wastes more. **A lone unit cannot
> overkill**: it fires, the target dies, it retargets. Nothing is wasted at any
> guard intensity.*

Every player has said "stop overkilling that zealot". It is also exactly what a
3s5z policy at a 90 % win rate has learned to avoid — which is why the dial has
something to take away from the existing algorithms.

| object | SMAC | check |
|---|---|---|
| medium | the enemy line's absorbable damage | — |
| element `e`, `cap_e` | an enemy unit; its **published** life + shields (Zealot 150, Stalker 160) | — |
| loading | `u_i = Σ_{j≠i} dmg_j→target(i) / cap`, **max** over the agent's own elements | NS-1.1 |
| operator `W[i,j]` | incidence ÷ capacity × published per-unit damage. **zero-diag, asym 0.545, spread 0.470, ratio 3.6×** | NS-1.2 |
| `r` classes | enemy unit **types** (2 on 3s5z). Independent of N and of the element count | P-1.1 |
| driver `A(t)` | the guard cycle; shrinks absorption, never adds to the loss; **exactly 0 for 75/150 steps** | NS-1.3, NS-2.5 |
| anchor | one SC2 **armour point** against a Stalker's 10-damage weapon = exactly **10 %** | NS-2.4 |
| harm | `excess = f(u/g)/f(u) − 1`, `f(u)=1+αu`, handed back as restored hit points | NS-1.4 |
| sensor | `realized/nominal − 1` → `relative_excess`, verbatim | P-2.1 |

Two identities hold **exactly**, and the suite asserts them rather than hoping:

```
sigma = 0   ->  excess == 0        the stock task, byte for byte
u_i   = 0   ->  harm  == 1.000000  at g = 1e-3, i.e. the harshest dial
```

The second is category C obtained *structurally*, from the `j != i`, not from a
number that happens to be small.

**Regime, not convention.** The medium runs at `u ≈ 0.213` — URB measures 0.219 —
two orders of magnitude below where a quartic BPR exponent means anything. So the
performance function is **linear**, `α = 2.28` declared and swept.

## Files

| file | what |
|---|---|
| `driver.py` | `A(t)`, the dial, `certify()` — the four NS-2 gates over the whole domain |
| `coupling.py` | the declared `W`, the `r` classes, the basis, the geometric centring |
| `layer.py` | the severity **mixin**: below the method, above the host |
| `ceiling.py` | Part C — the coordination gap, with **no training** |
| `selftest.py` | the conformance suite, 27 checks, **no StarCraft II** |
| `pact1_core.py` | **verbatim vendored** `pact1/core.py`. Do not edit |

The host gained exactly **one** no-op hook, `_ns_hook`, called between the engine
tick and `update_units` — the only point where harm can touch reward, observation,
termination and record together. NS-3.2 bought once instead of by patching three
call sites.

The trust head is **one scalar** in `StochasticPolicy`, applied as a logit bias
inside the softmax, so the host's clipped surrogate, entropy bonus and optimiser
are byte-for-byte unchanged and the gradient reaches it with no new sampled
dimension (P-6.2). The per-action predicted cost rides the **tail of the
observation** and the base network never sees it, so every PACT-family arm has the
identical host network.

## Runbook

**1. Conformance (seconds, no StarCraft II).** Ends `ALL CHECKS PASSED`.

```bash
python -m harl.envs.smac.smac_ns.selftest
```

**2. Part C, committed before any method run (NS-4.1).**

```bash
python -m harl.envs.smac.smac_ns.ceiling --map 3s5z --out ceiling.json
```

Measured: the coordination gap rises **24 % → 51 % → 78 % → 100 %** with
controllable share — NS-4.2's falsifiable prediction, confirmed with no training.
Quote the *curve*, not the 100 % endpoint: `Δ_fixed = 0` because every damage
source is an agent, and `Δ_own = 0` by construction of the sensor, so the endpoint
was never at risk of coming out small.

**3. The arms.** Five, through the identical wrapper, differing only in the trust
term (P-9.1).

```bash
python examples/train.py --algo pactoff --env smac --exp_name blind --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name pact --seed 1
```
```bash
python examples/train.py --algo pact_fixed --env smac --exp_name fixed --seed 1
```
```bash
python examples/train.py --algo pact_oracle --env smac --exp_name oracle --seed 1
```
```bash
python examples/train.py --algo pact_intercept --env smac --exp_name intercept --seed 1
```

`pactoff` is the honest baseline: trust forced to 0, the shift returns the logits
bit for bit, the trust parameter is never created — the base network is the stock
one. `oracle` is the ceiling; no method can beat it.

**4. The existing algorithms, inside the identical physics.** The dial is in the
env config, so every algorithm in the repo runs under it unchanged. These are what
should fall.

```bash
python examples/train.py --algo mappo --env smac --exp_name mappo_ns --seed 1
```
```bash
python examples/train.py --algo happo --env smac --exp_name happo_ns --seed 1
```
```bash
python examples/train.py --algo hatrpo --env smac --exp_name hatrpo_ns --seed 1
```
```bash
python examples/train.py --algo qmix --env smac --exp_name qmix_ns --seed 1
```

**5. B0 reference — stock SMAC, dial off.**

```bash
python examples/train.py --algo mappo --env smac --exp_name b0 --ns_on 0 --seed 1
```

**6. The severity ladder.** `σ > 1` is a **beyond-physical stress test** and must
be labelled as such in every table and figure.

```bash
for S in 0.0 0.5 1.0 2.0 3.0; do python examples/train.py --algo pact --env smac --exp_name sigma_$S --ns_severity $S --seed 1; done
```

**7. Ablations.**

```bash
python examples/train.py --algo pact --env smac --exp_name abl_mu --ns_mu 0.99 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name abl_alpha --ns_alpha 1.0 --seed 1
```
```bash
python examples/train.py --algo pact --env smac --exp_name abl_meanpres --ns_mean_preserving 1 --seed 1
```
```bash
python examples/train.py --algo pact_happo --env smac --exp_name pact_happo --seed 1
```

Run 5 seeds per arm. Compare host to host: `pactoff` vs `pact`, `pact_happo_off`
vs `pact_happo`.

## Reading `pact_debug.csv`

II.10's principle: **"is the method working" and "is it winning" are separate
columns**, because the method can work perfectly and still not win, and that is a
statement about how much headroom the domain has rather than a bug.

| column | question | healthy |
|---|---|---|
| `harmed_steps`, `hp_restored` | **did the dial fire at all?** | non-zero — a run with σ>0 and zero here refuses to be reported as a severity arm |
| `sigma` / `g` / `placebo` | is the dial live, and does it switch off? | `placebo ≈ 0.5` |
| `u` / `excess` | is the NS biting? | non-zero outside the placebo |
| `fit_gain` | does the reduction hold here? | rises, stays up |
| `cond_psi` | can β be *decomposed*, or only used? | low; **warn only** — a degenerate regressor still predicts |
| `trust_pol` / `trust_app` | policy-set vs applied reliance | together; divergence means the gate misfires |
| `x_std` | channel liveness | non-zero |
| `herd_index` | is steering creating the commons externality? | **logged, never acted on** |
| `clip_frac` | how much of the sensor is clipped? | small |
| `win_rate`, `ep_return` | is it winning? | — |

## Honest limits

1. **Steering only.** No inverse exists, so the claim is identification and
   steering, and the margin is capped by the coordination gap. Do not write
   "compensation".
2. **The driver's shape is injected.** The *amplitude* is published (SC2 armour),
   but StarCraft has no weather, and NS-1.3's "a practitioner names it unprompted"
   is met only in part. Say so in those words.
3. **The Part-C endpoint carries no information** (100 % by construction). The
   N-scaling curve is the result.
4. **`ns_augment` is on only for the PACT family.** The stock baselines run on
   byte-identical stock SMAC observations inside the same dial; within the PACT
   family every arm shares the identical host network, so `pactoff` is bit-identical
   to the untouched host. Those are two different comparisons and both are reported.
5. **The harm lands one engine tick late** — it is issued through `controller.debug`
   and applies on the next step, which keeps the engine authoritative and nothing
   desynchronised. Physically right for a regeneration-like effect; state it.
