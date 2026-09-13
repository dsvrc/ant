"""The actor side of PACT-1 -- the real StochasticPolicy, not a copy of its maths.

``selftest.py`` is numpy-only and checks ``pact1_core.steer_logits``.  The policy
re-implements that shift in torch (it has to: the trust gradient must flow through
the host's own objective, P-6.2), and a re-implementation can drift from the thing
it copies.  It did.  The core z-scores over a ``valid`` mask; the actor built that
mask from ``available_actions`` alone, so SMAC's stop/move rows -- zero because no
peer load is predicted for them -- entered the statistics as the CHEAPEST options.
Every attack was pushed down and every move up, on every step of the first run.
The numpy test could not see it: it passed ``valid[6:] = True`` by hand.

Run::

    python -m harl.envs.smac.smac_ns.check_actor
"""

import sys

import numpy as np

FAILS = []
OBS, K, FROM = 30, 14, 6          # 3s5z: 6 no-attack actions + 8 enemies


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


def _args(**pact):
    a = dict(hidden_sizes=[32, 32], activation_func="relu",
             use_feature_normalization=True, initialization_method="orthogonal_",
             gain=0.01, use_naive_recurrent_policy=False, use_recurrent_policy=False,
             recurrent_n=1, use_policy_active_masks=True, std_x_coef=1,
             std_y_coef=0.5)
    a.update(pact)
    return a


def main():
    import torch
    from gym import spaces

    from harl.models.policy_models.stochastic_policy import StochasticPolicy
    from harl.envs.smac.smac_ns.channel import spread, steer

    print("=" * 78)
    print("PACT-1 actor checks -- the torch StochasticPolicy itself")
    print("=" * 78)
    act_space = spaces.Discrete(K)

    def build(tail, **pact):
        torch.manual_seed(0)
        return StochasticPolicy(_args(**pact),
                                spaces.Box(-1, 1, (OBS + (K if tail else 0),)),
                                act_space)

    FLOOR = 0.01
    stock = build(False)
    off = build(True, pact_n_actions=K, pact_trust="off")
    pact = build(True, pact_n_actions=K, pact_trust="learned", pact_steer_from=FROM,
                 pact_steer_floor=FLOOR)

    B = 64
    rng = np.random.RandomState(3)
    obs = rng.randn(B, OBS).astype(np.float32)
    avail = np.ones((B, K), dtype=np.float32)
    avail[:, 0] = 0.0                                   # no-op: dead agents only
    avail[:, FROM:][rng.rand(B, K - FROM) < 0.3] = 0.0  # out-of-range enemies
    avail[:, FROM] = 1.0                                # keep >= 2 attack options
    avail[:, FROM + 1] = 1.0
    rnn = np.zeros((B, 1, 32), dtype=np.float32)
    masks = np.ones((B, 1), dtype=np.float32)
    act = np.zeros((B, 1), dtype=np.float32)

    def probs(model, tail):
        o = obs if tail is None else np.concatenate([obs, tail], 1)
        _, _, dist = model.evaluate_actions(o, rnn, act, masks, avail)
        return dist.probs.detach()

    def logits(model, tail):
        o = obs if tail is None else np.concatenate([obs, tail], 1)
        _, _, dist = model.evaluate_actions(o, rnn, act, masks, avail)
        return dist.logits.detach()

    host = probs(stock, None)

    # ---- P-9.1: the off arm IS the host -------------------------------------
    tail_noise = rng.rand(B, K).astype(np.float32)
    check("pactoff_is_bit_identical_to_the_stock_policy",
          torch.equal(probs(off, tail_noise), host)
          and sum(p.numel() for p in off.parameters())
          == sum(p.numel() for p in stock.parameters()),
          "same parameters, same probabilities, whatever the tail holds")

    # ---- the first run's bug, reproduced through the real actor --------------
    first = np.zeros((B, K), dtype=np.float32)
    first[:, FROM:] = 0.0027 + 0.0007 * rng.randn(B, K - FROM)   # its measured costs
    buggy = build(True, pact_n_actions=K, pact_trust="learned", pact_steer_from=0,
                  pact_steer_floor=0.0)
    lb, lh = logits(buggy, first), logits(stock, None)
    d = (lb - lh)
    # compare the LOG-ODDS of attacking vs moving, which normalisation cannot hide
    mv = avail[:, 1:FROM] > 0
    at = avail[:, FROM:] > 0
    shift_mv = float(d[:, 1:FROM][torch.from_numpy(mv)].mean())
    shift_at = float(d[:, FROM:][torch.from_numpy(at)].mean())
    check("unmasked_zscore_pushes_attacks_below_moves  (the bug)",
          shift_at - shift_mv < -1.0,
          "attack-vs-move log-odds shifted by %+.2f logits" % (shift_at - shift_mv))

    # ---- the fix, part 1: the first run's costs are below the floor ----------
    check("first_run_costs_are_below_the_floor_so_the_host_is_untouched",
          torch.equal(probs(pact, first), host),
          "spread ~%.4f < floor %.2f -> exactly the host"
          % (float(np.nanmean([spread(first[b], avail[b] * (np.arange(K) >= FROM))
                               for b in range(B)])), FLOOR))

    # ---- the fix, part 2: above the floor, moves never shift and the attack
    # shift is the core's, row by row
    live = np.zeros((B, K), dtype=np.float32)
    live[:, FROM:] = (0.12 * rng.rand(B, K - FROM)).astype(np.float32)
    live[: B // 4, FROM:] = (0.0027 + 0.0007 * rng.randn(B // 4, K - FROM))  # gated rows
    lp = logits(pact, live)
    dd = (lp - lh)
    mv_moves = float((dd[:, 1:FROM] - dd[:, 1:2]).abs().max())
    check("steering_never_reorders_a_move_against_a_move",
          mv_moves < 1e-5, "max relative shift among moves %.2e" % mv_moves)
    # relative to the moves, attacks shift by -g*kappa*z, which has zero mean
    rel = (dd[:, FROM:] - dd[:, 1:2])
    rel_mean = float(rel[torch.from_numpy(at)].mean())
    check("attacks_as_a_group_keep_their_odds_against_moves",
          abs(rel_mean) < 1e-4, "mean attack shift vs moves %+.2e (z has mean 0)"
          % rel_mean)
    g = float(torch.sigmoid(torch.tensor(2.2)))
    worst, n_open, n_shut, shut_exact = 0.0, 0, 0, True
    for b in range(B):
        valid = np.zeros(K, bool)
        valid[FROM:] = avail[b, FROM:] > 0
        ref = steer(np.zeros(K), live[b].astype(np.float64), g, 1.0, valid, FLOOR)
        got = (dd[b] - dd[b, 1]).numpy().astype(np.float64)
        worst = max(worst, float(np.max(np.abs((got - ref)[valid]))))
        if np.all(ref == 0.0):
            n_shut += 1
            shut_exact &= bool(torch.equal(lp[b], lh[b]))
        else:
            n_open += 1
    check("actor_shift_equals_channel_steer_row_by_row",
          worst < 1e-4 and n_open > 0 and n_shut > 0,
          "max|torch - numpy| %.2e over %d open and %d gated rows"
          % (worst, n_open, n_shut))
    check("gated_rows_are_the_host_bit_for_bit", shut_exact)

    flat = np.zeros((B, K), dtype=np.float32)
    flat[:, FROM:] = 0.0031                        # identical predictions
    check("flat_attack_costs_leave_the_host_bit_for_bit",
          torch.equal(probs(pact, flat), host),
          "moves at 0, attacks all equal -> exactly the host (P-7.1)")

    # ---- P-6.2: the trust head is reached through the ordinary objective -----
    pact.zero_grad()
    o = torch.from_numpy(np.concatenate([obs, live], 1))
    a_taken = torch.from_numpy(np.full((B, 1), FROM, dtype=np.float32))
    lpb, _, _ = pact.evaluate_actions(o, rnn, a_taken, masks, avail)
    lpb.sum().backward()
    gw = pact.pact_w.grad
    check("trust_head_receives_gradient",
          gw is not None and float(gw.abs().sum()) > 0.0,
          "d logp / d w = %.4g" % (float(gw.sum()) if gw is not None else float("nan")))

    fixed = build(True, pact_n_actions=K, pact_trust="fixed", pact_steer_from=FROM,
                  pact_steer_floor=FLOOR)
    check("fixed_arm_creates_no_trust_parameter", fixed.pact_w is None)

    for missing, why in (("pact_steer_from", "z-scores moves against attacks"),
                         ("pact_steer_floor", "steers on a residual in the placebo")):
        kw = dict(pact_n_actions=K, pact_trust="learned", pact_steer_from=FROM,
                  pact_steer_floor=FLOOR)
        kw.pop(missing)
        raised = False
        try:
            build(True, **kw)
        except ValueError:
            raised = True
        check("missing_%s_refuses_to_build" % missing, raised,
              "a silent default %s -- the first run's bug" % why)

    print("=" * 78)
    if FAILS:
        print("FAILED %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL ACTOR CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
