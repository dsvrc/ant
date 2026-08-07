"""G — the cross-context generalization matrix  [campaign spec §5.1].

Every **blind-obs** policy from {F0, F1a, F1b, F1c, F3a, X1 student} evaluated at
``FREEZE_A in {0, .25, .5, .75, 1.0}``, 40 episodes per cell, deterministic
actors. Oracle-schema policies (F2, F3b) get their own smaller matrix with the
oracle obs active — never mixed into the blind one (Prohibition 7: no protocol
mixing in any figure; a normalized-oracle policy and a blind policy do not even
share an input space).

What it reads off
-----------------
* **diagonal** = the per-slice attainable return (the constructive ceiling);
* **off-diagonal decay** = how context-specific the optima are;
* ``max_pi min_A G(pi, A)`` = the best single robust policy found — **gate V4**;
* **the path ceiling**::

      PC = mean_t G_diag(c(A(t)*sigma))   over one payload period

  with the diagonal interpolated piecewise-linearly in A and the average taken as
  a 40k-point trapezoid sum over the smoothstep A(t). **PC replaces the arbitrary
  "6500" as the principled target for any future method**:

      target := 0.9 * PC   (cycle-average, C4 protocol)

  This is the campaign's single most reusable number: it is what a method could
  achieve if it tracked the payload perfectly and paid nothing for switching.

    python scripts/diag_crosseval.py --out diag_out/g \\
        --policy f0:<run>/models --policy f1a:<run>/models@0.0 \\
        --policy f1b:<run>/models@0.5 --policy f1c:<run>/models@1.0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.envs.mamujoco.diag.report_io import (  # noqa: E402
    DebugReport, bootstrap_ci, fmt_ci, write_csv)
from scripts.diag_tier0 import load_run_config, run_cell  # noqa: E402

_P_PERIOD = 40000
_B = 0.2
_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def payload_at(clock):
    ph = (clock % _P_PERIOD) / _P_PERIOD
    x = ph / _B if ph < _B else (1.0 - ph) / (1.0 - _B)
    return x * x * (3.0 - 2.0 * x)


def parse_policy(s):
    """``name:path[@train_A]`` — ``@A`` marks this policy as the diagonal's expert
    at slice A (F1a@0.0, F1b@0.5, F1c@1.0). Policies without @A (F0, F3a, the X1
    student) are evaluated across the row but contribute no diagonal point."""
    name, _, rest = s.partition(":")
    path, _, a = rest.partition("@")
    return {"name": name, "path": path,
            "train_A": float(a) if a != "" else None}


def path_ceiling(diag_points, rep):
    """PC = mean over one payload period of the piecewise-linearly interpolated
    diagonal, evaluated at A(t) — a 40k-point trapezoid sum (spec §5.1)."""
    if len(diag_points) < 2:
        rep.line("  PC needs >= 2 diagonal experts; got "
                 f"{len(diag_points)}. Pass f1a/f1b/f1c with @A.")
        return float("nan")
    xs = np.array(sorted(diag_points))
    ys = np.array([diag_points[x] for x in xs])
    t = np.arange(_P_PERIOD)
    A_t = np.array([payload_at(int(x)) for x in t])
    if A_t.min() < xs.min() - 1e-9 or A_t.max() > xs.max() + 1e-9:
        rep.line(f"  note: A(t) spans [{A_t.min():.3f}, {A_t.max():.3f}] but the "
                 f"diagonal only covers [{xs.min():.2f}, {xs.max():.2f}]; "
                 f"np.interp CLAMPS outside, so PC is an approximation there.")
    G_t = np.interp(A_t, xs, ys)
    pc = float(np.trapz(G_t, t) / (len(t) - 1))
    return pc


def main(argv=None):
    ap = argparse.ArgumentParser(description="G — cross-context generalization matrix.")
    ap.add_argument("--policy", action="append", required=True,
                    help="'name:path/to/models[@train_A]' — repeatable")
    ap.add_argument("--oracle_policy", action="append", default=[],
                    help="oracle-schema policies (F2, F3b) — their own matrix")
    ap.add_argument("--out", default="./diag_out/g")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--b0", type=float, default=None,
                    help="B0 (F0's stationary return) for the V4 gate; defaults to "
                         "the best cell at A=0")
    args = ap.parse_args(argv)

    rep = DebugReport(os.path.join(args.out, "g_matrix.md"),
                      title="G — the cross-context generalization matrix",
                      subtitle="path geometry; gate V4; the path ceiling PC")

    def build(pol_specs, tag):
        rows, G = [], {}
        for spec in pol_specs:
            p = parse_policy(spec)
            cfg, _ = load_run_config(p["path"])
            G[p["name"]] = {}
            for A in _GRID:
                r = run_cell(cfg, p["path"], args.out, f"g_{tag}",
                             f"{p['name']}_A{A}", "identity", A=A,
                             episodes=args.episodes, device=args.device)
                G[p["name"]][A] = r
                rows.append([tag, p["name"], p["train_A"], A,
                             round(float(r["ret_mean"]), 2),
                             round(float(r["ret_lo"]), 2),
                             round(float(r["ret_hi"]), 2),
                             r["len_mean"], r["fall_rate"]])
        return G, rows

    rep.h2("blind-obs matrix")
    G, rows = build(args.policy, "blind")
    names = list(G.keys())
    rep.table(["policy \\ eval A"] + [f"A={a}" for a in _GRID] + ["min over A"],
              [[n] + [f"{G[n][a]['ret_mean']:.0f}" for a in _GRID]
               + [f"**{min(G[n][a]['ret_mean'] for a in _GRID):.0f}**"]
               for n in names])
    rep.line("(cell = mean return over "
             f"{args.episodes} deterministic episodes; CIs in g_matrix.csv)")

    # ---- V4 --------------------------------------------------------------
    rep.h2("V4 — robust single policy")
    worst = {n: min(G[n][a]["ret_mean"] for a in _GRID) for n in names}
    best_n = max(worst, key=worst.get)
    b0 = args.b0
    if b0 is None:
        b0 = max(G[n][0.0]["ret_mean"] for n in names)
        rep.line(f"  B0 not given; using the best A=0 cell = {b0:.1f}")
    mm = worst[best_n]
    rep.kv("max_pi min_A G", f"{mm:.1f}  (policy '{best_n}')")
    rep.kv("0.8 * B0 — the V4 bar", f"{0.8 * b0:.1f}")
    rep.verdict("V4 robust-single-policy (max_pi min_A G >= 0.8*B0)",
                mm >= 0.8 * b0)
    rep.note("Note the sign convention: for the BENCHMARK, V4-fail is *desired* "
             "(WP-5 path non-degeneracy — if one policy covers every slice the "
             "environment does not require adaptation at all). A V4 PASS is a "
             "strong negative result about PCR@0.9, not a method win; §8's leaf L5 "
             "says decide with the user, and do not silently ship it as the "
             "headline.")

    # ---- path ceiling ----------------------------------------------------
    rep.h2("PC — the path ceiling (the principled target)")
    diag = {}
    for spec in args.policy:
        p = parse_policy(spec)
        if p["train_A"] is not None and p["name"] in G:
            diag[p["train_A"]] = float(G[p["name"]][p["train_A"]]["ret_mean"])
    rep.table(["slice A", "expert", "G_diag(A)"],
              [[a, [parse_policy(s)["name"] for s in args.policy
                    if parse_policy(s)["train_A"] == a][0], f"{v:.1f}"]
               for a, v in sorted(diag.items())])
    pc = path_ceiling(diag, rep)
    rep.kv("PC (payload-period average of the interpolated diagonal)",
           f"{pc:.1f}")
    rep.kv("**target := 0.9 * PC** (cycle-average, C4 protocol)",
           f"**{0.9 * pc:.1f}**")
    rep.note("This number replaces the arbitrary 6500 for every future method "
             "spec. It is the return of an oracle that tracked the payload "
             "perfectly and paid nothing to switch — so it is an upper bound that "
             "is actually constructed, not guessed.")

    # ---- path geometry ---------------------------------------------------
    rep.h2("path geometry — value Lipschitz constant along the path (theory hook 5)")
    xs = np.array(sorted(diag))
    if xs.size >= 2:
        ys = np.array([diag[x] for x in xs])
        slopes = np.diff(ys) / np.diff(xs)
        rep.table(["A interval", "dG/dA"],
                  [[f"[{xs[i]:.2f}, {xs[i + 1]:.2f}]", f"{slopes[i]:.0f}"]
                   for i in range(len(slopes))])
        rep.kv("max |dG/dA| (feeds the T3 tracking bound with an actual number)",
               f"{np.max(np.abs(slopes)):.0f}")

    # ---- oracle-schema matrix -------------------------------------------
    orows = []
    if args.oracle_policy:
        rep.h2("oracle-schema matrix (kept separate — Prohibition 7)")
        Go, orows = build(args.oracle_policy, "oracle")
        rep.table(["policy \\ eval A"] + [f"A={a}" for a in _GRID],
                  [[n] + [f"{Go[n][a]['ret_mean']:.0f}" for a in _GRID]
                   for n in Go])
        rep.note("These policies read a privileged obs; their numbers are NOT "
                 "comparable to the blind matrix cell-by-cell and must never share "
                 "a figure with it.")

    path = write_csv(os.path.join(args.out, "g_matrix.csv"),
                     ["matrix", "policy", "train_A", "eval_A", "ret_mean",
                      "ret_lo", "ret_hi", "len_mean", "fall_rate"], rows + orows)
    # Machine-readable summary: X1 needs PC and diag_report needs it too. Writing
    # it here means neither has to scrape this file's prose for a number.
    import json

    summary = {"PC": pc, "target": 0.9 * pc, "B0": b0,
               "V4_max_min": mm, "V4_policy": best_n,
               "V4_pass": bool(mm >= 0.8 * b0),
               "diagonal": {str(k): v for k, v in sorted(diag.items())},
               "grid": list(_GRID), "episodes": args.episodes}
    spath = os.path.join(args.out, "g_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    rep.h2("artifacts")
    rep.kv("g_matrix.csv", path)
    rep.kv("g_summary.json (PC/target/V4 — read by X1 and the report)", spath)

    # ---- heatmap (optional; the markdown table above is always there) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        M = np.array([[G[n][a]["ret_mean"] for a in _GRID] for n in names])
        fig, ax = plt.subplots(figsize=(1.4 * len(_GRID) + 2, 0.6 * len(names) + 2))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(_GRID)))
        ax.set_xticklabels([f"A={a}" for a in _GRID])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(_GRID)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="w", fontsize=8)
        ax.set_title(f"G matrix (blind obs) — PC={pc:.0f}, target=0.9*PC={0.9 * pc:.0f}")
        fig.colorbar(im)
        fig.tight_layout()
        hp = os.path.join(args.out, "g_matrix.png")
        fig.savefig(hp, dpi=140)
        rep.kv("heatmap", hp)
    except Exception as e:
        rep.line(f"  (heatmap skipped: {e!r} — the markdown table above is the "
                 f"artifact of record)")

    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
