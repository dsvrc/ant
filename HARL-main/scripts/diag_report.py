"""The verdict  [campaign spec Part 7 / 8 / 9]. Ingests everything; emits one answer.

Reads every CSV/NPZ the campaign produced and emits ``verdict.md`` + ``verdict.json``
with the **eight axis readings**, the scalar facts, the **decision-tree walk** with
the fired leaf highlighted, and a ``diag_bundle/`` carrying everything the designer
needs for the next method spec.

| Axis | From | PASS |
|---|---|---|
| V1 existence-control        | E2       | max_beta R(A=1) >= 0.9*B0 |
| V2 existence-learning       | F1       | F1c >= 0.85*F1a  (and F1b >= 0.9*F1a) |
| V3 information-effect       | F2 vs F1c| gap >= 0.15*B0 => info-limited; |gap| < 0.05*B0 => info-irrelevant; F2 < F1c - 0.05*B0 => info-TOXIC |
| V4 robust-single-policy     | G        | max_pi min_A G >= 0.8*B0 |
| V5 conditioned-representation | X1     | drifting cycle-avg >= 0.9*PC |
| V6 decentralized observability | E5+E3-DOB | R^2(F-loc, L<=8, peak) >= 0.6 AND E3-DOB >= 0.85*R(exact beta*) |
| V7 retention                | F4 (+D3) | trough >= 0.8*B0 after 200k peak steps |
| V8 bandwidth budget         | E3       | report (k*, h*, r*); "tight" iff k* < 4 |

**The report states measurements and axis readings. It does not choose a method**
(Prohibition 8): the leaf is picked from Part 8's tree *with the user*. What this
prints is which leaf the tree *fires*, and every reading behind it.

    python scripts/diag_report.py --diag_out ./diag_out --results_root ./results \\
        --bundle ./diag_bundle
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harl.envs.mamujoco.diag.report_io import DebugReport  # noqa: E402

UNKNOWN = "UNKNOWN"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def find(diag_out, *parts):
    hits = sorted(glob.glob(os.path.join(diag_out, *parts), recursive=True))
    return hits[0] if hits else None


# ==========================================================================
#  ingest
# ==========================================================================
def load_tier0(diag_out):
    rows = []
    for p in sorted(glob.glob(os.path.join(diag_out, "**", "tier0_cells.csv"),
                              recursive=True)):
        rows += read_csv(p)
    return rows


def cells(rows, stage):
    return [r for r in rows if r.get("stage") == stage]


def final_eval(run_dir):
    """A training arm's converged eval: the cycle-average over the last fifth of
    the eval rounds in eval_debug.csv, else the tail of progress.txt."""
    ed = os.path.join(run_dir, "eval_debug.csv")
    if os.path.exists(ed):
        rs = read_csv(ed)
        if rs:
            steps = sorted({fnum(r["step"]) for r in rs})
            keep = set(steps[-max(1, len(steps) // 5):])
            v = [fnum(r["ep_return"]) for r in rs if fnum(r["step"]) in keep]
            p = [fnum(r["payload_end"]) for r in rs if fnum(r["step"]) in keep]
            v, p = np.array(v), np.array(p)
            ok = np.isfinite(v)
            out = {"cycle_avg": float(np.mean(v[ok])), "n": int(ok.sum()),
                   "source": "eval_debug.csv"}
            pf = p[ok & np.isfinite(p)]
            vf = v[ok & np.isfinite(p)]
            if pf.size >= 5:
                lo, hi = np.quantile(pf, 0.1), np.quantile(pf, 0.8)
                out["trough"] = float(np.mean(vf[pf <= lo])) if np.any(pf <= lo) else float("nan")
                out["peak"] = float(np.mean(vf[pf >= hi])) if np.any(pf >= hi) else float("nan")
            return out
    pt = os.path.join(run_dir, "progress.txt")
    if os.path.exists(pt):
        vals = []
        with open(pt, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2:
                    try:
                        vals.append(float(p[1]))
                    except ValueError:
                        pass
        if vals:
            tail = vals[-max(1, len(vals) // 5):]
            return {"cycle_avg": float(np.mean(tail)), "n": len(tail),
                    "source": "progress.txt (tail)"}
    return None


def load_arms(results_root, rep):
    """Map arm id -> its converged eval, by exp_name (diag_<arm>[_s<seed>])."""
    arms = {}
    for cfg_path in sorted(glob.glob(os.path.join(results_root, "**", "config.json"),
                                     recursive=True)):
        run_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        exp = cfg.get("main_args", {}).get("exp_name", "")
        if not exp.startswith("diag_"):
            continue
        arm = exp[len("diag_"):].split("_s")[0]
        fe = final_eval(run_dir)
        if fe is None:
            continue
        fe["run_dir"] = run_dir
        fe["seed"] = cfg.get("algo_args", {}).get("seed", {}).get("seed")
        arms.setdefault(arm, []).append(fe)
    return arms


def arm_mean(arms, name, key="cycle_avg"):
    if name not in arms:
        return float("nan")
    v = [a[key] for a in arms[name] if key in a and np.isfinite(a[key])]
    return float(np.mean(v)) if v else float("nan")


# ==========================================================================
#  axes
# ==========================================================================
def axis_v1(t0, b0, rep):
    e2 = cells(t0, "e2")
    peak = [r for r in e2 if abs(fnum(r["A"]) - 1.0) < 1e-9]
    if not peak or not np.isfinite(b0):
        return {"status": UNKNOWN, "why": "no E2 cells at A=1, or B0 unknown"}
    best = max(peak, key=lambda r: fnum(r["ret_mean"]))
    val = fnum(best["ret_mean"])
    return {"status": "PASS" if val >= 0.9 * b0 else "FAIL",
            "value": val, "bar": 0.9 * b0, "beta_star": best["cell"],
            "why": f"max_beta R(A=1) = {val:.0f} vs 0.9*B0 = {0.9 * b0:.0f}"}


def axis_v2(arms, rep):
    f1a, f1b, f1c = (arm_mean(arms, k) for k in ("f1a", "f1b", "f1c"))
    if not np.isfinite(f1a) or not np.isfinite(f1c):
        return {"status": UNKNOWN, "why": "F1a or F1c missing"}
    peak_ok = f1c >= 0.85 * f1a
    mid_ok = (not np.isfinite(f1b)) or (f1b >= 0.9 * f1a)
    return {"status": "PASS" if (peak_ok and mid_ok) else "FAIL",
            "f1a": f1a, "f1b": f1b, "f1c": f1c,
            "why": f"F1c {f1c:.0f} vs 0.85*F1a {0.85 * f1a:.0f} "
                   f"({'ok' if peak_ok else 'FAIL'}); F1b {f1b:.0f} vs 0.9*F1a "
                   f"{0.9 * f1a:.0f} ({'ok' if mid_ok else 'FAIL'})"}


def axis_v3(arms, b0, rep):
    f1c, f2 = arm_mean(arms, "f1c"), arm_mean(arms, "f2")
    f2b, f2c = arm_mean(arms, "f2b"), arm_mean(arms, "f2c")
    if not np.isfinite(f1c) or not np.isfinite(f2) or not np.isfinite(b0):
        return {"status": UNKNOWN, "why": "F1c, F2 or B0 missing"}
    gap = f2 - f1c
    if gap >= 0.15 * b0:
        read = "info-limited"
    elif abs(gap) < 0.05 * b0:
        read = "info-irrelevant"
    elif gap < -0.05 * b0:
        read = "info-TOXIC (H-C4)"
    else:
        read = "between thresholds (0.05*B0 <= gap < 0.15*B0): weakly info-limited"
    return {"status": read, "gap": gap, "f1c": f1c, "f2": f2, "f2b": f2b,
            "f2c": f2c,
            "why": f"F2 - F1c = {gap:+.0f} (0.05*B0 = {0.05 * b0:.0f}, "
                   f"0.15*B0 = {0.15 * b0:.0f})"}


def axis_v4(diag_out, b0, rep):
    p = find(diag_out, "**", "g_matrix.csv")
    if not p or not np.isfinite(b0):
        return {"status": UNKNOWN, "why": "g_matrix.csv or B0 missing"}
    rows = [r for r in read_csv(p) if r.get("matrix") == "blind"]
    if not rows:
        return {"status": UNKNOWN, "why": "no blind rows in g_matrix.csv"}
    worst = {}
    for r in rows:
        worst.setdefault(r["policy"], []).append(fnum(r["ret_mean"]))
    mm = {k: min(v) for k, v in worst.items()}
    best = max(mm, key=mm.get)
    return {"status": "PASS" if mm[best] >= 0.8 * b0 else "FAIL",
            "value": mm[best], "policy": best, "bar": 0.8 * b0,
            "why": f"max_pi min_A G = {mm[best]:.0f} ('{best}') vs 0.8*B0 = "
                   f"{0.8 * b0:.0f}"}


def axis_v5(diag_out, pc, rep):
    p = find(diag_out, "**", "x1_eval.csv")
    if not p or not np.isfinite(pc):
        return {"status": UNKNOWN, "why": "x1_eval.csv or PC missing"}
    v = [fnum(r["ret_mean"]) for r in read_csv(p) if r.get("protocol") == "drift"]
    if not v:
        return {"status": UNKNOWN, "why": "no drift rows in x1_eval.csv"}
    got = float(np.mean(v))
    return {"status": "PASS" if got >= 0.9 * pc else "FAIL", "value": got,
            "bar": 0.9 * pc,
            "why": f"X1 drifting cycle-avg {got:.0f} vs 0.9*PC = {0.9 * pc:.0f}"}


def axis_v6(diag_out, t0, rep):
    p = find(diag_out, "**", "e5_r2.csv")
    r2 = float("nan")
    if p:
        rows = read_csv(p)
        bins = sorted({r["bin"] for r in rows if r["bin"].startswith("bin")})
        if bins:
            peak = bins[-1]
            c = [fnum(r["r2_cv"]) for r in rows
                 if r["bin"] == peak and r["feature_set"] == "F-loc"
                 and fnum(r["L"]) <= 8 and np.isfinite(fnum(r["r2_cv"]))]
            r2 = max(c) if c else float("nan")
    dob = cells(t0, "e3dob")
    e3 = cells(t0, "e3")
    exact = [r for r in cells(t0, "e2")
             if abs(fnum(r["A"]) - 1.0) < 1e-9]
    dob_ok = UNKNOWN
    dob_val = ref = float("nan")
    if dob and exact:
        dob_val = fnum(dob[0]["ret_mean"])
        ref = max(fnum(r["ret_mean"]) for r in exact)
        dob_ok = "PASS" if dob_val >= 0.85 * ref else "FAIL"
    obs_ok = "PASS" if (np.isfinite(r2) and r2 >= 0.6) else (
        UNKNOWN if not np.isfinite(r2) else "FAIL")
    status = ("PASS" if (obs_ok == "PASS" and dob_ok == "PASS")
              else UNKNOWN if UNKNOWN in (obs_ok, dob_ok) else "FAIL")
    return {"status": status, "r2_floc_peak_L8": r2, "e3dob": dob_val,
            "e3dob_bar": 0.85 * ref if np.isfinite(ref) else float("nan"),
            "why": f"E5 F-loc R^2 (peak bin, L<=8) = {r2:.2f} vs 0.6 [{obs_ok}]; "
                   f"E3-DOB {dob_val:.0f} vs 0.85*R(exact) "
                   f"{0.85 * ref if np.isfinite(ref) else float('nan'):.0f} [{dob_ok}]"}


def axis_v7(diag_out, b0, rep):
    p = find(diag_out, "**", "f4_*.csv")
    if not p:
        return {"status": UNKNOWN, "why": "no f4_*.csv"}
    rows = read_csv(p)
    if not rows:
        return {"status": UNKNOWN, "why": "empty f4 csv"}
    ys = [fnum(r["trough_return_5ep"]) for r in rows]
    ref = ys[0]
    final = ys[-1]
    easy = final >= 0.8 * ref
    xs = [fnum(r["env_step"]) for r in rows]
    t10 = next((x for x, y in zip(xs, ys) if y <= 0.9 * ref), float("nan"))
    t50 = next((x for x, y in zip(xs, ys) if y <= 0.5 * ref), float("nan"))
    cls = ("easy" if easy else
           "cliff" if np.isfinite(t10) and t10 - xs[0] <= 2e4 else
           "fast" if np.isfinite(t50) and t50 - xs[0] <= 5e4 else "slow")
    return {"status": "PASS" if easy else "FAIL", "slope_class": cls,
            "trough_start": ref, "trough_final": final,
            "steps_to_lose_10pct": t10, "steps_to_lose_50pct": t50,
            "why": f"trough {ref:.0f} -> {final:.0f} after 200k peak steps "
                   f"(class: {cls})"}


def axis_v8(t0, rep):
    e3 = cells(t0, "e3")
    e2peak = [r for r in cells(t0, "e2") if abs(fnum(r["A"]) - 1.0) < 1e-9]
    if not e3 or not e2peak:
        return {"status": UNKNOWN, "why": "E3 or E2@A=1 missing"}
    ref = max(fnum(r["ret_mean"]) for r in e2peak)
    bar = 0.9 * ref
    def star(prefix, vals):
        good = []
        for v in vals:
            m = [r for r in e3 if r["cell"] == f"{prefix}{v}"]
            if m and fnum(m[0]["ret_mean"]) >= bar:
                good.append(v)
        return max(good) if good else 0
    k = star("delay", [1, 2, 4, 8, 16, 32])
    h = star("ema", [1, 4, 16, 64])
    r = max([v for v in (0.1, 0.2, 0.4)
             if any(x["cell"] == f"noise{v}" and fnum(x["ret_mean"]) >= bar
                    for x in e3)], default=0.0)
    dc = [x for x in e3 if x["cell"] == "dc64"]
    dc_ok = bool(dc) and fnum(dc[0]["ret_mean"]) >= bar
    return {"status": "tight" if k < 4 else "loose", "k_star": k, "h_star": h,
            "r_star": r, "dc64_pass": dc_ok, "ref": ref,
            "why": f"k*={k}, h*={h}, r*={r}; dc:64 "
                   f"{'PASS' if dc_ok else 'fail'} (bar = 0.9*R(exact) = {bar:.0f})"}


# ==========================================================================
#  the decision tree (spec Part 8) — total over the axis tuple
# ==========================================================================
LEAVES = {
    "L1": "Multi-agent privileged distillation of a local disturbance observer "
          "(MA-RMA/DOB) — train-time teacher supervises a per-agent adaptation "
          "module phi_i(own history) -> d_hat_i; policy consumes o_i (+) d_hat_i; "
          "execution fully decentralized.",
    "L1'": "As L1, but info is necessary even to HOLD — the observer is required "
           "for retention, not just for growth.",
    "L2": "Policy-path continuation by slice-training + online distillation — "
          "train experts on frozen/localized data and continually distill into one "
          "conditioned student (X1 proved the student class suffices).",
    "L3": "Ride-the-ramp: drift-as-curriculum + explicit retention — never train "
          "the peak cold; enter each peak from the trough policy and spend the "
          "design effort entirely on retention.",
    "L4": "Structure injection — scripted cancellation works but RL with the same "
          "information cannot learn it: give the learner the STRUCTURE (a fixed "
          "compensation layer a = pi_theta(o) - beta*d_hat), RL learns the residual.",
    "L5": "Robust-policy distillation — one policy covers the whole range. The "
          "honest conclusion is that PCR@0.9 does not REQUIRE adaptation: a strong "
          "negative result about the benchmark, not a method win. Decide with the "
          "user; do not silently ship as the headline.",
    "L6": "REDESIGN (severity/cap) — see Part 9. V1 fail means the benchmark is "
          "ill-posed under WP-1.",
    "L8/R-e": "Decentralized-unsolvable at the required bandwidth — redesign "
              "observability or rescope to CTDE-train/centralized-adapt.",
}


def walk(v1, v2, v3, v4, v5, v6, v7, v8, arms, rep):
    lines, leaf = [], None

    def say(s, fired=False):
        lines.append(("**==> " + s + "**") if fired else s)

    if v1["status"] == "FAIL":
        say("V1 fail ──► **L6** (redesign: severity/cap — Part 9)", True)
        return "L6", lines
    if v1["status"] == UNKNOWN:
        say("V1 UNKNOWN — E2 has not run. The tree cannot be walked: E2 is the "
            "existence experiment and every branch presupposes it.")
        return None, lines
    say("V1 pass.")
    if v2["status"] == "FAIL":
        say("V1 pass, V2 fail @peak:")
        f2, f1c = v3.get("f2", float("nan")), v3.get("f1c", float("nan"))
        f2_pass = np.isfinite(f2) and np.isfinite(f1c) and f2 >= 0.85 * v2["f1a"]
        f3a = arm_mean(arms, "f3a")
        f3b = arm_mean(arms, "f3b")
        f3a_pass = np.isfinite(f3a) and f3a >= 0.85 * v2["f1a"]
        f3b_pass = np.isfinite(f3b) and f3b >= 0.85 * v2["f1a"]
        if f2_pass:
            say("    F2 pass ──► **L1** (information deficit at learning time)", True)
            leaf = "L1"
        elif f3a_pass:
            say("    F2 fail, F3a pass ──► **L3** (grow-vs-hold: curriculum + "
                "retention)", True)
            leaf = "L3"
        elif f3b_pass:
            say("    F2 fail, F3a fail, F3b pass ──► **L1'** (info necessary even "
                "to hold)", True)
            leaf = "L1'"
        else:
            say("    all fail (but E2 passed!) ──► **L4** (RL-under-disturbance "
                "pathology: scripted control works, RL cannot — inject structure)",
                True)
            leaf = "L4"
        return leaf, lines
    if v2["status"] == UNKNOWN:
        say("V2 UNKNOWN — F1a/F1c have not run. Tree stops here.")
        return None, lines
    say("V1+V2 pass:")
    if v4["status"] == "PASS":
        say("    V4 pass ──► **L5** (robust policy exists; drift is a distractor; "
            "also consider R-hardening — see the caveat)", True)
        return "L5", lines
    if v4["status"] == UNKNOWN:
        say("    V4 UNKNOWN — G has not run. Tree stops here.")
        return None, lines
    if v5["status"] == "PASS":
        say("    V4 fail, V5 pass ──► **L2** (path exists, representable; RL "
            "optimization is the trap — slice-train-then-distill)", True)
        return "L2", lines
    if v5["status"] == UNKNOWN:
        say("    V4 fail, V5 UNKNOWN — X1 has not run. Tree stops here.")
        return None, lines
    say("    V4 fail, V5 fail:")
    if v6["status"] == "PASS":
        say("        V6 pass ──► **L1** (need fast local d_hat; memoryless c is "
            "not enough)", True)
        return "L1", lines
    if v6["status"] == UNKNOWN:
        say("        V6 UNKNOWN — E5/E3-DOB have not run. Tree stops here.")
        return None, lines
    say("        V6 fail ──► **L8/R-e** (decentralized-unsolvable at the required "
        "bandwidth — redesign observability or rescope)", True)
    return "L8/R-e", lines


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="The campaign verdict.")
    ap.add_argument("--diag_out", default="./diag_out")
    ap.add_argument("--results_root", default="./results")
    ap.add_argument("--bundle", default="./diag_bundle")
    ap.add_argument("--b0", type=float, default=None)
    ap.add_argument("--pc", type=float, default=None)
    args = ap.parse_args(argv)

    rep = DebugReport(os.path.join(args.bundle, "verdict.md"),
                      title="PCR diagnosis campaign — VERDICT",
                      subtitle="eight axis readings, the decision-tree walk, and "
                               "the bundle for the next method spec")

    t0 = load_tier0(args.diag_out)
    arms = load_arms(args.results_root, rep)
    rep.h2("inputs")
    rep.kv("Tier-0 cells found", len(t0))
    rep.kv("training arms found", ", ".join(sorted(arms)) or "(none)")

    # ---- scalar facts ----------------------------------------------------
    b0 = args.b0
    if b0 is None:
        e1_0 = [r for r in cells(t0, "e1") if abs(fnum(r["A"])) < 1e-9]
        if e1_0:
            b0 = fnum(e1_0[0]["ret_mean"])
        elif "f1a" in arms:
            b0 = arm_mean(arms, "f1a")
        else:
            b0 = float("nan")
    pc = args.pc
    if pc is None:
        pc = float("nan")
        gp = find(args.diag_out, "**", "g_summary.json")
        if gp:
            try:
                with open(gp, encoding="utf-8") as f:
                    pc = fnum(json.load(f).get("PC"))
            except Exception:
                pass
    rep.h2("scalar facts")
    rep.kv("B0 (stationary return)", f"{b0:.1f}")
    rep.kv("PC (path ceiling)", f"{pc:.1f}")
    rep.kv("target := 0.9 * PC (cycle-average, C4 protocol)",
           f"{0.9 * pc:.1f}" if np.isfinite(pc) else UNKNOWN)

    # ---- axes ------------------------------------------------------------
    V = {
        "V1": axis_v1(t0, b0, rep), "V2": axis_v2(arms, rep),
        "V3": axis_v3(arms, b0, rep), "V4": axis_v4(args.diag_out, b0, rep),
        "V5": axis_v5(args.diag_out, pc, rep),
        "V6": axis_v6(args.diag_out, t0, rep),
        "V7": axis_v7(args.diag_out, b0, rep), "V8": axis_v8(t0, rep),
    }
    names = {"V1": "existence-control", "V2": "existence-learning",
             "V3": "information-effect", "V4": "robust-single-policy",
             "V5": "conditioned-representation", "V6": "decentralized observability",
             "V7": "retention", "V8": "bandwidth budget"}
    rep.h2("the eight axes")
    rep.table(["axis", "reading", "detail"],
              [[f"**{k}** {names[k]}", f"**{V[k]['status']}**", V[k]["why"]]
               for k in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")])

    # ---- E1 deficit decomposition + fall causes --------------------------
    e1 = cells(t0, "e1")
    if e1:
        rep.h2("E1 — collapse profile, deficit decomposition and fall causes")
        base = [r for r in e1 if abs(fnum(r["A"])) < 1e-9]
        rows = []
        for r in sorted(e1, key=lambda x: fnum(x["A"])):
            A = fnum(r["A"])
            share = ""
            if base and A > 0:
                L0, LA = fnum(base[0]["len_mean"]), fnum(r["len_mean"])
                R0, RA = fnum(base[0]["ret_mean"]), fnum(r["ret_mean"])
                r0, rA = R0 / max(L0, 1e-9), RA / max(LA, 1e-9)
                ach, term = (rA - r0) * L0, rA * (LA - L0)
                share = f"{abs(term) / max(abs(ach) + abs(term), 1e-9):.0%}"
            rows.append([A, r["ret_mean"], r["len_mean"], r["fall_rate"],
                         r["sat_frac"], r["clip_frac_cmd"], share,
                         r["term_fall_low"], r["term_fall_high"],
                         r["term_nonfinite"], r["term_timeout"]])
        rep.table(["A", "return", "len", "fall rate", "sat_frac", "clip_cmd",
                   "termination share", "fall_low", "fall_high", "nonfinite",
                   "timeout"], rows)

    # ---- V8 / E4 / E2b / F2b ---------------------------------------------
    rep.h2("design-relevant scalars")
    rep.kv("V8 (k*, h*, r*)",
           f"k*={V['V8'].get('k_star')}, h*={V['V8'].get('h_star')}, "
           f"r*={V['V8'].get('r_star')}; dc:64 "
           f"{'PASS' if V['V8'].get('dc64_pass') else 'fail'}")
    e4 = cells(t0, "e4")
    if e4:
        rep.kv("E4 (project_sumzero)", ", ".join(
            f"A={r['A']}: {r['ret_mean']}" for r in e4))
    e2b = cells(t0, "e2b")
    if e2b and np.isfinite(b0):
        ok = [fnum(r["severity"]) for r in e2b if fnum(r["ret_mean"]) >= 0.9 * b0]
        rep.kv("E2b frontier sigma* (largest sigma still >= 0.9*B0)",
               f"{max(ok):.2f}" if ok else "none in the swept range")
    rep.kv("F2b/F2c three-way reading (V3)",
           f"F1c={V['V3'].get('f1c', float('nan')):.0f}  "
           f"F2(normalized d, both)={V['V3'].get('f2', float('nan')):.0f}  "
           f"F2b(raw d, critic only)={V['V3'].get('f2b', float('nan')):.0f}  "
           f"F2c(raw d, both)={V['V3'].get('f2c', float('nan')):.0f}")
    if np.isfinite(V["V3"].get("f2c", float("nan"))) and \
            np.isfinite(V["V3"].get("f2", float("nan"))):
        d = V["V3"]["f2c"] - V["V3"]["f2"]
        rep.note(f"**F2c - F2 = {d:+.0f}.** This is the cost of the obs "
                 f"normalization on the privileged arm: F2 receives "
                 f"(d - mean_t)/std_t, F2c receives d in torque units. If this gap "
                 f"is large, V3's 'info-toxic' reading would have been an artifact "
                 f"of the schema, not evidence for H-C4 — and E-5/E-7's 'the oracle "
                 f"failed' inherits the same doubt.")

    # ---- D2/F2 coupling + abort rules ------------------------------------
    d2, f2 = arm_mean(arms, "d2"), arm_mean(arms, "f2")
    if np.isfinite(d2) or np.isfinite(f2):
        rep.h2("D2 x F2 coupling (spec Part 6)")
        rep.kv("F2 (frozen peak, oracle)", f"{f2:.1f}")
        rep.kv("D2 (drifting, oracle, from scratch)", f"{d2:.1f}")
        if np.isfinite(f2) and np.isfinite(d2) and np.isfinite(V["V2"].get("f1a", np.nan)):
            if f2 >= 0.85 * V["V2"]["f1a"] and d2 < 0.85 * V["V2"]["f1a"]:
                rep.note("**F2 pass + D2 fail => drift itself breaks oracle "
                         "learning**: a retention pathology under FULL information. "
                         "Strong, surprising, and it indicts optimization directly.")

    # ---- the walk --------------------------------------------------------
    rep.h2("decision-tree walk (spec Part 8)")
    leaf, lines = walk(V["V1"], V["V2"], V["V3"], V["V4"], V["V5"], V["V6"],
                       V["V7"], V["V8"], arms, rep)
    rep.line("```")
    for ln in lines:
        rep.line(ln.replace("**", ""))
    rep.line("```")
    for ln in lines:
        if ln.startswith("**==>"):
            rep.line(ln)
    if leaf:
        rep.h3(f"fired leaf: {leaf}")
        rep.line(LEAVES.get(leaf, ""))
    else:
        rep.h3("no leaf fired")
        rep.line("The tree is total over a COMPLETE axis tuple; some axes are still "
                 "UNKNOWN, so the walk stops rather than guessing. Run the missing "
                 "stages (see the axis table).")
    rep.h3("overlays")
    if V["V7"]["status"] == "FAIL":
        rep.line(f"* **V7 hard ({V['V7'].get('slope_class')})** => add a retention "
                 f"module (KL-anchor / snapshot / modular heads) to whichever leaf "
                 f"fires.")
    if V["V8"]["status"] == "tight":
        rep.line(f"* **V8 tight (k*={V['V8'].get('k_star')} < 4)** => FORBID "
                 f"slow-chart designs in the method spec. Every windowed "
                 f"identifier, every c-conditioning scheme, ECL's envelope: all "
                 f"ruled out by the lag budget, regardless of their accuracy.")
    rep.note("**Prohibition 8**: this report states measurements and axis readings. "
             "The method choice is made from the tree above WITH THE USER, not "
             "inside this report.")

    # ---- WP certificates (spec §9.1) -------------------------------------
    rep.h2("well-posedness certificates (spec §9.1) — is PCR@0.9 a good benchmark?")
    wp = {
        "WP-1 existence-control (scripted compensator >= 0.9*B0 at peak)":
            V["V1"]["status"],
        "WP-2 slice learnability (base algo solves the hardest frozen slice to "
        ">= 0.85 of stationary)": V["V2"]["status"],
        "WP-3 graceful harm (<= 50% of the deficit is termination-mediated)":
            _wp3(t0),
        "WP-4 observability budget (privileged quantity recoverable within the E3 "
        "lag budget)": V["V6"]["status"],
        "WP-5 path non-degeneracy (NO single policy covers all slices — V4 FAIL is "
        "desired)": ("PASS" if V["V4"]["status"] == "FAIL"
                     else "FAIL" if V["V4"]["status"] == "PASS" else UNKNOWN),
    }
    rep.table(["certificate", "reading"], [[k, f"**{v}**"] for k, v in wp.items()])
    if V["V1"]["status"] == "FAIL":
        rep.note("**Public conclusion (spec §9.3):** *PCR at sigma=0.9 is ill-posed "
                 "under WP-1; the phase-boundary theory bounded the wrong regime "
                 "(saturation geometry binds before c=1).* That is a publishable "
                 "diagnosis, not a dead end: impossibility measurement + "
                 "certificate + repaired benchmark + (next cycle) a certified "
                 "method on the repaired benchmark is a coherent narrative. The "
                 "E2b frontier is the constructive replacement; the redesign is L6.")

    # ---- bundle ----------------------------------------------------------
    rep.h2("bundle")
    os.makedirs(args.bundle, exist_ok=True)
    copied = 0
    for pat in ("**/*.csv", "**/*.npz", "**/*.md", "**/*.png"):
        for p in glob.glob(os.path.join(args.diag_out, pat), recursive=True):
            rel = os.path.relpath(p, args.diag_out)
            dst = os.path.join(args.bundle, "diag_out", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(p, dst)
                copied += 1
            except OSError:
                pass
    for d in sorted(glob.glob(os.path.join(args.results_root, "**", "config.json"),
                              recursive=True)):
        rd = os.path.dirname(d)
        exp = os.path.basename(os.path.dirname(rd))
        if not exp.startswith("diag_"):
            continue
        for f in ("config.json", "progress.txt", "eval_debug.csv",
                  "diag_telemetry.csv", "diag_qcal.csv", "diag_probes.npz"):
            src = os.path.join(rd, f)
            if os.path.exists(src):
                dst = os.path.join(args.bundle, "runs",
                                   os.path.relpath(rd, args.results_root), f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError:
                    pass
    ck = [{"arm": k, "run_dir": a["run_dir"], "seed": a.get("seed"),
           "cycle_avg": a["cycle_avg"], "source": a["source"]}
          for k, v in arms.items() for a in v]
    with open(os.path.join(args.bundle, "checkpoint_index.json"), "w",
              encoding="utf-8") as f:
        json.dump(ck, f, indent=2)
    rep.kv("files bundled", copied)
    rep.kv("bundle", os.path.abspath(args.bundle))

    out = {"B0": b0, "PC": pc, "target": 0.9 * pc, "axes": V, "leaf": leaf,
           "wp": wp, "arms": {k: [a["cycle_avg"] for a in v] for k, v in arms.items()}}
    with open(os.path.join(args.bundle, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    rep.kv("verdict.json", os.path.join(args.bundle, "verdict.json"))
    rep.close()
    return 0


def _wp3(t0):
    e1 = cells(t0, "e1")
    base = [r for r in e1 if abs(fnum(r["A"])) < 1e-9]
    peak = [r for r in e1 if abs(fnum(r["A"]) - 1.0) < 1e-9]
    if not base or not peak:
        return UNKNOWN
    L0, LA = fnum(base[0]["len_mean"]), fnum(peak[0]["len_mean"])
    R0, RA = fnum(base[0]["ret_mean"]), fnum(peak[0]["ret_mean"])
    r0, rA = R0 / max(L0, 1e-9), RA / max(LA, 1e-9)
    ach, term = (rA - r0) * L0, rA * (LA - L0)
    share = abs(term) / max(abs(ach) + abs(term), 1e-9)
    return "PASS" if share <= 0.5 else "FAIL"


if __name__ == "__main__":
    sys.exit(main())
