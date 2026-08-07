"""A0 — forensics on existing artifacts  [campaign spec Part 1]. Zero compute; do first.

Walks the ``results/`` tree and answers, from what is already on disk:

 1. **Run inventory** (``a0_runs.csv``) — algo, exp_name, steps, seed, tag_oracle,
    any ``[ORACLE ARM]`` / ``[DIAG ENV]`` banners found in logs, the ant.py
    SEVERITY if echoed, and whether the eval protocol was de-aliased.
    **Explicitly resolves E-5**: was the "ECL oracle" run the *tag*-oracle
    (perfect tags -> tests the [L]/[A] mechanism) or the env *d*-oracle
    (``ANT_PCR_ORACLE=1`` -> tests existence)? The whole interpretation forks on
    this. Unrecoverable => marked `unknown`, and Tier 2 runs the unambiguous
    replacement (D2) anyway.
 2. **v2 identifier re-verification** — from ``ecl_debug.csv``:
    corr(``c_now``, ``pcr_payload``) whole-run and per-cycle, the ``c_max_seen``
    trajectory, ``lock_gain`` / ``clip_frac`` / ``tag_payload_corr``. Confirms or
    refutes E-4 ("identifier locked yet return did not improve") **with numbers**.
 3. **Per-phase re-slicing of every prior run** (``a0_phase_corrected.csv``) —
    cycle-average, trough-decile and peak-quintile returns. Where
    ``eval_debug.csv`` exists (v2 protocol) the payload is read directly. Where
    only ``progress.txt`` exists (v1, baselines) the phase of each eval row is
    **reconstructed** from the known lockstep clock and flagged as such
    (Prohibition 7: reconstructed numbers always carry the flag).
 4. **Checkpoint inventory** — every saved model dir; flags stationary-walker
    candidates for F0.
 5. ``a0_summary.md`` — blind / v1 / v2 / oracle side by side, and the resolved
    (or unresolved) identity of E-5.

Why this matters (spec §1): if A0 shows v2's cycle-average actually *matched* v1
while its trough slice improved, or that the "oracle" arm was the tag-oracle
rather than the d-oracle, the campaign's priorities shift (§10.2 abort rule 2:
a d-oracle at sigma=0.9 / 10M / de-aliased that already failed means D2 is
answered — run a 3M confirmation instead and promote F2/F2b to double seeds).
**Do not skip.**

Pandas-free (spec Part 1). Stdlib + numpy.

    python scripts/diag_a0_inventory.py --results_root ./results --out ./diag_out/a0
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harl.envs.mamujoco.diag.report_io import DebugReport, write_csv  # noqa: E402

_P = 40000          # PCR payload period, in env steps
_B = 0.2

_BANNER_RE = re.compile(r"\[(DIAG ENV|ORACLE ARM|DIAG ARM|DIAG RUN)\][^\n]*")
_SEV_RE = re.compile(r"SEVERITY=([0-9.]+)")
_ORACLE_RE = re.compile(r"ORACLE=(\d)")
_FREEZE_RE = re.compile(r"FREEZE_A=([^\s]+)")
_LOG_GLOBS = ("*.log", "*.out", "*.txt", "../*.log", "../*.out",
              "../../*.log", "../../*.out")


def payload_at(clock):
    """A(t) — must match ant.py exactly."""
    ph = (clock % _P) / _P
    x = ph / _B if ph < _B else (1.0 - ph) / (1.0 - _B)
    return x * x * (3.0 - 2.0 * x)


def read_csv_dict(path):
    """Tiny CSV reader -> dict of float arrays (strings kept where unparsable)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    hdr, body = rows[0], rows[1:]
    out = {}
    for i, h in enumerate(hdr):
        col = [r[i] if i < len(r) else "" for r in body]
        try:
            out[h] = np.array([float(c) if c != "" else np.nan for c in col])
        except ValueError:
            out[h] = np.array(col, dtype=object)
    return out


def find_logs(run_dir):
    txt = []
    for g in _LOG_GLOBS:
        for p in glob.glob(os.path.join(run_dir, g)):
            if os.path.basename(p) == "progress.txt":
                continue
            try:
                if os.path.getsize(p) > 40 * 1024 * 1024:
                    continue
                with open(p, encoding="utf-8", errors="replace") as f:
                    txt.append(f.read())
            except OSError:
                pass
    return "\n".join(txt)


# ==========================================================================
#  1. inventory
# ==========================================================================
def inventory(results_root, rep):
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(results_root, "**", "config.json"),
                                     recursive=True)):
        run_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            rep.line(f"  ! unreadable config: {cfg_path} ({e})")
            continue
        ma = cfg.get("main_args", {})
        aa = cfg.get("algo_args", {})
        ea = cfg.get("env_args", {})
        ecl_cfg = ea.get("ecl_cfg", {}) or {}

        logs = find_logs(run_dir)
        banners = _BANNER_RE.findall(logs) if logs else []
        banner_lines = sorted(set(
            m.group(0) for m in _BANNER_RE.finditer(logs))) if logs else []
        sev = _SEV_RE.search(logs).group(1) if logs and _SEV_RE.search(logs) else ""
        env_oracle = ""
        if logs:
            m = _ORACLE_RE.search(logs)
            if m:
                env_oracle = m.group(1)
        freeze = _FREEZE_RE.search(logs).group(1) if logs and _FREEZE_RE.search(logs) else ""

        # eval protocol: was the payload clock de-aliased?
        dephase = "?"
        if ea.get("ecl", False):
            dephase = str(bool(ecl_cfg.get("eval_dephase", True)))
        elif "pcr_eval_dephase" in ea:
            dephase = str(bool(ea["pcr_eval_dephase"]))
        elif os.path.exists(os.path.join(run_dir, "eval_debug.csv")):
            dephase = "likely-True(v2 artifact present)"
        else:
            dephase = "False/unknown(v1 protocol)"

        # ---- E-5 resolution ------------------------------------------------
        tag_oracle = bool(ecl_cfg.get("tag_oracle", False))
        has_oracle_banner = any("[ORACLE ARM]" in b for b in banner_lines)
        if tag_oracle or has_oracle_banner:
            oracle_kind = "tag_oracle (perfect TAGS -> tests the [L]/[A] mechanism)"
        elif env_oracle == "1":
            oracle_kind = "d_oracle (ANT_PCR_ORACLE=1 -> tests EXISTENCE)"
        elif logs and env_oracle == "0":
            oracle_kind = "blind (banner says ORACLE=0)"
        else:
            oracle_kind = "unknown (no banner; env-var status unrecoverable)"

        runs.append({
            "run_dir": os.path.relpath(run_dir, results_root),
            "algo": ma.get("algo", ""),
            "exp_name": ma.get("exp_name", ""),
            "num_env_steps": aa.get("train", {}).get("num_env_steps", ""),
            "seed": aa.get("seed", {}).get("seed", ""),
            "model_dir": aa.get("train", {}).get("model_dir", "") or "",
            "scenario": f"{ea.get('scenario', '')}-{ea.get('agent_conf', '')}",
            "tag_oracle": tag_oracle,
            "env_oracle_banner": env_oracle,
            "oracle_kind": oracle_kind,
            "severity_echoed": sev,
            "freeze_a_echoed": freeze,
            "eval_dephase": dephase,
            "banners": " | ".join(banner_lines)[:400],
            "has_eval_debug": os.path.exists(os.path.join(run_dir, "eval_debug.csv")),
            "has_ecl_debug": os.path.exists(os.path.join(run_dir, "ecl_debug.csv")),
            "has_progress": os.path.exists(os.path.join(run_dir, "progress.txt")),
            "has_models": os.path.isdir(os.path.join(run_dir, "models")),
            "_abs": run_dir,
        })
    return runs


# ==========================================================================
#  2. v2 identifier re-verification (E-4)
# ==========================================================================
def verify_identifier(run, rep):
    path = os.path.join(run["_abs"], "ecl_debug.csv")
    d = read_csv_dict(path)
    if not d or "c_now" not in d or "pcr_payload" not in d:
        return None
    c, p = d["c_now"], d["pcr_payload"]
    ok = np.isfinite(c) & np.isfinite(p)
    if ok.sum() < 10:
        return None
    corr = float(np.corrcoef(c[ok], p[ok])[0, 1])
    # per-cycle: env_step advances by n_rollout_threads per env step, and the
    # payload period is _P *env* steps per env, so a cycle is _P * threads on the
    # total-step axis. Use the payload's own zero-crossings instead of assuming
    # the thread count — robust to a mis-recorded config.
    step = d.get("env_step", np.arange(len(c)))
    cyc = []
    if ok.sum() > 50:
        lo = p[ok] < 0.05
        edges = np.flatnonzero(np.diff(lo.astype(int)) == 1)
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < 10:
                continue
            cc, pp = c[ok][a:b], p[ok][a:b]
            if np.std(cc) > 1e-9 and np.std(pp) > 1e-9:
                cyc.append(float(np.corrcoef(cc, pp)[0, 1]))
    out = {
        "corr_whole": corr,
        "corr_per_cycle_mean": float(np.mean(cyc)) if cyc else float("nan"),
        "corr_per_cycle_min": float(np.min(cyc)) if cyc else float("nan"),
        "n_cycles": len(cyc),
        "c_max_seen_final": float(d["c_max_seen"][ok][-1]) if "c_max_seen" in d else float("nan"),
        "c_now_final": float(c[ok][-1]),
    }
    for k in ("lock_gain", "clip_frac", "tag_payload_corr", "trough_frac",
              "anchor_payload"):
        if k in d:
            v = d[k][np.isfinite(d[k])]
            out[k + "_median"] = float(np.median(v)) if v.size else float("nan")
    out["step_range"] = f"{float(np.nanmin(step)):.0f}..{float(np.nanmax(step)):.0f}"
    return out


# ==========================================================================
#  3. per-phase re-slicing
# ==========================================================================
def slices_from_eval_debug(run):
    """v2 protocol: the payload is logged per episode — no reconstruction needed."""
    d = read_csv_dict(os.path.join(run["_abs"], "eval_debug.csv"))
    if not d or "ep_return" not in d or "payload_end" not in d:
        return None
    r, p, s = d["ep_return"], d["payload_end"], d.get("step", np.zeros(len(d["ep_return"])))
    ok = np.isfinite(r) & np.isfinite(p)
    if ok.sum() < 10:
        return None
    r, p, s = r[ok], p[ok], s[ok]
    # report over the LAST cycle's worth of eval rounds (the converged estimate)
    last = np.unique(s)[-max(1, len(np.unique(s)) // 5):]
    m = np.isin(s, last)
    return _slice_stats(r[m], p[m], reconstructed=False,
                        note=f"last {len(last)} eval round(s) of {len(np.unique(s))}")


def slices_from_progress(run):
    """v1 protocol: only (step, eval_avg_rew, eval_avg_len) exists.

    Reconstruct each eval round's phase from the lockstep clock (the v2-spec F6
    analysis). The eval envs are built once and their payload clock persists
    across rounds, advancing only while eval is running. A round steps every
    thread in lockstep until `eval_episodes` are done, i.e.
    ``ceil(eval_episodes / n_eval_threads)`` episodes each, so

        steps_in_round(r) ~= ceil(E/T) * avg_len(r)
        eval_clock(r)      = sum over rounds <= r

    and the round's payload is ``A(eval_clock(r) mod P)``. Approximate — a round
    sweeps ~2*avg_len ~ 2000 steps = 5% of the period, which is exactly the
    aliasing this reconstruction exposes: the v1 protocol measured ONE phase per
    round, and which phase it was drifted slowly across rounds. Always flagged
    `reconstructed=True` (Prohibition 7).
    """
    path = os.path.join(run["_abs"], "progress.txt")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    if len(rows) < 5:
        return None
    steps = np.array([x[0] for x in rows])
    rets = np.array([x[1] for x in rows])
    lens = np.array([x[2] for x in rows])
    cfg_path = os.path.join(run["_abs"], "config.json")
    eps_per_thread = 2
    try:
        with open(cfg_path, encoding="utf-8") as f:
            c = json.load(f)
        E = int(c["algo_args"]["eval"]["eval_episodes"])
        T = int(c["algo_args"]["eval"]["n_eval_rollout_threads"])
        eps_per_thread = max(1, int(np.ceil(E / max(T, 1))))
    except Exception:
        pass
    clock = np.cumsum(eps_per_thread * lens)
    pay = np.array([payload_at(cl) for cl in clock])
    return _slice_stats(rets, pay, reconstructed=True,
                        note=f"phase reconstructed from the lockstep clock "
                             f"({eps_per_thread} ep/thread/round, "
                             f"{len(rows)} rounds, "
                             f"{clock[-1] / _P:.1f} payload cycles swept)")


def _slice_stats(r, p, reconstructed, note):
    lo, hi = np.quantile(p, 0.1), np.quantile(p, 0.8)
    trough = float(np.mean(r[p <= lo])) if np.any(p <= lo) else float("nan")
    peak = float(np.mean(r[p >= hi])) if np.any(p >= hi) else float("nan")
    return {"cycle_avg": float(np.mean(r)), "trough_decile": trough,
            "peak_quintile": peak, "n": int(r.size),
            "payload_span": f"{p.min():.2f}..{p.max():.2f}",
            "reconstructed": reconstructed, "note": note}


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="A0 forensics on the results tree.")
    ap.add_argument("--results_root", default="./results")
    ap.add_argument("--out", default="./diag_out/a0")
    args = ap.parse_args(argv)

    rep = DebugReport(os.path.join(args.out, "a0_summary.md"),
                      title="A0 — forensics on existing artifacts",
                      subtitle="zero compute; resolves E-5 and re-slices every "
                               "prior run by phase")

    rep.h2("1. run inventory")
    runs = inventory(args.results_root, rep)
    rep.kv("results root", os.path.abspath(args.results_root))
    rep.kv("runs found", len(runs))
    if not runs:
        rep.note("No runs found. If the results tree lives elsewhere, pass "
                 "--results_root. A0 cannot resolve E-5 without it, and §10.2's "
                 "abort rule 2 then cannot fire — D2 must run at full 10M.")
        rep.close()
        return 0

    cols = ["run_dir", "algo", "exp_name", "num_env_steps", "seed", "model_dir",
            "scenario", "tag_oracle", "env_oracle_banner", "oracle_kind",
            "severity_echoed", "freeze_a_echoed", "eval_dephase", "banners",
            "has_eval_debug", "has_ecl_debug", "has_progress", "has_models"]
    write_csv(os.path.join(args.out, "a0_runs.csv"), cols,
              [[r[c] for c in cols] for r in runs])
    rep.kv("inventory csv", os.path.join(args.out, "a0_runs.csv"))
    rep.table(["algo", "exp_name", "steps", "seed", "oracle kind", "eval protocol"],
              [[r["algo"], r["exp_name"], r["num_env_steps"], r["seed"],
                r["oracle_kind"], r["eval_dephase"]] for r in runs])

    # ---- E-5 ------------------------------------------------------------
    rep.h2("2. E-5 resolution — what was the 'ECL oracle' run?")
    oracle_runs = [r for r in runs
                   if r["tag_oracle"] or r["env_oracle_banner"] == "1"
                   or "oracle" in (r["exp_name"] or "").lower()
                   or r["algo"] == "oracle"]
    if not oracle_runs:
        rep.line("  No oracle-flavoured run found in this tree.")
        rep.note("E-5 stays UNRESOLVED. D2 (d-oracle, drifting, from scratch, 10M, "
                 "instrumented, banner-labeled) runs at full budget — it is the "
                 "unambiguous replacement the campaign was designed to provide.")
    else:
        rep.table(["run_dir", "algo", "exp_name", "tag_oracle", "ORACLE banner",
                   "verdict"],
                  [[r["run_dir"], r["algo"], r["exp_name"], r["tag_oracle"],
                    r["env_oracle_banner"] or "-", r["oracle_kind"]]
                   for r in oracle_runs])
        kinds = {r["oracle_kind"].split(" ")[0] for r in oracle_runs}
        if "unknown" in kinds:
            rep.note("At least one oracle-flavoured run is UNCLASSIFIABLE: no "
                     "banner survived, so whether it tested the [L]/[A] mechanism "
                     "(tag_oracle) or EXISTENCE (d_oracle) cannot be recovered. "
                     "This is precisely the ambiguity the campaign's mandatory "
                     "banners (Prohibition 3) exist to prevent. Treat E-5 as "
                     "unresolved and run D2.")
        if any(k.startswith("d_oracle") for k in kinds):
            rep.note("ABORT RULE 2 (spec §10.2) MAY APPLY: a d-oracle run exists. "
                     "If it was at sigma=0.9, 10M, de-aliased, and it failed, then "
                     "D2 is already answered — run a 3M confirmation with telemetry "
                     "instead, and promote F2/F2b to double seeds (the "
                     "info-toxicity question becomes central). Check its severity, "
                     "steps and eval protocol in the table above before deciding.")

    # ---- identifier ------------------------------------------------------
    rep.h2("3. E-4 re-verification — did the v2 identifier actually lock?")
    id_rows = []
    for r in runs:
        if not r["has_ecl_debug"]:
            continue
        v = verify_identifier(r, rep)
        if v is None:
            continue
        id_rows.append([r["run_dir"], r["exp_name"],
                        f"{v['corr_whole']:.3f}",
                        f"{v['corr_per_cycle_mean']:.3f}",
                        f"{v['corr_per_cycle_min']:.3f}", v["n_cycles"],
                        f"{v['c_now_final']:.3f}", f"{v['c_max_seen_final']:.3f}",
                        f"{v.get('lock_gain_median', float('nan')):.3f}",
                        f"{v.get('tag_payload_corr_median', float('nan')):.3f}",
                        f"{v.get('trough_frac_median', float('nan')):.3f}"])
    if id_rows:
        rep.table(["run_dir", "exp", "corr(c,payload) whole", "per-cycle mean",
                   "per-cycle min", "#cycles", "c_now final", "c_max_seen",
                   "lock_gain med", "tag_payload_corr med", "trough_frac med"],
                  id_rows)
        rep.note("E-4 says the v2 identifier locked (near-1 payload correlation) "
                 "yet the return did not improve. The whole-run correlation above "
                 "is the claim's evidence; the PER-CYCLE numbers are the honest "
                 "test — a high whole-run correlation can come entirely from the "
                 "shared slow trend while the within-cycle tracking is poor. If "
                 "per-cycle mean << whole, E-4 is REFUTED and the identifier never "
                 "really locked.")
    else:
        rep.line("  No ecl_debug.csv found — E-4 cannot be re-verified from disk.")

    # ---- phase correction ------------------------------------------------
    rep.h2("4. per-phase re-slicing of every prior run")
    pc_rows = []
    for r in runs:
        s = None
        if r["has_eval_debug"]:
            s = slices_from_eval_debug(r)
        if s is None and r["has_progress"]:
            s = slices_from_progress(r)
        if s is None:
            continue
        pc_rows.append([r["run_dir"], r["algo"], r["exp_name"], r["oracle_kind"],
                        f"{s['cycle_avg']:.1f}", f"{s['trough_decile']:.1f}",
                        f"{s['peak_quintile']:.1f}", s["n"], s["payload_span"],
                        s["reconstructed"], s["note"]])
    write_csv(os.path.join(args.out, "a0_phase_corrected.csv"),
              ["run_dir", "algo", "exp_name", "oracle_kind", "cycle_avg",
               "trough_decile", "peak_quintile", "n", "payload_span",
               "reconstructed", "note"], pc_rows)
    if pc_rows:
        rep.table(["run", "algo", "exp", "cycle-avg", "trough-decile",
                   "peak-quintile", "n", "reconstructed?"],
                  [[x[0], x[1], x[2], x[4], x[5], x[6], x[7], x[9]]
                   for x in pc_rows])
    rep.note("Rows with reconstructed=True carry a phase inferred from the "
             "lockstep clock, not a logged payload. Never compare a reconstructed "
             "number to a measured one inside one figure (Prohibition 7); the "
             "reconstruction exists to show whether H-D1 (phase-aliased "
             "evaluation) can explain a prior verdict, not to replace measurement.")

    # ---- checkpoints -----------------------------------------------------
    rep.h2("5. checkpoint inventory — is there an F0 already?")
    ck_rows = []
    for r in runs:
        if not r["has_models"]:
            continue
        md = os.path.join(r["_abs"], "models")
        n_actors = len(glob.glob(os.path.join(md, "actor_agent*.pt")))
        stationary = ("off" in (r["freeze_a_echoed"] or "")
                      or r["severity_echoed"] in ("0", "0.0")
                      or "MASK=off" in (r["banners"] or "")
                      or r["freeze_a_echoed"] == "0.0")
        ck_rows.append([os.path.relpath(md, args.results_root), r["algo"],
                        r["exp_name"], r["num_env_steps"], n_actors,
                        "YES" if stationary else "no/unknown",
                        r["banners"][:120]])
    write_csv(os.path.join(args.out, "a0_checkpoints.csv"),
              ["models_dir", "algo", "exp_name", "steps", "n_actors",
               "stationary_walker_candidate", "banners"], ck_rows)
    if ck_rows:
        rep.table(["models_dir", "algo", "exp", "steps", "#actors",
                   "stationary candidate?"],
                  [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in ck_rows])
    cands = [x for x in ck_rows if x[5] == "YES" and x[1] in ("hasac", "hasac_diag")]
    if cands:
        rep.note(f"{len(cands)} HASAC stationary-walker candidate(s) found — Tier 0 "
                 f"can start immediately with one as F0 (spec Part 4 preamble). "
                 f"CONFIRM FROM THE BANNER, not the exp_name: mistaking which env a "
                 f"checkpoint was trained in is exactly the E-5 failure, one level "
                 f"down. If no banner survives, train F0 (5M).")
    else:
        rep.note("No confirmable SEVERITY=0 / MASK=off HASAC walker. **F0 must be "
                 "trained** (5M, spec Part 5) before Tier 0 can run — every Tier-0 "
                 "probe needs it, and B0 (its stationary return) is the denominator "
                 "of gates V1, V2, V4 and V7.")

    rep.h2("SUMMARY — the two load-bearing unknowns (spec §0.1)")
    rep.line("  (a) has EXISTENCE at sigma=0.9 ever actually been demonstrated?")
    rep.line("      -> E-6 says no, only theorized. A0 cannot settle it; **E2** "
             "(scripted privileged cancellation) is the experiment of record.")
    rep.line("  (b) what exactly failed in v2 (E-4 / E-5)?")
    rep.line("      -> see sections 2 and 3 above.")
    rep.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
