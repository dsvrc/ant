"""Resolve campaign artifacts to paths and numbers — the glue the runbook needs.

Every stage after the first depends on something an earlier stage produced, and
those paths are not predictable: a run dir is
``results/mamujoco/Ant-v2-4x2/hasac_diag/diag_<arm>/seed-<NNNNN>-<timestamp>/models``
and the timestamp is wall-clock. Hand-copying them is how you end up evaluating
the wrong checkpoint — the E-5 error one level down — so nothing in this campaign
should ask you to.

    eval "$(python scripts/diag_resolve.py --exports)"     # F0, F1A, F1C, ... set
    python scripts/diag_resolve.py --arm f0                # one path
    python scripts/diag_resolve.py --beta_star             # E2's beta*
    python scripts/diag_resolve.py --pc                    # G's path ceiling
    python scripts/diag_resolve.py --list                  # what exists so far

**It verifies, not just globs.** A checkpoint is only returned if
``models/actor_agent0.pt`` and the run's ``config.json`` exist, and — when a
``run.log`` survives — the run's ``[DIAG ENV]`` banner is checked against the env
vars that arm is *supposed* to have been launched with. A mismatch is reported
loudly rather than returned, because "which env was this checkpoint actually
trained in" is the exact question the whole campaign exists to stop guessing at.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diag_make_configs import ARMS, env_line  # noqa: E402

_SEED_RE = re.compile(r"seed-(\d+)-(.+)$")
_BANNER_RE = re.compile(r"\[DIAG ENV\][^\n]*")
_ARM_BY_ID = {a["id"]: a for a in ARMS}


# ==========================================================================
#  runs
# ==========================================================================
def find_runs(results_root="./results", arm=None, seed=None):
    """Every campaign run on disk, newest first per (arm, seed)."""
    out = []
    for cfg_path in glob.glob(os.path.join(results_root, "**", "config.json"),
                              recursive=True):
        run_dir = os.path.dirname(cfg_path)
        m = _SEED_RE.search(os.path.basename(run_dir))
        if not m:
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        exp = cfg.get("main_args", {}).get("exp_name", "")
        if not exp.startswith("diag_"):
            continue
        a = exp[len("diag_"):].split("_s")[0]
        if arm is not None and a != arm:
            continue
        s = int(m.group(1))
        if seed is not None and s != seed:
            continue
        models = os.path.join(run_dir, "models")
        out.append({
            "arm": a, "seed": s, "ts": m.group(2), "run_dir": run_dir,
            "models": models, "config": cfg,
            "complete": os.path.exists(os.path.join(models, "actor_agent0.pt")),
        })
    out.sort(key=lambda r: (r["arm"], r["seed"], r["ts"]), reverse=True)
    return out


def check_banner(run):
    """Compare the run's [DIAG ENV] banner to the env vars its arm should carry.

    Returns (ok, message). Missing log => (True, "no log") — absence of evidence,
    not a mismatch; but A0 will flag such a run as unclassifiable, which is a
    Prohibition-3 violation in its own right.
    """
    spec = _ARM_BY_ID.get(run["arm"])
    if spec is None:
        return True, "unknown arm (not in the manifest)"
    text = ""
    for p in glob.glob(os.path.join(run["run_dir"], "*.log")) + \
            glob.glob(os.path.join(run["run_dir"], "*.out")):
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text += f.read(2_000_000)
        except OSError:
            pass
    if not text:
        return True, "no log to check the banner against"
    m = _BANNER_RE.search(text)
    if not m:
        return True, "log present but no [DIAG ENV] banner found"
    banner = m.group(0)
    want = spec.get("env", {})
    bad = []
    for key, frag in (("ANT_PCR_FREEZE_A", "FREEZE_A"), ("ANT_PCR_MASK", "MASK"),
                      ("ANT_PCR_ORACLE", "ORACLE")):
        expect = want.get(key)
        got = re.search(rf"{frag}=(\S+)", banner)
        got = got.group(1) if got else "?"
        if expect is None:
            default = {"FREEZE_A": "None", "MASK": "both", "ORACLE": "0"}[frag]
            if got != default:
                bad.append(f"{frag}={got} (expected the default {default})")
        else:
            if got.rstrip("0").rstrip(".") != str(expect).rstrip("0").rstrip("."):
                bad.append(f"{frag}={got} (expected {expect})")
    if bad:
        return False, "banner disagrees with the manifest: " + "; ".join(bad)
    return True, "banner matches the manifest"


def resolve_models(results_root, arm, seed=None, strict=True):
    """The models/ dir of the newest COMPLETE run of ``arm``."""
    runs = [r for r in find_runs(results_root, arm, seed) if r["complete"]]
    if not runs:
        return None
    run = runs[0]
    ok, msg = check_banner(run)
    if not ok:
        m = (f"[resolve] REFUSING {arm}: {msg}\n"
             f"          run: {run['run_dir']}\n"
             f"          This is the E-5 failure one level down: the checkpoint is "
             f"not from the env its arm claims. Re-launch the arm with the right "
             f"env vars, or pass the path explicitly if you know better.")
        if strict:
            print(m, file=sys.stderr)
            return None
        print(m, file=sys.stderr)
    return os.path.abspath(run["models"])


# ==========================================================================
#  numbers
# ==========================================================================
def beta_star(diag_out="./diag_out"):
    """E2's beta*: the beta of the best A=1 cancel cell."""
    best, val = None, -float("inf")
    for p in glob.glob(os.path.join(diag_out, "**", "tier0_cells.csv"),
                       recursive=True):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("stage") != "e2":
                    continue
                try:
                    if abs(float(r["A"]) - 1.0) > 1e-9:
                        continue
                    v = float(r["ret_mean"])
                except (TypeError, ValueError):
                    continue
                if v > val:
                    val, best = v, r["cell"]
    if best is None:
        return None
    return float(str(best).split("_b")[-1])


def path_ceiling(diag_out="./diag_out"):
    """G's PC, from the machine-readable summary crosseval writes."""
    for p in glob.glob(os.path.join(diag_out, "**", "g_summary.json"),
                       recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                return float(json.load(f)["PC"])
        except Exception:
            continue
    return None


def b0(diag_out="./diag_out"):
    """B0: E1's A=0 return."""
    for p in glob.glob(os.path.join(diag_out, "**", "tier0_cells.csv"),
                       recursive=True):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("stage") == "e1":
                    try:
                        if abs(float(r["A"])) < 1e-9:
                            return float(r["ret_mean"])
                    except (TypeError, ValueError):
                        pass
    return None


def e5_best_r2(diag_out="./diag_out"):
    """Best F-loc R^2 at the peak bin with L<=8 on source (a) — abort rule 4."""
    best = None
    for p in glob.glob(os.path.join(diag_out, "**", "e5_r2.csv"), recursive=True):
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        src = [r for r in rows if str(r.get("source", "")).startswith("e1")]
        bins = sorted({r["bin"] for r in src if str(r["bin"]).startswith("bin")})
        if not bins:
            continue
        for r in src:
            if r["bin"] != bins[-1] or r["feature_set"] != "F-loc":
                continue
            try:
                if float(r["L"]) <= 8:
                    v = float(r["r2_cv"])
                    best = v if best is None else max(best, v)
            except (TypeError, ValueError):
                pass
    return best


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default="./results")
    ap.add_argument("--diag_out", default="./diag_out")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--exports", action="store_true",
                    help="print shell `export F0=...` lines for every resolved arm")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--beta_star", action="store_true")
    ap.add_argument("--pc", action="store_true")
    ap.add_argument("--b0", action="store_true")
    ap.add_argument("--no_strict", action="store_true",
                    help="warn on a banner mismatch instead of refusing")
    args = ap.parse_args(argv)

    if args.beta_star:
        v = beta_star(args.diag_out)
        print("" if v is None else v)
        return 0 if v is not None else 1
    if args.pc:
        v = path_ceiling(args.diag_out)
        print("" if v is None else v)
        return 0 if v is not None else 1
    if args.b0:
        v = b0(args.diag_out)
        print("" if v is None else v)
        return 0 if v is not None else 1
    if args.arm:
        p = resolve_models(args.results_root, args.arm, args.seed,
                           strict=not args.no_strict)
        print(p or "")
        return 0 if p else 1
    if args.exports:
        for a in ARMS:
            p = resolve_models(args.results_root, a["id"], strict=not args.no_strict)
            if p:
                print(f'export {a["id"].upper()}="{p}"')
        v = beta_star(args.diag_out)
        if v is not None:
            print(f'export BETA_STAR="{v}"')
        v = path_ceiling(args.diag_out)
        if v is not None:
            print(f'export PC="{v}"')
        return 0
    if args.list or True:
        runs = find_runs(args.results_root, args.arm, args.seed)
        print(f"{'arm':<6} {'seed':<5} {'done':<5} {'banner':<8} run_dir")
        print("-" * 100)
        for r in runs:
            ok, msg = check_banner(r)
            print(f"{r['arm']:<6} {r['seed']:<5} {str(r['complete']):<5} "
                  f"{'ok' if ok else 'MISMATCH':<8} {r['run_dir']}")
            if not ok:
                print(f"       ^ {msg}")
        print()
        for label, fn in (("B0", b0), ("beta*", beta_star), ("PC", path_ceiling),
                          ("E5 best F-loc R^2 (peak, L<=8)", e5_best_r2)):
            v = fn(args.diag_out)
            print(f"{label:<32} {'(not measured yet)' if v is None else v}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
