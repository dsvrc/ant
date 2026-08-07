"""Run the whole PCR diagnosis campaign unattended  [spec Part 10].

    nohup python -u scripts/diag_campaign.py --results_root ./results > campaign.out 2>&1 &
    # come back later:
    cat diag_out/_campaign/STATUS.md

This is **not** a list of commands. The campaign's ordering and abort rules
(spec §10.2) are the difference between 45M useful steps and 45M wasted ones, so
this driver implements them and **branches**:

* **V0 fail => hard stop.** `ant_diag.py` defaults must pass the golden test
  before any deployment (Prohibition 5). Nothing downstream means anything if the
  env is not the env.
* **Ordering rule 1 — E2 fail (V1 fail) => skip F1c/F2/F2b/F2c/F3a/F3b/D2 at
  sigma=0.9 ENTIRELY.** The campaign moves to the repaired-env track: run E2b to
  locate sigma*, and stop for the redesign decision (R-a or R-b), which is a
  human call (§9.2). Blindly training the peak arms after a V1 fail burns ~30M
  steps answering a question the campaign has already declared ill-posed.
* **Abort rule 3 — F2 fail => cap D2 at 3M.**
* **Abort rule 4 — E5 R^2 < 0.3 on source (a) => skip E3-DOB**, log V6 fail early.
* **Abort rule 2 — A0 finds a d-oracle run** => flagged in STATUS.md for a human
  call (is D2 already answered?). Deliberately NOT automated: it turns on whether
  that run *failed*, which is a judgement about a prior experiment, not a number
  this script can read.

Everything is resumable: each stage writes a marker, and a re-launch skips what
is already done. Each stage's stdout is tee'd to its own log. `STATUS.md` is
rewritten after every stage — that one file is what you read when you come back.

    --dry_run            print the plan and exit (do this first)
    --priority P0        P0 only (~45M steps); default P1 (~65M); P2 adds F4b
    --jobs N             parallelise the independent frozen arms (GPU memory!)
    --f0 <models>        use an inherited stationary walker instead of training one
    --only / --skip      stage-name filters
    --force              re-run stages that are already marked done
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import diag_resolve as R  # noqa: E402
from scripts.diag_make_configs import ARMS  # noqa: E402

_ARM = {a["id"]: a for a in ARMS}
_PRIO_RANK = {"P0": 0, "P1": 1, "P2": 2}


# ==========================================================================
#  plumbing
# ==========================================================================
class Campaign:
    def __init__(self, args):
        self.args = args
        self.diag = args.diag_out
        self.results = args.results_root
        self.state = os.path.join(self.diag, "_campaign")
        self.logs = os.path.join(self.state, "logs")
        os.makedirs(self.logs, exist_ok=True)
        self.status_path = os.path.join(self.state, "STATUS.md")
        self.history = []
        self.notes = []
        self.t0 = time.time()
        self.branch = "nominal"

    # -- markers -------------------------------------------------------
    def done_path(self, name):
        return os.path.join(self.state, f"{name}.done")

    def is_done(self, name):
        return os.path.exists(self.done_path(name)) and not self.args.force

    def mark(self, name, info):
        with open(self.done_path(name), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)

    def selected(self, name, prio="P0"):
        if self.args.only and name not in self.args.only:
            return False
        if name in self.args.skip:
            return False
        return _PRIO_RANK[prio] <= _PRIO_RANK[self.args.priority]

    # -- running -------------------------------------------------------
    def run(self, name, argv, env=None, prio="P0", critical=False, cwd=None):
        """Run one stage. Returns (ok, rc). Skips if filtered or already done."""
        if not self.selected(name, prio):
            self._record(name, "skipped", 0.0, "filtered out (--only/--skip/priority)")
            return True, 0
        if self.is_done(name):
            self._record(name, "cached", 0.0, "already done (delete its .done to redo)")
            return True, 0
        line = " ".join(shlex.quote(str(a)) for a in argv)
        envs = " ".join(f"{k}={v}" for k, v in (env or {}).items())
        pretty = (envs + " " if envs else "") + line
        if self.args.dry_run:
            self._record(name, "dry-run", 0.0, pretty)
            return True, 0
        log = os.path.join(self.logs, f"{name}.log")
        t = time.time()
        self._say(f"[{name}] START  {pretty}")
        full_env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"# {pretty}\n# started {datetime.now()}\n\n")
            fh.flush()
            rc = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT,
                                 env=full_env, cwd=cwd)
        dt = time.time() - t
        ok = rc == 0
        self._record(name, "ok" if ok else f"FAILED (rc={rc})", dt, pretty, log)
        self._say(f"[{name}] {'OK' if ok else 'FAILED rc=' + str(rc)}  "
                  f"({timedelta(seconds=int(dt))})  log: {log}")
        if ok:
            self.mark(name, {"argv": argv, "env": env, "seconds": dt, "log": log})
        elif critical:
            self.note(f"**{name} FAILED and is critical — campaign stopped.** "
                      f"See {log}.")
            self.write_status()
            sys.exit(1)
        return ok, rc

    def run_parallel(self, jobs, prio="P0"):
        """jobs: list of (name, argv, env). Runs up to --jobs at a time."""
        pending = [(n, a, e) for (n, a, e) in jobs
                   if self.selected(n, prio) and not self.is_done(n)]
        for (n, a, e) in jobs:
            if not self.selected(n, prio):
                self._record(n, "skipped", 0.0, "filtered out")
            elif self.is_done(n):
                self._record(n, "cached", 0.0, "already done")
        if self.args.dry_run:
            for (n, a, e) in pending:
                envs = " ".join(f"{k}={v}" for k, v in (e or {}).items())
                self._record(n, "dry-run", 0.0,
                             (envs + " " if envs else "")
                             + " ".join(shlex.quote(str(x)) for x in a))
            return
        running = []
        queue = list(pending)
        while queue or running:
            while queue and len(running) < max(1, self.args.jobs):
                n, a, e = queue.pop(0)
                log = os.path.join(self.logs, f"{n}.log")
                fh = open(log, "w", encoding="utf-8")
                envs = " ".join(f"{k}={v}" for k, v in (e or {}).items())
                pretty = ((envs + " " if envs else "")
                          + " ".join(shlex.quote(str(x)) for x in a))
                fh.write(f"# {pretty}\n# started {datetime.now()}\n\n")
                fh.flush()
                self._say(f"[{n}] START  {pretty}")
                p = subprocess.Popen(a, stdout=fh, stderr=subprocess.STDOUT,
                                     env={**os.environ,
                                          **{k: str(v) for k, v in (e or {}).items()}})
                running.append({"n": n, "p": p, "fh": fh, "t": time.time(),
                                "argv": a, "env": e, "log": log, "pretty": pretty})
            time.sleep(5)
            for r in list(running):
                rc = r["p"].poll()
                if rc is None:
                    continue
                running.remove(r)
                r["fh"].close()
                dt = time.time() - r["t"]
                ok = rc == 0
                self._record(r["n"], "ok" if ok else f"FAILED (rc={rc})", dt,
                             r["pretty"], r["log"])
                self._say(f"[{r['n']}] {'OK' if ok else 'FAILED rc=' + str(rc)}  "
                          f"({timedelta(seconds=int(dt))})")
                if ok:
                    self.mark(r["n"], {"argv": r["argv"], "env": r["env"],
                                       "seconds": dt, "log": r["log"]})
                self.write_status()

    # -- reporting -----------------------------------------------------
    def _say(self, s):
        print(f"{datetime.now():%H:%M:%S} {s}", flush=True)

    def _record(self, name, status, seconds, detail, log=None):
        self.history.append({"stage": name, "status": status, "seconds": seconds,
                             "detail": detail, "log": log,
                             "when": datetime.now().isoformat(timespec="seconds")})
        self.write_status()

    def note(self, s):
        self.notes.append(s)
        self._say("NOTE: " + s.replace("**", ""))
        self.write_status()

    def write_status(self):
        el = timedelta(seconds=int(time.time() - self.t0))
        L = ["# PCR diagnosis campaign — STATUS", "",
             f"`updated {datetime.now():%Y-%m-%d %H:%M:%S}`  ",
             f"`elapsed {el}`  `branch: {self.branch}`  ",
             f"`priority {self.args.priority}`  `jobs {self.args.jobs}`"
             + ("  **DRY RUN**" if self.args.dry_run else ""), ""]
        if self.notes:
            L += ["## Decisions and warnings", ""]
            L += [f"* {n}" for n in self.notes] + [""]
        L += ["## Stages", "",
              "| stage | status | elapsed | detail |", "|---|---|---|---|"]
        for h in self.history:
            d = h["detail"] if len(h["detail"]) < 150 else h["detail"][:147] + "..."
            L.append(f"| `{h['stage']}` | {h['status']} | "
                     f"{timedelta(seconds=int(h['seconds']))} | `{d}` |")
        L += ["", "## Measured so far", ""]
        for label, fn in (("B0 (stationary return)", R.b0),
                          ("beta* (E2)", R.beta_star),
                          ("PC (path ceiling)", R.path_ceiling),
                          ("E5 best F-loc R^2 (peak, L<=8)", R.e5_best_r2)):
            v = fn(self.diag)
            L.append(f"* **{label}**: {'—' if v is None else round(v, 4)}")
        L += ["", "## Checkpoints", ""]
        for a in ARMS:
            p = R.resolve_models(self.results, a["id"], strict=False)
            if p:
                L.append(f"* `{a['id']}` → `{p}`")
        L += ["", "---", "",
              "Logs: `" + self.logs + "/<stage>.log`  ",
              "Verdict (once the report stage runs): `diag_bundle/verdict.md`"]
        with open(self.status_path, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")


# ==========================================================================
#  the campaign
# ==========================================================================
def py(*a):
    return [sys.executable, "-u"] + [str(x) for x in a]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_out", default="./diag_out")
    ap.add_argument("--results_root", default="./results")
    ap.add_argument("--bundle", default="./diag_bundle")
    ap.add_argument("--priority", default="P1", choices=("P0", "P1", "P2"))
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel training arms. Each is a full HASAC run — raise "
                         "only if the GPU has room.")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--f0", default=None,
                    help="inherited stationary walker (models/ dir); skips training F0")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--device", default="cpu", help="Tier-0 eval device")
    args = ap.parse_args(argv)
    args.only = set(x for x in args.only.split(",") if x)
    args.skip = set(x for x in args.skip.split(",") if x)

    # Every stage's argv uses repo-relative paths, so anchor there rather than
    # trusting the launch dir (`nohup ... &` from anywhere must work).
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    C = Campaign(args)
    D = args.diag_out
    C._say(f"campaign -> {os.path.abspath(C.state)}  (STATUS.md is the file to read)")
    # Each arm declares its own ANT_PCR_* vars; a stray inherited one would
    # silently apply to every arm that did not override it (and would fail the
    # golden test outright). Clear them once, here.
    for k in ("ANT_PCR_SEVERITY", "ANT_PCR_FREEZE_A", "ANT_PCR_MASK",
              "ANT_PCR_DCAP", "ANT_PCR_ORACLE", "ANT_PCR_CORACLE"):
        if os.environ.pop(k, None) is not None:
            C.note(f"cleared inherited `{k}` from the environment — each arm sets "
                   f"its own (and the golden test requires a clean env).")

    # ---- 0. configs --------------------------------------------------
    C.run("configs", py("scripts/diag_make_configs.py"), critical=True)
    C.run("configs_verify", py("scripts/diag_make_configs.py", "--verify"),
          critical=True)

    # ---- 1. self-tests (no simulator) --------------------------------
    for n, mod in (("v0_probes", "harl.envs.mamujoco.diag.probes"),
                   ("v0_sysid", "harl.envs.mamujoco.diag.sysid")):
        C.run(n, py("-m", mod, "--selftest"), critical=True)
    C.run("v0_recorder", py("-m", "harl.envs.mamujoco.pcr_diag"), critical=True)
    C.run("v0_runner", py("-m", "harl.runners.off_policy_diag_runner"), critical=True)

    # ---- 2. deploy + golden ------------------------------------------
    # The golden test REQUIRES a clean environment: it asserts ant_diag with no
    # ANT_PCR_* set is byte-identical to the frozen ant_pcr_v1.
    if not args.dry_run and not C.is_done("v0_golden"):
        try:
            import gym.envs.mujoco.ant as _ant
            deployed = os.path.abspath(_ant.__file__)
            src = os.path.abspath("harl/envs/mamujoco/diag/ant_diag.py")
            if not hasattr(_ant, "set_freeze_a"):
                C.note(f"`ant_diag.py` is **not deployed** — `{deployed}` is some "
                       f"other ant. Deploy it and re-launch:\n"
                       f"      cp \"{src}\" \"{deployed}\"\n"
                       f"    (back the old one up first). Refusing to run the "
                       f"campaign against an unknown env.")
                C.write_status()
                return 1
            C.note(f"deployed env confirmed: `{deployed}` is ant_diag.")
        except Exception as e:
            C.note(f"could not import gym's ant module ({e!r}) — cannot verify the "
                   f"deployment. Stopping.")
            C.write_status()
            return 1
    C.run("v0_golden", py("-m", "harl.envs.mamujoco.diag.test_ant_diag"),
          critical=True)

    # ---- 3. A0 -------------------------------------------------------
    C.run("a0", py("scripts/diag_a0_inventory.py", "--results_root",
                   args.results_root, "--out", f"{D}/a0"))
    a0_sum = os.path.join(D, "a0", "a0_summary.md")
    if os.path.exists(a0_sum):
        txt = open(a0_sum, encoding="utf-8", errors="replace").read()
        if "ABORT RULE 2" in txt:
            C.note("**A0 found a d-oracle run — abort rule 2 MAY apply.** If it was "
                   "at sigma=0.9, 10M, de-aliased, and it FAILED, then D2 is already "
                   "answered: re-launch with `--skip d2_s1,d2_s2` and run a 3M "
                   "confirmation instead, and promote F2/F2b to double seeds. "
                   "Not automated: it turns on whether that run failed, which is a "
                   "judgement about a prior experiment, not a number. See "
                   f"`{a0_sum}`. **Proceeding with the full D2 for now.**")

    # ---- 4. F0 -------------------------------------------------------
    f0 = args.f0 or R.resolve_models(args.results_root, "f0", strict=False)
    if f0:
        C.note(f"F0 = `{f0}`" + (" (supplied via --f0)" if args.f0 else
                                 " (resolved from a previous run)"))
    else:
        C.run("f0", py("examples/train.py", "--load_config",
                       "tuned_configs/mamujoco/Ant-v2-4x2/diag/f0.json",
                       "--exp_name", "diag_f0", "--seed", 1),
              env={"ANT_PCR_MASK": "off"}, critical=True)
        f0 = R.resolve_models(args.results_root, "f0", strict=False)
    if not f0 and not args.dry_run:
        C.note("**No F0 and training it did not produce one — stopping.** Every "
               "Tier-0 probe needs it, and B0 is the denominator of V1/V2/V4/V7.")
        C.write_status()
        return 1
    f0 = f0 or "<F0>"

    # ---- 5. Tier 0 ---------------------------------------------------
    t0 = ["--ckpt", f0, "--episodes", args.episodes, "--device", args.device]
    C.run("e1", py("scripts/diag_tier0.py", *t0, "--stage", "e1", "--out", f"{D}/e1"))
    C.run("d0", py("scripts/diag_tier0.py", *t0, "--stage", "d0", "--out", f"{D}/d0"))
    C.run("e2", py("scripts/diag_tier0.py", *t0, "--stage", "e2", "--out", f"{D}/e2"),
          critical=True)

    # ---- GATE: V1 (ordering rule 1) ----------------------------------
    b0 = R.b0(D)
    beta = R.beta_star(D)
    if beta is None and args.dry_run:
        beta = "<BETA*>"      # so --dry_run prints the FULL plan, not a truncated one
    v1_pass = None
    e2_csv = os.path.join(D, "e2", "tier0_cells.csv")
    if b0 is not None and not args.dry_run and os.path.exists(e2_csv):
        import csv as _csv
        best = -float("inf")
        with open(e2_csv, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                try:
                    if r["stage"] == "e2" and abs(float(r["A"]) - 1.0) < 1e-9:
                        best = max(best, float(r["ret_mean"]))
                except (TypeError, ValueError):
                    continue
        if best > -float("inf"):
            v1_pass = best >= 0.9 * b0
            C.note(f"**V1 (existence-control): {'PASS' if v1_pass else 'FAIL'}** — "
                   f"max_beta R(A=1) = {best:.0f} vs 0.9*B0 = {0.9 * b0:.0f} "
                   f"(B0 = {b0:.0f}, beta* = {beta}). This is the campaign's "
                   f"hinge: it decides whether the peak arms run at all.")

    if beta is not None:
        bs = ["--beta_star", beta]
        C.run("e2b", py("scripts/diag_tier0.py", *t0, "--stage", "e2b", *bs,
                        "--out", f"{D}/e2b"))
        C.run("e2b_dcap", py("scripts/diag_tier0.py", *t0, "--stage", "e2b", *bs,
                             "--dcap_leg", "--out", f"{D}/e2b"), prio="P1")
        C.run("e3", py("scripts/diag_tier0.py", *t0, "--stage", "e3", *bs,
                       "--out", f"{D}/e3"))
    C.run("e4", py("scripts/diag_tier0.py", *t0, "--stage", "e4", "--out", f"{D}/e4"))
    C.run("e6", py("scripts/diag_tier0.py", *t0, "--stage", "e6", "--out", f"{D}/e6"))

    if v1_pass is False:
        C.branch = "ill-posed (V1 fail) — repaired-env track"
        C.note("**ORDERING RULE 1 FIRED (spec §10.2).** V1 failed at sigma=0.9, so "
               "F1c / F2 / F2b / F2c / F3a / F3b / D2 are **skipped entirely** — "
               "they would spend ~30M steps on a slice the campaign has just shown "
               "to be ill-posed under WP-1. E2b has located the feasible frontier "
               f"sigma*: read `{D}/e2b/tier0_e2b.md`.\n\n"
               "    **Next step is a human decision, not a run**: pick the redesign "
               "dial (R-a: sigma* minus a margin; R-b: engage DCAP and raise sigma) "
               "per §9.2, then re-run the Tier-0 suite + F1c on the repaired env. "
               "The campaign's public conclusion is already publishable: *PCR at "
               "sigma=0.9 is ill-posed under WP-1; the phase-boundary theory "
               "bounded the wrong regime (saturation geometry binds before c=1)* "
               "(§9.3).")
        # Ordering rule 1: the campaign continues on the REPAIRED env, which is a
        # human decision (§9.2). So halt after Tier 0 + E2b + the report — do NOT
        # spend steps on f1a/f1b/f4/g/x1/D-arms training slices of an env that has
        # just been shown ill-posed at sigma=0.9. E5/E3-DOB already ran (harmless,
        # eval-only) and their observability reading carries to the repaired env.
        for s in ("f1a", "f1b", "f1c_s1", "f1c_s2", "f2_s1", "f2_s2", "f2b",
                  "f2c", "f0o", "f3a", "f3b", "f4", "f4b_collect", "f4b", "g",
                  "x1", "d2_s1", "d2_s2", "d1", "d3"):
            args.skip.add(s)
        C.note("**Campaign HALTS after the report** (spec §10.2 rule 1): the next "
               "runs happen on the *repaired* env, and choosing the repair is a "
               "human decision. Read E2b for sigma* and E6 for the harm channel, "
               "pick the redesign dial (§9.2), then re-launch Tier 0 + F1c on the "
               "repaired env.")

    # ---- 6. E5 + E3-DOB ----------------------------------------------
    for A in ("0.5", "1.0"):
        C.run(f"e5_random_A{A}", py("scripts/diag_tier0.py", *t0, "--probe",
                                    "identity", "--A", A, "--episodes", 50,
                                    "--dump_traj", "--out", f"{D}/e5_random_A{A}"))
    C.run("e5", py("-m", "harl.envs.mamujoco.diag.sysid",
                   "--data", f"e1_frozen:{D}/e1/recorder/**/traj_*.npz",
                   "--data", f"blind_drift:{D}/d0/recorder/**/traj_*.npz",
                   "--data", f"random_excite:{D}/e5_random_A*/recorder/**/traj_*.npz",
                   "--out", f"{D}/e5", "--export_dob", f"{D}/e5/dob_filter.npz"))
    r2 = R.e5_best_r2(D)
    dob = os.path.join(D, "e5", "dob_filter.npz")
    if r2 is not None and r2 < 0.3:
        C.note(f"**ABORT RULE 4 FIRED**: E5's best F-loc R^2 on source (a) is "
               f"{r2:.3f} < 0.3 — E3-DOB is **skipped** and V6 is logged as an early "
               f"fail. Do not spend the eval pass.")
        args.skip.add("e3dob")
    if beta is not None and (os.path.exists(dob) or args.dry_run):
        C.run("e3dob", py("scripts/diag_tier0.py", *t0, "--stage", "e3dob",
                          "--beta_star", beta, "--dob", dob, "--out", f"{D}/e3dob"))

    # ---- 7. Tier 1 frozen arms (parallelisable) ----------------------
    def arm_job(arm_id, seed=1, prio=None, extra=None):
        a = _ARM[arm_id]
        exp = f"diag_{arm_id}" + (f"_s{seed}" if len(a["seeds"]) > 1 else "")
        argv = py("examples/train.py", "--load_config",
                  f"tuned_configs/mamujoco/Ant-v2-4x2/diag/{arm_id}.json",
                  "--exp_name", exp, "--seed", seed) + (extra or [])
        name = f"{arm_id}_s{seed}" if len(a["seeds"]) > 1 else arm_id
        return (name, argv, a.get("env", {}))

    C.run_parallel([arm_job("f1a"), arm_job("f1b"),
                    arm_job("f1c", 1), arm_job("f1c", 2),
                    arm_job("f2", 1), arm_job("f2", 2)], prio="P0")
    C.run_parallel([arm_job("f2b"), arm_job("f2c"), arm_job("f0o")], prio="P1")

    # ---- 8. pretrained-init arms (need f0 / f0o) ---------------------
    f0o = R.resolve_models(args.results_root, "f0o", strict=False)
    C.run("f3a", py("examples/train.py", "--load_config",
                    "tuned_configs/mamujoco/Ant-v2-4x2/diag/f3a.json",
                    "--exp_name", "diag_f3a", "--seed", 1, "--model_dir", f0),
          env=_ARM["f3a"]["env"])
    if f0o:
        C.run("f3b", py("examples/train.py", "--load_config",
                        "tuned_configs/mamujoco/Ant-v2-4x2/diag/f3b.json",
                        "--exp_name", "diag_f3b", "--seed", 1, "--model_dir", f0o),
              env=_ARM["f3b"]["env"])
    elif not args.dry_run and "f3b" not in args.skip:
        C.note("**F3b skipped**: it needs F0o (the oracle-schema stationary walker) "
               "as its init — it cannot load F0, whose obs is 8 dims narrower. "
               "Run with `--priority P1` to build F0o first.")

    # ---- 9. F4 / F4b -------------------------------------------------
    C.run("f4", py("scripts/diag_f4.py", "--config",
                   "tuned_configs/mamujoco/Ant-v2-4x2/diag/f4.json",
                   "--model_dir", f0, "--exp_name", "diag_f4", "--seed", 1,
                   "--out", f"{D}/f4"), env={"ANT_PCR_FREEZE_A": "1.0"})
    pool = os.path.join(D, "f4", "trough_pool.npz")
    C.run("f4b_collect", py("scripts/diag_f4.py", "--config",
                            "tuned_configs/mamujoco/Ant-v2-4x2/diag/f4.json",
                            "--model_dir", f0, "--collect_pool", pool,
                            "--pool_steps", 200000, "--out", f"{D}/f4"),
          env={"ANT_PCR_FREEZE_A": "0.0"}, prio="P2")
    C.run("f4b", py("scripts/diag_f4.py", "--config",
                    "tuned_configs/mamujoco/Ant-v2-4x2/diag/f4.json",
                    "--model_dir", f0, "--rehearse_pool", pool, "--rehearse", 0.25,
                    "--exp_name", "diag_f4b", "--seed", 1, "--out", f"{D}/f4"),
          env={"ANT_PCR_FREEZE_A": "1.0"}, prio="P2")

    # ---- 10. G (=> PC) then X1 ---------------------------------------
    pol = []
    for arm, tA in (("f0", None), ("f1a", 0.0), ("f1b", 0.5), ("f1c", 1.0),
                    ("f3a", None)):
        p = f0 if arm == "f0" else R.resolve_models(args.results_root, arm,
                                                    strict=False)
        if p:
            pol += ["--policy", f"{arm}:{p}" + (f"@{tA}" if tA is not None else "")]
    orc = []
    for arm in ("f2", "f3b"):
        p = R.resolve_models(args.results_root, arm, strict=False)
        if p:
            orc += ["--oracle_policy", f"{arm}:{p}"]
    if pol:
        C.run("g", py("scripts/diag_crosseval.py", "--out", f"{D}/g",
                      "--episodes", args.episodes, "--device", args.device,
                      *(["--b0", b0] if b0 else []), *pol, *orc))
    pc = R.path_ceiling(D)
    if pc is None and args.dry_run:
        pc = "<PC>"           # keep the dry-run plan complete through X1
    if pc is not None and not args.dry_run:
        C.note(f"**PC (path ceiling) = {pc:.0f}** ⇒ the principled target for any "
               f"future method is **0.9*PC = {0.9 * pc:.0f}** (cycle-average, C4 "
               f"protocol). This replaces the arbitrary 6500.")
    exp = []
    for arm, tA in (("f1a", 0.0), ("f1b", 0.5), ("f1c", 1.0)):
        p = R.resolve_models(args.results_root, arm, strict=False)
        if p:
            exp += ["--expert", f"{arm}:{p}@{tA}"]
    if len(exp) >= 4 and pc is not None:
        C.run("x1", py("scripts/diag_distill.py", "--out", f"{D}/x1", "--pc", pc,
                       "--episodes", args.episodes, *exp))
    elif not args.dry_run:
        C.note("**X1 skipped**: it needs the f1a/f1b/f1c experts and PC from G.")

    # ---- 11. GATE: F2 vs F1c => D2's budget (abort rule 3) -----------
    d2_steps = None
    f1c_v = _arm_return(args.results_root, "f1c")
    f2_v = _arm_return(args.results_root, "f2")
    f1a_v = _arm_return(args.results_root, "f1a")
    if None not in (f1c_v, f2_v, f1a_v):
        if f2_v < 0.85 * f1a_v:
            d2_steps = 3_000_000
            C.note(f"**ABORT RULE 3 FIRED**: F2 ({f2_v:.0f}) fails the 0.85*F1a bar "
                   f"({0.85 * f1a_v:.0f}) ⇒ **D2 capped at 3M** (F2 fail ⇒ D2 will "
                   f"fail too). Checkpoint-abort still applies by hand: if D2's "
                   f"2M-step trough slice is below F1c's at the same step, stop it.")

    # ---- 12. Tier 2 --------------------------------------------------
    d2_extra = ["--num_env_steps", d2_steps] if d2_steps else []
    C.run_parallel([("d2_s1", py("examples/train.py", "--load_config",
                                 "tuned_configs/mamujoco/Ant-v2-4x2/diag/d2.json",
                                 "--exp_name", "diag_d2_s1", "--seed", 1,
                                 *d2_extra), _ARM["d2"]["env"]),
                    ("d2_s2", py("examples/train.py", "--load_config",
                                 "tuned_configs/mamujoco/Ant-v2-4x2/diag/d2.json",
                                 "--exp_name", "diag_d2_s2", "--seed", 2,
                                 *d2_extra), _ARM["d2"]["env"])], prio="P0")
    C.run_parallel([("d1", py("examples/train.py", "--load_config",
                              "tuned_configs/mamujoco/Ant-v2-4x2/diag/d1.json",
                              "--exp_name", "diag_d1", "--seed", 1), {}),
                    ("d3", py("examples/train.py", "--load_config",
                              "tuned_configs/mamujoco/Ant-v2-4x2/diag/d3.json",
                              "--exp_name", "diag_d3", "--seed", 1,
                              "--model_dir", f0), {})], prio="P1")

    # ---- 13. verdict -------------------------------------------------
    C.run("report", py("scripts/diag_report.py", "--diag_out", D,
                       "--results_root", args.results_root, "--bundle",
                       args.bundle))
    C.note(f"**Campaign finished.** Verdict: `{args.bundle}/verdict.md`. "
           f"The method leaf is chosen from the spec's Part-8 tree **with the "
           f"user** (Prohibition 8) — the report prints which leaf fires and every "
           f"reading behind it, but it does not decide.")
    C.write_status()
    C._say(f"DONE. Read {C.status_path} and {args.bundle}/verdict.md")
    return 0


def _arm_return(results_root, arm):
    """An arm's converged cycle-average, for the gate decisions."""
    try:
        from scripts.diag_report import final_eval
    except Exception:
        return None
    vals = []
    for r in R.find_runs(results_root, arm):
        fe = final_eval(r["run_dir"])
        if fe:
            vals.append(fe["cycle_avg"])
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    sys.exit(main())
