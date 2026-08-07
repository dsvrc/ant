"""Generate (and verify) the campaign's per-arm configs  [spec §11.1 item 10].

Every arm's JSON is the **frozen** ``tuned_configs/mamujoco/Ant-v2-4x2/hasac/
config.json`` plus (a) the arm's diag env flags and (b) its step budget / init.
Generating them from that one file — instead of hand-maintaining a dozen
near-identical copies — makes campaign **Prohibition 2** ("host hyperparameters
identical across every arm; never retune per arm") a mechanical property rather
than a promise. ``--verify`` re-asserts it against the files on disk, so a later
hand-edit that quietly retunes an arm is caught before it costs 45M steps.

Only these keys may differ from the frozen host config, and each is checked:
    main_args.algo / exp_name        (the runner + the run's name)
    algo_args.train.num_env_steps    (the arm's budget)
    algo_args.train.model_dir        (F3/D3's pretrained init)
    algo_args.seed.seed              (per-seed launches)
    env_args.*                       (the arm's diagnostic flags)

Stdlib only — runs anywhere, including this machine.

    python scripts/diag_make_configs.py                 # write them
    python scripts/diag_make_configs.py --verify        # check them
    python scripts/diag_make_configs.py --print_manifest
"""

import argparse
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HOST = os.path.join(_ROOT, "tuned_configs", "mamujoco", "Ant-v2-4x2", "hasac",
                     "config.json")
_OUT = os.path.join(_ROOT, "tuned_configs", "mamujoco", "Ant-v2-4x2", "diag")

_M = 1_000_000

# Keys allowed to differ from the frozen host config (everything else must match).
_ALLOWED = {("algo_args", "train", "num_env_steps"),
            ("algo_args", "train", "model_dir"),
            ("algo_args", "seed", "seed")}


# ==========================================================================
#  THE RUN MANIFEST (spec Part 5 / Part 6 / §10.1)
# ==========================================================================
# env: the ANT_PCR_* env vars the launch line must carry (the env is a gym
#      drop-in; these are read at import, one arm per process).
# d_to: DiagMujocoMulti's raw-d schema (see diag/diag_mujoco.py). Mutually
#      exclusive with ANT_PCR_ORACLE=1, which is the *normalized*-d schema.
ARMS = [
    dict(id="f0", tier="1", prio="P0", steps=5 * _M, seeds=[1],
         env={"ANT_PCR_MASK": "off"}, drift=False,
         q="B0 reference + the Tier-0 policy (stationary walker)",
         note="If A0 finds an existing SEVERITY=0 HASAC checkpoint, skip this and "
              "Tier 0 starts immediately."),
    dict(id="f0o", tier="1", prio="P1", steps=5 * _M, seeds=[1],
         env={"ANT_PCR_MASK": "off", "ANT_PCR_ORACLE": "1"}, drift=False,
         q="ADDED ARM: stationary walker in the ORACLE obs schema — F3b's init",
         note="The spec gives F3b as 'F3a + d-oracle, model_dir=F0'. That cannot "
              "load: ANT_PCR_ORACLE=1 grows the obs by 8 dims, so F0's actor "
              "weights do not fit the oracle-schema net. Worse, MujocoMulti "
              "normalizes the WHOLE obs vector, so adding 8 coords changes every "
              "other coordinate too — an F0 policy is not even approximately the "
              "right init. F0o is the same 5M stationary walker trained in the "
              "oracle schema (MASK=off makes d==0, so the 8 oracle dims are "
              "constant zeros); F3b initializes from it. It is also the correct "
              "B0 reference for oracle-schema arms."),
    dict(id="f1a", tier="1", prio="P0", steps=3 * _M, seeds=[1],
         env={"ANT_PCR_FREEZE_A": "0"}, drift=False,
         q="sanity: == F0 modulo the clock (also the cleanest B0 if F0 is inherited)"),
    dict(id="f1b", tier="1", prio="P0", steps=3 * _M, seeds=[1],
         env={"ANT_PCR_FREEZE_A": "0.5"}, drift=False,
         q="mid-slice learnability"),
    dict(id="f1c", tier="1", prio="P0", steps=5 * _M, seeds=[1, 2],
         env={"ANT_PCR_FREEZE_A": "1.0"}, drift=False,
         q="LEARNING-LEVEL EXISTENCE at the hardest slice (axis V2)"),
    dict(id="f2", tier="1", prio="P0", steps=5 * _M, seeds=[1, 2],
         env={"ANT_PCR_FREEZE_A": "1.0", "ANT_PCR_ORACLE": "1"}, drift=False,
         q="does information rescue from-scratch peak learning? (axis V3)",
         note="d arrives NORMALIZED — appended inside AntEnv._get_obs, then "
              "rescaled by MujocoMulti's whole-vector normalization. Compare with "
              "f2c to separate 'info is toxic' from 'info arrived mangled'."),
    dict(id="f2b", tier="1", prio="P1", steps=5 * _M, seeds=[1], d_to="share",
         env={"ANT_PCR_FREEZE_A": "1.0"}, drift=False,
         q="H-C4: critic-side variance reduction alone (RAW d, share_obs only)"),
    dict(id="f2c", tier="1", prio="P1", steps=5 * _M, seeds=[1], d_to="both",
         env={"ANT_PCR_FREEZE_A": "1.0"}, drift=False,
         q="ADDED ARM: RAW d (torque units) to actor+critic — the F2 control",
         note="Isolates the obs-normalization confound on axis V3. f2 vs f2c = the "
              "units; f2c vs f2b = actor conditioning vs critic variance "
              "reduction; f2b vs f1c = the critic effect alone."),
    dict(id="f3a", tier="1", prio="P0", steps=1 * _M, seeds=[1], init="f0",
         env={"ANT_PCR_FREEZE_A": "1.0"}, drift=False,
         q="can a competent walker HOLD/adapt at the peak? (grow-vs-hold split)"),
    dict(id="f3b", tier="1", prio="P0", steps=1 * _M, seeds=[1], init="f0o",
         env={"ANT_PCR_FREEZE_A": "1.0", "ANT_PCR_ORACLE": "1"}, drift=False,
         q="does info enable holding if blind holding fails?",
         note="Initialized from f0o, not f0 — see the f0o note."),
    dict(id="f4", tier="1", prio="P0", steps=200_000, seeds=[1], init="f0",
         env={"ANT_PCR_FREEZE_A": "1.0"}, drift=False,
         q="H-C2 mechanism check — the forgetting curve",
         note="Driven by scripts/diag_f4.py, not examples/train.py: it trains at "
              "FREEZE_A=1 while scoring a 5-episode FREEZE_A=0 return every 10k "
              "steps. This config supplies the host hyperparameters."),
    dict(id="d1", tier="2", prio="P1", steps=10 * _M, seeds=[1], drift=True,
         env={},
         q="forensic re-run of the baseline failure under the §3.2 microscope"),
    dict(id="d2", tier="2", prio="P0", steps=10 * _M, seeds=[1, 2], drift=True,
         env={"ANT_PCR_ORACLE": "1"},
         q="existence-under-drift with full state — the instrumented, "
           "banner-labeled replacement for the ambiguous E-5 run"),
    dict(id="d3", tier="2", prio="P1", steps=5 * _M, seeds=[1], init="f0",
         drift=True, env={},
         q="the ratchet: does cyclical training erode a competent walker?"),
]


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def build(arm, host):
    cfg = copy.deepcopy(host)
    cfg["main_args"]["algo"] = "hasac_diag"
    cfg["main_args"]["exp_name"] = f"diag_{arm['id']}"
    cfg["algo_args"]["train"]["num_env_steps"] = int(arm["steps"])
    if arm.get("init"):
        # placeholder: the launcher fills it in (the path has a timestamp)
        cfg["algo_args"]["train"]["model_dir"] = f"<PATH_TO_{arm['init'].upper()}_MODELS>"

    ea = cfg["env_args"]
    ea["diag"] = True
    # C4 de-aliased eval: only meaningful when the payload actually drifts. On a
    # frozen slice every clock offset is the same slice, so the offset is a no-op —
    # off, so nothing about the frozen arms' eval is implied to be phase-corrected.
    ea["pcr_eval_dephase"] = bool(arm.get("drift", False))
    ea["pcr_period"] = 40000
    ea["diag_cfg"] = {
        "telemetry": True,
        "telemetry_interval": 10000,
        "bank_interval": 50000,
        "rank_interval": 200000,
        "bank_size": 512,
        "rank_batch": 1024,
        "d_to": arm.get("d_to", "none"),
    }
    ea["pcr_diag_cfg"] = {
        "dir": f"./diag_out/recorder/{arm['id']}",
        "interval": 10,                 # spec §5: flight recorder on at interval=10
        "dump_traj": False,             # E5's NPZ dumps come from Tier-0, not here
        "dump_every_k_episodes": 10,
        "dump_max_mb": 512,
    }
    return cfg


def env_line(arm):
    return " ".join(f"{k}={v}" for k, v in sorted(arm.get("env", {}).items()))


def launch_line(arm, seed):
    e = env_line(arm)
    pre = (e + " ") if e else ""
    exp = f"diag_{arm['id']}" + (f"_s{seed}" if len(arm["seeds"]) > 1 else "")
    script = ("python scripts/diag_f4.py --config" if arm["id"] == "f4"
              else "python examples/train.py --load_config")
    line = (f"{pre}{script} tuned_configs/mamujoco/Ant-v2-4x2/diag/{arm['id']}.json "
            f"--exp_name {exp} --seed {seed}")
    if arm.get("init"):
        line += f" \\\n    --model_dir results/mamujoco/Ant-v2-4x2/hasac_diag/" \
                f"diag_{arm['init']}/seed-00001-*/models"
    return line


def expected_banners(arm):
    ek = arm.get("env", {})
    return [
        "[DIAG ENV] SEVERITY=0.9 FREEZE_A=%s MASK=%s DCAP=None ORACLE=%s CORACLE=0"
        % (ek.get("ANT_PCR_FREEZE_A", "None"), ek.get("ANT_PCR_MASK", "both"),
           ek.get("ANT_PCR_ORACLE", "0")),
        "[DIAG ARM] d_to=%s" % arm.get("d_to", "none"),
        "[DIAG RUN] algo=hasac_diag telemetry=True",
    ]


def cmd_write(args):
    with open(_HOST, encoding="utf-8") as f:
        host = json.load(f)
    os.makedirs(_OUT, exist_ok=True)
    for arm in ARMS:
        path = os.path.join(_OUT, f"{arm['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(build(arm, host), f, indent=4, sort_keys=True)
            f.write("\n")
        print(f"wrote {os.path.relpath(path, _ROOT)}   "
              f"[{arm['prio']}] {int(arm['steps']) / 1e6:g}M x {len(arm['seeds'])} "
              f"seed(s) — {arm['q']}")
    print(f"\n{len(ARMS)} arm configs generated from {os.path.relpath(_HOST, _ROOT)}")
    print("Run `--verify` after any hand-edit.")
    return 0


def cmd_verify(args):
    with open(_HOST, encoding="utf-8") as f:
        host = json.load(f)
    bad = 0
    for arm in ARMS:
        path = os.path.join(_OUT, f"{arm['id']}.json")
        if not os.path.exists(path):
            print(f"MISSING {arm['id']}.json — run without --verify first")
            bad += 1
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        for section in ("algo", "device", "eval", "logger", "model", "render",
                        "train", "seed"):
            for k, v in host["algo_args"][section].items():
                if ("algo_args", section, k) in _ALLOWED:
                    continue
                got = cfg["algo_args"][section].get(k, "<missing>")
                if got != v:
                    print(f"DRIFT {arm['id']}: algo_args.{section}.{k} = {got!r} "
                          f"but the frozen host config says {v!r}  "
                          f"(Prohibition 2: never retune per arm)")
                    bad += 1
        if cfg["main_args"]["algo"] != "hasac_diag":
            print(f"DRIFT {arm['id']}: main_args.algo = "
                  f"{cfg['main_args']['algo']!r}, expected 'hasac_diag'")
            bad += 1
    if bad:
        print(f"\n{bad} problem(s). The campaign's arms are NOT comparable until "
              f"these are fixed.")
        return 1
    print(f"OK — all {len(ARMS)} arms carry the frozen host hyperparameters "
          f"(only steps / model_dir / seed / env_args differ).")
    return 0


def cmd_manifest(args):
    print(f"{'arm':<5} {'prio':<4} {'steps':>7} {'seeds':<7} {'drift':<6} "
          f"{'d_to':<6} {'init':<5} env vars")
    print("-" * 110)
    for a in ARMS:
        print(f"{a['id']:<5} {a['prio']:<4} {a['steps'] / 1e6:>6.2f}M "
              f"{str(a['seeds']):<7} {str(bool(a.get('drift'))):<6} "
              f"{a.get('d_to', 'none'):<6} {a.get('init', '-'):<5} "
              f"{env_line(a) or '(none — blind)'}")
    tot = sum(a["steps"] * len(a["seeds"]) for a in ARMS)
    p0 = sum(a["steps"] * len(a["seeds"]) for a in ARMS if a["prio"] == "P0")
    print("-" * 110)
    print(f"P0 only: {p0 / 1e6:.1f}M env steps   |   full: {tot / 1e6:.1f}M")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="assert every arm still carries the frozen host "
                         "hyperparameters")
    ap.add_argument("--print_manifest", action="store_true")
    args = ap.parse_args(argv)
    if args.print_manifest:
        return cmd_manifest(args)
    if args.verify:
        return cmd_verify(args)
    return cmd_write(args)


if __name__ == "__main__":
    sys.exit(main())
