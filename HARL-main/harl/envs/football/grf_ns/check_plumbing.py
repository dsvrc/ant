"""Config consistency: yaml <-> dataclasses <-> the keys the layer pops <-> the
runner arms <-> the registries.  numpy + pyyaml only; ~1 s.  Run BEFORE queueing
anything (porting brief, trap 15).

Every failure here is one that otherwise surfaces on the cluster after the
environment has been built: a yaml key nothing reads, a dataclass field nothing
sets, an arm name with no yaml, an algo listed in one registry and not another.

Run::

    python -m harl.envs.football.grf_ns.check_plumbing
"""

import json
import os
import re
import sys

import yaml

from .driver import DialParams, PitchDriver
from .keys import DIAL_FIELDS, NS_KWARGS, PACT_FIELDS, RUNNER_KEYS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)
    return ok


def _read(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.load(_read("harl/configs/envs_cfgs/football.yaml"), Loader=yaml.FullLoader)
    ns_in_yaml = sorted(k for k in cfg if str(k).startswith("ns_"))

    print("yaml <-> layer keys")
    extra = sorted(set(ns_in_yaml) - set(NS_KWARGS))
    missing = sorted(set(NS_KWARGS) - set(ns_in_yaml))
    check("every_yaml_ns_key_is_read_by_the_layer", not extra, "unread: %s" % extra)
    check("every_layer_key_has_a_yaml_value", not missing,
          "missing (a CLI override of a missing key is silently DROPPED by "
          "update_args): %s" % missing)
    layer_src = _read("harl/envs/football/grf_ns/layer.py")
    read_keys = set(re.findall(r'raw\.get\("(ns_[a-z0-9_]+)"', layer_src))
    check("layer_reads_exactly_the_declared_keys",
          read_keys == set(NS_KWARGS),
          "declared-not-read: %s  read-not-declared: %s"
          % (sorted(set(NS_KWARGS) - read_keys), sorted(read_keys - set(NS_KWARGS))))

    print("dataclasses <-> yaml")
    dial_fields = set(DialParams.__dataclass_fields__.keys())
    check("dial_fields_table_matches_the_dataclass", dial_fields == set(DIAL_FIELDS),
          "%s" % sorted(dial_fields ^ set(DIAL_FIELDS)))
    check("every_dial_field_has_a_yaml_key",
          all(("ns_" + f) in cfg for f in DIAL_FIELDS),
          "%s" % [f for f in DIAL_FIELDS if ("ns_" + f) not in cfg])
    from .channel import PactConfig
    pact_fields = set(PactConfig.__dataclass_fields__.keys())
    check("pact_fields_table_matches_the_dataclass", pact_fields == set(PACT_FIELDS),
          "%s" % sorted(pact_fields ^ set(PACT_FIELDS)))
    check("every_pact_field_has_a_yaml_key", all(v in cfg for v in PACT_FIELDS.values()))
    check("runner_keys_are_at_their_defaults_in_the_yaml",
          int(cfg.get("ns_pact", 0)) == 0 and str(cfg.get("ns_trust", "off")) == "off"
          and int(cfg.get("ns_oracle", 0)) == 0 and int(cfg.get("ns_intercept_only", 0)) == 0,
          "the runner sets %s; a yaml that sets them makes every baseline a PACT arm"
          % list(RUNNER_KEYS))

    print("the dial, built from the yaml values")
    try:
        p = DialParams(**{f: type(getattr(DialParams(), f))(cfg["ns_" + f]) for f in DIAL_FIELDS})
        facts = PitchDriver(p).certify()
        check("yaml_dial_certifies", True, "period=%d placebo=%d L=%.3f sigma=%.2f"
              % (facts["period"], facts["placebo_steps"], facts["loss"], p.severity))
    except Exception as ex:  # noqa: BLE001
        check("yaml_dial_certifies", False, repr(ex))
    check("severity_is_labelled_beyond_physical_when_above_1",
          float(cfg["ns_severity"]) <= 1.0 or "beyond" in _read(
              "harl/configs/envs_cfgs/football.yaml").lower(),
          "sigma=%.2f" % float(cfg["ns_severity"]))

    print("the host")
    fe = _read("harl/envs/football/football_env.py")
    m = re.search(r'"%s":\s*(\d+)' % re.escape(cfg["env_name"]), fe)
    n_tbl = int(m.group(1)) if m else None
    check("env_name_is_in_the_host_agent_table", n_tbl is not None, cfg["env_name"])
    check("controlled_players_match_the_agent_table",
          n_tbl == int(cfg["number_of_left_players_agent_controls"]),
          "table=%s yaml=%s" % (n_tbl, cfg["number_of_left_players_agent_controls"]))
    cj = os.path.join(here, "ceiling.json")
    if os.path.exists(cj) and cfg.get("ns_load_norm") not in (None, "~"):
        with open(cj, "r", encoding="utf-8") as f:
            ceil = json.load(f)
        ok = (ceil.get("env_name") == cfg["env_name"]
              and abs(float(ceil["load_norm"]) - float(cfg["ns_load_norm"]))
              <= 0.01 * float(ceil["load_norm"]))
        check("committed_load_norm_matches_ceiling_json", ok,
              "yaml=%s ceiling.json=%.6f (%s)" % (cfg["ns_load_norm"], ceil["load_norm"],
                                                   ceil.get("env_name")))
    else:
        check("committed_load_norm_matches_ceiling_json", cfg.get("ns_load_norm") in (None, "~"),
              "no ceiling.json: ns_load_norm must then be ~ (computed at first reset)")

    print("arms <-> registries")
    runner_src = _read("harl/runners/on_policy_grf_ns_runner.py")
    arms = re.findall(r'^\s+"([a-z_]+)": _mk\(', runner_src, flags=re.M)
    check("runner_declares_the_five_arm_names", set(arms) >= {
        "pact", "pactoff", "pact_fixed", "pact_oracle", "pact_intercept"}, "%s" % arms)
    trusts = set(re.findall(r'\("(off|fixed|learned)", \d, \d\)', runner_src))
    check("runner_trust_values_are_ones_the_layer_accepts", trusts <= {"off", "fixed"},
          "%s" % sorted(trusts))
    reg = _read("harl/runners/__init__.py")
    train = _read("examples/train.py")
    actors = _read("harl/algorithms/actors/__init__.py")
    for name in arms:
        in_reg = re.search(r'"%s":\s*_by_env\("%s"\)|"%s":\s*_pact_runner' % (name, name, name), reg)
        check("%s_is_in_RUNNER_REGISTRY_via_env_dispatch" % name, in_reg is not None)
        check("%s_is_a_train.py_choice" % name, ('"%s"' % name) in train)
        check("%s_has_an_algo_yaml" % name,
              os.path.exists(os.path.join(ROOT, "harl/configs/algos_cfgs/%s.yaml" % name)))
        check("%s_is_in_the_actor_registry" % name, ('"%s":' % name) in actors)
    check("football_branch_in__pact_runner", 'args["env"] == "football"' in reg)
    tools = _read("harl/utils/envs_tools.py")
    check("envs_tools_routes_football_through_the_ns_factory",
          tools.count("make_football_ns_env(env_args, rank, n_threads)") == 2
          and "make_football_ns_env(env_args, 0, 1)" in tools)

    if FAILS:
        print("\nFAILED %d check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("\nPLUMBING OK")


if __name__ == "__main__":
    main()
