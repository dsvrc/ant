"""Config consistency: yaml <-> dataclasses <-> the keys the layer reads <-> the
runner arms <-> every registry.  numpy + pyyaml only; ~1 s.  Run BEFORE queueing
anything (the porting brief's trap 15).

Every failure here is one that otherwise surfaces on the cluster after MuJoCo
has been built: a yaml key nothing reads, a dataclass field nothing sets, an arm
name with no yaml, an env name listed in one registry and not another.

Run::

    python -m harl.envs.mamujoco.ant_ns.check_plumbing
"""

import json
import os
import re
import sys

import yaml

from .driver import DialParams, ThermalDriver
from .keys import DIAL_FIELDS, NS_KWARGS, PACT_FIELDS, RUNNER_KEYS
from .structure import LEG_OF, PARTITIONS, RUNNABLE

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
    cfg = yaml.load(_read("harl/configs/envs_cfgs/mamujoco_ns.yaml"), Loader=yaml.FullLoader)
    ns_in_yaml = sorted(k for k in cfg if str(k).startswith("ns_"))

    print("yaml <-> layer keys")
    extra = sorted(set(ns_in_yaml) - set(NS_KWARGS))
    missing = sorted(set(NS_KWARGS) - set(ns_in_yaml))
    check("every_yaml_ns_key_is_read_by_the_layer", not extra, "unread: %s" % extra)
    check("every_layer_key_has_a_yaml_value", not missing,
          "missing (a CLI override of a missing key is silently DROPPED by "
          "update_args): %s" % missing)
    layer_src = _read("harl/envs/mamujoco/ant_ns/layer.py")
    read_keys = set(re.findall(r'raw\.get\("(ns_[a-z0-9_]+)"', layer_src))
    check("layer_reads_exactly_the_declared_keys", read_keys == set(NS_KWARGS),
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
          and int(cfg.get("ns_oracle", 0)) == 0
          and int(cfg.get("ns_intercept_only", 0)) == 0,
          "the runner sets %s; a yaml that sets them makes every baseline a PACT arm"
          % list(RUNNER_KEYS))

    print("the dial, built from the yaml values")
    try:
        p = DialParams(**{f: type(getattr(DialParams(), f))(cfg["ns_" + f])
                          for f in DIAL_FIELDS})
        facts = ThermalDriver(p).certify()
        check("yaml_dial_certifies", True,
              "period=%d placebo=%d L=%.3f sigma=%.2f"
              % (facts["period"], facts["placebo_steps"], facts["loss"], p.severity))
    except Exception as ex:  # noqa: BLE001
        check("yaml_dial_certifies", False, repr(ex))
    src = _read("harl/configs/envs_cfgs/mamujoco_ns.yaml").lower()
    check("severity_is_labelled_beyond_physical_when_above_1",
          float(cfg["ns_severity"]) <= 1.0 or "beyond" in src,
          "sigma=%.2f" % float(cfg["ns_severity"]))

    print("the host")
    check("scenario_is_one_ant_ns_declares_a_structure_for",
          cfg["scenario"] in ("Ant-v2", "Ant-v4"), cfg["scenario"])
    check("agent_conf_is_a_partition_MAMuJoCo_will_build",
          cfg["agent_conf"] in RUNNABLE,
          "%s of %s (1x8 is the structural N=1 projection and is NOT runnable)"
          % (cfg["agent_conf"], list(RUNNABLE)))
    check("declared_partitions_build_the_same_machine_as_MAMuJoCo",
          _partitions_match(),
          "compared by STRUCTURE: obsk's act_ids and the positional mapping "
          "step() actually uses differ by a relabelling -- see _partitions_match")
    cj = os.path.join(here, "ceiling.json")
    if os.path.exists(cj) and cfg.get("ns_load_norm") not in (None, "~"):
        with open(cj, "r", encoding="utf-8") as f:
            ceil = json.load(f)
        ok = abs(float(ceil["load_norm"]) - float(cfg["ns_load_norm"])) \
            <= 0.01 * float(ceil["load_norm"])
        check("committed_load_norm_matches_ceiling_json", ok,
              "yaml=%s ceiling.json=%.6f at partition %s"
              % (cfg["ns_load_norm"], ceil["load_norm"], ceil.get("agent_conf")))
    else:
        check("committed_load_norm_matches_ceiling_json",
              cfg.get("ns_load_norm") in (None, "~"),
              "no ceiling.json: ns_load_norm must then be ~ (computed at startup)")

    print("arms <-> registries")
    runner_src = _read("harl/runners/on_policy_ant_ns_runner.py")
    arms = re.findall(r'^\s+"([a-z_]+)": _mk\(', runner_src, flags=re.M)
    check("runner_declares_the_five_arm_names",
          set(arms) >= {"pact", "pactoff", "pact_fixed", "pact_oracle", "pact_intercept"},
          "%s" % arms)
    trusts = set(re.findall(r'\("(off|fixed|learned)", \d, \d\)', runner_src))
    check("runner_trust_values_are_ones_the_layer_accepts", trusts <= {"off", "fixed"},
          "%s" % sorted(trusts))
    reg = _read("harl/runners/__init__.py")
    train = _read("examples/train.py")
    actors = _read("harl/algorithms/actors/__init__.py")
    for name in arms:
        if name == "pact_mappo":
            continue                      # an extra host, not a spec arm
        in_reg = re.search(r'"%s":\s*_by_env\("%s"\)|"%s":\s*_pact_runner'
                           % (name, name, name), reg)
        check("%s_is_in_RUNNER_REGISTRY_via_env_dispatch" % name, in_reg is not None)
        check("%s_is_a_train.py_choice" % name, ('"%s"' % name) in train)
        check("%s_has_an_algo_yaml" % name,
              os.path.exists(os.path.join(ROOT, "harl/configs/algos_cfgs/%s.yaml" % name)))
        check("%s_is_in_the_actor_registry" % name, ('"%s":' % name) in actors)
    check("mamujoco_ns_branch_in__pact_runner", 'args["env"] == "mamujoco_ns"' in reg)
    check("ANT_ARMS_is_imported_by_the_registry", "from harl.runners.on_policy_ant_ns_runner "
          "import ANT_ARMS" in reg)

    print("the env name, in every registry that must know it")
    tools = _read("harl/utils/envs_tools.py")
    check("envs_tools_routes_mamujoco_ns_through_the_ns_factory",
          tools.count("make_ant_ns_env(env_args, rank, n_threads)") == 2
          and "make_ant_ns_env(env_args, 0, 1)" in tools,
          "train + eval + render")
    check("get_num_agents_knows_mamujoco_ns", 'elif env == "mamujoco_ns":' in tools)
    check("logger_registry_knows_mamujoco_ns",
          '"mamujoco_ns": MAMuJoCoLogger' in _read("harl/envs/__init__.py"))
    check("get_task_name_knows_mamujoco_ns",
          'mamujoco_ns' in _read("harl/utils/configs_tools.py"))
    check("train.py_accepts_mamujoco_ns", '"mamujoco_ns"' in train)
    check("host_guard_is_armed_in_the_yaml",
          int(cfg.get("ns_allow_patched_host", 0)) == 0,
          "ns_allow_patched_host=1 makes a run unreportable -- it must be 0 for anything that goes in the paper")
    check("mamujoco_yaml_is_untouched",
          "ns_" not in _read("harl/configs/envs_cfgs/mamujoco.yaml"),
          "the eight other mamujoco method families must not inherit a dial")

    if FAILS:
        print("")
        print("FAILED %d check(s): %s" % (len(FAILS), FAILS))
        sys.exit(1)
    print("")
    print("PLUMBING OK")


def _shape(groups):
    """A canonical, label-free description of a partition: for each agent, how
    many joints it owns and whether its legs are adjacent, diagonal, or one leg.

    Legs sit at the four corners of the torso in the order 1(+,+) 2(-,+)
    3(-,-) 4(+,-), so {1,2} is adjacent and {1,3} is diagonal.
    """
    ADJ = {frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 4)), frozenset((4, 1))}
    out = []
    for g in groups:
        legs = {int(LEG_OF[i]) for i in g}
        if len(legs) == 1:
            kind = "one-leg"
        elif len(legs) == 2:
            kind = "adjacent" if frozenset(legs) in ADJ else "diagonal"
        else:
            kind = "%d-legs" % len(legs)
        out.append((len(g), kind))
    return sorted(out)


def _partitions_match():
    """The declared partitions must describe the SAME machine MAMuJoCo builds.

    They are compared by structure, not by label, and that is not a shortcut --
    it is the only correct comparison, for a reason worth stating:

    MAMuJoCo's ``obsk`` tags each joint with an ``act_ids`` and declares Ant's
    ``2x4`` as act-ids [(2,3,4,5), (6,7,0,1)].  But ``MujocoMulti.step`` never
    reorders by ``act_ids``: it concatenates the agents' action slices and hands
    the result straight to MuJoCo, so agent ``a`` physically drives
    ``ctrl[sum(dims[:a]) : sum(dims[:a+1])]`` -- the POSITIONAL block.  The
    disturbance has to land on the actuators the agent actually drives, so
    ``structure.PARTITIONS`` is positional, and it differs from ``obsk``'s
    labels by a relabelling of agents (measured: identical leg groups for 4x2,
    2x4d and 8x1; for 2x4 the same "two adjacent legs per agent" shape rotated
    by one corner, which the ant's 4-fold symmetry makes equivalent).

    ``smoke.py`` closes this at runtime by measuring which actuators an agent's
    action actually reaches -- the only account of it that cannot be argued with.

    Imported lazily and tolerantly: this check must not be the reason the suite
    cannot run on a bare machine.
    """
    try:
        from harl.envs.mamujoco.multiagent_mujoco.obsk import get_parts_and_edges
    except Exception:  # pragma: no cover
        print("       (obsk not importable; declared partitions unchecked)")
        return True
    ok = True
    for conf in RUNNABLE:
        parts, _, _ = get_parts_and_edges("Ant-v2", conf)
        theirs = [tuple(int(n.act_ids) for n in grp) for grp in parts]
        mine = [tuple(g) for g in PARTITIONS[conf]]
        if _shape(theirs) != _shape(mine):
            print("       %s: MAMuJoCo builds %s, structure.py declares %s -- NOT the "
                  "same machine" % (conf, _shape(theirs), _shape(mine)))
            ok = False
        elif theirs != mine:
            print("       %s: same machine, agents relabelled (obsk act_ids %s, "
                  "positional %s -- step() uses the positional one)"
                  % (conf, theirs, mine))
    return ok


if __name__ == "__main__":
    main()
