"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def set_pcr_clock_offset(env, offset):
    """C4.1 de-aliased eval: pin a mamujoco env's PCR payload clock to ``offset``.

    Stratifying the eval envs' clocks across the payload cycle is what makes an
    eval round a true cycle-average instead of a phase-aliased snapshot (the
    measurement error H-D1 blames prior verdicts on). ``MujocoMulti.env`` is the
    AntEnv itself; a non-PCR env simply has no ``_clock`` and is left alone.
    """
    if not offset:
        return
    inner = getattr(env, "env", env)
    tgt = getattr(inner, "unwrapped", inner)
    try:
        tgt._clock = int(offset)
    except Exception:
        print(f"[PCR] WARNING: could not set pcr_clock_offset={offset} on {env}.")


def _pcr_eval_env_args(env_args, rank, n_threads):
    """Per-rank copy of ``env_args`` carrying this eval env's clock offset.

    Generalized out of the ecl-only branch (spec §3.3 / 11.1 item 8) so ANY
    mamujoco algo gets the C4 protocol via ``env_args.pcr_eval_dephase: true``.
    Copies the dict per rank — never mutates the shared one.
    """
    ea = dict(env_args)
    if not ea.get("pcr_eval_dephase", False):
        return ea, 0
    period = int(ea.get("pcr_period", 40000))
    offset = (rank * period) // max(1, n_threads)
    cfg = dict(ea.get("diag_cfg", {}))
    cfg["pcr_clock_offset"] = offset
    ea["diag_cfg"] = cfg
    return ea, offset


def _snd_dephase(env_args, rank, n_threads):
    """Per-rank copy of ``env_args`` carrying this SMAC-CWO env's driver phase.

    The CWO driver A(t) is a raised cosine of period 5000 env steps -- ~30x a rollout
    (episode_length=160) and ~40x an episode.  With every parallel env on the same
    clock, a whole PPO batch sees ONE phase of the driver (the critic chases a moving
    target) and, worse, a whole EVAL round is a single-phase snapshot: 40 episodes /
    10 threads advances the eval clock by only ~4*ep_len ~= 250-600 of 5000 steps, so
    consecutive evals crawl around the cycle and the reported win-rate becomes a slow
    square wave measuring the driver phase rather than the policy.

    Spreading rank r to phase r/n_threads makes every batch and every eval round a
    true cycle-average.  Same idea, same reason as ``_pcr_eval_env_args`` for
    mamujoco; ``StarCraft2_Env`` reads ``snd_phase`` and ignores it when
    ``SMAC_SND_DEPHASE=0``.  Copies the dict per rank -- never mutates the shared one.
    """
    if n_threads <= 1:
        return env_args
    return {**env_args, "snd_phase": float(rank) / float(n_threads)}


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(_snd_dephase(env_args, rank, n_threads))
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                if env_args.get("echor", False):
                    from harl.envs.mamujoco.echor.echor_mujoco import (
                        EchoRMujocoMulti,
                    )

                    env = EchoRMujocoMulti(env_args=env_args)
                elif env_args.get("diag", False):
                    from harl.envs.mamujoco.diag.diag_mujoco import DiagMujocoMulti

                    env = DiagMujocoMulti(env_args=env_args)
                elif env_args.get("pcr_diag", False):
                    from harl.envs.mamujoco.pcr_diag import PcrDiagMujocoMulti

                    env = PcrDiagMujocoMulti(env_args=env_args)
                elif env_args.get("ecl", False):
                    from harl.envs.mamujoco.ecl.ecl_mujoco import EclMujocoMulti

                    env = EclMujocoMulti(env_args=env_args)
                elif env_args.get("recon", False):
                    from harl.envs.mamujoco.recon.recon_mujoco import (
                        ReconMujocoMulti,
                    )

                    env = ReconMujocoMulti(env_args=env_args)
                elif env_args.get("omax", False):
                    from harl.envs.mamujoco.omax.omax_mujoco import OmaxMujocoMulti

                    env = OmaxMujocoMulti(env_args=env_args)
                elif env_args.get("pact1", False):
                    from harl.envs.mamujoco.pact.pact1_mujoco import Pact1MujocoMulti

                    env = Pact1MujocoMulti(env_args=env_args)
                elif env_args.get("pact", False):
                    from harl.envs.mamujoco.pact.pact_mujoco import PactMujocoMulti

                    env = PactMujocoMulti(env_args=env_args)
                else:
                    from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                        MujocoMulti,
                    )

                    env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                assert env_args["scenario"] in [
                    "simple_v2",
                    "simple_spread_v2",
                    "simple_reference_v2",
                    "simple_speaker_listener_v3",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    # SMAC-CWO: eval envs SKIP the training warmup curriculum -- they always use the
    # full severity so eval measures the harmed win rate.  Harmless for other envs
    # (only StarCraft2_Env reads "snd_eval").
    env_args = {**env_args, "snd_eval": 1}

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(_snd_dephase(env_args, rank, n_threads))
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                # C4.1 de-aliased eval: stratify each eval env's payload clock
                # across the cycle so every eval round is a true cycle-average.
                # `pcr_eval_dephase: true` activates this for ANY mamujoco algo
                # (spec §3.3); the ecl branch keeps its own ecl_cfg spelling.
                ea, pcr_offset = _pcr_eval_env_args(env_args, rank, n_threads)
                if env_args.get("echor", False):
                    from harl.envs.mamujoco.echor.echor_mujoco import (
                        EchoRMujocoMulti,
                    )

                    env = EchoRMujocoMulti(env_args=ea)
                    set_pcr_clock_offset(env, pcr_offset)
                elif env_args.get("diag", False):
                    from harl.envs.mamujoco.diag.diag_mujoco import DiagMujocoMulti

                    # DiagMujocoMulti applies diag_cfg.pcr_clock_offset itself
                    env = DiagMujocoMulti(env_args=ea)
                elif env_args.get("pcr_diag", False):
                    from harl.envs.mamujoco.pcr_diag import PcrDiagMujocoMulti

                    env = PcrDiagMujocoMulti(env_args=ea)
                    set_pcr_clock_offset(env, pcr_offset)
                elif env_args.get("ecl", False):
                    from harl.envs.mamujoco.ecl.ecl_mujoco import EclMujocoMulti

                    # unchanged: ECL carries the offset in its own ecl_cfg
                    ea_ecl = dict(env_args)
                    ecl_cfg = dict(ea_ecl.get("ecl_cfg", {}))
                    if ecl_cfg.get("eval_dephase", True):
                        period = int(ecl_cfg.get("pcr_period", 40000))
                        ecl_cfg["pcr_clock_offset"] = (rank * period) // max(1, n_threads)
                        ea_ecl["ecl_cfg"] = ecl_cfg
                    env = EclMujocoMulti(env_args=ea_ecl)
                elif env_args.get("recon", False):
                    from harl.envs.mamujoco.recon.recon_mujoco import (
                        ReconMujocoMulti,
                    )

                    # RECON uses the generic C4 path: `pcr_eval_dephase: true`
                    env = ReconMujocoMulti(env_args=ea)
                    set_pcr_clock_offset(env, pcr_offset)
                elif env_args.get("omax", False):
                    from harl.envs.mamujoco.omax.omax_mujoco import OmaxMujocoMulti

                    # carry the de-alias offset into omax_cfg so the wrapper pins
                    # the eval env's payload clock at construction
                    ea_omax = dict(env_args)
                    ocfg = dict(ea_omax.get("omax_cfg", {}))
                    ocfg["pcr_clock_offset"] = pcr_offset
                    ea_omax["omax_cfg"] = ocfg
                    env = OmaxMujocoMulti(env_args=ea_omax)
                elif env_args.get("pact1", False):
                    from harl.envs.mamujoco.pact.pact1_mujoco import Pact1MujocoMulti

                    ea_p1 = dict(env_args)
                    p1cfg = dict(ea_p1.get("pact1_cfg", {}))
                    p1cfg["pcr_clock_offset"] = pcr_offset
                    ea_p1["pact1_cfg"] = p1cfg
                    env = Pact1MujocoMulti(env_args=ea_p1)
                elif env_args.get("pact", False):
                    from harl.envs.mamujoco.pact.pact_mujoco import PactMujocoMulti

                    # carry the de-alias offset into pact_cfg so the wrapper pins
                    # the eval env's payload clock at construction (never training)
                    ea_pact = dict(env_args)
                    pcfg = dict(ea_pact.get("pact_cfg", {}))
                    pcfg["pcr_clock_offset"] = pcr_offset
                    ea_pact["pact_cfg"] = pcfg
                    env = PactMujocoMulti(env_args=ea_pact)
                else:
                    from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                        MujocoMulti,
                    )

                    env = MujocoMulti(env_args=ea)
                    set_pcr_clock_offset(env, pcr_offset)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(args=env_args)
        manual_render = (
            False  # smac does not support manually calling the render() function
        )
        # instead, it use save_replay()
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2Env

        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "mamujoco":
        if env_args.get("echor", False):
            from harl.envs.mamujoco.echor.echor_mujoco import EchoRMujocoMulti

            env = EchoRMujocoMulti(env_args=env_args)
        elif env_args.get("diag", False):
            from harl.envs.mamujoco.diag.diag_mujoco import DiagMujocoMulti

            env = DiagMujocoMulti(env_args=env_args)
        elif env_args.get("pcr_diag", False):
            from harl.envs.mamujoco.pcr_diag import PcrDiagMujocoMulti

            env = PcrDiagMujocoMulti(env_args=env_args)
        elif env_args.get("ecl", False):
            from harl.envs.mamujoco.ecl.ecl_mujoco import EclMujocoMulti

            env = EclMujocoMulti(env_args=env_args)
        elif env_args.get("recon", False):
            from harl.envs.mamujoco.recon.recon_mujoco import ReconMujocoMulti

            env = ReconMujocoMulti(env_args=env_args)
        elif env_args.get("omax", False):
            from harl.envs.mamujoco.omax.omax_mujoco import OmaxMujocoMulti

            env = OmaxMujocoMulti(env_args=env_args)
        elif env_args.get("pact1", False):
            from harl.envs.mamujoco.pact.pact1_mujoco import Pact1MujocoMulti

            env = Pact1MujocoMulti(env_args=env_args)
        elif env_args.get("pact", False):
            from harl.envs.mamujoco.pact.pact_mujoco import PactMujocoMulti

            env = PactMujocoMulti(env_args=env_args)
        else:
            from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

            env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GYMEnv

        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
        manual_render = False  # football renders automatically
        env.seed(seed * 60000)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv

        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False  # dexhands renders automatically
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env_num = 64
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LAGEnv

        env = LAGEnv(env_args)
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smac":
        from harl.envs.smac.smac_maps import get_map_params

        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "lag":
        return envs.n_agents
