"""Runner for DORAEMON (Domain Randomization via Entropy Maximization).

DORAEMON is an *automatic domain-randomization curriculum* that wraps an
otherwise-unchanged RL learner -- here HAPPO.  The policy, critic, observations
and buffers are all standard HAPPO; DORAEMON only acts on the **environment side**
by maintaining a distribution over bounded dynamics factors and gradually widening
it (maximizing its entropy) as long as the policy keeps succeeding.

Per DORAEMON iteration (a fixed number of HARL training episodes) this runner:

1. trains HAPPO under the current dynamics distribution ``nu_phi_i`` (each episode
   the env samples fresh dynamics from ``nu_phi_i`` and applies them);
2. harvests, from the ``info`` dict, the dynamics ``xi`` and episode return of
   every *completed* episode that was sampled under the current distribution;
3. solves the DORAEMON optimization (``DoraemonUpdater``) to obtain ``nu_phi_{i+1}``
   -- minimal KL to the max-entropy target s.t. an importance-sampling success
   constraint and a KL trust region -- and broadcasts it to the training envs.

Evaluation is intentionally run with DR **off** (on the plain test env), so the
DORAEMON eval number is directly comparable to the other non-stationary-Ant
baselines.  Domain randomization is a training-time curriculum only.
"""

import numpy as np

from harl.algorithms.doraemon import DomainRandDistribution, DoraemonUpdater
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


# Default dynamics-randomization dimensions for the mamujoco Ant. Each entry is a
# multiplicative factor (around the nominal model value) bounded in [m, M]; the
# DORAEMON curriculum expands each Beta from a narrow init toward a wide target.
_DEFAULT_DR_SPEC = [
    {"param": "mass", "m": 0.5, "M": 1.5},
    {"param": "damping", "m": 0.5, "M": 1.5},
    {"param": "friction", "m": 0.5, "M": 1.5},
    {"param": "gravity", "m": 0.8, "M": 1.2},
    {"param": "gain", "m": 0.5, "M": 1.5},
]


class OnPolicyDoraemonRunner(OnPolicyHARunner):
    """Runner for the DORAEMON algorithm (HAPPO + entropy-maximizing DR curriculum)."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyDoraemonRunner, self).__init__(args, algo_args, env_args)

        cfg = algo_args["algo"]
        dr_spec = cfg.get("doraemon_dr_spec", _DEFAULT_DR_SPEC)
        init_beta = float(cfg.get("doraemon_init_beta", 100.0))
        target_beta = float(cfg.get("doraemon_target_beta", 1.0))

        self.doraemon_iter_episodes = int(cfg.get("doraemon_iter_episodes", 10))
        self.doraemon_max_buffer = int(cfg.get("doraemon_max_buffer", 2000))
        self.doraemon_min_samples = int(cfg.get("doraemon_min_samples", 50))

        # DORAEMON randomizes bounded *dynamics* parameters of the environment.
        # That is only available for mamujoco (MujocoMulti exposes
        # ``set_dr_distribution`` and writes ``dr_dynamics``/``dr_return`` to info).
        # SMAC/SMACv2 expose no randomizable dynamics, so there is nothing to
        # randomize -- DORAEMON then degrades to plain HAPPO. We detect that and
        # skip the (inert) DR curriculum with a clear notice.
        self.dr_enabled = args["env"] == "mamujoco"
        if not self.dr_enabled:
            print(
                "[DORAEMON] env '%s' exposes no randomizable dynamics; DORAEMON "
                "reduces to plain HAPPO (no domain-randomization curriculum "
                "applied)." % args["env"]
            )

        # static per-dim mapping (which mujoco property / indices each dim scales)
        self.dr_static_spec = []
        init_list, target_list = [], []
        for dim in dr_spec:
            static = {"param": dim["param"]}
            if "indices" in dim:
                static["indices"] = dim["indices"]
            self.dr_static_spec.append(static)
            init_list.append(
                {"m": dim["m"], "M": dim["M"], "a": init_beta, "b": init_beta}
            )
            target_list.append(
                {"m": dim["m"], "M": dim["M"], "a": target_beta, "b": target_beta}
            )

        self.init_distr = DomainRandDistribution(init_list)
        self.target_distr = DomainRandDistribution(target_list)

        self.updater = DoraemonUpdater(
            init_distr=self.init_distr,
            target_distr=self.target_distr,
            kl_upper_bound=float(cfg.get("doraemon_kl_bound", 0.5)),
            alpha=float(cfg.get("doraemon_alpha", 0.5)),
            return_threshold=float(cfg.get("doraemon_return_threshold", 1000.0)),
            success_mode=cfg.get("doraemon_success_mode", "success_rate"),
            robust_estimate=bool(cfg.get("doraemon_robust_estimate", False)),
            alpha_ci=float(cfg.get("doraemon_alpha_ci", 0.9)),
            train_until_performance_lb=bool(
                cfg.get("doraemon_train_until_lb", True)
            ),
            min_dynamics_samples=self.doraemon_min_samples,
            init_beta_param=init_beta,
            verbose=1,
        )

        # per-iteration (dynamics, return) collection buffers
        self._dyn_buf = []
        self._ret_buf = []
        self.dr_version = 0
        self.doraemon_iter = 0
        self.episodes_since_update = 0
        self._global_steps = 0

        if self.algo_args["render"]["use_render"] or not self.dr_enabled:
            return

        # push the initial (narrow) distribution to the training envs so the very
        # first rollouts already sample dynamics from nu_phi_0
        self.envs.set_dr_distribution(
            self._full_spec(self.updater.current_distr), self.dr_version
        )
        if self.writter is not None:
            self._log_doraemon(
                {
                    "updated": False,
                    "n_samples": 0,
                    "entropy": float(self.init_distr.entropy().item()),
                    "kl_from_target": float(
                        self.init_distr.kl_divergence(self.target_distr).item()
                    ),
                    "kl_step": 0.0,
                    "train_success_rate": 0.0,
                    "est_success": 0.0,
                }
            )

    # ------------------------------------------------------------------ helpers
    def _full_spec(self, distr):
        """Merge the static param mapping with the distribution's current Beta params."""
        d = distr.get()
        spec = []
        for i, static in enumerate(self.dr_static_spec):
            spec.append(
                {
                    **static,
                    "m": d[i]["m"],
                    "M": d[i]["M"],
                    "a": d[i]["a"],
                    "b": d[i]["b"],
                }
            )
        return spec

    def _harvest(self, infos):
        """Collect (dynamics, return) of completed episodes from the info dicts."""
        for info in infos:
            info0 = info[0] if isinstance(info, (list, tuple, np.ndarray)) else info
            if not isinstance(info0, dict) or "dr_dynamics" not in info0:
                continue
            if info0.get("dr_version", -1) != self.dr_version:
                continue  # episode sampled under a stale distribution -> skip
            dyn = info0.get("dr_dynamics")
            if dyn is None:
                continue
            self._dyn_buf.append(np.asarray(dyn, dtype=np.float64))
            self._ret_buf.append(float(info0.get("dr_return", 0.0)))

        # cap the buffer to the most recent samples
        if len(self._dyn_buf) > self.doraemon_max_buffer:
            self._dyn_buf = self._dyn_buf[-self.doraemon_max_buffer :]
            self._ret_buf = self._ret_buf[-self.doraemon_max_buffer :]

    def _log_doraemon(self, info):
        if self.writter is None:
            return
        step = self._global_steps
        for key in [
            "entropy",
            "kl_from_target",
            "kl_step",
            "train_success_rate",
            "est_success",
            "n_samples",
        ]:
            self.writter.add_scalar("doraemon/" + key, float(info[key]), step)
        self.writter.add_scalar("doraemon/updated", float(bool(info["updated"])), step)
        self.writter.add_scalar("doraemon/iter", self.doraemon_iter, step)
        distr = self.updater.current_distr.get()
        for i, d in enumerate(distr):
            mean_factor = d["m"] + (d["M"] - d["m"]) * d["a"] / (d["a"] + d["b"])
            self.writter.add_scalar(f"doraemon/dim{i}_a", float(d["a"]), step)
            self.writter.add_scalar(f"doraemon/dim{i}_b", float(d["b"]), step)
            self.writter.add_scalar(f"doraemon/dim{i}_mean", float(mean_factor), step)

    # ------------------------------------------------------------------- insert
    def insert(self, data):
        # standard HAPPO insert (obs / critic / buffers unchanged)
        super(OnPolicyDoraemonRunner, self).insert(data)
        # then harvest per-episode dynamics + returns reported in info
        if self.dr_enabled:
            self._harvest(data[4])

    # -------------------------------------------------------------------- train
    def train(self):
        actor_train_infos, critic_train_info = super(
            OnPolicyDoraemonRunner, self
        ).train()

        self._global_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.episodes_since_update += 1

        # one DORAEMON iteration = doraemon_iter_episodes HARL episodes, provided
        # enough completed episodes have been collected for a stable IS estimate
        if self.dr_enabled and self.episodes_since_update >= self.doraemon_iter_episodes:
            if len(self._dyn_buf) >= self.doraemon_min_samples:
                self._doraemon_update()
                self.episodes_since_update = 0
            # else: keep collecting; re-check next episode (counter stays tripped)

        return actor_train_infos, critic_train_info

    def _doraemon_update(self):
        """Solve the DORAEMON optimization and broadcast the new distribution."""
        dynamics = np.stack(self._dyn_buf, axis=0)
        returns = np.asarray(self._ret_buf, dtype=np.float64)

        info = self.updater.update(dynamics, returns)
        self.doraemon_iter += 1

        # push the (possibly updated) distribution under a fresh version tag, and
        # clear the buffer so the next iteration only uses freshly sampled dynamics
        self.dr_version += 1
        self.envs.set_dr_distribution(
            self._full_spec(self.updater.current_distr), self.dr_version
        )
        self._dyn_buf.clear()
        self._ret_buf.clear()

        self._log_doraemon(info)
        if info["updated"]:
            print(
                f"[DORAEMON] iter {self.doraemon_iter}: entropy={info['entropy']:.3f} "
                f"kl_from_target={info['kl_from_target']:.3f} "
                f"kl_step={info['kl_step']:.3f} "
                f"train_succ={info['train_success_rate']:.3f} "
                f"est_succ={info['est_success']:.3f} | "
                f"{self.updater.current_distr.to_string()}"
            )
