"""Runner for LCPO (HAPPO + Locally Constrained Policy Optimization).

LCPO only modifies the *actor update*: it adds a soft KL penalty that anchors the
policy on out-of-distribution (past-context) states. Therefore this runner is a
thin wrapper around the on-policy HA runner -- rollout, the critic, and the
buffers are all unchanged. The only additions are:

* a per-agent OOD buffer of past observations, and
* a ``train`` override that, before the HAPPO update, feeds the rollout's
  observations into each agent's OOD buffer, samples distant (past-context)
  anchor states, and hands them to the agent's LCPO actor (which then applies the
  local KL constraint inside its update).
"""

import numpy as np

from harl.common.lcpo_ood_buffer import OODBuffer
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class OnPolicyLcpoRunner(OnPolicyHARunner):
    """Runner for the LCPO algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyLcpoRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.lcpo_ood_batch = int(algo_cfg.get("lcpo_ood_batch", 2048))
        ood_capacity = int(algo_cfg.get("lcpo_ood_capacity", 100000))
        ood_window = int(algo_cfg.get("lcpo_recent_window", 20000))
        ood_type = algo_cfg.get("lcpo_ood_type", "diag_mahala")
        ood_thresh = float(algo_cfg.get("lcpo_ood_thresh", -2.0))
        ood_thresh_mode = algo_cfg.get("lcpo_ood_thresh_mode", "absolute")
        ood_percentile = float(algo_cfg.get("lcpo_ood_percentile", 20.0))

        self.ood_rng = np.random.RandomState(seed=algo_args["seed"]["seed"])
        self.lcpo_steps = 0
        self._lcpo_iter = 0

        # cache the knobs so the diagnostic header shows exactly what is running
        self._lcpo_cfg = {
            "kappa": float(algo_cfg.get("lcpo_kappa", 1.0)),
            "ood_type": ood_type,
            "thresh_mode": ood_thresh_mode,
            "thresh": ood_thresh,
            "percentile": ood_percentile,
            "ood_batch": self.lcpo_ood_batch,
            "ood_minibatch": int(algo_cfg.get("lcpo_ood_minibatch", 1024)),
            "recent_window": ood_window,
            "capacity": ood_capacity,
        }

        if self.algo_args["render"]["use_render"]:
            return

        # one OOD buffer per agent
        self.ood_buffers = []
        for agent_id in range(self.num_agents):
            obs_dim = self.envs.observation_space[agent_id].shape[0]
            self.ood_buffers.append(
                OODBuffer(
                    obs_len=obs_dim,
                    recent_window=ood_window,
                    capacity=ood_capacity,
                    dist_type=ood_type,
                    thresh=ood_thresh,
                    thresh_mode=ood_thresh_mode,
                    percentile=ood_percentile,
                )
            )

    def train(self):
        """LCPO training: prepare OOD anchors, then run the HAPPO update."""
        self._lcpo_iter += 1

        # 1) update each agent's OOD buffer with the freshly-collected observations
        #    and sample distant (past-context) anchor states for the local constraint.
        ood_sizes = []
        for agent_id in range(self.num_agents):
            ab = self.actor_buffer[agent_id]
            obs_np = ab.obs[:-1].reshape(-1, *ab.obs.shape[2:]).astype(np.float32)
            self.ood_buffers[agent_id].add_many(obs_np, self.ood_rng)
            ood = self.ood_buffers[agent_id].get(self.ood_rng, self.lcpo_ood_batch)
            self.actor[agent_id].set_ood(ood)
            ood_sizes.append(len(ood))

        # 2) standard HAPPO actor (now with the LCPO OOD KL penalty) + critic update
        actor_train_infos, critic_train_info = super(OnPolicyLcpoRunner, self).train()

        # 3) log LCPO diagnostics (tensorboard every iter, console every log_interval)
        self.lcpo_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if self.writter is not None:
            for agent_id in range(self.num_agents):
                st = self.ood_buffers[agent_id].last_stats
                act = self.actor[agent_id]
                self.writter.add_scalar(
                    "lcpo/ood_size_agent%i" % agent_id,
                    ood_sizes[agent_id],
                    self.lcpo_steps,
                )
                self.writter.add_scalar(
                    "lcpo/ood_kl_agent%i" % agent_id,
                    act.last_ood_kl_avg,
                    self.lcpo_steps,
                )
                self.writter.add_scalar(
                    "lcpo/ood_penalty_ratio_agent%i" % agent_id,
                    act.last_ood_ratio,
                    self.lcpo_steps,
                )
                if st is not None:
                    self.writter.add_scalar(
                        "lcpo/frac_distant_agent%i" % agent_id,
                        st["frac_distant"],
                        self.lcpo_steps,
                    )
                    self.writter.add_scalar(
                        "lcpo/score_median_agent%i" % agent_id,
                        st["pct"][50],
                        self.lcpo_steps,
                    )

        if self._lcpo_iter % self.algo_args["train"]["log_interval"] == 0:
            self._print_lcpo_diag(ood_sizes)

        # 4) clear the anchors for the next iteration
        for agent_id in range(self.num_agents):
            self.actor[agent_id].clear_ood()

        return actor_train_infos, critic_train_info

    def _print_lcpo_diag(self, ood_sizes):
        """Print a per-agent LCPO diagnostic block (the values used for tuning).

        Read this to tune:
          * ``lcpo_ood_thresh``  -> ``distant X/N (pp%)`` + the score percentiles.
            For ``diag_mahala``/``mahala_full`` a state is OOD when ll < thresh, so
            to flag ~p% of candidates set thresh near the p-th score percentile.
            If frac_distant is ~0, LCPO is silently degenerating to plain HAPPO.
          * ``lcpo_kappa``  -> ``ood_kl`` and ``pen/|pol|`` (how hard the constraint
            bites relative to the policy-gradient loss). A few % to ~30% is a sane
            operating range; >>100% means kappa is over-constraining.
          * buffer fills tell you whether ``recent_window`` / ``capacity`` are filled.
        Early training (before ~half a day/night cycle of history) legitimately
        shows frac_distant ~ 0 -- the FIFO has not yet seen other-phase states.
        """
        c = self._lcpo_cfg
        if c["thresh_mode"] == "percentile":
            sel = "mode=percentile p={p}%".format(p=c["percentile"])
        else:
            sel = "mode=absolute thresh={th}".format(th=c["thresh"])
        print(
            "\n===== LCPO diag | iter {it} | env-steps {step} | "
            "kappa={k} type={t} {sel} ood_batch={ob} recent_window={rw} =====".format(
                it=self._lcpo_iter,
                step=self.lcpo_steps,
                k=c["kappa"],
                t=c["ood_type"],
                sel=sel,
                ob=c["ood_batch"],
                rw=c["recent_window"],
            )
        )
        for agent_id in range(self.num_agents):
            act = self.actor[agent_id]
            st = self.ood_buffers[agent_id].last_stats
            if st is None:
                print(
                    "  agent {a} | OOD buffer warming up (no candidates yet) | "
                    "anchors={n}".format(a=agent_id, n=ood_sizes[agent_id])
                )
                continue
            p = st["pct"]
            print(
                "  agent {a} | seen={seen} fifo={ff} recent={rf} | "
                "distant {nd}/{nc} ({fd:.1f}%) cut={cut:.3f} | "
                "{lbl}[min/p5/p10/p25/p50/max]="
                "{mn:.3f}/{p5:.3f}/{p10:.3f}/{p25:.3f}/{p50:.3f}/{mx:.3f} | "
                "anchors={anch} ood_kl={kl:.4f} pen={pen:.4f} "
                "(pen/|pol|={ratio:.1f}% of |pol|={pol:.4f})".format(
                    a=agent_id,
                    seen=st["total_seen"],
                    ff=st["fifo_fill"],
                    rf=st["recent_fill"],
                    nd=st["n_distant"],
                    nc=st["n_candidates"],
                    fd=100.0 * st["frac_distant"],
                    cut=st["eff_thresh"],
                    lbl=st["label"],
                    mn=p[0],
                    p5=p[5],
                    p10=p[10],
                    p25=p[25],
                    p50=p[50],
                    mx=p[100],
                    anch=ood_sizes[agent_id],
                    kl=act.last_ood_kl_avg,
                    pen=act.last_ood_pen_avg,
                    ratio=100.0 * act.last_ood_ratio,
                    pol=act.last_pol_loss_avg,
                )
            )
        print("=" * 70, flush=True)
