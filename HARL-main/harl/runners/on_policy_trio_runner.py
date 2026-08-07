"""Runner for TRIO (HAPPO + task-tracking variational inference).

TRIO conditions the policy on a task latent. Following the original method:

* During TRAINING the policy/critic are conditioned on the **oracle** task -- the
  env's exposed non-stationary context (``info["ambient"]`` for the TCC Ant) --
  appended to the observation, and the per-step task labels are collected.
* After each HAPPO update the per-agent VI network is trained **supervised** to
  predict the task from a (action, reward, next_obs) context window.
* During EVALUATION the oracle is replaced by an online **Thompson sample** from
  the inferred posterior (the VI Bayesian filter), so the policy runs without any
  privileged information. An optional linear forecaster anticipates the next
  latent (the continuous analogue of TRIO's GP task-tracking).

Because the policy is oracle-conditioned at train time, TRIO's true
(inference-only) performance is the EVAL curve, not the training reward.
"""

import numpy as np
import torch

from gym.spaces import Box
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.critics.v_critic import VCritic
from harl.models.trio.trio_modules import compute_linear_weight
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


class OnPolicyTrioRunner(OnPolicyHARunner):
    """Runner for the TRIO algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyTrioRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.latent_dim = int(algo_cfg.get("trio_latent_dim", 1))
        self.task_keys = algo_cfg.get("trio_task_keys", ["ambient"])
        if isinstance(self.task_keys, str):
            self.task_keys = [self.task_keys]
        assert len(self.task_keys) == self.latent_dim, (
            "trio_latent_dim must equal the number of trio_task_keys."
        )
        self.use_trio_critic = bool(algo_cfg.get("use_trio_critic", True))
        self.prior_mu = float(algo_cfg.get("trio_prior_mu", 0.5))
        self.use_forecast = bool(algo_cfg.get("trio_use_forecast", True))
        self.forecast_len = int(algo_cfg.get("trio_forecast_len", 10))
        self.trio_steps = 0

        if self.use_forecast:
            self._fw = compute_linear_weight(self.forecast_len).to(self.device)

        if self.algo_args["render"]["use_render"]:
            return

        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.raw_share_dim = self.envs.share_observation_space[0].shape[0]
        self.episode_length = self.algo_args["train"]["episode_length"]
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        # actor buffers store augmented observations [raw_obs, task_latent]
        self.actor_buffer = []
        for agent_id in range(self.num_agents):
            aug_space = self._augmented_box(self.raw_obs_dims[agent_id] + self.latent_dim)
            self.actor_buffer.append(
                OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    aug_space,
                    self.envs.action_space[agent_id],
                )
            )

        if self.use_trio_critic:
            aug_share_dim = self.raw_share_dim + self.num_agents * self.latent_dim
            aug_share_space = self._augmented_box(aug_share_dim)
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]},
                aug_share_space,
                device=self.device,
            )
            if self.state_type == "EP":
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    aug_share_space,
                )
            elif self.state_type == "FP":
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    aug_share_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

        # per-iteration task labels (the oracle task per transition) for VI training
        self.trio_labels = np.zeros(
            (self.episode_length, n_threads, self.latent_dim), dtype=np.float32
        )
        self.cur_task = np.full(
            (n_threads, self.latent_dim), self.prior_mu, dtype=np.float32
        )
        self._t_idx = 0

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _augmented_box(dim):
        low = np.full(dim, -np.inf, dtype=np.float32)
        return Box(low=low, high=-low, dtype=np.float32)

    def _extract_task(self, infos):
        """Pull the oracle task out of the env info dicts.

        Returns (n_threads, latent_dim). The configured ``trio_task_keys`` are
        used per latent dimension (Ant: "ambient"). For a scalar task
        (latent_dim == 1) we fall back to the known SMAC/SMACv2 non-stationarity
        drivers ("snd_payload" / "cwd_payload") when the configured key is absent,
        so the same config runs across all three envs. Missing keys default to
        the prior mean, in which case TRIO degrades gracefully to plain HAPPO.
        """
        n_threads = len(infos)
        task = np.full((n_threads, self.latent_dim), self.prior_mu, dtype=np.float32)
        for i in range(n_threads):
            info_i = infos[i]
            # infos[i] is the per-agent list of info dicts; take agent 0's dict
            if isinstance(info_i, (list, tuple, np.ndarray)):
                info_i = info_i[0]
            if not isinstance(info_i, dict):
                continue
            for d, k in enumerate(self.task_keys):
                if k in info_i:
                    task[i, d] = float(info_i[k])
                elif self.latent_dim == 1:
                    alt = next(
                        (kk for kk in ("snd_payload", "cwd_payload") if kk in info_i),
                        None,
                    )
                    if alt is not None:
                        task[i, d] = float(info_i[alt])
        return task

    def _all_task(self):
        """Concatenate every agent's task latent -> (n_threads, n_agents*latent_dim).

        During training every agent uses the shared oracle task ``self.cur_task``.
        """
        return np.tile(self.cur_task, (1, self.num_agents))

    def _augment_share(self, share_obs, all_task):
        n_threads, n_agents = share_obs.shape[0], share_obs.shape[1]
        all_task_b = np.repeat(all_task[:, None, :], n_agents, axis=1)
        return np.concatenate([share_obs, all_task_b], axis=-1)

    # ======================================================================
    # warmup
    # ======================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self.cur_task = np.full(
            (n_threads, self.latent_dim), self.prior_mu, dtype=np.float32
        )
        self._t_idx = 0

        for agent_id in range(self.num_agents):
            aug_obs = np.concatenate([obs[:, agent_id], self.cur_task], axis=-1)
            self.actor_buffer[agent_id].obs[0] = aug_obs.copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        if self.use_trio_critic:
            aug_share = self._augment_share(share_obs, self._all_task())
            if self.state_type == "EP":
                self.critic_buffer.share_obs[0] = aug_share[:, 0].copy()
            elif self.state_type == "FP":
                self.critic_buffer.share_obs[0] = aug_share.copy()
        else:
            if self.state_type == "EP":
                self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
            elif self.state_type == "FP":
                self.critic_buffer.share_obs[0] = share_obs.copy()

    # ======================================================================
    # insert (oracle task augmentation + label collection)
    # ======================================================================
    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # oracle task for this transition (and label for VI training)
        self.cur_task = self._extract_task(infos)
        if self._t_idx < self.episode_length:
            self.trio_labels[self._t_idx] = self.cur_task
        self._t_idx += 1

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            aug_obs = np.concatenate([obs[:, agent_id], self.cur_task], axis=-1)
            self.actor_buffer[agent_id].insert(
                aug_obs,
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if self.use_trio_critic:
            aug_share = self._augment_share(share_obs, self._all_task())
        else:
            aug_share = share_obs

        if self.state_type == "EP":
            self.critic_buffer.insert(
                aug_share[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                aug_share, rnn_states_critic, values, rewards, masks, bad_masks
            )

    def after_update(self):
        super(OnPolicyTrioRunner, self).after_update()
        self._t_idx = 0

    # ======================================================================
    # train: HAPPO update + supervised VI update
    # ======================================================================
    def train(self):
        actor_train_infos, critic_train_info = super(OnPolicyTrioRunner, self).train()

        vae_agents = [0] if self.share_param else list(range(self.num_agents))
        vi_accum = {}
        n_vi = 0
        for agent_id in vae_agents:
            ab = self.actor_buffer[agent_id]
            obs_seq = ab.obs[:, :, : self.raw_obs_dims[agent_id]]  # (T+1, threads, raw)
            act_seq = ab.actions  # (T, threads, act_dim)
            rew_seq = self.critic_buffer.rewards  # (T, threads, 1), shared reward
            mask_seq = ab.masks[1:]
            for _ in range(self.actor[agent_id].num_vi_updates):
                info = self.actor[agent_id].update_vi(
                    obs_seq, act_seq, rew_seq, self.trio_labels, mask_seq
                )
                n_vi += 1
                for k, v in info.items():
                    vi_accum[k] = vi_accum.get(k, 0.0) + v

        self.trio_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if n_vi > 0 and self.writter is not None:
            for k, v in vi_accum.items():
                self.writter.add_scalar("trio/" + k, v / n_vi, self.trio_steps)

        return actor_train_infos, critic_train_info

    # ======================================================================
    # eval: online Bayesian filter + Thompson-sampled task latent
    # ======================================================================
    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # per-agent belief (prior), GRU hidden, sample count, and mu history
        priors = [self.actor[a].init_belief(n_eval) for a in range(self.num_agents)]
        hiddens = [None for _ in range(self.num_agents)]
        seq_lens = [0 for _ in range(self.num_agents)]
        z_samples = [priors[a][:, : self.latent_dim].copy() for a in range(self.num_agents)]
        mu_hist = [[] for _ in range(self.num_agents)]

        eval_rnn_states = np.zeros(
            (n_eval, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id], z_samples[agent_id]], axis=-1
                )
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    aug_obs,
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            # online Bayesian filter: update belief and Thompson-sample the latent
            for agent_id in range(self.num_agents):
                posterior, z, hiddens[agent_id], seq_lens[agent_id] = self.actor[
                    agent_id
                ].filter_step(
                    eval_actions[:, agent_id],
                    eval_rewards[:, agent_id],
                    eval_obs[:, agent_id],
                    priors[agent_id],
                    hiddens[agent_id],
                    seq_lens[agent_id],
                )
                z_samples[agent_id] = z
                # carry posterior forward as next prior (tracking)
                priors[agent_id] = posterior.copy()
                # optional forecast: anticipate the next latent mean
                if self.use_forecast:
                    mu_hist[agent_id].append(posterior[:, : self.latent_dim].copy())
                    if len(mu_hist[agent_id]) >= self.forecast_len:
                        recent = np.stack(
                            mu_hist[agent_id][-self.forecast_len :], axis=0
                        )  # (L, n_eval, latent)
                        recent_t = torch.as_tensor(
                            recent, dtype=torch.float32, device=self.device
                        )
                        # least-squares line per (thread, dim), extrapolate one step
                        L = self.forecast_len
                        flat = recent_t.reshape(L, -1)  # (L, n_eval*latent)
                        w = torch.matmul(self._fw, flat)  # (2, n_eval*latent)
                        fut = torch.tensor(
                            [[float(L + 1), 1.0]], device=self.device
                        )
                        pred = torch.matmul(fut, w).reshape(
                            n_eval, self.latent_dim
                        )
                        priors[agent_id][:, : self.latent_dim] = pred.cpu().numpy()

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(eval_data)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(n_eval):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

    # ======================================================================
    # save / restore (also persist the VI networks)
    # ======================================================================
    def save(self):
        super(OnPolicyTrioRunner, self).save()
        for agent_id in range(self.num_agents):
            torch.save(
                self.actor[agent_id].vi.state_dict(),
                str(self.save_dir) + "/trio_vi_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        super(OnPolicyTrioRunner, self).restore()
        for agent_id in range(self.num_agents):
            self.actor[agent_id].vi.load_state_dict(
                torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/trio_vi_agent"
                    + str(agent_id)
                    + ".pt"
                )
            )
