"""Runner for ESCP (HAPPO + Environment-Sensitive Contextual Policy).

ESCP conditions the policy on an environment embedding produced by an Environment
Probe (EP), trained by the RMDM loss. This runner mirrors the COREP runner:

* it augments every agent's observation (and the critic's state) with the EP
  embedding (detached -- the EP is never trained by the policy gradient),
* carries the per-agent EP GRU hidden state across the rollout (reset every
  ``history_len`` steps -> history truncation, and on episode end),
* collects discretized ``ambient`` bins as pseudo-task ids, and
* after each HAPPO update trains the EP with the RMDM loss on those bins.
"""

import numpy as np
import torch
from gym.spaces import Box

from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.critics.v_critic import VCritic
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


class OnPolicyEscpRunner(OnPolicyHARunner):
    """Runner for the ESCP algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyEscpRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.ep_dim = int(algo_cfg.get("escp_ep_dim", 2))
        self.history_len = int(algo_cfg.get("escp_history_len", 16))
        self.num_task_bins = int(algo_cfg.get("escp_num_task_bins", 8))
        # Privileged non-stationarity signal used to form pseudo-task bins. Ant
        # exposes "ambient"; SMAC-SND exposes "snd_payload"; SMACv2-CWD exposes
        # "cwd_payload". We try the configured key first, then fall back through
        # the known env keys, so the same config works across all three envs. If
        # none is present, bins stay 0 (invalid) and RMDM degrades to plain HAPPO.
        self.task_key = algo_cfg.get("escp_task_key", "ambient")
        self.task_keys = [self.task_key, "snd_payload", "cwd_payload", "ambient"]
        self.task_min = float(algo_cfg.get("escp_task_min", 0.0))
        self.task_max = float(algo_cfg.get("escp_task_max", 1.0))
        self.use_escp_critic = bool(algo_cfg.get("use_escp_critic", True))
        self.escp_steps = 0

        if self.algo_args["render"]["use_render"]:
            return

        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.raw_share_dim = self.envs.share_observation_space[0].shape[0]
        self.episode_length = self.algo_args["train"]["episode_length"]
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        # actor buffers store augmented observations [raw_obs, ep]
        self.actor_buffer = []
        for agent_id in range(self.num_agents):
            aug_space = self._augmented_box(self.raw_obs_dims[agent_id] + self.ep_dim)
            self.actor_buffer.append(
                OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    aug_space,
                    self.envs.action_space[agent_id],
                )
            )

        if self.use_escp_critic:
            aug_share_dim = self.raw_share_dim + self.num_agents * self.ep_dim
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

        # per-agent EP rollout state
        self.escp_ep = [None for _ in range(self.num_agents)]
        self.escp_hidden = [None for _ in range(self.num_agents)]
        # discretized ambient bins per transition (for RMDM)
        self.task_bins = np.zeros(
            (self.episode_length, n_threads), dtype=np.int64
        )
        self._t_idx = 0
        self._ep_count = 0

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _augmented_box(dim):
        low = np.full(dim, -np.inf, dtype=np.float32)
        return Box(low=low, high=-low, dtype=np.float32)

    def _all_ep(self):
        return np.concatenate(
            [self.escp_ep[a] for a in range(self.num_agents)], axis=-1
        )

    def _augment_share(self, share_obs, all_ep):
        n_threads, n_agents = share_obs.shape[0], share_obs.shape[1]
        all_ep_b = np.repeat(all_ep[:, None, :], n_agents, axis=1)
        return np.concatenate([share_obs, all_ep_b], axis=-1)

    def _extract_bins(self, infos):
        """Discretize the env's ambient context into task bins (1..K; 0=invalid)."""
        n_threads = len(infos)
        bins = np.zeros(n_threads, dtype=np.int64)
        span = max(self.task_max - self.task_min, 1e-6)
        for i in range(n_threads):
            info_i = infos[i]
            if isinstance(info_i, (list, tuple, np.ndarray)):
                info_i = info_i[0]
            if not isinstance(info_i, dict):
                continue
            key = next((k for k in self.task_keys if k in info_i), None)
            if key is not None:
                val = float(info_i[key])
                frac = (val - self.task_min) / span
                b = int(np.clip(np.floor(frac * self.num_task_bins), 0, self.num_task_bins - 1))
                bins[i] = b + 1  # 1..K (0 reserved for invalid)
        return bins

    # ======================================================================
    # warmup
    # ======================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self._t_idx = 0
        self._ep_count = 0

        # zero "no previous action" vector, sized to the *encoded* action dim
        # (act_dim = shape[0] for Box, n for one-hot Discrete).
        zero_action = [
            np.zeros((n_threads, self.actor[a].act_dim), dtype=np.float32)
            for a in range(self.num_agents)
        ]
        for agent_id in range(self.num_agents):
            ep, hidden = self.actor[agent_id].step_ep(
                obs[:, agent_id], zero_action[agent_id], None, deterministic=False
            )
            self.escp_ep[agent_id] = ep
            self.escp_hidden[agent_id] = hidden
            aug_obs = np.concatenate([obs[:, agent_id], ep], axis=-1)
            self.actor_buffer[agent_id].obs[0] = aug_obs.copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        if self.use_escp_critic:
            aug_share = self._augment_share(share_obs, self._all_ep())
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
    # insert
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

        # task bins for RMDM
        bins = self._extract_bins(infos)
        if self._t_idx < self.episode_length:
            self.task_bins[self._t_idx] = bins
        self._t_idx += 1
        self._ep_count += 1
        reset_ep_hidden = (self._ep_count % self.history_len) == 0

        dones_env = np.all(dones, axis=1)

        # advance each agent's environment probe (history-truncated).
        # Encode discrete actions to one-hot before feeding the probe (no-op Box).
        for agent_id in range(self.num_agents):
            last_a = self.actor[agent_id].encode_last_action(actions[:, agent_id])
            ep, hidden = self.actor[agent_id].step_ep(
                obs[:, agent_id],
                last_a,
                self.escp_hidden[agent_id],
                deterministic=False,
            )
            self.escp_ep[agent_id] = ep
            # reset hidden on episode end or at window boundaries (truncation)
            if hidden is not None:
                done_mask = torch.as_tensor(
                    dones[:, agent_id].astype(np.float32), device=hidden.device
                ).reshape(1, -1, 1)
                hidden = hidden * (1 - done_mask)
            if reset_ep_hidden:
                hidden = None
            self.escp_hidden[agent_id] = hidden

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
            aug_obs = np.concatenate([obs[:, agent_id], self.escp_ep[agent_id]], axis=-1)
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

        if self.use_escp_critic:
            aug_share = self._augment_share(share_obs, self._all_ep())
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
        super(OnPolicyEscpRunner, self).after_update()
        self._t_idx = 0

    # ======================================================================
    # train: HAPPO update + RMDM environment-probe update
    # ======================================================================
    def train(self):
        actor_train_infos, critic_train_info = super(OnPolicyEscpRunner, self).train()

        rmdm_agents = [0] if self.share_param else list(range(self.num_agents))
        accum = {}
        n_upd = 0
        for agent_id in rmdm_agents:
            ab = self.actor_buffer[agent_id]
            obs_seq = ab.obs[:, :, : self.raw_obs_dims[agent_id]]  # (T+1, threads, raw)
            act_seq = ab.actions  # (T, threads, act)
            for _ in range(self.actor[agent_id].num_rmdm_updates):
                info = self.actor[agent_id].update_rmdm(obs_seq, act_seq, self.task_bins)
                n_upd += 1
                for k, v in info.items():
                    accum[k] = accum.get(k, 0.0) + v

        self.escp_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if n_upd > 0 and self.writter is not None:
            for k, v in accum.items():
                self.writter.add_scalar("escp/" + k, v / n_upd, self.escp_steps)
            n_bins_seen = len(self.actor[0].rmdm.mean_vector)
            self.writter.add_scalar("escp/n_bins_seen", n_bins_seen, self.escp_steps)

        return actor_train_infos, critic_train_info

    # ======================================================================
    # eval: deterministic environment probe
    # ======================================================================
    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_ep = [None for _ in range(self.num_agents)]
        eval_hidden = [None for _ in range(self.num_agents)]
        ep_count = 0
        for agent_id in range(self.num_agents):
            zero_a = np.zeros((n_eval, self.actor[agent_id].act_dim), dtype=np.float32)
            ep, hidden = self.actor[agent_id].step_ep(
                eval_obs[:, agent_id], zero_a, None, deterministic=True
            )
            eval_ep[agent_id] = ep
            eval_hidden[agent_id] = hidden

        eval_rnn_states = np.zeros(
            (n_eval, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id], eval_ep[agent_id]], axis=-1
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

            ep_count += 1
            reset_ep = (ep_count % self.history_len) == 0
            for agent_id in range(self.num_agents):
                last_a = self.actor[agent_id].encode_last_action(eval_actions[:, agent_id])
                ep, hidden = self.actor[agent_id].step_ep(
                    eval_obs[:, agent_id],
                    last_a,
                    eval_hidden[agent_id],
                    deterministic=True,
                )
                eval_ep[agent_id] = ep
                if hidden is not None:
                    done_mask = torch.as_tensor(
                        eval_dones[:, agent_id].astype(np.float32), device=hidden.device
                    ).reshape(1, -1, 1)
                    hidden = hidden * (1 - done_mask)
                if reset_ep:
                    hidden = None
                eval_hidden[agent_id] = hidden

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
    # save / restore (also persist the environment probes)
    # ======================================================================
    def save(self):
        super(OnPolicyEscpRunner, self).save()
        for agent_id in range(self.num_agents):
            torch.save(
                self.actor[agent_id].ep.state_dict(),
                str(self.save_dir) + "/escp_ep_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        super(OnPolicyEscpRunner, self).restore()
        for agent_id in range(self.num_agents):
            self.actor[agent_id].ep.load_state_dict(
                torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/escp_ep_agent"
                    + str(agent_id)
                    + ".pt"
                )
            )
