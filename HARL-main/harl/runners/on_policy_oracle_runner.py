"""Runner for the Oracle baseline (HAPPO + privileged hidden context).

This runner augments every agent's observation (and the centralized critic state)
with the true non-stationary context read from the env ``info`` dict -- e.g.
``ambient`` (day/night phase), ``heat`` (mean motor heat), ``derate`` (mean motor
gain), or, with the instrumented env, the full per-leg vectors. The augmentation
is applied at BOTH training and evaluation (the oracle is privileged everywhere),
so the resulting return is the performance ceiling for the task.

Everything else is plain HAPPO -- there is no encoder, no representation loss, and
no inference; ``train`` is inherited unchanged.
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


class OnPolicyOracleRunner(OnPolicyHARunner):
    """Runner for the Oracle baseline."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyOracleRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.oracle_dim = int(algo_cfg.get("oracle_dim", 3))
        self.oracle_keys = algo_cfg.get("oracle_keys", ["ambient", "heat", "derate"])
        if isinstance(self.oracle_keys, str):
            self.oracle_keys = [self.oracle_keys]
        self.use_oracle_critic = bool(algo_cfg.get("use_oracle_critic", True))

        if self.algo_args["render"]["use_render"]:
            return

        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.raw_share_dim = self.envs.share_observation_space[0].shape[0]
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        # actor buffers store augmented observations [raw_obs, oracle_context]
        self.actor_buffer = []
        for agent_id in range(self.num_agents):
            aug_space = self._augmented_box(self.raw_obs_dims[agent_id] + self.oracle_dim)
            self.actor_buffer.append(
                OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    aug_space,
                    self.envs.action_space[agent_id],
                )
            )

        if self.use_oracle_critic:
            aug_share_dim = self.raw_share_dim + self.oracle_dim
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

        self.cur_oracle = np.zeros((n_threads, self.oracle_dim), dtype=np.float32)

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _augmented_box(dim):
        low = np.full(dim, -np.inf, dtype=np.float32)
        return Box(low=low, high=-low, dtype=np.float32)

    def _extract_oracle(self, infos):
        """Read the privileged context vector from the env info dicts.

        Each key may be a scalar or a list/array; all are flattened and
        concatenated, then padded/truncated to ``oracle_dim``.
        Returns (n_threads, oracle_dim).
        """
        n_threads = len(infos)
        out = np.zeros((n_threads, self.oracle_dim), dtype=np.float32)
        for i in range(n_threads):
            info_i = infos[i]
            if isinstance(info_i, (list, tuple, np.ndarray)):
                info_i = info_i[0]
            if not isinstance(info_i, dict):
                continue
            vals = []
            for k in self.oracle_keys:
                if k in info_i:
                    v = info_i[k]
                    if np.isscalar(v):
                        vals.append(float(v))
                    else:
                        vals.extend(
                            [float(x) for x in np.asarray(v, dtype=np.float32).flatten()]
                        )
            if len(vals) > 0:
                v = np.asarray(vals[: self.oracle_dim], dtype=np.float32)
                out[i, : len(v)] = v
        return out

    def _augment_share(self, share_obs, oracle):
        """Append the (global) oracle context to each agent's share-obs slot."""
        n_threads, n_agents = share_obs.shape[0], share_obs.shape[1]
        oracle_b = np.repeat(oracle[:, None, :], n_agents, axis=1)
        return np.concatenate([share_obs, oracle_b], axis=-1)

    # ======================================================================
    # warmup
    # ======================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self.cur_oracle = np.zeros((n_threads, self.oracle_dim), dtype=np.float32)

        for agent_id in range(self.num_agents):
            aug_obs = np.concatenate([obs[:, agent_id], self.cur_oracle], axis=-1)
            self.actor_buffer[agent_id].obs[0] = aug_obs.copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        if self.use_oracle_critic:
            aug_share = self._augment_share(share_obs, self.cur_oracle)
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

        # read the true hidden context for this transition
        self.cur_oracle = self._extract_oracle(infos)

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
            aug_obs = np.concatenate([obs[:, agent_id], self.cur_oracle], axis=-1)
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

        if self.use_oracle_critic:
            aug_share = self._augment_share(share_obs, self.cur_oracle)
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

    # ======================================================================
    # eval: augment with the true context too (oracle is privileged everywhere)
    # ======================================================================
    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_oracle = np.zeros((n_eval, self.oracle_dim), dtype=np.float32)

        eval_rnn_states = np.zeros(
            (n_eval, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id], eval_oracle], axis=-1
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
            eval_oracle = self._extract_oracle(eval_infos)

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
