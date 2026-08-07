"""Runner for WISDOM (HAPPO + Wavelet Predictive Representations).

WISDOM conditions the policy on a wavelet predictive representation ``pred_z`` of
the task. This runner mirrors the COREP / ESCP runners:

* it augments every agent's observation (and the centralized critic's state) with
  the agent's wavelet representation ``pred_z`` (dim ``latent_dim``),
* carries each agent's previous raw observation across the rollout so the encoder
  can be fed full transitions ``(prev_obs, action, reward, obs)``,
* after each HAPPO update, trains the encoder + wavelet networks with the WISDOM
  objective (encoder KL-to-prior; wavelet prediction loss + wavelet TD loss).
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


class OnPolicyWisdomRunner(OnPolicyHARunner):
    """Runner for the WISDOM algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyWisdomRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.latent_dim = int(algo_cfg.get("wisdom_latent_dim", 5))
        self.use_wisdom_critic = bool(algo_cfg.get("use_wisdom_critic", True))
        self.wisdom_steps = 0

        if self.algo_args["render"]["use_render"]:
            return

        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.raw_share_dim = self.envs.share_observation_space[0].shape[0]
        self.episode_length = self.algo_args["train"]["episode_length"]
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        # actor buffers store augmented observations [raw_obs, pred_z]
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

        if self.use_wisdom_critic:
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

        # per-agent rollout state: current wavelet representation + previous raw obs
        self.wisdom_pred_z = [None for _ in range(self.num_agents)]
        self.wisdom_prev_obs = [None for _ in range(self.num_agents)]
        # rollout reward / validity sequences for the representation update
        self.wisdom_rewards = np.zeros((self.episode_length, n_threads, 1), dtype=np.float32)
        self.wisdom_mask = np.ones((self.episode_length, n_threads, 1), dtype=np.float32)
        self._t_idx = 0

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _augmented_box(dim):
        low = np.full(dim, -np.inf, dtype=np.float32)
        return Box(low=low, high=-low, dtype=np.float32)

    def _all_z(self):
        return np.concatenate(
            [self.wisdom_pred_z[a] for a in range(self.num_agents)], axis=-1
        )

    def _augment_share(self, share_obs, all_z):
        n_agents = share_obs.shape[1]
        all_z_b = np.repeat(all_z[:, None, :], n_agents, axis=1)
        return np.concatenate([share_obs, all_z_b], axis=-1)

    # ======================================================================
    # warmup
    # ======================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self._t_idx = 0

        for agent_id in range(self.num_agents):
            pred_z = self.actor[agent_id].init_latent(n_threads)
            self.wisdom_pred_z[agent_id] = pred_z
            self.wisdom_prev_obs[agent_id] = obs[:, agent_id].copy()
            aug_obs = np.concatenate([obs[:, agent_id], pred_z], axis=-1)
            self.actor_buffer[agent_id].obs[0] = aug_obs.copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        if self.use_wisdom_critic:
            aug_share = self._augment_share(share_obs, self._all_z())
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

        dones_env = np.all(dones, axis=1)

        # advance each agent's wavelet representation from the latest transition
        for agent_id in range(self.num_agents):
            pred_z = self.actor[agent_id].step_latent(
                self.wisdom_prev_obs[agent_id],
                actions[:, agent_id],
                rewards[:, agent_id],
                obs[:, agent_id],
            )
            self.wisdom_pred_z[agent_id] = pred_z
            self.wisdom_prev_obs[agent_id] = obs[:, agent_id].copy()

        # store reward / validity sequences for the representation update
        if self._t_idx < self.episode_length:
            self.wisdom_rewards[self._t_idx] = rewards[:, 0]
            self.wisdom_mask[self._t_idx] = (1.0 - dones_env.astype(np.float32)).reshape(
                -1, 1
            )
        self._t_idx += 1

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
            aug_obs = np.concatenate(
                [obs[:, agent_id], self.wisdom_pred_z[agent_id]], axis=-1
            )
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

        if self.use_wisdom_critic:
            aug_share = self._augment_share(share_obs, self._all_z())
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
        super(OnPolicyWisdomRunner, self).after_update()
        self._t_idx = 0

    # ======================================================================
    # train: HAPPO update + WISDOM representation update
    # ======================================================================
    def train(self):
        actor_train_infos, critic_train_info = super(
            OnPolicyWisdomRunner, self
        ).train()

        repr_agents = [0] if self.share_param else list(range(self.num_agents))
        accum = {}
        n_upd = 0
        for agent_id in repr_agents:
            ab = self.actor_buffer[agent_id]
            obs_seq = ab.obs[:, :, : self.raw_obs_dims[agent_id]]  # (T+1, B, raw)
            act_seq = ab.actions  # (T, B, act)
            info = self.actor[agent_id].update_representation(
                obs_seq, act_seq, self.wisdom_rewards, self.wisdom_mask
            )
            n_upd += 1
            for k, v in info.items():
                accum[k] = accum.get(k, 0.0) + v

        self.wisdom_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if n_upd > 0 and self.writter is not None:
            for k, v in accum.items():
                self.writter.add_scalar("wisdom/" + k, v / n_upd, self.wisdom_steps)

        return actor_train_infos, critic_train_info

    # ======================================================================
    # eval: deterministic wavelet representation
    # ======================================================================
    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_pred_z = [
            self.actor[a].init_latent(n_eval) for a in range(self.num_agents)
        ]
        eval_prev_obs = [eval_obs[:, a].copy() for a in range(self.num_agents)]

        eval_rnn_states = np.zeros(
            (n_eval, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id], eval_pred_z[agent_id]], axis=-1
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

            for agent_id in range(self.num_agents):
                eval_pred_z[agent_id] = self.actor[agent_id].step_latent(
                    eval_prev_obs[agent_id],
                    eval_actions[:, agent_id],
                    eval_rewards[:, agent_id],
                    eval_obs[:, agent_id],
                )
                eval_prev_obs[agent_id] = eval_obs[:, agent_id].copy()

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
    # save / restore (also persist the WISDOM modules)
    # ======================================================================
    def save(self):
        super(OnPolicyWisdomRunner, self).save()
        for agent_id in range(self.num_agents):
            torch.save(
                self.actor[agent_id].encoder.state_dict(),
                str(self.save_dir) + "/wisdom_encoder_agent" + str(agent_id) + ".pt",
            )
            torch.save(
                self.actor[agent_id].z_model.state_dict(),
                str(self.save_dir) + "/wisdom_zmodel_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        super(OnPolicyWisdomRunner, self).restore()
        for agent_id in range(self.num_agents):
            self.actor[agent_id].encoder.load_state_dict(
                torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/wisdom_encoder_agent"
                    + str(agent_id)
                    + ".pt"
                )
            )
            self.actor[agent_id].z_model.load_state_dict(
                torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/wisdom_zmodel_agent"
                    + str(agent_id)
                    + ".pt"
                )
            )
