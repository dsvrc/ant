"""Runner for COREP (HAPPO + Causal-Origin REPresentation).

This runner extends the on-policy HARL runner with the COREP machinery:

* It augments every agent's observation (and, by default, the centralized critic's
  state) with the agent's causal-origin latent ``(mu, logvar)`` produced by the
  COREP encoder.
* During rollout/eval it carries the per-agent GRU belief hidden state across
  steps and episodes, resetting it on ``done`` (exactly like VariBAD).
* After each HAPPO update it (a) trains the COREP VAE (reward+state reconstruction,
  KL, and the graph/sparsity/magnitude graph-regularization losses) and (b) runs
  the guided-updating test on the shared critic TD-error to decide whether the
  stable GAT should be frozen for the next iteration.
"""

from collections import deque

import numpy as np
import torch
from gym.spaces import Box

from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.critics.v_critic import VCritic
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


class OnPolicyCorepRunner(OnPolicyHARunner):
    """Runner for the COREP algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyCorepRunner, self).__init__(args, algo_args, env_args)

        # ---- COREP / guided-updating configuration ----------------------------
        algo_cfg = algo_args["algo"]
        self.latent_dim = int(algo_cfg.get("corep_latent_dim", 8))
        self.latent_concat_dim = 2 * self.latent_dim
        self.use_corep_critic = bool(algo_cfg.get("use_corep_critic", True))

        # guided-updating (TD-error gated freeze of the stable GAT)
        self.td_buffer = deque(maxlen=int(algo_cfg.get("td_buffer_size", 2000)))
        self.td_recent_size = float(algo_cfg.get("td_recent_size", 0.1))
        self.confidence = float(algo_cfg.get("confidence", 1.96))
        self.freeze_eps = float(algo_cfg.get("freeze_eps", 0.05))
        self.freeze = False

        self.corep_steps = 0  # for tensorboard x-axis of COREP-specific scalars

        if self.algo_args["render"]["use_render"]:
            return

        # ---- rebuild buffers / critic on the AUGMENTED spaces -----------------
        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.raw_share_dim = self.envs.share_observation_space[0].shape[0]

        # actor buffers store augmented observations [raw_obs, latent]
        self.actor_buffer = []
        for agent_id in range(self.num_agents):
            aug_obs_space = self._augmented_box(
                self.raw_obs_dims[agent_id] + self.latent_concat_dim
            )
            self.actor_buffer.append(
                OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    aug_obs_space,
                    self.envs.action_space[agent_id],
                )
            )

        # critic uses augmented share-obs [raw_share, all-agent latents]
        if self.use_corep_critic:
            aug_share_dim = self.raw_share_dim + self.num_agents * self.latent_concat_dim
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

        # ---- per-agent rollout belief state -----------------------------------
        self.corep_hidden = [None for _ in range(self.num_agents)]
        self.corep_latent = [None for _ in range(self.num_agents)]

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _augmented_box(dim):
        low = np.full(dim, -np.inf, dtype=np.float32)
        return Box(low=low, high=-low, dtype=np.float32)

    def _all_latents(self):
        """Concatenate every agent's latent -> (n_threads, n_agents * 2 * latent_dim)."""
        return np.concatenate(
            [self.corep_latent[a] for a in range(self.num_agents)], axis=-1
        )

    def _augment_share(self, share_obs, all_lat):
        """Augment share_obs with the concatenated latents.

        share_obs: EP (n_threads, n_agents, raw_share) or FP same shape.
        all_lat:   (n_threads, n_agents * 2 * latent_dim).
        Returns the augmented share_obs with the same leading shape.
        """
        n_threads, n_agents = share_obs.shape[0], share_obs.shape[1]
        all_lat_b = np.repeat(all_lat[:, None, :], n_agents, axis=1)
        return np.concatenate([share_obs, all_lat_b], axis=-1)

    # ======================================================================
    # warmup: initialise buffers AND the prior belief
    # ======================================================================
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        # prior latents / hidden states
        for agent_id in range(self.num_agents):
            lat, hid = self.actor[agent_id].init_latent(n_threads)
            self.corep_latent[agent_id] = lat
            self.corep_hidden[agent_id] = hid

        # augmented actor observations
        for agent_id in range(self.num_agents):
            aug_obs = np.concatenate(
                [obs[:, agent_id], self.corep_latent[agent_id]], axis=-1
            )
            self.actor_buffer[agent_id].obs[0] = aug_obs.copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        # augmented critic state
        if self.use_corep_critic:
            aug_share = self._augment_share(share_obs, self._all_latents())
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
    # insert: advance belief then store augmented data
    # ======================================================================
    def insert(self, data):
        (
            obs,  # (n_threads, n_agents, raw_obs_dim)
            share_obs,  # (n_threads, n_agents, raw_share_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # list
            available_actions,
            values,
            actions,  # (n_threads, n_agents, act_dim)
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # --- advance the per-agent belief with the just-observed transition ---
        for agent_id in range(self.num_agents):
            lat, hid = self.actor[agent_id].step_latent(
                actions[:, agent_id],
                rewards[:, agent_id],
                obs[:, agent_id],
                dones[:, agent_id],
                self.corep_hidden[agent_id],
            )
            self.corep_latent[agent_id] = lat
            self.corep_hidden[agent_id] = hid

        dones_env = np.all(dones, axis=1)  # env done iff all agents done

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

        # --- store augmented actor observations -------------------------------
        for agent_id in range(self.num_agents):
            aug_obs = np.concatenate(
                [obs[:, agent_id], self.corep_latent[agent_id]], axis=-1
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

        # --- store augmented critic state -------------------------------------
        if self.use_corep_critic:
            aug_share = self._augment_share(share_obs, self._all_latents())
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
    # train: HAPPO update + COREP VAE update + guided-updating test
    # ======================================================================
    def train(self):
        # 1) standard HAPPO actor update (sequential, factor) + critic update
        actor_train_infos, critic_train_info = super(OnPolicyCorepRunner, self).train()

        # 2) COREP VAE update (reward+state recon, KL, graph regularization)
        rew_seq = self.critic_buffer.rewards  # (T, n_threads, 1), shared reward
        vae_agents = [0] if self.share_param else list(range(self.num_agents))
        vae_info_accum = {}
        n_vae = 0
        for agent_id in vae_agents:
            ab = self.actor_buffer[agent_id]
            obs_seq = ab.obs[:, :, : self.raw_obs_dims[agent_id]]  # (T+1, threads, raw)
            act_seq = ab.actions  # (T, threads, act_dim)
            mask_seq = ab.masks[1:]  # (T, threads, 1): 0 at episode boundaries
            for _ in range(self.actor[agent_id].num_vae_updates):
                info = self.actor[agent_id].update_vae(
                    obs_seq, act_seq, rew_seq, mask_seq, self.freeze
                )
                n_vae += 1
                for k, v in info.items():
                    vae_info_accum[k] = vae_info_accum.get(k, 0.0) + v

        # 3) guided-updating: decide whether to freeze the stable GAT next time
        td = float(critic_train_info["value_loss"])
        if len(self.td_buffer) > 0:
            td_mean = float(np.mean(self.td_buffer))
            td_std = float(np.std(self.td_buffer))
        else:
            td_mean, td_std = 0.0, 0.0
        self.td_buffer.append(td)
        recent_n = max(1, int(len(self.td_buffer) * self.td_recent_size))
        recent_td = float(np.mean(np.array(self.td_buffer)[-recent_n:]))
        if (
            (recent_td < td_mean + self.confidence * td_std)
            and (recent_td > td_mean - self.confidence * td_std)
            and (np.random.rand() > self.freeze_eps)
        ):
            self.freeze = True
        else:
            self.freeze = False

        # 4) log COREP-specific scalars
        self.corep_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if n_vae > 0 and self.writter is not None:
            for k, v in vae_info_accum.items():
                self.writter.add_scalar(
                    "corep/" + k, v / n_vae, self.corep_steps
                )
            self.writter.add_scalar("corep/freeze", float(self.freeze), self.corep_steps)
            self.writter.add_scalar("corep/td_recent", recent_td, self.corep_steps)

        return actor_train_infos, critic_train_info

    # ======================================================================
    # eval: maintain belief states during deterministic evaluation
    # ======================================================================
    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_hidden = [None for _ in range(self.num_agents)]
        eval_latent = [None for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            lat, hid = self.actor[agent_id].init_latent(n_eval)
            eval_latent[agent_id] = lat
            eval_hidden[agent_id] = hid

        eval_rnn_states = np.zeros(
            (n_eval, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_eval, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                aug_obs = np.concatenate(
                    [eval_obs[:, agent_id], eval_latent[agent_id]], axis=-1
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

            # advance belief states with the new transition
            for agent_id in range(self.num_agents):
                lat, hid = self.actor[agent_id].step_latent(
                    eval_actions[:, agent_id],
                    eval_rewards[:, agent_id],
                    eval_obs[:, agent_id],
                    eval_dones[:, agent_id],
                    eval_hidden[agent_id],
                )
                eval_latent[agent_id] = lat
                eval_hidden[agent_id] = hid

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

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
    # save / restore (also persist the COREP modules)
    # ======================================================================
    def save(self):
        super(OnPolicyCorepRunner, self).save()
        for agent_id in range(self.num_agents):
            torch.save(
                self.actor[agent_id].encoder.state_dict(),
                str(self.save_dir) + "/corep_encoder_agent" + str(agent_id) + ".pt",
            )
            if self.actor[agent_id].reward_decoder is not None:
                torch.save(
                    self.actor[agent_id].reward_decoder.state_dict(),
                    str(self.save_dir) + "/corep_rew_dec_agent" + str(agent_id) + ".pt",
                )
            if self.actor[agent_id].state_decoder is not None:
                torch.save(
                    self.actor[agent_id].state_decoder.state_dict(),
                    str(self.save_dir) + "/corep_state_dec_agent" + str(agent_id) + ".pt",
                )

    def restore(self):
        super(OnPolicyCorepRunner, self).restore()
        for agent_id in range(self.num_agents):
            enc_path = (
                str(self.algo_args["train"]["model_dir"])
                + "/corep_encoder_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].encoder.load_state_dict(torch.load(enc_path))
            if self.actor[agent_id].reward_decoder is not None:
                self.actor[agent_id].reward_decoder.load_state_dict(
                    torch.load(
                        str(self.algo_args["train"]["model_dir"])
                        + "/corep_rew_dec_agent"
                        + str(agent_id)
                        + ".pt"
                    )
                )
            if self.actor[agent_id].state_decoder is not None:
                self.actor[agent_id].state_decoder.load_state_dict(
                    torch.load(
                        str(self.algo_args["train"]["model_dir"])
                        + "/corep_state_dec_agent"
                        + str(agent_id)
                        + ".pt"
                    )
                )
