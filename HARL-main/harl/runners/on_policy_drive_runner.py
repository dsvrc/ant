"""Runner for DRIVE (HAPPO + Dynamic Reward Incentives for Variable Exchange).

DRIVE (Altmann et al., AAMAS 2026, "Dynamic Incentivized Cooperation under
Changing Rewards") is a decentralized peer-incentivization mechanism: agents
reciprocally exchange *reward differences* (relative to a running epoch-average)
to shape each other's rewards.  Because every incentive is expressed in the same
units as the environment reward, the mechanism is invariant to reward scaling
and shifting -- the exact kind of non-stationarity (thermal-coupling reward
drift) studied on this Ant benchmark -- without any fixed incentive magnitude or
learned incentive function.

How it is adapted here (documented deviations from the social-dilemma original):

* Backbone is HAPPO/EP, identical to the other baselines, so the comparison is
  apples-to-apples.  The DRIVE actor IS the HAPPO actor; everything DRIVE-specific
  lives in this runner.
* DRIVE needs a *per-agent* value function for its TD-advantage gate.  The HARL
  mamujoco ``share_obs`` is identical across agents, so the centralized critic
  would gate everybody the same way (DRIVE no-op).  We therefore add small
  per-agent value nets ``V_i(obs_i)`` -- each agent's ``obs`` carries an agent-id
  one-hot, so the gates genuinely differ.  This mirrors the original DRIVE/MATE
  per-agent critics.
* Reward shaping (paper Algorithm 2) is applied per environment step inside
  ``insert``.  Because EP feeds the critic a single team reward, the critic is
  fed the *mean* of the per-agent DRIVE-shaped rewards (= true team reward + the
  DRIVE incentive).  The per-agent shaped rewards train the per-agent value nets.
* Evaluation and the episodic-return logging are left untouched, so reported
  performance is always the *true* environment return.
"""

import numpy as np
import torch
import torch.nn.functional as F

from harl.models.drive.drive_modules import DriveValueNet, drive_shape_rewards
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class OnPolicyDriveRunner(OnPolicyHARunner):
    """Runner for the DRIVE algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyDriveRunner, self).__init__(args, algo_args, env_args)

        algo_cfg = algo_args["algo"]
        self.drive_coef = float(algo_cfg.get("drive_incentive_coef", 1.0))
        self.drive_value_lr = float(algo_cfg.get("drive_value_lr", 5e-4))
        self.drive_value_epochs = int(algo_cfg.get("drive_value_epochs", 5))
        self.drive_value_hidden = list(
            algo_cfg.get("drive_value_hidden_sizes", algo_args["model"]["hidden_sizes"])
        )
        self.drive_comm_failure_prob = float(
            algo_cfg.get("drive_comm_failure_prob", 0.0)
        )
        self.drive_gamma = float(algo_cfg["gamma"])

        self.drive_steps = 0  # tensorboard x-axis for DRIVE-specific scalars

        if self.algo_args["render"]["use_render"]:
            return

        # ---- per-agent value nets V_i(obs_i) used for TD gating --------------
        value_args = {**algo_args["model"]}
        value_args["hidden_sizes"] = self.drive_value_hidden
        self.drive_value_nets = []
        self.drive_value_optims = []
        for agent_id in range(self.num_agents):
            obs_dim = self.envs.observation_space[agent_id].shape[0]
            net = DriveValueNet(value_args, obs_dim, device=self.device)
            self.drive_value_nets.append(net)
            self.drive_value_optims.append(
                torch.optim.Adam(net.parameters(), lr=self.drive_value_lr)
            )

        # ---- per-step storage of the DRIVE-shaped per-agent rewards ----------
        self.drive_shaped_rewards = np.zeros(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                self.num_agents,
            ),
            dtype=np.float32,
        )

        # running epoch-average reward bar_u_i and rollout diagnostics
        self._reset_epoch_stats()

        # base __init__ already ran restore() for the actors/critic before the
        # value nets existed; load the value-net weights now if resuming.
        if self.algo_args["train"]["model_dir"] is not None:
            self._restore_value_nets()

    # ======================================================================
    # epoch (= one HARL rollout iteration) bookkeeping
    # ======================================================================
    def _reset_epoch_stats(self):
        """Reset the running epoch-average reward and rollout diagnostics."""
        self.drive_rew_sum = np.zeros(self.num_agents, dtype=np.float64)
        self.drive_rew_count = 0
        # diagnostics accumulated over the rollout
        self._stat_steps = 0
        self._stat_win = 0.0
        self._stat_ureq = 0.0
        self._stat_ures = 0.0
        self._stat_true_r = 0.0
        self._stat_shaped_r = 0.0
        self._token_accum = np.zeros(self.num_agents, dtype=np.float64)

    def _current_ubar(self):
        """Running epoch-average reward bar_u_i over the rollout so far."""
        if self.drive_rew_count == 0:
            return np.zeros(self.num_agents, dtype=np.float32)
        return (self.drive_rew_sum / self.drive_rew_count).astype(np.float32)

    def prep_rollout(self):
        super(OnPolicyDriveRunner, self).prep_rollout()
        if hasattr(self, "drive_value_nets"):
            for net in self.drive_value_nets:
                net.eval()

    def prep_training(self):
        super(OnPolicyDriveRunner, self).prep_training()
        for net in self.drive_value_nets:
            net.train()

    def after_update(self):
        super(OnPolicyDriveRunner, self).after_update()
        self._reset_epoch_stats()

    # ======================================================================
    # DRIVE shaping for one environment step
    # ======================================================================
    def _drive_step(self, cur_obs, next_obs, rewards, dones):
        """Run the DRIVE protocol for a single env step (vectorized over threads).

        Args:
            cur_obs:  list over agents of (B, obs_dim) -- s_t.
            next_obs: (B, N, obs_dim) -- s_{t+1}.
            rewards:  (B, N) raw per-agent reward of this step.
            dones:    (B, N) bool.
        Returns:
            shaped: (B, N) DRIVE-shaped per-agent rewards.
        """
        B = rewards.shape[0]
        N = self.num_agents

        # 1) per-agent TD advantage with the (frozen) per-agent value nets
        v_cur = np.zeros((B, N), dtype=np.float32)
        v_next = np.zeros((B, N), dtype=np.float32)
        for i in range(N):
            v_cur[:, i] = self.drive_value_nets[i].values_np(cur_obs[i])
            v_next[:, i] = self.drive_value_nets[i].values_np(next_obs[:, i])
        not_done = 1.0 - dones.astype(np.float32)
        td = rewards + self.drive_gamma * v_next * not_done - v_cur
        winners = td >= 0.0

        # optional communication-failure dropout of requests (default off)
        if self.drive_comm_failure_prob > 0.0:
            drop = np.random.rand(B, N) < self.drive_comm_failure_prob
            winners = winners & (~drop)

        # 2) update running epoch-average reward BEFORE shaping (Algorithm 1)
        self.drive_rew_sum += rewards.sum(axis=0)
        self.drive_rew_count += B
        ubar = self._current_ubar()

        # 3) reward-difference exchange (Algorithm 2)
        shaped, u_req, u_res = drive_shape_rewards(
            rewards, ubar, winners, coef=self.drive_coef
        )

        # 4) diagnostics
        self._stat_steps += 1
        self._stat_win += float(winners.mean())
        self._stat_ureq += float(u_req.mean())
        self._stat_ures += float(u_res.mean())
        self._stat_true_r += float(rewards.mean())
        self._stat_shaped_r += float(shaped.mean())
        self._token_accum += (u_res - u_req).mean(axis=0)
        return shaped

    # ======================================================================
    # insert: shape rewards, then store data exactly like the base runner
    # ======================================================================
    def insert(self, data):
        (
            obs,  # (n_threads, n_agents, obs_dim) == s_{t+1}
            share_obs,
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # --- DRIVE reward shaping --------------------------------------------
        cur_step = self.actor_buffer[0].step  # s_t lives at obs[cur_step]
        cur_obs = [
            self.actor_buffer[i].obs[cur_step] for i in range(self.num_agents)
        ]
        raw_rewards = rewards[:, :, 0]  # (B, N)
        shaped = self._drive_step(cur_obs, obs, raw_rewards, dones)
        self.drive_shaped_rewards[cur_step] = shaped

        # --- standard book-keeping (mirrors OnPolicyBaseRunner.insert) -------
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
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
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
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        # --- feed the critic buffer the DRIVE-shaped reward ------------------
        if self.state_type == "EP":
            # EP uses a single team reward: mean of the per-agent shaped rewards
            # (= true team reward + the mean DRIVE incentive).
            team_reward = shaped.mean(axis=1, keepdims=True)  # (B, 1)
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                team_reward,
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs,
                rnn_states_critic,
                values,
                shaped[:, :, None],
                masks,
                bad_masks,
            )

    # ======================================================================
    # train: HAPPO update + per-agent value-net update + token diagnostics
    # ======================================================================
    def train(self):
        # 1) standard HAPPO actor/critic update on the DRIVE-shaped returns
        actor_train_infos, critic_train_info = super(
            OnPolicyDriveRunner, self
        ).train()

        # 2) update the per-agent DRIVE value nets on per-agent shaped returns
        value_loss = self._update_value_nets()

        # 3) per-epoch dynamic token value (diagnostic, paper's update_step)
        n = max(1, self._stat_steps)
        token_values = (self._token_accum / n).astype(np.float32)

        # 4) log DRIVE-specific scalars
        self.drive_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if self.writter is not None and self._stat_steps > 0:
            self.writter.add_scalar(
                "drive/winner_frac", self._stat_win / n, self.drive_steps
            )
            self.writter.add_scalar(
                "drive/u_req_mean", self._stat_ureq / n, self.drive_steps
            )
            self.writter.add_scalar(
                "drive/u_res_mean", self._stat_ures / n, self.drive_steps
            )
            self.writter.add_scalar(
                "drive/incentive_mean",
                (self._stat_ures - self._stat_ureq) / n,
                self.drive_steps,
            )
            self.writter.add_scalar(
                "drive/true_train_reward", self._stat_true_r / n, self.drive_steps
            )
            self.writter.add_scalar(
                "drive/shaped_train_reward",
                self._stat_shaped_r / n,
                self.drive_steps,
            )
            self.writter.add_scalar(
                "drive/ubar_mean", float(self._current_ubar().mean()), self.drive_steps
            )
            self.writter.add_scalar("drive/value_loss", value_loss, self.drive_steps)
            for i in range(self.num_agents):
                self.writter.add_scalar(
                    "drive/token_value_agent{}".format(i),
                    float(token_values[i]),
                    self.drive_steps,
                )

        return actor_train_infos, critic_train_info

    def _update_value_nets(self):
        """Regress each per-agent value net to its DRIVE-shaped discounted returns."""
        gamma = self.drive_gamma
        total_loss = 0.0
        n_updates = 0
        agents = [0] if self.share_param else list(range(self.num_agents))
        for agent_id in agents:
            net = self.drive_value_nets[agent_id]
            optim = self.drive_value_optims[agent_id]
            ab = self.actor_buffer[agent_id]
            obs_all = ab.obs  # (T+1, B, obs_dim)
            masks = ab.masks  # (T+1, B, 1): 0 at episode boundaries
            T, B = self.drive_shaped_rewards.shape[0], self.drive_shaped_rewards.shape[1]
            shaped_r = self.drive_shaped_rewards[:, :, agent_id]  # (T, B)

            # bootstrap value at the rollout end with the (frozen) current net
            with torch.no_grad():
                v_last = net.values_np(obs_all[-1])  # (B,)

            returns = np.zeros((T + 1, B), dtype=np.float32)
            returns[T] = v_last
            for t in reversed(range(T)):
                returns[t] = (
                    shaped_r[t] + gamma * masks[t + 1, :, 0] * returns[t + 1]
                )

            obs_flat = obs_all[:T].reshape(-1, obs_all.shape[-1])
            target = torch.as_tensor(
                returns[:T].reshape(-1), dtype=torch.float32, device=self.device
            )
            for _ in range(self.drive_value_epochs):
                pred = net(obs_flat).squeeze(-1)
                loss = F.smooth_l1_loss(pred, target)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optim.step()
                total_loss += float(loss.item())
                n_updates += 1

        # parameter sharing: copy agent 0's net into the others
        if self.share_param:
            ref = self.drive_value_nets[0].state_dict()
            for agent_id in range(1, self.num_agents):
                self.drive_value_nets[agent_id].load_state_dict(ref)

        return total_loss / max(1, n_updates)

    # ======================================================================
    # save / restore (also persist the per-agent value nets)
    # ======================================================================
    def save(self):
        super(OnPolicyDriveRunner, self).save()
        for agent_id in range(self.num_agents):
            torch.save(
                self.drive_value_nets[agent_id].state_dict(),
                str(self.save_dir) + "/drive_value_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        super(OnPolicyDriveRunner, self).restore()
        # ``drive_value_nets`` may not exist yet: base ``__init__`` calls
        # ``restore`` before the subclass builds them. They are loaded later
        # via ``_restore_value_nets`` at the end of ``__init__``.
        if self.algo_args["render"]["use_render"] or not hasattr(
            self, "drive_value_nets"
        ):
            return
        self._restore_value_nets()

    def _restore_value_nets(self):
        for agent_id in range(self.num_agents):
            state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/drive_value_agent"
                + str(agent_id)
                + ".pt"
            )
            self.drive_value_nets[agent_id].load_state_dict(state_dict)
