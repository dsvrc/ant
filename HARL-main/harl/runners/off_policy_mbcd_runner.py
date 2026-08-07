"""Runner for MBCD (Model-Based Context Detection) on top of HASAC.

MBCD (Alegre et al., AAMAS 2021) detects non-stationary context changes online via
a CUSUM changepoint statistic on dynamics-model log-likelihoods, and maintains a
growing *library* of context-specific dynamics models and policies. This runner
adapts it onto HARL's off-policy multi-agent SAC (HASAC):

* A single probabilistic dynamics-model ensemble per detected context predicts the
  *global* transition ``(share_obs, joint_action) -> (reward, delta share_obs)``;
  the CUSUM detector runs on the thread-0 transition stream (the non-stationary
  context is global, shared by all parallel envs in lock-step).
* When a change is detected the runner saves the current context's policy (all
  agents' actors + the twin critic + entropy temperatures) into the library and
  either loads the matching known context's policy (switch) or warm-starts a new
  context from the current policy (spawn) -- mirroring the reference, whose
  new-model branch keeps the current policy and only resets the dynamics model.
* The dynamics models are trained periodically on each context's data and are used
  purely for detection (MBCD's contribution). The MBPO model-rollout augmentation
  of the original is intentionally omitted: feeding model-generated rollouts into
  HARL's per-agent buffer would require reconstructing each agent's normalized
  observation from a predicted global state, which is env-specific and fragile --
  so the RL backbone is HASAC's standard off-policy learning on real transitions.

Everything else (the HASAC critic/actor updates) is the unchanged off-policy HA
runner. When no change is ever detected MBCD degrades gracefully to HASAC.
"""

from copy import deepcopy

import numpy as np
import torch

from harl.algorithms.mbcd import MBCDDetector
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.utils.discrete_util import get_encoded_act_dim


class OffPolicyMbcdRunner(OffPolicyHARunner):
    """Runner for the MBCD algorithm (HASAC + model-based context detection)."""

    def __init__(self, args, algo_args, env_args):
        super(OffPolicyMbcdRunner, self).__init__(args, algo_args, env_args)

        cfg = algo_args["algo"]
        self.mbcd_clear_buffer_on_change = bool(
            cfg.get("mbcd_clear_buffer_on_change", False)
        )
        self.auto_alpha = bool(cfg.get("auto_alpha", False))

        # Joint-action dimensionality for the dynamics model. Continuous Box
        # actions contribute shape[0]; discrete (SMAC/SMACv2) actions are one-hot
        # encoded, contributing n per agent. ``_encode_joint_action`` builds the
        # matching vector from the stored (integer-index) discrete actions.
        self.mbcd_act_spaces = self.envs.action_space
        state_dim = self.envs.share_observation_space[0].shape[0]
        action_dim = int(
            sum(
                get_encoded_act_dim(self.mbcd_act_spaces[a])
                for a in range(self.num_agents)
            )
        )

        self.detector = MBCDDetector(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            cusum_threshold=float(cfg.get("mbcd_cusum_threshold", 100.0)),
            max_std=float(cfg.get("mbcd_max_std", 0.5)),
            num_stds=float(cfg.get("mbcd_num_stds", 2.0)),
            min_steps=int(cfg.get("mbcd_min_steps", 1000)),
            memory_capacity=int(cfg.get("mbcd_memory_capacity", 100000)),
            num_networks=int(cfg.get("mbcd_num_networks", 5)),
            num_elites=int(cfg.get("mbcd_num_elites", 2)),
            hidden_size=int(cfg.get("mbcd_dynamics_hidden", 200)),
            num_layers=int(cfg.get("mbcd_dynamics_layers", 4)),
            lr=float(cfg.get("mbcd_dynamics_lr", 1e-3)),
        )

        self.mbcd_current = 0
        self.policy_lib = {}
        self._mbcd_env_steps = 0
        self._n_threads = self.algo_args["train"]["n_rollout_threads"]

        # cache for (optional) fresh-buffer construction on context change
        self._buffer_ctor_args = (
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )

    # ======================================================================
    # per-context policy library (state-dict snapshots)
    # ======================================================================
    def _snapshot_policy(self):
        snap = {
            "actors": [
                deepcopy(self.actor[a].actor.state_dict())
                for a in range(self.num_agents)
            ],
            "critic": deepcopy(self.critic.critic.state_dict()),
            "critic2": deepcopy(self.critic.critic2.state_dict()),
            "target_critic": deepcopy(self.critic.target_critic.state_dict()),
            "target_critic2": deepcopy(self.critic.target_critic2.state_dict()),
        }
        if self.auto_alpha:
            snap["log_alpha"] = [
                self.log_alpha[a].detach().clone() for a in range(self.num_agents)
            ]
        return snap

    def _load_policy(self, snap):
        for a in range(self.num_agents):
            self.actor[a].actor.load_state_dict(snap["actors"][a])
        self.critic.critic.load_state_dict(snap["critic"])
        self.critic.critic2.load_state_dict(snap["critic2"])
        self.critic.target_critic.load_state_dict(snap["target_critic"])
        self.critic.target_critic2.load_state_dict(snap["target_critic2"])
        if self.auto_alpha and "log_alpha" in snap:
            for a in range(self.num_agents):
                with torch.no_grad():
                    self.log_alpha[a].copy_(snap["log_alpha"][a])
                self.alpha[a] = torch.exp(self.log_alpha[a].detach())

    def _fresh_buffer(self):
        if self.state_type == "EP":
            return OffPolicyBufferEP(*self._buffer_ctor_args)
        return OffPolicyBufferFP(*self._buffer_ctor_args)

    def _encode_joint_action(self, act0):
        """Build the dynamics-model joint action for one env thread.

        Args:
            act0: (np.ndarray) (n_agents, stored_act_dim) actions for thread 0.
                  Discrete actions are stored as integer indices (stored dim 1).
        Returns:
            (np.ndarray) flat joint action; discrete sub-actions one-hot encoded,
            continuous sub-actions concatenated as-is.
        """
        parts = []
        for a in range(self.num_agents):
            sp = self.mbcd_act_spaces[a]
            if sp.__class__.__name__ == "Discrete":
                idx = int(np.asarray(act0[a]).reshape(-1)[0])
                onehot = np.zeros(int(sp.n), dtype=np.float32)
                onehot[idx] = 1.0
                parts.append(onehot)
            else:
                parts.append(np.asarray(act0[a], dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    # ======================================================================
    # detection + context switching, woven into the off-policy collection loop
    # ======================================================================
    def insert(self, data):
        # 1) standard HASAC buffer insert (also corrects terminal next-states in place)
        super(OffPolicyMbcdRunner, self).insert(data)

        # 2) extract the thread-0 GLOBAL transition for the detector
        share_obs = data[0]  # (n_threads, n_agents, share_dim)
        actions = data[2]  # (n_agents, n_threads, act_dim)
        rewards = data[4]  # (n_threads, n_agents, 1)
        dones = data[5]  # (n_threads, n_agents)
        next_share_obs = data[7]  # (n_threads, n_agents, share_dim) [corrected]

        state = np.asarray(share_obs[0, 0], dtype=np.float32)
        # thread-0 joint action, one-hot encoded for discrete action spaces
        joint_action = self._encode_joint_action(actions[:, 0, :])
        reward = float(rewards[0, 0, 0])
        next_state = np.asarray(next_share_obs[0, 0], dtype=np.float32)
        done = bool(np.all(dones[0]))

        # 3) CUSUM detection step
        changed, current, is_new = self.detector.step(
            state, joint_action, reward, next_state, done
        )

        # 4) on change: save the old context's policy and switch / spawn
        if changed:
            old = self.mbcd_current
            self.policy_lib[old] = self._snapshot_policy()
            if is_new:
                # spawn: warm-start the new context from the current policy
                # (mirrors the reference, whose new-model branch keeps the policy)
                pass
            elif current in self.policy_lib:
                self._load_policy(self.policy_lib[current])
            self.mbcd_current = current
            if self.mbcd_clear_buffer_on_change:
                self.buffer = self._fresh_buffer()
            print(
                f"[MBCD] context change -> model {current} "
                f"({'NEW' if is_new else 'switch'}) at env step "
                f"{self._mbcd_env_steps}; num_models={self.detector.num_models}"
            )

        # 5) add the transition to the current context's dynamics dataset
        self.detector.add_experience(state, joint_action, reward, next_state)

        # 6) periodically (re)train the current context's dynamics model
        counter = self.detector.counter
        if counter < 250:
            model_train_freq = 10
        elif counter < 5000:
            model_train_freq = 100
        elif counter < 40000:
            model_train_freq = 250
        else:
            model_train_freq = 2000
        if (changed and counter > 10) or (counter % model_train_freq == 0):
            self.detector.train_current_model(
                batch_size=int(self.algo_args["algo"].get("mbcd_model_batch", 256))
            )

        # 7) diagnostics
        self._mbcd_env_steps += self._n_threads
        if self.writter is not None and (counter % 200 == 0):
            self.writter.add_scalar(
                "mbcd/num_models", self.detector.num_models, self._mbcd_env_steps
            )
            self.writter.add_scalar(
                "mbcd/current_model", self.mbcd_current, self._mbcd_env_steps
            )
            self.writter.add_scalar(
                "mbcd/S_max", float(max(self.detector.S.values())), self._mbcd_env_steps
            )
            self.writter.add_scalar(
                "mbcd/var_mean_current",
                float(self.detector.var_mean[self.detector.current_model]),
                self._mbcd_env_steps,
            )

    # ======================================================================
    # use the HASAC code paths (get_actions / train branch on algo == "hasac")
    # ======================================================================
    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            return super(OffPolicyMbcdRunner, self).get_actions(
                obs, available_actions, add_random
            )
        finally:
            self.args["algo"] = orig

    def train(self):
        # guard training against an empty buffer right after an optional clear
        if self.buffer.cur_size < self.buffer.batch_size:
            return
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            super(OffPolicyMbcdRunner, self).train()
        finally:
            self.args["algo"] = orig
