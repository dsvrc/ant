"""Payload-aligned replay buffer for the ``hasac_diag`` telemetry runner (§3.2.1).

``OffPolicyBufferEP`` + two parallel arrays filled at insertion:

* ``payload_diag[slot]`` — the true ``pcr_payload`` of that transition;
* ``insert_step[slot]`` — the env step at which it was inserted.

Together they give the campaign's two phase-resolved telemetry readings:
**TD-error by phase** (H-C5 / the average-game trap) and **replay-age by phase**
(H-C2 / who-overwrites-whom).

**The sampler never reads either array.** They are diagnostics, not method
components (campaign Prohibition 1 — no replay shaping). The pattern is copied
from ``EclOffPolicyBufferEP`` rather than imported, per the same rule: the ECL
buffer *does* steer sampling, and inheriting from it would smuggle a method into
a measurement.

``gather()`` mirrors ``sample()`` for caller-supplied indices **without touching
any RNG**, so telemetry can look at the replay distribution without shifting the
training run's random stream by a single draw.
"""

import numpy as np
import torch

from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP


class DiagOffPolicyBufferEP(OffPolicyBufferEP):
    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        super().__init__(args, share_obs_space, num_agents, obs_spaces, act_spaces)
        self.payload_diag = np.full(self.buffer_size, np.nan, dtype=np.float32)
        self.insert_step = np.zeros(self.buffer_size, dtype=np.int64)
        self.total_inserted = 0
        self._cur_payload = None
        self._cur_env_step = 0

    # ---- runner hooks ----------------------------------------------------
    def stash_meta(self, payload, env_step):
        """Called by the runner just BEFORE insert (idx not yet advanced)."""
        self._cur_payload = np.asarray(payload, dtype=np.float32)
        self._cur_env_step = int(env_step)

    def _stash_into(self, arr, vals):
        vals = np.asarray(vals)
        length = vals.shape[0]
        start = self.idx
        end = start + length
        if end <= self.buffer_size:
            arr[start:end] = vals
        else:
            n1 = self.buffer_size - start
            arr[start:] = vals[:n1]
            arr[: end - self.buffer_size] = vals[n1:]

    def insert(self, data):
        length = data[0].shape[0]
        if self._cur_payload is not None and self._cur_payload.shape[0] == length:
            self._stash_into(self.payload_diag, self._cur_payload)
        self._stash_into(self.insert_step, np.full(length, self._cur_env_step,
                                                   dtype=np.int64))
        super().insert(data)
        self.total_inserted += length

    # ---- sampling --------------------------------------------------------
    def sample(self):
        """Unchanged math; records the drawn indices for telemetry."""
        out = super().sample()
        return out

    def gather(self, indice):
        """``sample()`` for caller-supplied indices, consuming NO randomness.

        Mirrors ``OffPolicyBufferEP.sample()`` line for line from the n-step
        accumulation onward (continuous-action path only — this runner exists for
        the mamujoco campaign). If the base's sample() ever changes, this must be
        re-synced; the runner's identity self-test compares the two.
        """
        assert self.act_spaces[0].__class__.__name__ == "Box", (
            "DiagOffPolicyBufferEP.gather supports the continuous-action path only."
        )
        indice = np.asarray(indice, dtype=np.int64)
        n = indice.shape[0]
        self.update_end_flag()

        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array([self.obs[a][indice] for a in range(self.num_agents)])
        sp_actions = np.array([self.actions[a][indice] for a in range(self.num_agents)])
        sp_valid_transitions = np.array(
            [self.valid_transitions[a][indice] for a in range(self.num_agents)]
        )

        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [self.next_obs[a][indices[-1]] for a in range(self.num_agents)]
        )

        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((n, 1))
        gammas = np.full(n, self.n_step)
        for k in range(self.n_step - 1, -1, -1):
            now = indices[k]
            gammas[self.end_flag[now] > 0] = k + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(n, 1)

        return (sp_share_obs, sp_obs, sp_actions, None, sp_reward, sp_done,
                sp_valid_transitions, sp_term, sp_next_share_obs, sp_next_obs,
                None, sp_gamma)

    # ---- diagnostics -----------------------------------------------------
    def payload_of(self, indice):
        return self.payload_diag[np.asarray(indice, dtype=np.int64)]

    def age_of(self, indice, env_step):
        return env_step - self.insert_step[np.asarray(indice, dtype=np.int64)]
