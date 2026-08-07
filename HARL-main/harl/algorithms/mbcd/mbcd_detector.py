"""Model-Based Context Detection (CUSUM changepoint detector) for MBCD.

Faithful port of the detection logic in the reference ``mbcd/mbcd.py`` (Alegre et
al., AAMAS 2021), decoupled from the RL backbone. It maintains, per detected
context, a probabilistic dynamics-model ensemble and a dynamics dataset, and runs
the online CUSUM statistic over transition log-likelihood ratios to decide, at
every step, whether to (a) stay in the current context, (b) switch to a known
context, or (c) spawn a brand-new context.

Operates on *global* transitions ``(state, action) -> (reward, next_state)`` where
``state`` is the centralized multi-agent state (share_obs) and ``action`` is the
joint action. The policy / replay-buffer side of the per-context library lives in
``OffPolicyMbcdRunner``; this class returns the detection signals
``(changed, current_model, is_new)`` so the runner can swap policies/buffers.
"""

import numpy as np
import torch

from harl.algorithms.mbcd.dynamics_ensemble import ProbabilisticEnsemble


class _DynamicsDataset:
    """Ring buffer of (X=[state, action], Y=[reward, delta_state]) for one context."""

    def __init__(self, in_dim, out_dim, capacity):
        self.capacity = capacity
        self.X = np.zeros((capacity, in_dim), dtype=np.float32)
        self.Y = np.zeros((capacity, out_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, x, y):
        self.X[self.ptr] = x
        self.Y[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def remove_last_n(self, n):
        n = min(n, self.size)
        self.ptr = (self.ptr - n) % self.capacity
        self.size -= n

    def to_train_batch(self):
        if self.size < self.capacity:
            return self.X[: self.size].copy(), self.Y[: self.size].copy()
        return self.X.copy(), self.Y.copy()


class MBCDDetector:
    def __init__(
        self,
        state_dim,
        action_dim,
        device=torch.device("cpu"),
        cusum_threshold=100.0,
        max_std=0.5,
        num_stds=2.0,
        min_steps=5000,
        memory_capacity=100000,
        num_networks=5,
        num_elites=2,
        hidden_size=200,
        num_layers=4,
        lr=1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_dim = state_dim + action_dim
        self.out_dim = state_dim + 1
        self.device = device

        self.threshold = float(cusum_threshold)
        self.max_std = float(max_std)
        self.num_stds = float(num_stds)
        self.min_steps = int(min_steps)
        self.memory_capacity = int(memory_capacity)

        self._model_kwargs = dict(
            num_networks=num_networks,
            num_elites=num_elites,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lr=lr,
            device=device,
        )

        self.num_models = 1
        self.current_model = 0
        self.changed = False
        self.step = 0
        self.steps_per_context = {0: 0}

        self.models = {0: self._build_model()}
        self.datasets = {0: _DynamicsDataset(self.in_dim, self.out_dim, self.memory_capacity)}

        self.log_prob = {0: 0.0}
        self.var_mean = {0: 0.0}
        self.mean, self.variance = {}, {}
        self.S = {0: 0.0, -1: 0.0}  # -1 == new-model statistic

    # ------------------------------------------------------------------ models
    def _build_model(self):
        return ProbabilisticEnsemble(self.state_dim, self.action_dim, **self._model_kwargs)

    @property
    def counter(self):
        return self.steps_per_context[self.current_model]

    @property
    def memory(self):
        return self.datasets[self.current_model]

    # --------------------------------------------------------------- log-probs
    def get_logprob2(self, x, means, variances):
        """GMM log-likelihood of x under the ensemble + epistemic disagreement.

        x: [batch, out_dim]; means/variances: [num_networks, batch, out_dim].
        Returns log_prob [batch], var_mean [batch], mean [batch, out_dim],
        variance [batch, out_dim].
        """
        k = x.shape[-1]
        mean = np.mean(means, axis=0)
        variance = (np.mean(means ** 2 + variances, axis=0) - mean ** 2) + 1e-6
        log_prob = -0.5 * (
            k * np.log(2 * np.pi)
            + np.log(variance).sum(-1)
            + (np.power(x - mean, 2) / variance).sum(-1)
        )  # shape [batch]
        # Reference reduces over the batch axis here (detection uses batch == 1),
        # yielding a single combined-Gaussian log-likelihood for the transition.
        prob = np.exp(log_prob).sum(axis=0)  # scalar for batch == 1
        log_prob = np.log(prob + 1e-8)  # scalar
        var_mean = np.linalg.norm(np.std(means, axis=0), axis=-1)  # [batch]
        return log_prob, var_mean, mean, variance

    def _model_logprob(self, model_id, inputs, obs, true_output):
        means, varis = self.models[model_id].predict(inputs, factored=True)
        means = means.copy()
        means[:, :, 1:] += obs  # delta -> next-state prediction
        return self.get_logprob2(true_output, means, varis)

    # ------------------------------------------------------------------- step
    def step(self, obs, action, reward, next_obs, done):
        """Run one CUSUM detection step on a global transition (thread 0).

        Returns (changed, current_model, is_new).
        """
        obs = np.asarray(obs, dtype=np.float32)[None]
        action = np.asarray(action, dtype=np.float32)[None]
        inputs = np.concatenate((obs, action), axis=-1)
        true_output = np.concatenate(([np.float32(reward)], np.asarray(next_obs, dtype=np.float32)))[None]

        if self.changed:  # reset CUSUM right after a change
            self.S = {m: 0.0 for m in range(self.num_models)}
            self.S[-1] = 0.0

        for i in range(self.num_models):
            lp, vm, mu, var = self._model_logprob(i, inputs, obs, true_output)
            self.log_prob[i] = float(lp)
            self.var_mean[i] = float(vm[0])
            self.mean[i] = mu
            self.variance[i] = var

        gate = (
            self.var_mean[self.current_model] < self.max_std
            and self.counter > self.min_steps
        )

        # CUSUM for each known (alternative) model
        for i in (m for m in range(self.num_models) if m != self.current_model):
            if gate:
                log_ratio = self.log_prob[i] - self.log_prob[self.current_model]
                self.S[i] = max(0.0, self.S[i] + log_ratio)

        # new-model statistic: likelihood of a transition that is consistently
        # num_stds standard deviations away from the current model's prediction
        var_cur = self.variance[self.current_model]
        new_model_log_pdf = -0.5 * (
            (self.state_dim + 1) * np.log(2 * np.pi)
            + np.log(var_cur).sum(-1)
            + (
                np.power(true_output - (true_output + self.num_stds * np.sqrt(var_cur)), 2)
                / var_cur
            ).sum(-1)
        )
        new_model_log_pdf = float(np.log(np.exp(new_model_log_pdf).sum(0) + 1e-8))
        if gate:
            log_ratio = new_model_log_pdf - self.log_prob[self.current_model]
            self.S[-1] = max(0.0, self.S[-1] + log_ratio)

        changed = False
        is_new = False
        maxm = max(self.S.values())
        if maxm > self.threshold:
            changed = True
            # recent transitions may belong to the new context -> drop them
            self.memory.remove_last_n(100)
            if maxm == self.S[-1]:  # spawn a NEW context model
                newm = self.new_model()
                is_new = True
                self.set_model(newm, reset_dataset=True)
            else:  # switch to the best-matching known context
                newm = max(self.S, key=lambda key: self.S[key])
                self.set_model(newm)

        self.changed = changed
        self.step += 1
        self.steps_per_context[self.current_model] += 1
        return changed, self.current_model, is_new

    def add_experience(self, obs, action, reward, next_obs):
        """Add a transition to the current context's dynamics dataset."""
        x = np.concatenate(
            (np.asarray(obs, dtype=np.float32), np.asarray(action, dtype=np.float32))
        )
        y = np.concatenate(
            ([np.float32(reward)], np.asarray(next_obs, dtype=np.float32) - np.asarray(obs, dtype=np.float32))
        )
        self.memory.push(x, y)

    def train_current_model(self, batch_size=256, holdout_ratio=0.1):
        X, Y = self.memory.to_train_batch()
        self.models[self.current_model].train(
            X, Y, batch_size=batch_size, holdout_ratio=holdout_ratio
        )

    # ------------------------------------------------------------- bookkeeping
    def new_model(self):
        self.steps_per_context[self.num_models] = 0
        self.models[self.num_models] = self._build_model()
        self.datasets[self.num_models] = _DynamicsDataset(
            self.in_dim, self.out_dim, self.memory_capacity
        )
        self.log_prob[self.num_models] = 0.0
        self.var_mean[self.num_models] = 0.0
        self.S[self.num_models] = 0.0
        self.num_models += 1
        return self.num_models - 1

    def set_model(self, model_id, reset_dataset=False):
        self.current_model = model_id
        if reset_dataset:
            self.datasets[model_id] = _DynamicsDataset(
                self.in_dim, self.out_dim, self.memory_capacity
            )
