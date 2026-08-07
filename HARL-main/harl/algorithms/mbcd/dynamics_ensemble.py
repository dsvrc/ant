"""Probabilistic dynamics-model ensemble for MBCD (PyTorch port of the BNN/PE).

A small ensemble of probabilistic neural networks that, given ``[state, action]``,
predict a diagonal-Gaussian over ``[reward, next_state - state]`` (reward first,
then the per-dimension state delta). This mirrors the MBPO/PETS ensemble used by
the reference MBCD implementation (``mbcd/models/bnn.py``): each network outputs a
mean and a (bounded) log-variance, trained with the Gaussian negative
log-likelihood; ``num_elites`` best networks are tracked for model rollouts.

``predict(inputs, factored=True)`` returns the per-network means and variances
(shape ``[num_networks, batch, 1 + state_dim]``), exactly the interface the MBCD
detector and the (optional) MBPO rollout expect.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GaussianMLP(nn.Module):
    """One probabilistic network: outputs a mean and log-variance per output dim."""

    def __init__(self, in_dim, out_dim, hidden_size, num_layers):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(last, hidden_size), nn.SiLU()]
            last = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, out_dim)
        self.logvar_head = nn.Linear(last, out_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.mean_head(h), self.logvar_head(h)


class _RunningScaler:
    """Standardizes inputs using statistics fit on the training data."""

    def __init__(self, dim, device):
        self.mu = torch.zeros(dim, device=device)
        self.sigma = torch.ones(dim, device=device)
        self.fitted = False

    def fit(self, x):
        self.mu = x.mean(dim=0)
        self.sigma = x.std(dim=0)
        self.sigma[self.sigma < 1e-6] = 1.0
        self.fitted = True

    def transform(self, x):
        return (x - self.mu) / self.sigma


class ProbabilisticEnsemble:
    """Ensemble of probabilistic dynamics networks (aleatoric + epistemic)."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_networks=5,
        num_elites=2,
        hidden_size=200,
        num_layers=4,
        lr=1e-3,
        device=torch.device("cpu"),
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_dim = state_dim + action_dim
        self.out_dim = state_dim + 1  # [reward, delta_state]
        self.num_networks = num_networks
        self.num_elites = min(num_elites, num_networks)
        self.device = device

        self.nets = [
            _GaussianMLP(self.in_dim, self.out_dim, hidden_size, num_layers).to(device)
            for _ in range(num_networks)
        ]
        params = []
        for net in self.nets:
            params += list(net.parameters())
        # bounds on the predicted log-variance (PETS/MBPO style), learnable
        self.max_logvar = nn.Parameter(
            torch.ones(self.out_dim, device=device) * 0.5
        )
        self.min_logvar = nn.Parameter(
            torch.ones(self.out_dim, device=device) * -10.0
        )
        params += [self.max_logvar, self.min_logvar]
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.scaler = _RunningScaler(self.in_dim, device)
        self.elite_inds = list(range(self.num_elites))

    # ------------------------------------------------------------------ helpers
    def _bound_logvar(self, logvar):
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return logvar

    def random_inds(self, batch_size):
        """Random elite-network index per sample (for model rollouts)."""
        return np.random.choice(self.elite_inds, size=batch_size)

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, inputs, factored=True):
        """Predict means and variances for each network.

        Args:
            inputs: (np.ndarray) [batch, state_dim + action_dim].
        Returns:
            means: (np.ndarray) [num_networks, batch, 1 + state_dim].
            variances: (np.ndarray) [num_networks, batch, 1 + state_dim].
        """
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        x = self.scaler.transform(x)
        means, varis = [], []
        for net in self.nets:
            mean, logvar = net(x)
            logvar = self._bound_logvar(logvar)
            means.append(mean)
            varis.append(torch.exp(logvar))
        means = torch.stack(means, dim=0).cpu().numpy()
        varis = torch.stack(varis, dim=0).cpu().numpy()
        return means, varis

    # ------------------------------------------------------------------- train
    def train(self, X, Y, batch_size=256, holdout_ratio=0.1, max_epochs=40, patience=5):
        """Train the ensemble on (X, Y) with a Gaussian NLL loss.

        Uses a holdout split for early stopping (bounds latency) and elite
        selection.

        Args:
            X: (np.ndarray) [N, state_dim + action_dim] inputs.
            Y: (np.ndarray) [N, 1 + state_dim] targets ([reward, delta_state]).
        """
        N = X.shape[0]
        if N < 10:
            return
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)

        self.scaler.fit(X)

        # train / holdout split for early stopping and elite selection
        n_holdout = max(1, int(N * holdout_ratio))
        perm = torch.randperm(N, device=self.device)
        holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
        Xtr, Ytr = X[train_idx], Y[train_idx]
        Xho, Yho = X[holdout_idx], Y[holdout_idx]
        n_train = Xtr.shape[0]

        def holdout_mses():
            with torch.no_grad():
                xho = self.scaler.transform(Xho)
                return [float(((net(xho)[0] - Yho) ** 2).mean().item()) for net in self.nets]

        best = float(np.mean(holdout_mses()))
        no_improve = 0
        for _ in range(max_epochs):
            # each network sees its own bootstrap order
            orders = [torch.randperm(n_train, device=self.device) for _ in self.nets]
            for start in range(0, n_train, batch_size):
                self.optimizer.zero_grad()
                total_loss = 0.0
                for k, net in enumerate(self.nets):
                    idx = orders[k][start : start + batch_size]
                    xb = self.scaler.transform(Xtr[idx])
                    yb = Ytr[idx]
                    mean, logvar = net(xb)
                    logvar = self._bound_logvar(logvar)
                    inv_var = torch.exp(-logvar)
                    nll = ((mean - yb) ** 2 * inv_var + logvar).mean()
                    total_loss = total_loss + nll
                # logvar-bound regularization (encourages tight, calibrated bounds)
                total_loss = total_loss + 0.01 * (
                    self.max_logvar.sum() - self.min_logvar.sum()
                )
                total_loss.backward()
                self.optimizer.step()

            cur = float(np.mean(holdout_mses()))
            if cur < best * 0.99:
                best = cur
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # select elites by holdout MSE
        mses = holdout_mses()
        self.elite_inds = list(np.argsort(mses)[: self.num_elites])

    # ------------------------------------------------------------------ params
    def state_dict(self):
        return {
            "nets": [net.state_dict() for net in self.nets],
            "max_logvar": self.max_logvar.detach().clone(),
            "min_logvar": self.min_logvar.detach().clone(),
            "scaler_mu": self.scaler.mu.clone(),
            "scaler_sigma": self.scaler.sigma.clone(),
            "scaler_fitted": self.scaler.fitted,
            "elite_inds": list(self.elite_inds),
        }

    def load_state_dict(self, sd):
        for net, nsd in zip(self.nets, sd["nets"]):
            net.load_state_dict(nsd)
        with torch.no_grad():
            self.max_logvar.copy_(sd["max_logvar"])
            self.min_logvar.copy_(sd["min_logvar"])
        self.scaler.mu = sd["scaler_mu"].clone()
        self.scaler.sigma = sd["scaler_sigma"].clone()
        self.scaler.fitted = sd["scaler_fitted"]
        self.elite_inds = list(sd["elite_inds"])
