"""Neural modules for TRIO (Meta-RL by Tracking task non-stationarity, IJCAI 2021)
adapted to HARL.

Faithful re-implementation of the official repo's task-inference machinery
(https://github.com/.../trio-non-stationary-meta-rl): a GRU-based variational
inference network that infers a task latent ``z`` from a context of
(action, reward, next_state), taking the *previous posterior as its prior input*
-- i.e. a Bayesian filter that tracks the task over time. The network is trained
*supervised* against the true task with a closed-form Gaussian loss
(``loss_inference_closed_form``): ``MSE(mu_hat, z) + mean(sum(var_hat)) +
KL(posterior || prior)``.

A small linear forecaster is also provided: in the original TRIO, a Gaussian
Process over the inferred-latent history predicts the next task's prior. For the
continuous, per-step Ant setting (where a per-step GP over thousands of points is
infeasible) we substitute an equivalent least-squares linear extrapolation of the
recent posterior means -- the same pragmatic substitution used for the forecaster
in FSAC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceNetwork(nn.Module):
    """GRU variational task encoder with the previous posterior fed in as prior.

    Input per step: context = [action, reward, next_state] (dim ``ctx_dim``),
    concatenated with the prior (mu, logvar) of size ``2 * z_dim``. The GRU's last
    output, the prior, and a "trust" scalar (1 / number-of-samples-seen) are
    combined to produce the posterior (mu, logvar) over the task latent.
    """

    def __init__(self, ctx_dim, z_dim, gru_hidden=64, hidden2=16):
        super(InferenceNetwork, self).__init__()
        self.z_dim = z_dim
        self.ctx_dim = ctx_dim
        self.n_in = ctx_dim + 2 * z_dim  # context + prior appended each step

        self.gru = nn.GRU(
            input_size=self.n_in, hidden_size=gru_hidden, num_layers=1, batch_first=True
        )
        self.enc3 = nn.Linear(gru_hidden + 2 * z_dim + 1, hidden2)  # +prior +trust
        self.enc41 = nn.Linear(hidden2, z_dim)  # mu
        self.enc42 = nn.Linear(hidden2, z_dim)  # logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, context, prior, hidden=None, seq_len_so_far=0):
        """Encode a context window into a posterior over the task latent.

        Args:
            context: (n_batch, seq_len, ctx_dim).
            prior: (n_batch, 2 * z_dim) = [mu_prior, logvar_prior].
            hidden: (1, n_batch, gru_hidden) or None (for online filtering).
            seq_len_so_far: (int) total number of samples seen before this call
                (for the "trust" term during online filtering).
        Returns:
            mu, logvar: (n_batch, z_dim) posterior parameters.
            hidden: (1, n_batch, gru_hidden) updated GRU state.
            total_len: (int) updated number of samples seen.
        """
        n_batch, seq_len = context.shape[0], context.shape[1]
        prior_rep = prior.unsqueeze(1).repeat(1, seq_len, 1)
        inp = torch.cat([context, prior_rep], dim=2)

        if hidden is not None:
            out, hidden = self.gru(inp, hidden)
        else:
            out, hidden = self.gru(inp)
        t = F.elu(out[:, -1, :])  # last output of the sequence

        total_len = seq_len_so_far + seq_len
        trust = torch.full((n_batch, 1), 1.0 / max(1, total_len), dtype=t.dtype, device=t.device)
        h = torch.cat([t, prior, trust], dim=1)
        h = F.elu(self.enc3(h))
        mu = self.enc41(h)
        logvar = self.enc42(h)
        return mu, logvar, hidden, total_len


def loss_inference_closed_form(
    z, mu_hat, logvar_hat, mu_prior, logvar_prior, n_samples, use_decay, decay_param
):
    """TRIO's supervised closed-form inference loss.

    Args:
        z: (n_batch, z_dim) true task.
        mu_hat, logvar_hat: (n_batch, z_dim) predicted posterior.
        mu_prior, logvar_prior: (n_batch, z_dim) prior.
        n_samples: (int) context length (for KL decay).
        use_decay: (bool) decay the KL weight as more samples are seen.
        decay_param: (float) initial KL weight.
    Returns:
        loss, kld (float), mse (float)
    """
    mse_direct = F.mse_loss(mu_hat, z)
    mse_var = torch.mean(torch.sum(logvar_hat.exp(), dim=1))
    mse = mse_direct + mse_var

    kld_1 = torch.sum(logvar_prior - logvar_hat, dim=1)
    kld_2 = torch.sum(
        -1
        + (mu_hat - mu_prior).pow(2) * (1 / logvar_prior.exp())
        + (logvar_hat.exp() * (1 / logvar_prior.exp())),
        dim=1,
    )
    kld = 0.5 * torch.mean(kld_1 + kld_2)
    if use_decay:
        kld = kld * (decay_param / max(1, n_samples))

    return mse + kld, float(kld.item()), float(mse.item())


def compute_linear_weight(past_length):
    """Least-squares matrix (2, past_length) mapping past values -> [slope, intercept].

    Identical construction to the FSAC forecaster: design matrix rows [i+1, 1].
    """
    X = np.array([[i + 1, 1] for i in range(past_length)], dtype=np.float64)
    XtX_inv = np.linalg.inv(np.matmul(np.transpose(X), X))
    W = np.matmul(XtX_inv, np.transpose(X))
    return torch.FloatTensor(W)
