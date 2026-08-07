"""Network modules for WISDOM (Wavelet Predictive Representations for
Non-Stationary RL), ported from the reference ``wisdom/`` package.

Two modules:

* ``WisdomEncoder`` -- a PEARL/CEMRL-style task-belief encoder: an MLP that maps a
  transition ``(obs, action, reward, next_obs)`` to the parameters of a diagonal
  Gaussian posterior ``q(z | c)`` over a low-dimensional task latent ``z``, with a
  reparameterized sample and a KL-to-prior regularizer.

* ``WaveletYNetwork`` -- the wavelet predictive representation (the paper's
  ``Y_Network``): a *learnable* multi-scale wavelet decomposition implemented with
  dilated, length-preserving 1-D convolutions. Given a latent ``z`` (treated as a
  1-D signal over its ``latent_dim`` channels) it returns a refined representation
  ``pred_z`` (a learned weighting of the detail coefficients at every scale plus the
  final approximation and a residual) and the low-frequency approximation
  ``res_lo`` used by the wavelet temporal-difference operator.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wavelet predictive representation (Y_Network)
# ---------------------------------------------------------------------------
def _forward_fading(x, h0, h1, w, depth, kernel_size):
    """Multi-scale (fading-memory) wavelet decomposition.

    x: (B, d_model, L). Returns (y, res_lo) each (B, d_model, L).
    Exact port of the reference ``forward_fading``.
    """
    res_lo = x
    y = 0.0
    dilation = 1
    groups = x.shape[1]
    for i in range(depth, 0, -1):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = F.pad(res_lo, (padding, 0), "constant", 0)
        # detail (high-pass) and approximation (low-pass) coefficients
        res_hi = F.conv1d(res_lo_pad, h1, dilation=dilation, groups=groups)
        res_lo = F.conv1d(res_lo_pad, h0, dilation=dilation, groups=groups)
        y = y + w[:, i : i + 1] * res_hi
        dilation *= 2
    y = y + w[:, :1] * res_lo
    y = y + x * w[:, -1:]
    return y, res_lo


class WaveletYNetwork(nn.Module):
    """Learnable multi-scale wavelet representation of the task latent."""

    def __init__(self, d_model=1, kernel_size=2, depth=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.depth = depth
        self.m = depth + 1

        # learnable low-/high-pass filters (per channel) and per-scale weights
        scale = math.sqrt(2.0 / (kernel_size * 2))
        self.h0 = nn.Parameter(
            torch.empty(d_model, 1, kernel_size).uniform_(-1.0, 1.0) * scale
        )
        self.h1 = nn.Parameter(
            torch.empty(d_model, 1, kernel_size).uniform_(-1.0, 1.0) * scale
        )
        w_init = torch.empty(d_model, self.m + 1).uniform_(-1.0, 1.0) * math.sqrt(
            2.0 / (2 * self.m + 2)
        )
        self.w = nn.Parameter(w_init)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, z):
        """z: (B, latent_dim) -> (pred_z, res_lo), each (B, latent_dim).

        The latent is treated as a single-channel 1-D signal of length
        ``latent_dim`` (matching the reference, where ``d_model == 1``).
        """
        x = z.unsqueeze(1)  # (B, 1, latent_dim)
        y, res_lo = _forward_fading(
            x, self.h0, self.h1, self.w, self.depth, self.kernel_size
        )
        y = self.dropout(self.activation(y))
        return y.squeeze(1), res_lo.squeeze(1)


# ---------------------------------------------------------------------------
# Task-belief encoder
# ---------------------------------------------------------------------------
class WisdomEncoder(nn.Module):
    """MLP encoder of a transition into a diagonal-Gaussian task-latent posterior."""

    def __init__(self, input_dim, latent_dim, hidden_sizes=(200, 200, 200)):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last, latent_dim * 2)

    def forward(self, x):
        return self.head(self.trunk(x))

    def infer_posterior(self, encoder_input, deterministic=False):
        """encoder_input: (B, input_dim) -> (z, mu, var), each (B, latent_dim)."""
        params = self.forward(encoder_input)
        mu = params[..., : self.latent_dim]
        var = F.softplus(params[..., self.latent_dim :]) + 1e-7
        if deterministic:
            z = mu
        else:
            z = mu + torch.sqrt(var) * torch.randn_like(mu)
        return z, mu, var

    @staticmethod
    def kl_to_prior(mu, var):
        """KL( N(mu, var) || N(0, I) ), summed over latent dims, mean over batch."""
        kl = 0.5 * (var + mu.pow(2) - 1.0 - torch.log(var)).sum(dim=-1)
        return kl.mean()
