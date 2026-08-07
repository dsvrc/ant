"""RECON [F]+[DI] — the amortized local filter f_ψ(h_i) -> ℓ̂_i (spec §2.3).

One **weight-shared causal estimator** per run. Input at step t is
``o_i(t) ⊕ u_i(t−1)`` — strictly local quantities (the agent's own observation
as HARL provides it, and its own last *executed* action) — and the output is the
agent's liability estimate ``ℓ̂_i(t) ∈ R^k`` (Ant: k=2, own hip/ankle). It is
trained by plain MSE against the hindsight labels ℓ̃ minted by [ID]+[RE], with
**its own Adam, fully decoupled from every RL loss** (Prohibition 4: no joint
objective, no trade-off coefficient).

Why a ~5-step-memory recurrent map and not something bigger: T1. The liability
obeys ``ℓ(t+1) = ρℓ(t) + (1−ρ)cΦ(u_{−i}(t))`` — a stable first-order filter — so
a causal map of depth O(1/(1−ρ)) suffices to approximate the conditional mean
E[ℓ_i | h_i]. At the benchmark's ρ=0.8 that is ≈5–15 steps. The architecture is
*derived*, not tuned; ``hidden: 64`` and ``lr: 3e-4`` are the spec's constants.

Two arch options behind ``filter.arch`` (identical API):
  * ``gru``  — a GRU cell over the rollout, per-agent hidden state (default).
  * ``mlp``  — the sanctioned fallback: an MLP over a stacked window of the last
               ``stack_len`` frames.

Decentralization note. Every tensor this module sees is per-agent: the batch axis
carries (thread × agent) and nothing is ever mixed across the agent axis. Running
N copies batched inside the trainer's process is a *computational* choice — it is
arithmetically identical to N independent filters running on N machines, which is
what "the filter is part of the agent" means at execution (spec §2.3).

Zero-init output head. ℓ̂ ≡ 0 at initialization, so a fresh `recon` run *starts*
byte-equivalent to plain HAPPO ([CP] is then the identity) and the compensation
fades in exactly as fast as the filter earns it. This is also what makes U4
("host untouched") a statement about the code rather than about luck.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  cores — each owns a recurrent state and a (state, x) -> (feature, state) step
# ---------------------------------------------------------------------------
class _GruCore(nn.Module):
    """Causal GRU cell. State = h (B, hidden)."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(in_dim, hidden)

    def init_state(self, batch, device):
        return torch.zeros(batch, self.hidden, device=device)

    def step(self, x, state, mask):
        """x: (B, in_dim); state: (B, hidden); mask: (B, 1), 0 => episode start."""
        state = self.cell(x, state * mask)
        return state, state


class _StackCore(nn.Module):
    """Stacked-frame MLP fallback. State = the last ``stack_len`` frames
    (B, L, in_dim); an episode start zeroes the history before the new frame is
    pushed, so the window never straddles a reset."""

    def __init__(self, in_dim, hidden, stack_len):
        super().__init__()
        self.in_dim = in_dim
        self.L = stack_len
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * stack_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def init_state(self, batch, device):
        return torch.zeros(batch, self.L, self.in_dim, device=device)

    def step(self, x, state, mask):
        state = state * mask.unsqueeze(-1)               # (B,1,1) broadcast
        state = torch.cat([state[:, 1:], x.unsqueeze(1)], dim=1)
        return self.mlp(state.reshape(state.shape[0], -1)), state


class ReconFilterNet(nn.Module):
    """core -> ℓ̂ head (+ optional predicted-squared-error head, default OFF)."""

    def __init__(self, obs_dim, act_dim, k, arch="gru", hidden=64, stack_len=16,
                 var_head=False):
        super().__init__()
        in_dim = int(obs_dim) + int(act_dim)
        self.in_dim = in_dim
        self.k = int(k)
        self.arch = str(arch)
        if self.arch == "gru":
            self.core = _GruCore(in_dim, hidden)
        elif self.arch == "mlp":
            self.core = _StackCore(in_dim, hidden, stack_len)
        else:
            raise ValueError(f"recon filter.arch must be 'gru' or 'mlp' (got {arch!r})")
        self.head = nn.Linear(hidden, self.k)
        # ℓ̂ ≡ 0 at init => `recon` starts as plain HAPPO (see the module docstring).
        # This zeroes the core's gradient on the FIRST backward only (dL/dcore ∝
        # head.weight): the head itself always gets a gradient, so from step 2 the
        # core is live, and Adam's per-parameter normalization means a small
        # head.weight does not translate into small core steps.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.var_head = nn.Linear(hidden, self.k) if var_head else None

    def init_state(self, batch, device):
        return self.core.init_state(batch, device)

    def step(self, x, state, mask):
        """One causal step. x: (B, in_dim); mask: (B, 1) with 0 at an episode start.
        Returns (ℓ̂ (B, k), new_state)."""
        feat, state = self.core.step(x, state, mask)
        return self.head(feat), state

    def forward_seq(self, x_seq, state, masks):
        """Unroll over a rollout with episode-reset masking.

        x_seq: (T, B, in_dim); masks: (T, B, 1) with 0 marking an episode start
        (HARL's convention: masks[t] gates the state carried *into* step t).
        Returns (ℓ̂ (T, B, k), final_state, predicted-squared-error or None).
        """
        outs, vars_ = [], []
        for t in range(x_seq.shape[0]):
            feat, state = self.core.step(x_seq[t], state, masks[t])
            outs.append(self.head(feat))
            if self.var_head is not None:
                vars_.append(self.var_head(feat))
        return (
            torch.stack(outs, dim=0),
            state,
            torch.stack(vars_, dim=0) if vars_ else None,
        )


# ---------------------------------------------------------------------------
#  [DI] — the filter + its own optimizer + the (optional) learned β
# ---------------------------------------------------------------------------
class ReconFilter:
    """Owns the net, its decoupled Adam, and the [CP] gain β.

    ``update(...)`` is the whole of [DI]: one plain MSE regression of f_ψ(h_i)
    onto the manufactured labels ℓ̃, restricted to locked windows (masked, never
    zeroed — an unlocked window contributes no training signal rather than a
    fake ℓ̃=0 target).
    """

    def __init__(self, obs_dim, act_dim, k, cfg, device):
        cfg = dict(cfg or {})
        self.device = device
        self.k = int(k)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.arch = str(cfg.get("arch", "gru"))
        self.hidden = int(cfg.get("hidden", 64))
        self.stack_len = int(cfg.get("stack_len", 16))
        # NB: `adam_lr`/`grad_clip`, not `lr`/`max_grad_norm`. HARL's update_args
        # rewrites CLI overrides into *every* nested dict by key name, so a
        # `--lr 1e-4` meant for the host would silently retune the filter too.
        self.lr = float(cfg.get("adam_lr", 3e-4))
        self.epochs = int(cfg.get("epochs", 4))
        self.max_grad_norm = float(cfg.get("grad_clip", 10.0))
        self.var_head = bool(cfg.get("var_head", False))

        self.net = ReconFilterNet(
            obs_dim, act_dim, k, arch=self.arch, hidden=self.hidden,
            stack_len=self.stack_len, var_head=self.var_head,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # ---- [CP] gain β ∈ [0,1]^k, one per joint type (Ant: hip, ankle) ----
        self.beta_mode = str(cfg.get("beta_mode", "fixed"))
        self.beta_fixed = float(cfg.get("beta_fixed", 1.0))
        if self.beta_mode == "learned":
            # a learnable sigmoid gain, trained on the *cancellation* objective
            # min_β E‖ℓ̃ − β⊙ℓ̂‖² — the least-squares gain that best cancels the
            # liability given the filter's own error. It is the shrinkage factor
            # T3 predicts (β* = E[ℓℓ̂]/E[ℓ̂²] -> 1 as the filter's MSE -> 0) and
            # the mechanism T5 blames for the measured β-inversion at high σ.
            # Trainer-side only: it consumes ℓ̃, never a network input.
            self.beta_logit = torch.zeros(
                self.k, device=device, requires_grad=True
            )  # sigmoid(0) = 0.5
            self.beta_optimizer = torch.optim.Adam([self.beta_logit], lr=self.lr)
        else:
            self.beta_logit = None
            self.beta_optimizer = None

    # ------------------------------------------------------------------ β
    def beta(self):
        """The [CP] gain as numpy (k,)."""
        if self.beta_mode == "learned":
            with torch.no_grad():
                return torch.sigmoid(self.beta_logit).cpu().numpy()
        return np.full(self.k, self.beta_fixed, dtype=np.float64)

    # ------------------------------------------------------ execution (rollout)
    def init_state(self, batch):
        return self.net.init_state(batch, self.device)

    @torch.no_grad()
    def step_np(self, obs, u_prev, state, mask):
        """One causal filter step on numpy inputs — the execution path.

        obs:    (B, obs_dim)  the agent's own observation, as HARL provides it
        u_prev: (B, act_dim)  the agent's own last executed action
        mask:   (B, 1)        0 at an episode start (resets the state)
        Returns (ℓ̂ (B, k) numpy, new_state).
        """
        x = torch.as_tensor(
            np.concatenate([obs, u_prev], axis=-1), dtype=torch.float32,
            device=self.device,
        )
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        lhat, state = self.net.step(x, state, m)
        return lhat.cpu().numpy(), state

    # ------------------------------------------------------------------ [DI]
    def update(self, obs_seq, u_prev_seq, labels, masks, label_valid):
        """One [DI] pass over the current rollout (spec §2.3).

        All arrays are numpy with a leading time axis and a flat batch axis
        carrying (thread × agent):
          obs_seq:     (T, B, obs_dim)
          u_prev_seq:  (T, B, act_dim)
          labels:      (T, B, k)     ℓ̃ from [RE]
          masks:       (T, B, 1)     0 at an episode start
          label_valid: (T, B, 1)     1 where the window's ĉ locked
        Returns a dict of diagnostics.
        """
        x = torch.as_tensor(
            np.concatenate([obs_seq, u_prev_seq], axis=-1), dtype=torch.float32,
            device=self.device,
        )
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(label_valid, dtype=torch.float32, device=self.device)
        denom = w.sum() * self.k
        if float(denom) < 1.0:
            # nothing locked this iteration: hold (mask, don't zero — §2.1)
            return {"filter_mse": float("nan"), "filter_mse_first": float("nan"),
                    "filter_grad_norm": 0.0, "n_label_valid": 0.0}

        first_mse = None
        last_mse = float("nan")
        gnorm = 0.0
        for _ in range(max(1, self.epochs)):
            state = self.net.init_state(x.shape[1], self.device)
            pred, _, err_pred = self.net.forward_seq(x, state, m)
            se = (pred - y) ** 2
            mse = (se * w).sum() / denom
            last_mse = float(mse.detach())          # read it BEFORE backward frees
            if first_mse is None:
                first_mse = last_mse
            loss = mse
            if err_pred is not None:
                # optional uncertainty head: regress the *realized* squared error.
                # It never gates anything (default OFF) — it is a reporting head,
                # and .detach() keeps it from perturbing the ℓ̂ head's gradient.
                loss = loss + (((err_pred - se.detach()) ** 2) * w).sum() / denom
            self.optimizer.zero_grad()
            loss.backward()
            gnorm = float(
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            )
            self.optimizer.step()

        info = {
            # each epoch's loss is measured BEFORE that epoch's step, so:
            "filter_mse": last_mse,            # after epochs-1 of this iteration's fits
            "filter_mse_first": first_mse,     # before any of them — the honest,
                                               # on-policy, held-out-in-time MSE
            "filter_grad_norm": gnorm,
            "n_label_valid": float(w.sum()),
        }
        if self.beta_mode == "learned":
            info.update(self._update_beta(x, y, m, w, denom))
        return info

    def _update_beta(self, x, y, m, w, denom):
        """Fit β on min_β E‖ℓ̃ − β⊙ℓ̂‖² over the same locked batch."""
        with torch.no_grad():
            state = self.net.init_state(x.shape[1], self.device)
            pred, _, _ = self.net.forward_seq(x, state, m)
        beta = torch.sigmoid(self.beta_logit)
        loss = (((y - beta * pred) ** 2) * w).sum() / denom
        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()
        return {"beta_loss": float(loss.detach())}

    # ------------------------------------------------------------ save/restore
    def save(self, path):
        blob = {"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}
        if self.beta_logit is not None:
            blob["beta_logit"] = self.beta_logit.detach().cpu()
        torch.save(blob, path)

    def restore(self, path):
        blob = torch.load(path, map_location=self.device)
        self.net.load_state_dict(blob["net"])
        if "opt" in blob:
            self.optimizer.load_state_dict(blob["opt"])
        if self.beta_logit is not None and "beta_logit" in blob:
            with torch.no_grad():
                self.beta_logit.copy_(blob["beta_logit"].to(self.device))

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))
