"""Runner for MAMT (HAPPO + trust-region decomposition).

MAMT does not change the observation or the rollout -- it adds, on top of HAPPO,
an *adaptive per-agent local trust region* whose size is learned to keep the joint
policy divergence (a proxy for non-stationarity) bounded. This runner therefore
leaves the base rollout / buffers untouched and only orchestrates the MAMT block
after each HAPPO update:

1. HAPPO sequential policy update -- each MAMT actor adds a Tsallis-KL trust-region
   penalty ``tr_scale * KL_q(pi||pi_old) / local_tr`` (local_tr current value).
2. coordination coefficients ``ccs`` from the per-agent advantages.
3. non-stationarity ``d_ns`` -- each agent's teammate-modeling policy predicts its
   tightest teammate's policy; ``d_ns = KL(model || teammate)``.
4. the TRD-Net estimates each agent's joint-divergence contribution ``kl_hat`` from
   the state-action features + local trust regions over the coordination graph, and
   is trained to match ``d_ns`` (``ns_loss``).
5. the local trust regions are adapted by minimizing
   ``f_loss = sum_i tr_scale * tr_term_i / local_tr_i + kl_hat.mean()`` and clamped.
6. the old-policy reference is refreshed periodically.
"""

import numpy as np
import torch
import torch.nn.functional as F

from harl.models.mamt import TRDNet, ModelingPolicy
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.discrete_util import get_encoded_act_dim, encode_actions_torch


class OnPolicyMamtRunner(OnPolicyHARunner):
    """Runner for the MAMT algorithm."""

    def __init__(self, args, algo_args, env_args):
        super(OnPolicyMamtRunner, self).__init__(args, algo_args, env_args)

        cfg = algo_args["algo"]
        self.mamt_batch_size = int(cfg.get("mamt_batch_size", 256))
        self.mamt_old_update_interval = int(cfg.get("mamt_old_update_interval", 100))
        self.tr_scale = float(cfg.get("mamt_tr_scale", 10.0))
        self.mamt_sparse = float(cfg.get("mamt_sparse", 0.05))
        trd_hidden = int(cfg.get("mamt_trd_hidden", 32))
        lr = float(cfg.get("mamt_lr", 1e-3))
        self._mamt_updates = 0
        self._mamt_steps = 0

        if self.algo_args["render"]["use_render"]:
            return

        # Action-space handling: continuous Box (mamujoco) vs Discrete (SMAC/SMACv2).
        # For discrete, the TRD-Net state-action features and the teammate model use
        # one-hot actions, so the "action dim" is the encoded (one-hot) width.
        self.mamt_act_spaces = self.envs.action_space
        self.action_type = self.mamt_act_spaces[0].__class__.__name__
        self.discrete = self.action_type != "Box"
        self.raw_obs_dims = [
            self.envs.observation_space[a].shape[0] for a in range(self.num_agents)
        ]
        self.act_dims = [
            get_encoded_act_dim(self.mamt_act_spaces[a]) for a in range(self.num_agents)
        ]
        feat_dim = self.raw_obs_dims[0] + self.act_dims[0]

        # TRD-Net (shared, symmetric agents) + teammate modeling policies (per agent)
        self.trd_net = TRDNet(feat_dim, hidden_dim=trd_hidden, sparse=self.mamt_sparse).to(
            self.device
        )
        self.trd_optimizer = torch.optim.Adam(
            self.trd_net.parameters(), lr=lr, weight_decay=1e-3
        )
        self.modeling_policies = [
            ModelingPolicy(
                self.raw_obs_dims[a],
                self.num_agents,
                self.act_dims[a],
                discrete=self.discrete,
            ).to(self.device)
            for a in range(self.num_agents)
        ]
        self.modeling_optimizers = [
            torch.optim.Adam(mp.parameters(), lr=lr) for mp in self.modeling_policies
        ]
        # one optimizer over all agents' learnable local trust regions
        self.local_tr_optimizer = torch.optim.Adam(
            [self.actor[a].local_tr for a in range(self.num_agents)],
            lr=lr,
            weight_decay=1e-3,
        )

    # ======================================================================
    # train: HAPPO update + MAMT trust-region-decomposition block
    # ======================================================================
    def train(self):
        actor_train_infos, critic_train_info = super(
            OnPolicyMamtRunner, self
        ).train()

        # refresh the old-policy trust-region reference periodically
        self._mamt_updates += 1
        if self._mamt_updates % self.mamt_old_update_interval == 0:
            for a in range(self.num_agents):
                self.actor[a].snapshot_old()

        info = self._mamt_block()

        self._mamt_steps += (
            self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        if self.writter is not None and info is not None:
            for k, v in info.items():
                self.writter.add_scalar("mamt/" + k, v, self._mamt_steps)

        return actor_train_infos, critic_train_info

    def _advantages(self):
        if self.value_normalizer is not None:
            adv = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            adv = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        if self.state_type == "EP":
            adv = np.repeat(adv, self.num_agents, axis=2)  # (T, B, N)
        else:
            adv = adv[..., 0]
        return adv

    def _mamt_block(self):
        device = self.device
        N = self.num_agents
        T = self.algo_args["train"]["episode_length"]
        B = self.algo_args["train"]["n_rollout_threads"]

        # sample a minibatch of transitions from the rollout
        n_total = T * B
        bs = min(self.mamt_batch_size, n_total)
        idx = np.random.randint(0, n_total, size=bs)

        adv = self._advantages()  # (T, B, N)
        adv_flat = adv.reshape(n_total, N)[idx]  # (bs, N)
        A = torch.as_tensor(adv_flat, dtype=torch.float32, device=device)

        obs_list, act_list = [], []
        for a in range(N):
            ob = self.actor_buffer[a].obs[:-1].reshape(n_total, -1)[idx]
            ac = self.actor_buffer[a].actions.reshape(n_total, -1)[idx]
            obs_list.append(torch.as_tensor(ob, dtype=torch.float32, device=device))
            ac_t = torch.as_tensor(ac, dtype=torch.float32, device=device)
            # one-hot encode discrete actions for the TRD-Net features (no-op Box)
            act_list.append(encode_actions_torch(ac_t, self.mamt_act_spaces[a]))

        # ---- coordination coefficients ccs[b,i,j] = softmax_i |A_i - A_j| ----
        ccs = torch.abs(A.unsqueeze(2) - A.unsqueeze(1))  # (bs, N, N)
        eye = torch.eye(N, device=device).bool().unsqueeze(0)
        ccs = ccs.masked_fill(eye, float("-inf"))
        ccs = F.softmax(ccs, dim=1)
        ccs = torch.nan_to_num(ccs, nan=0.0)

        # ---- non-stationarity d_ns via teammate modeling ----
        tightest = torch.argmax(ccs.sum(dim=0), dim=1).cpu().numpy()  # per agent i
        d_ns_cols = []
        model_losses = []
        for a in range(N):
            teammate = int(tightest[a])
            onehot = torch.zeros(bs, N, device=device)
            onehot[:, teammate] = 1.0
            model_params = self.modeling_policies[a](obs_list[a], onehot)
            with torch.no_grad():
                teammate_params = self.actor[teammate].teammate_repr(
                    obs_list[teammate]
                )
            # KL between the teammate model and the teammate's real policy (d_ns);
            # dispatches Gaussian (Box) vs categorical (Discrete) internally.
            d_ns = ModelingPolicy.dist_kl(self.discrete, model_params, teammate_params)
            d_ns_cols.append(d_ns.detach().unsqueeze(1))
            # train the modeling policy to predict the teammate's policy
            model_losses.append(
                ModelingPolicy.model_loss(self.discrete, model_params, teammate_params)
            )
        d_ns = torch.cat(d_ns_cols, dim=1)  # (bs, N)

        for a in range(N):
            self.modeling_optimizers[a].zero_grad()
            model_losses[a].backward()
            self.modeling_optimizers[a].step()

        # ---- TRD-Net estimates kl_hat; trained to match d_ns (ns_loss) ----
        sa_feats = torch.stack(
            [torch.cat([obs_list[a], act_list[a]], dim=-1) for a in range(N)], dim=1
        )  # (bs, N, feat)
        local_trs = torch.stack([self.actor[a].local_tr for a in range(N)])  # (N,)

        kl_hat = self.trd_net(sa_feats, local_trs.detach(), ccs)  # (bs, N)
        ns_loss = F.mse_loss(kl_hat, d_ns.detach())
        self.trd_optimizer.zero_grad()
        ns_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trd_net.parameters(), 10.0 * N)
        self.trd_optimizer.step()

        # ---- adaptive local trust-region update (f_loss) ----
        kl_hat_tr = self.trd_net(sa_feats.detach(), local_trs, ccs.detach())  # (bs, N)
        tr_terms = torch.stack(
            [self.actor[a].last_tr_term.detach() for a in range(N)]
        )  # (N,)
        tr_cost = (
            self.tr_scale
            * tr_terms
            / local_trs.clamp(min=self.actor[0].local_tr_min)
        ).sum()
        f_loss = tr_cost + kl_hat_tr.mean()
        self.local_tr_optimizer.zero_grad()
        f_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.actor[a].local_tr for a in range(N)], 0.5
        )
        self.local_tr_optimizer.step()
        for a in range(N):
            self.actor[a].clamp_local_tr()
        # discard the stale TRD-Net grads accumulated by f_loss
        self.trd_optimizer.zero_grad()

        return {
            "ns_loss": float(ns_loss.item()),
            "f_loss": float(f_loss.item()),
            "kl_hat_mean": float(kl_hat.mean().item()),
            "d_ns_mean": float(d_ns.mean().item()),
            "model_loss": float(sum(ml.item() for ml in model_losses) / N),
            "local_tr_mean": float(local_trs.mean().item()),
        }

    # ======================================================================
    # save / restore (also persist the MAMT modules)
    # ======================================================================
    def save(self):
        super(OnPolicyMamtRunner, self).save()
        torch.save(self.trd_net.state_dict(), str(self.save_dir) + "/mamt_trd_net.pt")
        for a in range(self.num_agents):
            torch.save(
                self.modeling_policies[a].state_dict(),
                str(self.save_dir) + "/mamt_modeling_agent" + str(a) + ".pt",
            )
            torch.save(
                self.actor[a].local_tr.detach().cpu(),
                str(self.save_dir) + "/mamt_local_tr_agent" + str(a) + ".pt",
            )

    def restore(self):
        super(OnPolicyMamtRunner, self).restore()
        model_dir = str(self.algo_args["train"]["model_dir"])
        self.trd_net.load_state_dict(torch.load(model_dir + "/mamt_trd_net.pt"))
        for a in range(self.num_agents):
            self.modeling_policies[a].load_state_dict(
                torch.load(model_dir + "/mamt_modeling_agent" + str(a) + ".pt")
            )
