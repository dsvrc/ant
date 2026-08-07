"""``hasac_diag`` — HASAC + the campaign's training telemetry  [spec §3.2].

A thin subclass of the HASAC off-policy runner that **changes no training math**.
Every addition is a read-only measurement, and every measurement is wrapped in
``_rng_frozen()``, which saves and restores the torch / CUDA / numpy random
states. That buys a property stronger than the spec asks for:

    with telemetry flags OFF, this is behaviourally identical to ``hasac``;
    **with them ON, it is still identical** — same seed, same trajectory, bit for
    bit (modulo cuDNN non-determinism, which the tuned config already pins via
    ``cuda_deterministic``).

That matters for D1 ("a forensic re-run of the baseline failure with the §3.2
microscope"): the microscope must not be part of the experiment. A telemetry
probe that quietly consumed a few thousand Gaussian samples per window would
make D1 a *different run* from the baseline it is supposed to explain.

What it records
---------------
``diag_telemetry.csv``  one row per ``telemetry_interval`` env steps:
  * **TD-error by phase** — median |TD| over a uniform replay draw, binned by
    payload quintile. The replay draw is independent of (and identical in
    distribution to) the training minibatch, so no sampling is duplicated.
  * **replay-age by phase** — how stale each phase's transitions are (H-C2).
  * **collect fractions** — which phases were being *collected* this window; the
    report crosses these with the drift columns to build the empirical
    who-overwrites-whom matrix.
  * **alpha / entropy per bank** (H-C5). Note the tuned config has
    ``auto_alpha: false`` (alpha pinned at 0.2), so the "one auto-tuned alpha
    cannot serve trough and peak" half of H-C5 is *moot for this host* — the
    column is logged anyway (it would be a constant) and the live half is the
    per-phase **entropy**. The report says so rather than reporting a flat line
    as evidence.
  * **action drift per bank** — ``||pi_t(bank) - pi_{t-1}(bank)||`` (H-C2).
  * **critic feature rank** — 99%-energy effective rank of the critic's
    penultimate activations (plasticity loss, P2).
``diag_probes.npz``     the frozen probe banks + the per-checkpoint mean action
                        vectors (the raw material for the drift matrix).
``diag_qcal.csv``       per eval episode: Q(s0,a0) vs the realized discounted
                        return, with the episode's payload (§3.2.4).
``eval_debug.csv``      the C4 per-episode eval log (§3.3), as in the ECL runner.

HASAC code paths are selected by temporarily aliasing ``args["algo"]`` (repo
convention, see ``OffPolicyEclRunner`` / ``OffPolicyMbcdRunner``).
"""

import contextlib
import csv
import os

import numpy as np
import torch

from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.common.buffers.diag_off_policy_buffer import DiagOffPolicyBufferEP

_TELE_COLS = [
    "env_step", "reward_mean", "payload_mean",
    # TD / age by payload quintile
    "td_q0", "td_q1", "td_q2", "td_q3", "td_q4",
    "age_q0", "age_q1", "age_q2", "age_q3", "age_q4",
    "n_q0", "n_q1", "n_q2", "n_q3", "n_q4",
    # what was collected this window (the who-overwrites-whom row)
    "collect_q0", "collect_q1", "collect_q2", "collect_q3", "collect_q4",
    # per-bank policy telemetry (bank k == payload quintile k)
    "drift_b0", "drift_b1", "drift_b2", "drift_b3", "drift_b4",
    "entropy_b0", "entropy_b1", "entropy_b2", "entropy_b3", "entropy_b4",
    "alpha_mean", "auto_alpha", "feature_rank", "banks_ready",
]

_QCAL_COLS = ["env_step", "thread", "payload_start", "payload_end", "q0",
              "disc_return", "ep_return", "ep_len"]


class OffPolicyDiagRunner(OffPolicyHARunner):
    """HASAC + read-only, RNG-transparent telemetry."""

    def __init__(self, args, algo_args, env_args):
        super(OffPolicyDiagRunner, self).__init__(args, algo_args, env_args)
        if self.algo_args["render"]["use_render"]:
            return

        cfg = dict(env_args.get("diag_cfg", {}))
        self.telemetry = bool(cfg.get("telemetry", True))
        self.tele_interval = int(cfg.get("telemetry_interval", 10000))
        self.bank_interval = int(cfg.get("bank_interval", 50000))
        self.rank_interval = int(cfg.get("rank_interval", 200000))
        self.bank_size = int(cfg.get("bank_size", 512))
        self.rank_batch = int(cfg.get("rank_batch", 1024))
        self.gamma = float(self.algo_args["algo"]["gamma"])

        if self.state_type != "EP":
            raise NotImplementedError(
                "hasac_diag's buffer is implemented for state_type EP (mamujoco)."
            )

        # swap in the payload-aligned buffer (same math; two extra parallel arrays)
        import gc

        self.buffer = None
        gc.collect()
        self.buffer = DiagOffPolicyBufferEP(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )

        self._tele_rng = np.random.default_rng(12345 + int(algo_args["seed"]["seed"]))
        self._last_tele = 0
        self._last_bank = 0
        self._last_rank = 0
        self.banks = None
        self._prev_bank_actions = None
        self._bank_action_log = []       # (env_step, bank, agent, mean action vec)
        self._rank_probe = None
        self._win_payloads = []          # payloads collected since the last row
        self._last_reward = float("nan")
        self._last_payload = float("nan")
        self._feature_rank = float("nan")

        banner = ("[DIAG RUN] algo=hasac_diag telemetry=%s interval=%d bank=%d "
                  "rank=%d | auto_alpha=%s alpha=%s | eval_dephase=%s d_to=%s"
                  % (self.telemetry, self.tele_interval, self.bank_interval,
                     self.rank_interval, self.algo_args["algo"]["auto_alpha"],
                     self.algo_args["algo"].get("alpha"),
                     env_args.get("pcr_eval_dephase", False),
                     cfg.get("d_to", "none")))
        print("\n" + "=" * 78 + "\n" + banner + "\n" + "=" * 78, flush=True)

        self._tele_f = self._eval_f = self._qcal_f = None
        if self.telemetry and getattr(self, "run_dir", None):
            self._tele_f, self._tele_w = self._open_csv("diag_telemetry.csv", _TELE_COLS)
            self._qcal_f, self._qcal_w = self._open_csv("diag_qcal.csv", _QCAL_COLS)
            self._eval_f, self._eval_w = self._open_csv(
                "eval_debug.csv", ["step", "thread", "payload_end", "ep_return",
                                   "ep_len"])

    def _open_csv(self, name, cols):
        path = os.path.join(str(self.run_dir), name)
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(cols)
        print(f"[DIAG] {name} -> {os.path.abspath(path)}", flush=True)
        return f, w

    # ==================================================================
    #  RNG transparency — the property the whole runner rests on
    # ==================================================================
    @contextlib.contextmanager
    def _rng_frozen(self):
        """Any random draw inside this block is rolled back on exit, so telemetry
        cannot shift the training run's random stream."""
        cpu = torch.get_rng_state()
        npy = np.random.get_state()
        cuda = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
        try:
            yield
        finally:
            torch.set_rng_state(cpu)
            np.random.set_state(npy)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)

    # ==================================================================
    #  HASAC path selection (temporary alias, per repo convention)
    # ==================================================================
    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            return super(OffPolicyDiagRunner, self).get_actions(
                obs, available_actions, add_random
            )
        finally:
            self.args["algo"] = orig

    def insert(self, data):
        """Stash the true payload + env step alongside each transition, then
        insert unchanged. (``data[6]`` = infos, ``data[4]`` = rewards.)"""
        infos, rewards = data[6], data[4]
        pays = []
        for it in infos:
            d = it[0] if isinstance(it, (list, tuple, np.ndarray)) else it
            pays.append(float(d.get("pcr_payload", np.nan))
                        if isinstance(d, dict) else np.nan)
        pays = np.asarray(pays, dtype=np.float32)
        self.buffer.stash_meta(pays, self.buffer.total_inserted)
        self._last_reward = float(np.mean(np.asarray(rewards)))
        self._last_payload = float(np.nanmean(pays)) if pays.size else float("nan")
        self._win_payloads.append(pays)
        super(OffPolicyDiagRunner, self).insert(data)

    def train(self):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            super(OffPolicyDiagRunner, self).train()
        finally:
            self.args["algo"] = orig
        if self.telemetry:
            with self._rng_frozen():
                try:
                    self._maybe_telemetry()
                except Exception as e:      # telemetry must never kill a 10M run
                    if not getattr(self, "_tele_warned", False):
                        print(f"[DIAG] telemetry disabled after error: {e!r}",
                              flush=True)
                        self._tele_warned = True

    # ==================================================================
    #  telemetry
    # ==================================================================
    def _quintile_bins(self, pay):
        """Payload quintile index per element. Under a FROZEN arm the payload is
        constant, so there is exactly one bin — the report must not read the four
        empty ones as 'no data at those phases'."""
        p = np.asarray(pay, dtype=np.float64)
        ok = np.isfinite(p)
        bins = np.full(p.shape, -1, dtype=np.int64)
        if ok.sum() == 0:
            return bins
        vals = p[ok]
        if np.ptp(vals) < 1e-9:                 # frozen slice
            bins[ok] = 0
            return bins
        qs = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
        bins[ok] = np.digitize(vals, qs)
        return bins

    def _maybe_telemetry(self):
        step = self.buffer.total_inserted
        if step - self._last_tele < self.tele_interval:
            return
        self._last_tele = step

        # ---- TD + age by phase, on an independent uniform replay draw --------
        cur = int(self.buffer.cur_size)
        n = min(self.algo_args["algo"]["batch_size"], max(cur, 1))
        idx = self._tele_rng.integers(0, max(cur, 1), size=n)
        td = self._td_abs(idx)
        pay = self.buffer.payload_of(idx)
        age = self.buffer.age_of(idx, step)
        bins = self._quintile_bins(pay)
        td_q, age_q, n_q = [], [], []
        for k in range(5):
            m = bins == k
            td_q.append(float(np.median(td[m])) if m.any() else float("nan"))
            age_q.append(float(np.mean(age[m])) if m.any() else float("nan"))
            n_q.append(int(m.sum()))

        # ---- what was collected this window ---------------------------------
        win = (np.concatenate(self._win_payloads) if self._win_payloads
               else np.array([np.nan]))
        self._win_payloads = []
        wb = self._quintile_bins(win)
        collect = [float(np.mean(wb == k)) if wb.size else float("nan")
                   for k in range(5)]

        # ---- banks: drift + entropy -----------------------------------------
        drift = [float("nan")] * 5
        entropy = [float("nan")] * 5
        if self.banks is None:
            self._build_banks()
        if self.banks is not None and step - self._last_bank >= self.bank_interval:
            self._last_bank = step
            drift, entropy = self._bank_pass(step)

        # ---- critic feature rank (P2) ---------------------------------------
        if self._rank_probe is not None and step - self._last_rank >= self.rank_interval:
            self._last_rank = step
            self._feature_rank = self._critic_feature_rank()

        alpha_mean = float(np.mean([float(a) for a in self.alpha]))
        self._tele_w.writerow(
            [step, round(self._last_reward, 5), round(self._last_payload, 5)]
            + [round(v, 5) for v in td_q] + [round(v, 1) for v in age_q] + n_q
            + [round(v, 4) for v in collect]
            + [round(v, 5) for v in drift] + [round(v, 5) for v in entropy]
            + [round(alpha_mean, 5), int(bool(self.algo_args["algo"]["auto_alpha"])),
               round(self._feature_rank, 2) if np.isfinite(self._feature_rank)
               else "", int(self.banks is not None)]
        )
        self._tele_f.flush()
        if getattr(self, "writter", None) is not None:
            for k in range(5):
                if np.isfinite(td_q[k]):
                    self.writter.add_scalar(f"diag/td_q{k}", td_q[k], step)
            self.writter.add_scalar("diag/alpha", alpha_mean, step)
            if np.isfinite(self._feature_rank):
                self.writter.add_scalar("diag/feature_rank", self._feature_rank, step)

    @torch.no_grad()
    def _td_abs(self, idx):
        """|TD| per sampled element, recomputed with the SAME target formula the
        critic uses (soft twin-Q, n-step, proper time limits). Read-only."""
        data = self.buffer.gather(idx)
        (sp_share_obs, sp_obs, sp_actions, _, sp_reward, sp_done, _, sp_term,
         sp_next_share_obs, sp_next_obs, _, sp_gamma) = data
        tpdv = dict(dtype=torch.float32, device=self.device)
        next_actions, next_logp = [], []
        for a in range(self.num_agents):
            act, logp = self.actor[a].get_actions_with_logprobs(sp_next_obs[a])
            next_actions.append(act)
            next_logp.append(logp)
        nq1 = self.critic.target_critic(
            torch.as_tensor(sp_next_share_obs, **tpdv),
            torch.cat(next_actions, dim=-1).to(**tpdv))
        nq2 = self.critic.target_critic2(
            torch.as_tensor(sp_next_share_obs, **tpdv),
            torch.cat(next_actions, dim=-1).to(**tpdv))
        nq = torch.min(nq1, nq2)
        logp = torch.sum(torch.cat(next_logp, dim=-1), dim=-1, keepdim=True).to(**tpdv)
        r = torch.as_tensor(sp_reward, **tpdv)
        g = torch.as_tensor(sp_gamma, **tpdv)
        end = torch.as_tensor(
            sp_term if self.critic.use_proper_time_limits else sp_done,
        ).to(**tpdv)
        alpha = self.critic.alpha
        alpha = float(alpha) if not torch.is_tensor(alpha) else float(alpha.item())
        target = r + g * (nq - alpha * logp) * (1.0 - end)
        acts = torch.cat([torch.as_tensor(sp_actions[a], **tpdv)
                          for a in range(self.num_agents)], dim=-1)
        q = self.critic.get_values(sp_share_obs, acts)
        return torch.abs(target - q).squeeze(-1).cpu().numpy()

    def _build_banks(self):
        """Freeze ``bank_size`` per-agent obs from each payload quintile.

        Drawn from the replay buffer rather than fresh rollouts: those are real
        on-distribution states, cost nothing, and cannot perturb the run."""
        cur = int(self.buffer.cur_size)
        pay = self.buffer.payload_diag[:cur]
        ok = np.isfinite(pay)
        if ok.sum() < 5 * self.bank_size:
            return
        ids = np.flatnonzero(ok)
        bins = self._quintile_bins(pay[ok])
        banks = []
        for k in range(5):
            sel = ids[bins == k]
            if sel.size == 0:
                continue
            take = self._tele_rng.choice(sel, size=min(self.bank_size, sel.size),
                                         replace=False)
            banks.append({
                "bin": k,
                "payload": float(np.nanmean(self.buffer.payload_diag[take])),
                "obs": np.array([self.buffer.obs[a][take]
                                 for a in range(self.num_agents)]).copy(),
            })
        if not banks:
            return
        self.banks = banks
        rk = self._tele_rng.integers(0, cur, size=min(self.rank_batch, cur))
        self._rank_probe = (self.buffer.share_obs[rk].copy(),
                            np.concatenate([self.buffer.actions[a][rk]
                                            for a in range(self.num_agents)],
                                           axis=-1).copy())
        print(f"[DIAG] probe banks frozen: {len(self.banks)} bank(s), "
              f"payloads {[round(b['payload'], 3) for b in self.banks]}, "
              f"{self.bank_size} states each; rank probe {self._rank_probe[0].shape}",
              flush=True)

    @torch.no_grad()
    def _bank_pass(self, step):
        """Deterministic actions + entropy on every frozen bank."""
        drift = [float("nan")] * 5
        entropy = [float("nan")] * 5
        cur_actions = {}
        for b in self.banks:
            k = b["bin"]
            d_agents, e_agents = [], []
            for a in range(self.num_agents):
                obs = b["obs"][a]
                det = self.actor[a].get_actions(obs, stochastic=False).cpu().numpy()
                cur_actions[(k, a)] = det
                # MC entropy estimate E[-log pi(a|s)] over the bank (the squashed
                # Gaussian has no closed form; this is the standard estimator)
                _, logp = self.actor[a].get_actions_with_logprobs(obs)
                e_agents.append(float(-logp.mean().item()))
                if self._prev_bank_actions is not None and \
                        (k, a) in self._prev_bank_actions:
                    prev = self._prev_bank_actions[(k, a)]
                    d_agents.append(float(np.linalg.norm(det - prev, axis=-1).mean()))
                self._bank_action_log.append(
                    (step, k, a, det.mean(axis=0).astype(np.float32)))
            if d_agents:
                drift[k] = float(np.mean(d_agents))
            entropy[k] = float(np.mean(e_agents))
        self._prev_bank_actions = cur_actions
        return drift, entropy

    @torch.no_grad()
    def _critic_feature_rank(self):
        """99%-energy effective rank of the critic's penultimate activations."""
        share_obs, actions = self._rank_probe
        tpdv = dict(dtype=torch.float32, device=self.device)
        x = torch.cat([torch.as_tensor(share_obs, **tpdv),
                       torch.as_tensor(actions, **tpdv)], dim=-1)
        layers = list(self.critic.critic.mlp.mlp)
        # PlainMLP is [Linear, act, Linear, act, ..., Linear, final_act]; the
        # penultimate activation is everything except the last Linear + its act.
        if len(layers) < 3 or not isinstance(layers[-2], torch.nn.Linear):
            return float("nan")
        for layer in layers[:-2]:
            x = layer(x)
        s = torch.linalg.svdvals(x.double())
        e = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2)
        return float(torch.searchsorted(e, torch.tensor(0.99, dtype=e.dtype,
                                                        device=e.device)).item() + 1)

    # ==================================================================
    #  eval — C4 protocol + Q-calibration
    # ==================================================================
    @torch.no_grad()
    def eval(self, step):
        """De-aliased eval (C4): the eval envs' payload clocks are stratified
        across the cycle by ``pcr_eval_dephase`` (envs_tools), so each round is a
        true cycle-average, not a phase-aliased snapshot. Also logs per-episode
        payload/return/length, the trough-decile and peak-quintile slices, and
        the Q(s0,a0)-vs-realized-return calibration (§3.2.4)."""
        if self._eval_f is None:
            return super(OffPolicyDiagRunner, self).eval(step)

        n = self.algo_args["eval"]["n_eval_rollout_threads"]
        one_ep_rew = [[] for _ in range(n)]
        one_ep_len = np.zeros(n, dtype=np.int64)
        disc = np.zeros(n)
        gpow = np.ones(n)
        q0 = np.full(n, np.nan)
        pay0 = np.full(n, np.nan)
        at_start = np.ones(n, dtype=bool)
        ep_returns, ep_payloads, ep_lens = [], [], []
        eval_episode = 0

        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        while True:
            eval_actions = self.get_actions(eval_obs, available_actions=eval_avail,
                                            add_random=False)
            if at_start.any():
                so = np.asarray(eval_share_obs)[:, 0]          # EP: agents share it
                at = np.concatenate([eval_actions[:, a, :]
                                     for a in range(self.num_agents)], axis=-1)
                qv = self.critic.get_values(so, at).squeeze(-1).cpu().numpy()
                q0[at_start] = qv[at_start]
                at_start[:] = False

            (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos,
             eval_avail) = self.eval_envs.step(eval_actions)
            r = np.asarray(eval_rewards).reshape(n, -1)[:, 0]
            for i in range(n):
                one_ep_rew[i].append(eval_rewards[i])
                if np.isnan(pay0[i]) and isinstance(eval_infos[i][0], dict):
                    pay0[i] = float(eval_infos[i][0].get("pcr_payload", np.nan))
            disc += gpow * r
            gpow *= self.gamma
            one_ep_len += 1

            for i in range(n):
                if np.all(eval_dones[i]):
                    eval_episode += 1
                    ep_ret = float(np.mean(np.sum(one_ep_rew[i], axis=0)))
                    pay = (float(eval_infos[i][0].get("pcr_payload", np.nan))
                           if isinstance(eval_infos[i][0], dict) else np.nan)
                    ep_returns.append(ep_ret)
                    ep_payloads.append(pay)
                    ep_lens.append(int(one_ep_len[i]))
                    self._eval_w.writerow([step, i, round(pay, 5), round(ep_ret, 3),
                                           int(one_ep_len[i])])
                    self._qcal_w.writerow(
                        [step, i, round(float(pay0[i]), 5), round(pay, 5),
                         round(float(q0[i]), 3), round(float(disc[i]), 3),
                         round(ep_ret, 3), int(one_ep_len[i])])
                    one_ep_rew[i] = []
                    one_ep_len[i] = 0
                    disc[i] = 0.0
                    gpow[i] = 1.0
                    q0[i] = np.nan
                    pay0[i] = np.nan
                    at_start[i] = True
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                break
        self._eval_f.flush()
        self._qcal_f.flush()

        rets = np.asarray(ep_returns, dtype=np.float64)
        pays = np.asarray(ep_payloads, dtype=np.float64)
        avg = float(np.mean(rets))
        avg_len = float(np.mean(ep_lens))
        trough_slice = peak_slice = float("nan")
        finite = np.isfinite(pays)
        if finite.sum() >= 5:
            pf, rf = pays[finite], rets[finite]
            lo, hi = np.quantile(pf, 0.1), np.quantile(pf, 0.8)
            if np.any(pf <= lo):
                trough_slice = float(np.mean(rf[pf <= lo]))
            if np.any(pf >= hi):
                peak_slice = float(np.mean(rf[pf >= hi]))
        print(f"[DIAG eval] step={step} cycle-avg={avg:.1f} len={avg_len:.0f} "
              f"trough-slice={trough_slice:.1f} peak-slice={peak_slice:.1f}",
              flush=True)
        self.log_file.write(",".join(map(str, [step, avg, avg_len])) + "\n")
        self.log_file.flush()
        if getattr(self, "writter", None) is not None:
            self.writter.add_scalar("eval_average_episode_rewards", avg, step)
            self.writter.add_scalar("eval_average_episode_length", avg_len, step)
            if np.isfinite(trough_slice):
                self.writter.add_scalar("eval/trough_slice", trough_slice, step)
            if np.isfinite(peak_slice):
                self.writter.add_scalar("eval/peak_slice", peak_slice, step)

    def save(self):
        super(OffPolicyDiagRunner, self).save()
        self._dump_probes()

    def _dump_probes(self):
        if not self.telemetry or self.banks is None or not getattr(self, "run_dir", None):
            return
        try:
            path = os.path.join(str(self.run_dir), "diag_probes.npz")
            np.savez_compressed(
                path,
                bank_payloads=np.array([b["payload"] for b in self.banks]),
                bank_bins=np.array([b["bin"] for b in self.banks]),
                bank_obs=np.stack([b["obs"] for b in self.banks]),
                log_step=np.array([r[0] for r in self._bank_action_log]),
                log_bank=np.array([r[1] for r in self._bank_action_log]),
                log_agent=np.array([r[2] for r in self._bank_action_log]),
                log_action=(np.stack([r[3] for r in self._bank_action_log])
                            if self._bank_action_log else np.zeros((0, 1))),
            )
        except Exception as e:
            print(f"[DIAG] probe dump skipped: {e!r}", flush=True)

    def close(self):
        self._dump_probes()
        for f in (self._tele_f, self._eval_f, self._qcal_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        super(OffPolicyDiagRunner, self).close()


# ==========================================================================
#  unit test (§11.1 item 7) — no simulator, no run
# ==========================================================================
def selftest():
    """Checks the three claims that are checkable without a simulator:

      T1  ``buffer.gather(idx)`` reproduces ``buffer.sample()`` exactly for the
          same indices — gather is the one place base math was re-implemented.
      T2  the sampler ignores ``payload_diag`` / ``insert_step`` (scramble them and
          the sample is unchanged) — i.e. the stash is a diagnostic, not replay
          shaping (Prohibition 1).
      T3  ``_rng_frozen`` restores the torch and numpy streams exactly — the
          property that makes the telemetry-ON run identical to plain hasac.

    "train math identical" then holds by construction: ``train()`` is
    ``super().train()`` plus a telemetry call that is both read-only and
    RNG-transparent.
    """
    from gym.spaces import Box
    from harl.envs.mamujoco.diag.report_io import DebugReport

    rep = DebugReport(os.path.join("diag_out", "v0", "v0_runner.md"),
                      title="V0 — hasac_diag runner/buffer self-test",
                      subtitle="gather==sample, sampler ignores the stash, "
                               "telemetry is RNG-transparent")
    ok_all = True
    N, OB, AC, NT = 2, 6, 2, 4
    args = {"buffer_size": 400, "batch_size": 32, "n_step": 3, "gamma": 0.99,
            "n_rollout_threads": NT}
    share = Box(-10, 10, (OB,))
    obs_sp = [Box(-10, 10, (OB,)) for _ in range(N)]
    act_sp = [Box(-1, 1, (AC,)) for _ in range(N)]
    buf = DiagOffPolicyBufferEP(args, share, N, obs_sp, act_sp)

    rng = np.random.default_rng(0)
    for t in range(40):
        buf.stash_meta(np.full(NT, 0.1 * (t % 10), dtype=np.float32), t * NT)
        buf.insert((
            rng.normal(size=(NT, OB)).astype(np.float32),
            rng.normal(size=(N, NT, OB)).astype(np.float32),
            rng.normal(size=(N, NT, AC)).astype(np.float32),
            None,
            rng.normal(size=(NT, 1)).astype(np.float32),
            (rng.random((NT, 1)) < 0.1),
            np.ones((N, NT, 1), dtype=np.float32),
            (rng.random((NT, 1)) < 0.05),
            rng.normal(size=(NT, OB)).astype(np.float32),
            rng.normal(size=(N, NT, OB)).astype(np.float32),
            None,
        ))

    rep.h2("T1 — gather(idx) == sample() on the same indices")
    seen = {}
    real_randperm = torch.randperm

    def _spy(n, **kw):
        p = real_randperm(n, **kw)
        seen["idx"] = p.numpy()[: buf.batch_size]
        return p

    torch.randperm = _spy
    try:
        s = buf.sample()
    finally:
        torch.randperm = real_randperm
    g = buf.gather(seen["idx"])
    diffs, names = [], []
    for k, (a, b) in enumerate(zip(s, g)):
        if a is None and b is None:
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype == bool or b.dtype == bool:
            # sp_done / sp_term are bool arrays (buffer.dones/.terms are
            # np.full(..., False)); numpy refuses `-` on booleans, so compare them
            # exactly rather than by subtraction.
            diffs.append(0.0 if np.array_equal(a, b) else 1.0)
        else:
            diffs.append(float(np.max(np.abs(a.astype(np.float64)
                                             - b.astype(np.float64)))))
        names.append(k)
    ok = max(diffs) == 0.0
    ok_all &= ok
    rep.line(f"  compared {len(diffs)} of the 12 returned arrays (2 are None for "
             f"the continuous-action path)")
    rep.line(f"  max|Δ| = {max(diffs):.3e} (must be exactly 0)   "
             f"{'OK' if ok else 'FAIL'}")
    if not ok:
        bad = [names[i] for i, d in enumerate(diffs) if d != 0.0]
        rep.line(f"  disagreeing tuple positions: {bad}")
    rep.verdict("T1 gather mirrors sample", ok)

    rep.h2("T2 — the sampler ignores payload_diag / insert_step")
    torch.manual_seed(7)
    a1 = buf.sample()[0]
    buf.payload_diag[:] = rng.normal(size=buf.buffer_size).astype(np.float32)
    buf.insert_step[:] = rng.integers(0, 10 ** 6, size=buf.buffer_size)
    torch.manual_seed(7)
    a2 = buf.sample()[0]
    ok = float(np.max(np.abs(a1 - a2))) == 0.0
    ok_all &= ok
    rep.line(f"  scrambling both stash arrays changes the sample by "
             f"{float(np.max(np.abs(a1 - a2))):.3e} (must be 0 — the stash is a "
             f"diagnostic, not replay shaping)   {'OK' if ok else 'FAIL'}")
    rep.verdict("T2 stash never steers sampling", ok)

    rep.h2("T3 — _rng_frozen restores the torch and numpy streams")

    class _Shim:
        _rng_frozen = OffPolicyDiagRunner._rng_frozen

    sh = _Shim()
    torch.manual_seed(11)
    np.random.seed(11)
    want_t = torch.randn(3).tolist()
    want_n = np.random.rand(3).tolist()
    torch.manual_seed(11)
    np.random.seed(11)
    with sh._rng_frozen():
        torch.randn(1000)                      # what a telemetry pass would burn
        np.random.rand(1000)
    got_t = torch.randn(3).tolist()
    got_n = np.random.rand(3).tolist()
    ok = np.allclose(want_t, got_t) and np.allclose(want_n, got_n)
    ok_all &= ok
    rep.line(f"  torch stream after a frozen block: {np.allclose(want_t, got_t)}")
    rep.line(f"  numpy stream after a frozen block: {np.allclose(want_n, got_n)}")
    rep.line("  => telemetry ON and telemetry OFF produce the SAME training "
             "trajectory for a given seed. D1 is a re-run of the baseline, not a "
             "different experiment.")
    rep.verdict("T3 RNG transparency", ok)

    rep.h2("SUMMARY")
    rep.verdict("V0 runner/buffer self-test", ok_all)
    rep.close()
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys

    sys.exit(selftest())
