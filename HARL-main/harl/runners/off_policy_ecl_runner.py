"""Runner for ECL v2 on the off-policy (HASAC) host — the full method.

The envelope [E] is supplied by the env shim (``EclMujocoMulti``); this runner
adds the trainer-side pieces:

* [I] identifier — a windowed clip-aware known-structure fit over the replay
  buffer's joint actions (spec C1), refreshed every ``W`` steps; its output
  ``ĉ_now`` tags incoming transitions and steers the sampler. Never a net input.
* [L] localizer + [A] anchor — live in the replaced replay buffer
  (``EclOffPolicyBufferEP``): kernel-localized around ``ĉ_now`` + trough rehearsal
  + a uniform floor, with retro-retagging (C3.1) and a kernel floor (C3.2).
* Diagnostics/eval — payload-purity metering (C3.3), de-aliased per-episode eval
  (C4), an oracle-tag arm (C5), and a flag-gated KL trough-anchor (C6.1).

The host (HASAC) losses/critic/update are untouched (P6) except the optional,
default-OFF KL anchor. HASAC code paths are selected by temporarily aliasing
``args["algo"]`` (repo convention, see ``OffPolicyMbcdRunner``).
"""

import os
import csv
import copy

import numpy as np
import torch

from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.common.buffers.ecl_off_policy_buffer import EclOffPolicyBufferEP
from harl.envs.mamujoco.ecl.ecl_identifier import ECLIdentifier

_CSV_COLUMNS = [
    "env_step", "c_now", "c_max_seen", "pcr_payload", "eps_mean", "cond_number",
    "trough_frac", "reward_mean", "corr_x1y", "corr_x2y", "r2", "b1", "b2",
    "scan_best_corr", "scan_best_offset", "n_valid",
    # v2 additions (C1/C3):
    "clip_frac", "lock_gain", "c_now_ratio", "tag_payload_corr", "anchor_payload",
    "sumzero_frac", "eps_norm_mean", "scan_best_offset_ankle",
]


class OffPolicyEclRunner(OffPolicyHARunner):
    def __init__(self, args, algo_args, env_args):
        super(OffPolicyEclRunner, self).__init__(args, algo_args, env_args)
        if self.algo_args["render"]["use_render"]:
            return

        ecl_cfg = dict(env_args.get("ecl_cfg", {}))
        self.ecl_W = int(ecl_cfg.get("W", 2000))
        self.ecl_warmup_windows = int(ecl_cfg.get("warmup_windows", 5))
        # C5 oracle-tag arm
        self.tag_oracle = bool(ecl_cfg.get("tag_oracle", False))
        self.ecl_severity = float(ecl_cfg.get("severity", 0.9))
        # C3.1 retro-retagging cadence
        self.retag_interval = int(ecl_cfg.get("retag_interval", 100000))
        self._ecl_last_retag = 0
        # C4 de-aliased eval
        self.eval_dephase = bool(ecl_cfg.get("eval_dephase", True))
        self.pcr_period = int(ecl_cfg.get("pcr_period", 40000))
        # C6.1 KL trough-anchor (default OFF)
        self.kl_anchor = bool(ecl_cfg.get("kl_anchor", False))
        self.kl_anchor_coef = float(ecl_cfg.get("kl_anchor_coef", 0.3))
        self._kl_snapshot = None
        self._kl_low_streak = 0

        if self.state_type != "EP":
            raise NotImplementedError("ECL buffer implemented for state_type EP (mamujoco).")
        import gc

        self.buffer = None
        gc.collect()
        buf_args = {
            **algo_args["train"], **algo_args["model"], **algo_args["algo"],
            "ecl_cfg": ecl_cfg,
        }
        self.buffer = EclOffPolicyBufferEP(
            buf_args,
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )

        self.identifier = ECLIdentifier(self.num_agents, ecl_cfg)
        self._ecl_last_refresh = 0

        # ---- labeled oracle-arm banners (must be unmissable, spec Part 5 sanity) --
        if self.tag_oracle:
            print("\n" + "=" * 64 + "\n[ORACLE ARM] tag_oracle=True — tags come from "
                  f"SEVERITY({self.ecl_severity})×pcr_payload, NOT the identifier.\n"
                  "             This is the Stage-E ablation (iv), never the headline.\n" + "=" * 64)

        # ---- human-readable debug trace (CSV, per run) ----
        self._ecl_last_payload = float("nan")
        self._ecl_last_eps = float("nan")
        self._ecl_last_eps_norm = float("nan")
        self._ecl_last_reward = float("nan")
        self._ecl_dbg = None
        self._eval_dbg = None
        if bool(ecl_cfg.get("debug", True)) and getattr(self, "run_dir", None):
            path = os.path.join(str(self.run_dir), "ecl_debug.csv")
            self._ecl_dbg = open(path, "w", newline="", encoding="utf-8")
            self._ecl_dbg_writer = csv.writer(self._ecl_dbg)
            self._ecl_dbg_writer.writerow(_CSV_COLUMNS)
            print(f"[ECL] identifier/debug trace -> {os.path.abspath(path)}")
            # C4.2 per-episode eval log
            epath = os.path.join(str(self.run_dir), "eval_debug.csv")
            self._eval_dbg = open(epath, "w", newline="", encoding="utf-8")
            self._eval_dbg_writer = csv.writer(self._eval_dbg)
            self._eval_dbg_writer.writerow(
                ["step", "thread", "payload_end", "ep_return", "ep_len"]
            )
            print(f"[ECL] per-episode eval trace -> {os.path.abspath(epath)}")

    # -------- HASAC path selection (temporary alias, per repo convention) ----
    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            return super(OffPolicyEclRunner, self).get_actions(
                obs, available_actions, add_random
            )
        finally:
            self.args["algo"] = orig

    def insert(self, data):
        """Capture diagnostics, stash the raw readouts (hip+ankle) and the true
        payload, then insert. (`data[6]` = infos, `data[4]` = rewards.)"""
        infos, rewards = data[6], data[4]
        pays, epss, epss_norm = [], [], []
        raw_hip, raw_ank, pay_stash = [], [], []
        for it in infos:
            d = it[0] if isinstance(it, (list, tuple, np.ndarray)) else it
            if isinstance(d, dict):
                pay_stash.append(float(d.get("pcr_payload", np.nan)))
                if "pcr_payload" in d:
                    pays.append(float(d["pcr_payload"]))
                if "ecl_eps_mean" in d:
                    epss.append(float(d["ecl_eps_mean"]))
                if "ecl_eps_norm_mean" in d:
                    epss_norm.append(float(d["ecl_eps_norm_mean"]))
                if "ecl_raw_y" in d:
                    raw_hip.append(d["ecl_raw_y"])
                raw_ank.append(d.get("ecl_raw_y_ankle", None))
            else:
                pay_stash.append(np.nan)
                raw_ank.append(None)
        if pays:
            self._ecl_last_payload = float(np.mean(pays))
        if epss:
            self._ecl_last_eps = float(np.mean(epss))
        if epss_norm:
            self._ecl_last_eps_norm = float(np.mean(epss_norm))
        self._ecl_last_reward = float(np.mean(np.asarray(rewards)))

        # stash raw readout (n_threads, N, 2) and payload (n_threads,) BEFORE insert
        if len(raw_hip) == len(infos) and hasattr(self.buffer, "stash_raw_readout"):
            hip = np.asarray(raw_hip, dtype=np.float64)                  # (nt, N)
            if all(a is not None for a in raw_ank) and len(raw_ank) == len(infos):
                ank = np.asarray(raw_ank, dtype=np.float64)
            else:
                ank = np.zeros_like(hip)
            self.buffer.stash_raw_readout(np.stack([hip, ank], axis=-1))
        if len(pay_stash) == len(infos) and hasattr(self.buffer, "stash_payload"):
            self.buffer.stash_payload(np.asarray(pay_stash, dtype=np.float64))
        super(OffPolicyEclRunner, self).insert(data)

    # ------------------------------------------------------------------ retag
    def _retag(self):
        """C3.1 — re-run the clipfit identifier over the whole buffer in
        consecutive W-windows and rewrite tags + tag_valid. Heals first-lock
        mistags, cross-refresh scale drift, and anchor contamination."""
        buf = self.buffer
        cur = int(buf.cur_size)
        nt = int(buf.n_rollout_threads)
        margin = self.identifier.margin
        n_steps = cur // max(1, nt)
        if n_steps <= self.ecl_W // 2 + margin:
            return
        bsz = buf.buffer_size
        base = buf.idx if cur >= bsz else 0            # oldest occupied slot
        new_tags = np.array(buf.tags, dtype=np.float32)
        new_valid = np.array(buf.tag_valid, dtype=bool)
        mu = self.identifier.mu_c
        smoothed = 0.0
        started = False
        start_step = 0
        while start_step + margin + 10 < n_steps:
            wsteps_read = min(self.ecl_W + margin, n_steps - start_step)
            start_slot = (base + start_step * nt) % bsz
            c_hat, lock_gain = self.identifier.fit_window(buf, start_slot, wsteps_read)
            valid = lock_gain > self.identifier.lock_min_gain
            if valid:
                smoothed = c_hat if not started else (1 - mu) * smoothed + mu * c_hat
                started = True
            tag_val = float(smoothed if started else 0.0)
            L = wsteps_read * nt
            sl = (start_slot + np.arange(L)) % bsz
            new_tags[sl] = tag_val
            new_valid[sl] = valid
            start_step += wsteps_read                  # non-overlapping tiling
        buf.retag(new_tags, new_valid)

    def _maybe_refresh_identifier(self):
        """Refresh ĉ_now once W new steps have accumulated (§3.6); retag on cadence."""
        if self.buffer.total_inserted - self._ecl_last_refresh < self.ecl_W:
            return
        c_now = self.identifier.refresh(self.buffer)     # always run (logging + A/B)

        if self.tag_oracle:
            # C5: tags/localization come from the true driver, not the identifier
            oracle_c = self.ecl_severity * (
                self._ecl_last_payload if np.isfinite(self._ecl_last_payload) else 0.0
            )
            self.buffer.set_cnow(oracle_c, self.ecl_severity, True)
        else:
            ready = self.identifier.n_good >= self.ecl_warmup_windows
            self.buffer.set_cnow(c_now, self.identifier.c_max_seen, ready)
        self._ecl_last_refresh = self.buffer.total_inserted

        # retro-retag on cadence (skip in oracle-tag mode — tags aren't identifier's)
        if (not self.tag_oracle
                and self.buffer.total_inserted - self._ecl_last_retag >= self.retag_interval):
            self._retag()
            self._ecl_last_retag = self.buffer.total_inserted

        step = self.buffer.total_inserted
        di = self.identifier.diagnostics()
        db = self.buffer.ecl_diagnostics()
        if self._ecl_dbg is not None:
            self._ecl_dbg_writer.writerow([
                step, round(c_now, 6), round(di["c_max_seen"], 6),
                round(self._ecl_last_payload, 6), round(self._ecl_last_eps, 6),
                round(di["cond_number"], 3), round(db["trough_frac"], 4),
                round(self._ecl_last_reward, 4),
                round(di["corr_x1y"], 4), round(di["corr_x2y"], 4), round(di["r2_med"], 4),
                round(di["b1_med"], 5), round(di["b2_med"], 5),
                round(di["scan_best_corr"], 4), di["scan_best_offset"], di["n_valid"],
                round(di["clip_frac"], 4), round(di["lock_gain"], 4),
                round(di["c_now_ratio"], 6), round(db["tag_payload_corr"], 4),
                round(db["anchor_payload"], 4), round(di["sumzero_frac"], 4),
                round(self._ecl_last_eps_norm, 4), di["scan_best_offset_ankle"],
            ])
            self._ecl_dbg.flush()
        if hasattr(self, "writter") and self.writter is not None:
            for k in ("c_now", "c_max_seen", "lock_gain", "clip_frac", "sumzero_frac"):
                self.writter.add_scalar(f"ecl/{k}", di.get(k, c_now if k == "c_now" else 0), step)
            self.writter.add_scalar("ecl/tag_payload_corr", db["tag_payload_corr"], step)
            self.writter.add_scalar("ecl/trough_frac", db["trough_frac"], step)

    # ------------------------------------------------------------------ train
    def train(self):
        self._maybe_refresh_identifier()
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            super(OffPolicyEclRunner, self).train()
            if self.kl_anchor:
                self._kl_anchor_step()
        finally:
            self.args["algo"] = orig

    def _kl_anchor_step(self):
        """C6.1 (flag-gated, default OFF) — pull each actor toward a frozen trough
        snapshot on the anchor slice. The snapshot is (re)frozen whenever the
        smoothed tag has sat at a trough for a full window. Implemented as a stable
        action-space L2 anchor (a tractable surrogate for KL given the actor API
        exposes deterministic actions, not distribution parameters). Guarded so it
        can never crash the (default) headline run."""
        try:
            c_low = self.buffer.ecl_clow_frac * self.buffer.ecl_cmax
            if self.buffer.ecl_cnow <= c_low:
                self._kl_low_streak += 1
            else:
                self._kl_low_streak = 0
            if self._kl_snapshot is None or self._kl_low_streak == 1:
                self._kl_snapshot = [copy.deepcopy(a) for a in self.actor]
                for snap in self._kl_snapshot:
                    snap.turn_off_grad()
            if self._kl_snapshot is None:
                return
            data = self.buffer.sample()
            sp_obs = data[1]
            mask = getattr(self.buffer, "last_anchor_mask", None)
            if mask is None or mask.sum() < 8:
                return
            idx = np.where(mask)[0]
            for aid in range(self.num_agents):
                obs_np = np.asarray(sp_obs[aid])[idx]          # anchor-slice obs
                with torch.no_grad():
                    tgt = self._kl_snapshot[aid].get_actions(obs_np, stochastic=False)
                self.actor[aid].turn_on_grad()
                cur = self.actor[aid].get_actions(obs_np, stochastic=False)
                loss = self.kl_anchor_coef * ((cur - tgt.detach()) ** 2).mean()
                self.actor[aid].actor_optimizer.zero_grad()
                loss.backward()
                self.actor[aid].actor_optimizer.step()
                self.actor[aid].turn_off_grad()
        except Exception as e:  # never let the contingency break training
            if not getattr(self, "_kl_warned", False):
                print(f"[ECL] KL-anchor step disabled (error: {e}).")
                self._kl_warned = True

    # ------------------------------------------------------------------ eval (C4)
    @torch.no_grad()
    def eval(self, step):
        """De-aliased eval: eval-env clocks are stratified across the payload cycle
        (C4.1, in envs_tools), so each round is a true cycle-average. Also logs each
        episode's end-payload/return/length and the trough- & peak-slice returns."""
        if self._eval_dbg is None:
            return super(OffPolicyEclRunner, self).eval(step)

        n = self.algo_args["eval"]["n_eval_rollout_threads"]
        eval_episode_rewards = [[] for _ in range(n)]
        one_episode_rewards = [[] for _ in range(n)]
        one_episode_len = np.zeros(n, dtype=np.int64)
        ep_returns, ep_payloads, ep_lens = [], [], []
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False
            )
            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_available_actions) = self.eval_envs.step(eval_actions)
            for i in range(n):
                one_episode_rewards[i].append(eval_rewards[i])
            one_episode_len += 1
            eval_dones_env = np.all(eval_dones, axis=1)
            for i in range(n):
                if eval_dones_env[i]:
                    eval_episode += 1
                    # EP mamujoco: per-step reward is (n_agents, 1), shared across
                    # agents — sum over steps then reduce the agent axis to a scalar.
                    ep_ret = float(np.mean(np.sum(one_episode_rewards[i], axis=0)))
                    eval_episode_rewards[i].append(ep_ret)
                    pay = float(eval_infos[i][0].get("pcr_payload", np.nan)) \
                        if isinstance(eval_infos[i][0], dict) else np.nan
                    ep_returns.append(ep_ret)
                    ep_payloads.append(pay)
                    ep_lens.append(int(one_episode_len[i]))
                    self._eval_dbg_writer.writerow(
                        [step, i, round(pay, 5), round(ep_ret, 3), int(one_episode_len[i])]
                    )
                    one_episode_rewards[i] = []
                    one_episode_len[i] = 0
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                break
        self._eval_dbg.flush()

        rets = np.asarray(ep_returns, dtype=np.float64)
        pays = np.asarray(ep_payloads, dtype=np.float64)
        avg = float(np.mean(rets))
        avg_len = float(np.mean(ep_lens))
        # ratchet + collapse-depth slices (spec Stage E)
        trough_slice = peak_slice = float("nan")
        finite = np.isfinite(pays)
        if finite.sum() >= 5:
            pf, rf = pays[finite], rets[finite]
            lo = np.quantile(pf, 0.1)
            hi = np.quantile(pf, 0.8)
            if np.any(pf <= lo):
                trough_slice = float(np.mean(rf[pf <= lo]))
            if np.any(pf >= hi):
                peak_slice = float(np.mean(rf[pf >= hi]))
        print(f"[ECL eval] step={step} cycle-avg={avg:.1f} len={avg_len:.0f} "
              f"trough-slice={trough_slice:.1f} peak-slice={peak_slice:.1f}")
        self.log_file.write(",".join(map(str, [step, avg, avg_len])) + "\n")
        self.log_file.flush()
        if hasattr(self, "writter") and self.writter is not None:
            self.writter.add_scalar("eval_average_episode_rewards", avg, step)
            self.writter.add_scalar("eval_average_episode_length", avg_len, step)
            if np.isfinite(trough_slice):
                self.writter.add_scalar("eval/trough_slice", trough_slice, step)
            if np.isfinite(peak_slice):
                self.writter.add_scalar("eval/peak_slice", peak_slice, step)

    def close(self):
        for f in (getattr(self, "_ecl_dbg", None), getattr(self, "_eval_dbg", None)):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        super(OffPolicyEclRunner, self).close()
