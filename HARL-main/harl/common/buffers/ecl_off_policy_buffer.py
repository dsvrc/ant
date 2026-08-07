"""ECL off-policy replay buffer (EP) — spec [L] localizer + [A] anchor (§3.3-3.4).

Subclasses HARL's ``OffPolicyBufferEP`` and changes exactly one thing: **which
indices a minibatch is drawn from**. Every transition is stamped at insertion
with the central identifier's `tag = ĉ_now`. Each minibatch is composed with
exact fractions:

  * `1 − β_A − β_U`  kernel-localized around `ĉ_now` (Gaussian in the tag) — so no
    gradient ever averages across distant contexts (P3, deletes the average-game
    trap);
  * `β_A`            rehearsed trough experience (transitions with `tag ≤ c_low`)
    so the stationary competence is never surrendered (P4, the anchor);
  * `β_U`            a uniform floor for off-policy coverage / critic stability.

Implementation notes / small HARL-fit deviations (core theory preserved):
  * Localization uses **kernel-weighted importance resampling from a uniform
    candidate pool** rather than maintaining 24 explicit per-bin index lists — it
    realizes the exact Gaussian tag-kernel of §3.3 in O(pool) without the
    circular-buffer bookkeeping of live bins.
  * The anchor rehearses trough-tagged transitions **from the main buffer**
    rather than a separate reservoir. On PCR this is faithful: the trough game is
    the *stationary* base task (so recent trough data is as good as old) and the
    1e6 buffer always holds ≳1 payload cycle of trough experience — the
    separate-reservoir/reservoir-sampling machinery guards against rare or
    drifting troughs, which PCR does not have. (v2: dedicated reservoir.)
Everything downstream (n-step returns, gathering) is the parent's, unchanged.
"""

import numpy as np
import torch

from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP


class EclOffPolicyBufferEP(OffPolicyBufferEP):
    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        super().__init__(args, share_obs_space, num_agents, obs_spaces, act_spaces)
        ecfg = dict(args.get("ecl_cfg", {}))
        self.ecl_beta_A = float(ecfg.get("beta_A", 0.25))
        self.ecl_beta_U = float(ecfg.get("beta_U", 0.10))
        self.ecl_h_frac = float(ecfg.get("h_frac", 0.15))    # h = h_frac * c_max_seen
        self.ecl_h_min = float(ecfg.get("h_min_abs", 0.05))  # kernel floor (C3.2)
        self.ecl_clow_frac = float(ecfg.get("c_low_frac", 0.10))
        self.ecl_pool = int(ecfg.get("pool", 8192))
        self.ecl_coldstart = int(ecfg.get("coldstart_min", 5)) * self.batch_size

        # per-slot RAW readouts (un-normalized own hip & ankle qvel deltas, per
        # agent) stashed by the runner just before each insert — the identifier
        # regresses on these instead of the per-timestep normalized stored obs.
        # Shape (buffer_size, N, 2): channel 0 = hip, channel 1 = ankle (C1.2).
        self.raw_readout = np.zeros((self.buffer_size, num_agents, 2), dtype=np.float32)
        # per-slot true payload (oracle diagnostic only — NEVER read by the sampler;
        # hygiene: logging/gating allowed, sampling decisions never touch it, C3.3).
        self.payload_diag = np.full(self.buffer_size, np.nan, dtype=np.float32)

        # per-slot tag (the ĉ at insertion) + live localizer state (set by runner)
        self.tags = np.zeros(self.buffer_size, dtype=np.float32)
        # per-slot validity: was the identifier locked on when this tag was stamped?
        # transitions collected during warmup / before lock carry a meaningless
        # ĉ (≈0) and must NOT feed the localizer kernel or the trough anchor.
        self.tag_valid = np.zeros(self.buffer_size, dtype=bool)
        self.cur_tag = 0.0
        self.cur_tag_valid = False
        self.ecl_cnow = 0.0
        self.ecl_cmax = max(1e-6, float(ecfg.get("c_max_init", 0.1)))
        self.ecl_ready = False          # runner flips on after identifier warm-up
        self.total_inserted = 0

        # distribution-relative trough threshold + kernel width (recomputed each
        # refresh in set_cnow). The identifier's c_now carries a large offset from
        # the gait's PHYSICAL inter-leg coupling (c_now≈0.5 even at troughs), so an
        # ABSOLUTE c_low=0.1·c_max never fires ⇒ the anchor starves (trough_frac=0).
        # Percentiles of the live tag distribution are immune to that offset.
        self.ecl_c_low = self.ecl_clow_frac * self.ecl_cmax
        self.ecl_h = max(self.ecl_h_min, self.ecl_h_frac * self.ecl_cmax)

        # Adaptive trough threshold: the fixed c_low=0.1·c_max assumes c_now reaches
        # ~0 at troughs, but a weakly-locked identifier floors c_now well above 0 and
        # starves the anchor (trough_frac→0). When on, the trough is the bottom
        # c_low_frac QUANTILE of the live tags (set in set_cnow), so the anchor
        # always rehearses the lowest-severity transitions the identifier emits.
        self.ecl_adaptive_trough = bool(ecfg.get("adaptive_trough", True))

        # anchor-purity meter (diagnostic): EMA of the true payload of anchor draws
        self.anchor_payload = float("nan")
        self._anchor_pay_mu = 0.02
        # composition mask of the last sampled batch (True on the anchor slice) —
        # read by the optional KL trough-anchor (C6.1); never affects sampling.
        self.last_anchor_mask = None

    # ---- runner hooks -----------------------------------------------------
    def set_cnow(self, c_now, c_max_seen, ready):
        """Called by the runner after each identifier refresh."""
        self.ecl_cnow = float(c_now)
        self.cur_tag = float(c_now)                    # future insertions get this tag
        self.cur_tag_valid = bool(ready)               # ...and this validity stamp
        self.ecl_cmax = max(1e-6, float(c_max_seen))
        self.ecl_ready = bool(ready)

        # recompute distribution-relative trough threshold (c_low_frac quantile of
        # the valid tags) and kernel width (h_frac × the interdecile tag spread).
        # Robust to the identifier's physical-coupling offset (which starves an
        # absolute-threshold anchor); percentiles need no c_now-envelope EMA.
        if self.ecl_adaptive_trough:
            pool = np.random.randint(0, self.cur_size, size=min(self.cur_size, 8192)) \
                if self.cur_size > 0 else np.array([], dtype=np.int64)
            vt = self.tags[pool][self.tag_valid[pool]] if pool.size else np.array([])
            if vt.size > 200:
                q10, q90 = np.quantile(vt, 0.1), np.quantile(vt, 0.9)
                self.ecl_c_low = float(np.quantile(vt, self.ecl_clow_frac))
                self.ecl_h = max(self.ecl_h_min,
                                 self.ecl_h_frac * max(float(q90 - q10), 1e-6) / 0.8)
                return
        # fallback (adaptive off, or cold start): absolute scaling
        self.ecl_c_low = self.ecl_clow_frac * self.ecl_cmax
        self.ecl_h = max(self.ecl_h_min, self.ecl_h_frac * self.ecl_cmax)

    def _c_low(self):
        """Trough threshold: the c_low_frac quantile of the live valid tags
        (distribution-relative — immune to the identifier's offset), computed in
        set_cnow; falls back to 0.1·c_max before enough tags exist."""
        return self.ecl_c_low

    # ---- raw readout / payload: stashed by the runner just BEFORE insert --------
    def _stash_into(self, arr, vals):
        """Write ``vals`` (length n_threads on axis 0) at the current ``idx`` with
        the same wraparound as insert (idx not yet advanced)."""
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

    def stash_raw_readout(self, raw_y):
        """``raw_y``: (n_threads, num_agents, 2) — hip & ankle raw qvel deltas."""
        self._stash_into(self.raw_readout, np.asarray(raw_y, dtype=np.float32))

    def stash_payload(self, payload):
        """``payload``: (n_threads,) true pcr_payload (diagnostic only, C3.3)."""
        self._stash_into(self.payload_diag, np.asarray(payload, dtype=np.float32))

    # ---- retro-retagging: bulk overwrite of tags + validity (C3.1) --------------
    def retag(self, new_tags, new_valid):
        """Overwrite the whole tag history (called by the runner's retag sweep)."""
        self.tags[:] = np.asarray(new_tags, dtype=np.float32)
        self.tag_valid[:] = np.asarray(new_valid, dtype=bool)

    # ---- insertion: stamp tags -------------------------------------------
    def insert(self, data):
        length = data[0].shape[0]
        start = self.idx
        super().insert(data)
        end = start + length
        if end <= self.buffer_size:
            self.tags[start:end] = self.cur_tag
            self.tag_valid[start:end] = self.cur_tag_valid
        else:
            self.tags[start:] = self.cur_tag
            self.tags[: end - self.buffer_size] = self.cur_tag
            self.tag_valid[start:] = self.cur_tag_valid
            self.tag_valid[: end - self.buffer_size] = self.cur_tag_valid
        self.total_inserted += length

    # ---- ECL index composition (§3.3) ------------------------------------
    def _ecl_indices(self, n):
        cur = self.cur_size
        # cold start (or identifier not warmed up): plain uniform
        if (not self.ecl_ready) or cur < self.ecl_coldstart:
            self.last_anchor_mask = np.zeros(n, dtype=bool)
            return np.random.randint(0, cur, size=n)

        n_anc = int(round(self.ecl_beta_A * n))
        n_uni = int(round(self.ecl_beta_U * n))
        n_loc = n - n_anc - n_uni
        h = self.ecl_h              # distribution-relative width (set in set_cnow), floored
        c_low = self._c_low()       # c_low_frac quantile of the live valid tags
        parts = []

        # localized: uniform candidate pool, restricted to VALID tags, then
        # resample ∝ Gaussian tag kernel (invalid warmup tags would otherwise
        # pull the kernel toward their meaningless ĉ≈0).
        if n_loc > 0:
            pool = np.random.randint(0, cur, size=min(cur, self.ecl_pool))
            pool = pool[self.tag_valid[pool]]
            w = np.exp(-((self.tags[pool] - self.ecl_cnow) ** 2) / (2.0 * h * h)) if pool.size else None
            sw = float(w.sum()) if w is not None else 0.0
            if sw <= 1e-12:
                parts.append(np.random.randint(0, cur, size=n_loc))
            else:
                parts.append(np.random.choice(pool, size=n_loc, replace=True, p=w / sw))

        # anchor: rehearse trough-tagged transitions (valid tags only)
        anc_idx = None
        if n_anc > 0:
            pool2 = np.random.randint(0, cur, size=min(cur, self.ecl_pool))
            trough = pool2[(self.tags[pool2] <= c_low) & self.tag_valid[pool2]]
            if trough.shape[0] >= 1:
                anc_idx = np.random.choice(trough, size=n_anc, replace=True)
            else:
                anc_idx = np.random.randint(0, cur, size=n_anc)
            parts.append(anc_idx)
            # anchor-purity meter (diagnostic only): EMA of anchor draws' true payload
            pay = self.payload_diag[anc_idx]
            pay = pay[np.isfinite(pay)]
            if pay.size:
                m = float(np.mean(pay))
                self.anchor_payload = (m if not np.isfinite(self.anchor_payload)
                                       else (1 - self._anchor_pay_mu) * self.anchor_payload
                                       + self._anchor_pay_mu * m)

        # uniform floor
        if n_uni > 0:
            parts.append(np.random.randint(0, cur, size=n_uni))

        indice = np.concatenate(parts)
        # composition mask: True on the anchor slice (for the optional KL anchor).
        # parts order is [localized, anchor, uniform]; the anchor slice is the
        # n_anc entries following the n_loc localized ones.
        mask = np.zeros(indice.shape[0], dtype=bool)
        if anc_idx is not None:
            lo = max(0, n_loc)
            mask[lo: lo + anc_idx.shape[0]] = True
        self.last_anchor_mask = mask
        return indice

    # ---- sample: parent's gather + n-step, but ECL-composed indices -------
    def sample(self):
        self.update_end_flag()
        indice = self._ecl_indices(self.batch_size)

        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [self.obs[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [self.valid_transitions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_available_actions = np.array(
                [self.available_actions[agent_id][indice] for agent_id in range(self.num_agents)]
            )

        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [self.next_obs[agent_id][indices[-1]] for agent_id in range(self.num_agents)]
        )
        if self.act_spaces[0].__class__.__name__ == "Discrete":
            sp_next_available_actions = np.array(
                [self.next_available_actions[agent_id][indices[-1]]
                 for agent_id in range(self.num_agents)]
            )

        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)

        if self.act_spaces[0].__class__.__name__ == "Discrete":
            return (
                sp_share_obs, sp_obs, sp_actions, sp_available_actions, sp_reward,
                sp_done, sp_valid_transitions, sp_term, sp_next_share_obs,
                sp_next_obs, sp_next_available_actions, sp_gamma,
            )
        return (
            sp_share_obs, sp_obs, sp_actions, None, sp_reward, sp_done,
            sp_valid_transitions, sp_term, sp_next_share_obs, sp_next_obs, None,
            sp_gamma,
        )

    def ecl_diagnostics(self):
        cur = max(1, self.cur_size)
        c_low = self._c_low()
        # the anchor can only draw VALID trough tags — report that (real supply)
        trough_frac = float(np.mean((self.tags[:cur] <= c_low) & self.tag_valid[:cur]))
        # anchor-purity D1 gate (C3.3): corr(tag, true payload) over valid slots
        v = self.tag_valid[:cur] & np.isfinite(self.payload_diag[:cur])
        if int(v.sum()) > 100:
            tp = np.corrcoef(self.tags[:cur][v], self.payload_diag[:cur][v])[0, 1]
            tag_payload_corr = float(tp) if np.isfinite(tp) else 0.0
        else:
            tag_payload_corr = 0.0
        return {
            "trough_frac": trough_frac,
            "valid_frac": float(np.mean(self.tag_valid[:cur])),
            "tag_mean": float(np.mean(self.tags[:cur])),
            "tag_max": float(np.max(self.tags[:cur])),
            "tag_payload_corr": tag_payload_corr,
            "anchor_payload": self.anchor_payload,
        }
