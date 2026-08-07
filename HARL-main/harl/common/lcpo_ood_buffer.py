"""Out-of-distribution (OOD) state buffer for LCPO.

Faithful re-implementation of ``OutOfDSampler`` from the official LCPO repo
(https://github.com/pouyahmdn/LCPO, ``windy-gym/buffer/buffer_ood.py``):

* a reservoir-sampled all-time FIFO of past observations (``states_fifo``), and
* a ring buffer of the most recent observations (``recent_states``) that
  represents the *current* context.

``get()`` samples candidate states from the all-time FIFO and keeps only those
that are *distant* from the recent window -- i.e. states that belong to a
different (past) context. These are the anchor states on which LCPO constrains
the policy to stay unchanged.

Difference from the original: the LCPO paper computes the distance over the
``only_context`` slice of the observation (the part that encodes the
non-stationary context). In our setting the context (motor heat / time-of-day)
is *hidden* -- it is not in the observation -- so we measure novelty over the
**full observation** (the repo's own ``mahala_full`` option), made robust to
high-dimensional / constant observation dimensions.
"""

import numpy as np


class OODBuffer(object):
    def __init__(
        self,
        obs_len,
        recent_window,
        capacity,
        dist_type="diag_mahala",
        thresh=-2.0,
        thresh_mode="absolute",
        percentile=20.0,
        eps=1e-6,
    ):
        """Initialize the OOD buffer.

        Args:
            obs_len: (int) observation dimensionality.
            recent_window: (int) size of the recent-context ring buffer.
            capacity: (int) capacity of the all-time reservoir FIFO.
            dist_type: (str) "l2" | "diag_mahala" | "mahala_full".
            thresh: (float) ABSOLUTE distance threshold (used when
                ``thresh_mode == 'absolute'``). For "l2" a candidate is distant if
                its mean-squared distance > thresh; for the Mahalanobis variants a
                candidate is distant if its log-likelihood ``ll = -d/2 < thresh``.
            thresh_mode: (str) "absolute" | "percentile".
                * "absolute": compare the raw score against ``thresh`` (a fixed
                  number). Simple, but the score distribution drifts as the policy
                  specializes per context, so the flagged fraction is not stable.
                * "percentile": each call flags the most-distant ``percentile`` %% of
                  candidates (an adaptive cutoff). The anchored fraction is then
                  constant by construction -- the robust knob to tune here.
            percentile: (float) target distant fraction in %% for "percentile" mode
                (e.g. 20 -> anchor the most out-of-distribution 20%% of past states).
            eps: (float) numerical floor for variances / covariance ridge.
        """
        self.obs_len = obs_len
        self.win = recent_window
        self.cap = capacity
        self.dist_type = dist_type
        self.thresh = thresh
        self.thresh_mode = thresh_mode
        self.percentile = float(percentile)
        self.eps = eps

        self.states_fifo = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.recent_states = np.zeros([self.win, self.obs_len], dtype=np.float32)

        self.num_samples_so_far = 0
        self.i_win = 0

        # diagnostics from the most recent ``get()`` call (read by the runner for
        # console / tensorboard logging; see ``_record_stats``).
        self.last_stats = None

    # ------------------------------------------------------------------
    # distance / OOD test
    # ------------------------------------------------------------------
    def _raw_scores(self, data, base):
        """Score every ``data`` row by novelty w.r.t. the recent window ``base``.

        Returns ``(scores, higher_is_distant)``:
          * ``l2``: mean-squared distance to the recent mean (HIGHER == more OOD).
          * ``diag_mahala`` / ``mahala_full``: a Gaussian log-likelihood ``ll``
            (LOWER == more OOD).
        No thresholding here -- the raw score distribution is independent of the
        threshold, which is what lets the threshold be calibrated from the logs.
        """
        if self.dist_type == "l2":
            mu = base.mean(axis=0)
            scores = ((data - mu) ** 2).mean(axis=-1)
            return scores.astype(np.float32), True
        elif self.dist_type == "diag_mahala":
            mu = base.mean(axis=0)
            std = base.std(axis=0) + self.eps
            z = (data - mu) / std
            d = (z ** 2).mean(axis=-1)  # average squared z-score over dims
            ll = -d / 2
            return ll.astype(np.float32), False
        elif self.dist_type == "mahala_full":
            mu = base.mean(axis=0)
            cov = np.cov(base, rowvar=False) + self.eps * np.eye(self.obs_len)
            cen = data - mu
            try:
                lu = np.linalg.cholesky(cov)
                y = np.linalg.solve(lu, cen.T)
                d = np.einsum("ij,ij->j", y, y)
            except np.linalg.LinAlgError:
                # fall back to diagonal whitening if cov is not PD
                std = base.std(axis=0) + self.eps
                d = (((data - mu) / std) ** 2).sum(axis=-1)
            ll = -d / self.obs_len / 2
            return ll.astype(np.float32), False
        else:
            raise ValueError(f"Unknown OOD distance type: {self.dist_type}")

    def _distant_scores(self, data, base):
        """Score ``data`` and flag the distant (OOD) rows under the active mode.

        Returns ``(scores, mask, eff_thresh)`` where ``mask`` selects the distant
        rows and ``eff_thresh`` is the score cutoff actually applied (the fixed
        ``thresh`` in absolute mode, or the data-driven percentile cutoff otherwise).
        """
        if len(base) < 2:
            # not enough context to define a distribution -> nothing is "distant"
            n = len(data)
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=bool),
                float(self.thresh),
            )

        scores, higher_is_distant = self._raw_scores(data, base)

        if self.thresh_mode == "percentile" and scores.size > 0:
            if higher_is_distant:
                cut = float(np.percentile(scores, 100.0 - self.percentile))
                mask = scores >= cut
            else:
                cut = float(np.percentile(scores, self.percentile))
                mask = scores <= cut
            return scores, mask, cut

        # absolute mode
        if higher_is_distant:
            mask = scores > self.thresh
        else:
            mask = scores < self.thresh
        return scores, mask, float(self.thresh)

    def is_distant(self, data, base):
        """Boolean mask of which ``data`` rows are distant from ``base``."""
        return self._distant_scores(data, base)[1]

    def _record_stats(self, scores, mask, eff_thresh, recents, alls):
        """Stash a diagnostic summary of the candidate score distribution.

        For the mahala variants the low-tail percentiles of ``scores`` are exactly
        where a useful absolute ``thresh`` should sit; ``eff_thresh`` is the cutoff
        actually applied this call (in percentile mode it is the data-driven cutoff,
        which drifts as the policy specializes -- that drift is *why* percentile mode
        exists). ``frac_distant`` tells you whether the threshold is flagging too
        little (-> degenerates to plain HAPPO) or too much (-> over-constrained).
        """
        higher_is_distant = self.dist_type == "l2"
        if scores.size > 0:
            qs = [0, 1, 5, 10, 25, 50, 75, 90, 100]
            pct = {q: float(np.percentile(scores, q)) for q in qs}
        else:
            pct = {q: float("nan") for q in [0, 1, 5, 10, 25, 50, 75, 90, 100]}
        self.last_stats = {
            "label": "l2" if higher_is_distant else "ll",
            "higher_is_distant": higher_is_distant,
            "mode": self.thresh_mode,
            "thresh": float(self.thresh),
            "eff_thresh": float(eff_thresh),
            "n_candidates": int(scores.size),
            "n_distant": int(mask.sum()),
            "frac_distant": float(mask.mean()) if scores.size else 0.0,
            "pct": pct,
            "recent_fill": int(len(recents)),
            "fifo_fill": int(len(alls)),
            "total_seen": int(self.num_samples_so_far),
        }

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def get(self, rng, batch_size):
        """Sample up to ``batch_size`` past states distant from the recent window.

        Returns a (M, obs_len) float32 array (M <= batch_size), or an empty
        array if not enough distant states could be found.
        """
        self.last_stats = None
        if self.num_samples_so_far == 0:
            return np.zeros((0, self.obs_len), dtype=np.float32)
        recents = self.recent_states[: min(self.num_samples_so_far, self.win)]
        alls = self.states_fifo[: min(self.num_samples_so_far, self.cap)]
        rets = []
        resamples = 0
        while len(rets) < batch_size and resamples < 5:
            cand = alls[rng.choice(len(alls), size=batch_size)]
            scores, distant, eff_thresh = self._distant_scores(cand, recents)
            if resamples == 0:
                # record the score distribution of a representative candidate batch
                # (the first pass) so the threshold can be calibrated from the logs.
                self._record_stats(scores, distant, eff_thresh, recents, alls)
            if distant.any():
                rets.extend(cand[distant])
            resamples += 1
        if len(rets) >= batch_size:
            return np.asarray(rets[:batch_size], dtype=np.float32)
        elif len(rets) > 0:
            # return whatever distant states we found (still useful for anchoring)
            return np.asarray(rets, dtype=np.float32)
        else:
            return np.zeros((0, self.obs_len), dtype=np.float32)

    # ------------------------------------------------------------------
    # insertion (reservoir sampling for the FIFO + ring buffer for recents)
    # ------------------------------------------------------------------
    def add_many(self, states, rng):
        """Insert a batch of states (reservoir sampling)."""
        states = np.asarray(states, dtype=np.float32)
        n = len(states)
        if n == 0:
            return

        if self.num_samples_so_far + n <= self.cap:
            self.states_fifo[self.num_samples_so_far: self.num_samples_so_far + n] = states
        else:
            for i in range(n):
                if self.num_samples_so_far + i < self.cap:
                    self.states_fifo[self.num_samples_so_far + i] = states[i]
                else:
                    idx = rng.choice(self.num_samples_so_far + i + 1)
                    if idx < self.cap:
                        self.states_fifo[idx] = states[i]

        # recent ring buffer
        if n >= self.win:
            self.recent_states[:] = states[-self.win:]
            self.i_win = 0
        elif self.i_win + n <= self.win:
            self.recent_states[self.i_win: self.i_win + n] = states
            self.i_win = (self.i_win + n) % self.win
        else:
            first = self.win - self.i_win
            self.recent_states[self.i_win:] = states[:first]
            self.recent_states[: n - first] = states[first:]
            self.i_win = (self.i_win + n) % self.win

        self.num_samples_so_far += n
