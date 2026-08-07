"""RECON [RE] — hindsight relabeling, and the window [ID] reads (spec §2.2/§2.1).

**Pure numpy, no torch, no mujoco** — so U1 can pin it to 1e-10 in a second.

The whole idea of RECON lives in ``relabel()``. The C-class liability obeys

    (L)   ℓ_i(t+1) = ρ·ℓ_i(t) + (1−ρ)·c(t)·Φ_i(u_{−i}(t)),    ℓ_i(0) = 0

and every quantity on the right except the scalar ``c`` is already sitting in the
CTDE buffer: Φ_i is a function of the *other agents' executed actions*, which the
trainer stores as a matter of course. Identify ``c`` from the same buffer ([ID],
``ECLIdentifier`` in clipfit mode) and (L) turns into an **exact, zero-variance
label** for every stored step — a teacher manufactured from replay rather than
handed over by the simulator. That is the step RMA and every other
teacher–student method needs simulator privilege for.

Ant-PCR instantiation (spec §4): ``Φ_i`` per joint type is ``Σ_{j≠i} u_{type,j}``
— exactly ``s`` in ``ant.py``:

    hip, ank = tau[0::2], tau[1::2]
    s[0::2] = hip.sum() - hip      # Σ_{j≠i} hip_j
    s[1::2] = ank.sum() - ank      # Σ_{j≠i} ankle_j
    self._d = _RHO*self._d + (1-_RHO)*(A*SEVERITY*s)

so with agent i = leg i and k=2 = (own hip, own ankle), ``ℓ̃_i`` reconstructs
``(d[2i], d[2i+1])`` and ``c = A(t)·SEVERITY``. ``u`` here is the **executed**
action (spec §2.1 ext 2): once [CP] is on, the policy's sample ``a`` and what the
env integrates differ, and it is the executed torque that feeds Φ.

Exactness. ``ĉ`` is constant over a window, so the spec's factored form
``ℓ̃ = ĉ·F_ρ̂[Φ]`` is the exact solution of (L) on that window; the within-window
drift of the true ``c`` (≤ Δ_c·W) is precisely the ``ε_id`` term T3 bounds the
return gap with, and it is measured live (``label_r2`` in ``recon_debug.csv``).
The recursion state is *carried across iterations* rather than restarted at each
rollout boundary, so labels are exact from the first step of every rollout — the
on-policy buffer is contiguous by construction, which is why HAPPO is the
reference host.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  [RE] — the reconstruction
# ---------------------------------------------------------------------------
def exertion(u):
    """Φ_i(u_{−i}) per joint type = Σ_{j≠i} u_j.

    u: (..., N, k) executed actions -> same shape. Category-C irreducibility is
    visible right here: agent i's own action is excluded, so N=1 ⇒ Φ ≡ 0 ⇒ ℓ ≡ 0.
    """
    return u.sum(axis=-2, keepdims=True) - u


def leak(S, dones, rho, carry=None):
    """x2(t) = F_ρ[S](t), the unit-severity liability: the (L) recursion with c=1.

    S:     (T, nt, N, k) exertion at each step
    dones: (T, nt)       1 if the episode ENDED at that step
    carry: None, or (x2_prev, S_prev, done_prev) from the previous rollout —
           (nt, N, k), (nt, N, k), (nt,). None ⇒ start of an episode (x2=0).
    Returns (x2 (T, nt, N, k), new_carry).

    x2(t) = 0 if the previous step ended the episode, else
            ρ·x2(t−1) + (1−ρ)·S(t−1)   — matching ant.py's `_d` update exactly,
    including its reset (``reset_model`` sets ``self._d[:] = 0``).
    """
    S = np.asarray(S, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    T = S.shape[0]
    x2 = np.zeros_like(S)
    if carry is None:
        x2_prev = np.zeros(S.shape[1:], dtype=np.float64)
        S_prev = np.zeros(S.shape[1:], dtype=np.float64)
        done_prev = np.ones(S.shape[1], dtype=np.float64)     # fresh episode
    else:
        x2_prev, S_prev, done_prev = carry
    if T == 0:
        return x2, (x2_prev, S_prev, done_prev)

    x2[0] = np.where(
        (done_prev > 0.5)[:, None, None], 0.0, rho * x2_prev + (1.0 - rho) * S_prev
    )
    for t in range(1, T):
        x2[t] = np.where(
            (dones[t - 1] > 0.5)[:, None, None],
            0.0,
            rho * x2[t - 1] + (1.0 - rho) * S[t - 1],
        )
    return x2, (x2[T - 1].copy(), S[T - 1].copy(), dones[T - 1].copy())


def relabel(u, dones, rho, c, carry=None):
    """[RE]: exact per-step labels ℓ̃_i(t) = ĉ · F_ρ̂[Φ_i(u_{−i})](t).

    u:     (T, nt, N, k) EXECUTED actions
    dones: (T, nt)
    rho:   ρ̂ from [ID]'s grid;  c: ĉ for this window (scalar)
    Returns (labels (T, nt, N, k), new_carry). O(T·N), one numpy pass.
    """
    x2, carry = leak(exertion(np.asarray(u, dtype=np.float64)), dones, rho, carry)
    return float(c) * x2, carry


# ---------------------------------------------------------------------------
#  self-supervised disturbance-observer target (label_mode: self_supervised)
# ---------------------------------------------------------------------------
def nominal_gain(readout, own_act):
    """Per-agent, per-channel own-action gain ĝ_i = <y_i, a_i> / <a_i, a_i>.

    readout, own_act: (T, nt, N, kc) for the readout channels that pair with the
    own action (Ant: hip=0, ankle=1). Returns (N, kc) — legs can have different
    gains (and signs), so this is NOT averaged across agents.

    On the coordinated gait ĝ is inflated (in magnitude) by the coupling that
    correlates with the own action, and the inflation GROWS with the payload — so
    a single-window ĝ is payload-dependent. ``calibrate_gain`` takes the TROUGH
    (smallest-magnitude) value across windows (where c≈0 ⇒ ĝ = the true nominal
    gain), the same trough-calibration idea as c_floor but for the gain.
    """
    y = np.asarray(readout, dtype=np.float64)
    a = np.asarray(own_act, dtype=np.float64)
    N, kc = y.shape[2], y.shape[3]
    g = np.zeros((N, kc))
    for i in range(N):
        for c in range(kc):
            yc = y[:, :, i, c].ravel()
            ac = a[:, :, i, c].ravel()
            aa = float(ac @ ac)
            g[i, c] = float(ac @ yc) / aa if aa > 1e-12 else 0.0
    return g


def calibrate_gain(g_hist):
    """Robust GLOBAL nominal gain (median) from a history of per-window ĝ
    (list/array of (N, kc)). Returns (N, kc).

    Why the median (a global constant), not a per-window or trough value: on the
    coordinated gait the disturbance is partly indistinguishable from an own-gain
    change (d ≈ −c·leak(a) has a large own-action-aligned component — this is the
    measured anti-observability). A per-window ĝ therefore *absorbs* the
    load-dependent part of the disturbance and its ERROR is anti-correlated with
    the load — exactly the central identifier's failure. A single global ĝ0
    instead leaves only a CONSTANT own-action leak in the target d̃ = (y−ĝ0·a)/ĝ0:
    that leak is phase-independent, so it merely rescales the executed action a
    hair (the policy adapts), rather than injecting phase-wrong compensation. The
    median over ~2 payload cycles is the robust global estimate; the disturbance's
    distinguishing LAG (d is the *leaked* own/coupling history, a is current) is
    what the causal filter then uses to separate d from that residual a-term."""
    h = np.asarray(g_hist, dtype=np.float64)          # (H, N, kc)
    g = np.median(h, axis=0)
    return np.where(np.abs(g) < 1e-6, 1e-6, g)


def disturbance_target(readout, own_act, g0):
    """The self-supervised disturbance estimate d̃_i(t) = (y_i − ĝ0_i·a_i) / ĝ0_i.

    The whole point of RECON's local path: the readout y_i ≈ ĝ_i·(a_i + d_i) (joint
    velocity = plant gain × delivered torque). Remove the OWN-action part with the
    trough-calibrated nominal gain ĝ0 and what's left, rescaled, is the disturbance
    in torque units — with NO central severity c anywhere. Because the residual
    y_i − ĝ0·a_i → 0 wherever d_i → 0, this target VANISHES at the payload trough
    on its own (fixing run #3/#4's trough injection), and it is not anti-correlated
    with the load the way the central identifier's ĉ is (measured −0.35), because
    it never has to separate own-gain from coupling inside one regression.

    readout, own_act: (T, nt, N, kc);  g0: (N, kc) nominal gains.
    Returns d̃ (T, nt, N, kc), torque units.
    """
    y = np.asarray(readout, dtype=np.float64)
    a = np.asarray(own_act, dtype=np.float64)
    g = np.asarray(g0, dtype=np.float64).reshape(1, 1, *g0.shape)
    g = np.where(np.abs(g) < 1e-6, 1e-6, g)
    return (y - g * a) / g


# ---------------------------------------------------------------------------
#  [ID]'s window — the last (W + margin) per-thread steps, shaped for ECLIdentifier
# ---------------------------------------------------------------------------
class _EclBufferView:
    """Exactly the attribute surface ``ECLIdentifier`` reads off a replay buffer.

    The identifier is reused verbatim as a library (spec §2.1); this view is what
    lets an *on-policy* host feed it. The flat slot layout is C-order over
    (step, thread) — identical to the off-policy buffer's — so the identifier's
    ``inds = (idx - L + arange(L)) % buffer_size`` slicing and its
    ``.reshape(wsteps, nt)`` both mean what they mean there, wrap included.

    ``obs``/``next_obs`` are deliberately absent: the readout is supplied directly
    via ``raw_readout``, and the index scan is turned off (``do_scan: false``) in
    favour of ``scan_readout_offset`` on the fresh rollout.
    """

    def __init__(self, actions, dones, raw_readout, nt, cap, pos, count):
        self.actions = actions              # list of N arrays (cap*nt, k)
        self.dones = dones                  # (cap*nt, 1)
        self.raw_readout = raw_readout      # (cap*nt, N, 2)
        self.n_rollout_threads = int(nt)
        self.buffer_size = int(cap * nt)
        self.cur_size = int(count * nt)
        self.idx = int((pos % cap) * nt)    # next slot to be written


class ReconIdWindow:
    """Rolling window of the last ``W + margin`` per-thread steps of (executed
    actions, raw readouts, dones) — [ID]'s regression data.

    Why a window and not just the latest rollout: ``W`` is set by the driver's
    drift rate (Δ_c·W is T3's ε_id budget), not by the host's rollout length. At
    the benchmark's 200-step rollouts one iteration is only 4k samples, whereas
    W=2000 per-thread steps is the 40k-sample regime in which this identifier was
    measured to lock at corr ≈ 1 — and it is still only 5% of a payload cycle.
    """

    def __init__(self, n_agents, act_dim, n_threads, capacity):
        self.N = int(n_agents)
        self.k = int(act_dim)
        self.nt = int(n_threads)
        self.cap = int(capacity)
        self.u = np.zeros((self.cap, self.nt, self.N, self.k), dtype=np.float64)
        self.y = np.zeros((self.cap, self.nt, self.N, 2), dtype=np.float64)
        self.done = np.zeros((self.cap, self.nt), dtype=np.float64)
        self.pos = 0
        self.count = 0

    def push(self, u, y, done):
        """One step: u (nt, N, k), y (nt, N, 2), done (nt,)."""
        p = self.pos % self.cap
        self.u[p] = u
        self.y[p] = y
        self.done[p] = done
        self.pos = (self.pos + 1) % self.cap
        self.count = min(self.count + 1, self.cap)

    def push_rollout(self, u, y, done):
        """A whole rollout: u (T, nt, N, k), y (T, nt, N, 2), done (T, nt)."""
        for t in range(u.shape[0]):
            self.push(u[t], y[t], done[t])

    def view(self):
        acts = [
            self.u[:, :, i, :].reshape(self.cap * self.nt, self.k)
            for i in range(self.N)
        ]
        return _EclBufferView(
            actions=acts,
            dones=self.done.reshape(self.cap * self.nt, 1),
            raw_readout=self.y.reshape(self.cap * self.nt, self.N, 2),
            nt=self.nt,
            cap=self.cap,
            pos=self.pos,
            count=self.count,
        )


# ---------------------------------------------------------------------------
#  readout-index scan (ECL's `_scan_response`-style assert, on the fresh rollout)
# ---------------------------------------------------------------------------
def readout_corr_profile(dobs, tau):
    """|corr| of every raw-obs-delta column with each agent's own torque.

    dobs: (T, nt, D) raw obs deltas (shared across agents — it is the global
    state); tau: (T, nt, N) own torque on ONE channel. Returns (N, D).

    This is what auto-selection uses: for each agent, the column with the highest
    |corr| is the coordinate its own torque actually drives — the right readout for
    the self-supervised disturbance observer, whatever the obs layout and whichever
    of velocity/position dominates at the current gait (they swap as the walker gets
    competent, which is why a FIXED index is fragile — measured: velocity cols fell
    to corr 0.11 while position cols held 0.60)."""
    dobs = np.asarray(dobs, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    D = dobs.shape[-1]
    flat = dobs.reshape(-1, D)
    flat = flat - flat.mean(axis=0, keepdims=True)
    den_o = np.sqrt((flat * flat).sum(axis=0)) + 1e-12
    N = tau.shape[-1]
    prof = np.zeros((N, D))
    for i in range(N):
        x = tau[..., i].reshape(-1)
        x = x - x.mean()
        nx = np.sqrt(float(x @ x)) + 1e-12
        prof[i] = np.abs((x @ flat) / (den_o * nx))
    return prof


def scan_readout_offset(dobs, tau, idx):
    """Which raw-obs coordinate actually responds to each agent's own torque?

    dobs: (T, nt, D) per-step raw (un-normalized) obs deltas, correctly paired
          across episode boundaries; tau: (T, nt, N) own torque on one channel;
          idx: (N,) the configured readout indices for that channel.
    Returns (modal_offset, median_best_corr, median_corr_at_idx).

    THREE numbers, because the argmax alone is a treacherous test. Own torque
    moves a joint's *velocity* AND (integrated over frame_skip) its *position*, so
    two equally-valid readouts exist one obs-block apart, and the single best
    column flips between them with the gait — which is why an argmax-only gate
    false-fired mid-run. The load-bearing quantity is ``corr_at_idx``: whether the
    *configured* column responds to own torque (a valid readout), not whether it
    is the global winner. ``modal_offset``/``best_corr`` stay for the suggestion.
    """
    dobs = np.asarray(dobs, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    D = dobs.shape[-1]
    idx = np.asarray(idx)
    flat = dobs.reshape(-1, D)
    flat = flat - flat.mean(axis=0, keepdims=True)
    den_o = np.sqrt((flat * flat).sum(axis=0))
    best_idx, best_cor, cor_at_idx = [], [], []
    for i in range(tau.shape[-1]):
        x = tau[..., i].reshape(-1)
        x = x - x.mean()
        nx = np.sqrt(float(x @ x)) + 1e-12
        cor = np.abs((x @ flat) / (den_o * nx + 1e-12))
        b = int(np.argmax(cor))
        best_idx.append(b)
        best_cor.append(float(cor[b]))
        if i < idx.shape[0] and 0 <= int(idx[i]) < D:
            cor_at_idx.append(float(cor[int(idx[i])]))
    off = np.asarray(best_idx) - idx[: len(best_idx)]
    vals, cnts = np.unique(off, return_counts=True)
    return (
        int(vals[int(np.argmax(cnts))]),
        float(np.median(best_cor)),
        float(np.median(cor_at_idx)) if cor_at_idx else 0.0,
    )
