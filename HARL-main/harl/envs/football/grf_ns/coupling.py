"""The declared coupling operator and the PACT basis.  Structure only, never fitted.

`PACT_NS_SPEC` NS-1.2 and II.3.  Everything here is computable from public
geometry -- where the teammates are, which way I am heading, who has the ball --
before a single episode runs, and nothing is fitted from run data.

THE MEDIUM: MY RUNNING LANE
--------------------------------------------------------------------------------
    element         the lane ahead of player i -- the cone within 135 degrees of
                    his heading, weighted by proximity
    occupancy       teammates standing or running in that lane (j != i)
    capacity        how much of that traffic the surface lets him ignore: high on
                    a firm pitch, low on a greasy one -- this is what the driver
                    derates
    harm            he swerves off his line, away from the traffic

    kappa(d)        = 1 / (1 + (d / lambda)^2)                proximity, declared
    sector(i, j)    = front  if  cos(angle(h_i, p_j - p_i)) >=  cos 45
                    = flank  if  -cos 45 <= cos(...) < cos 45
                    = (not in the lane) otherwise                 -- excluded
    W[i, j]         = recv_i * kappa(|p_j - p_i|) * 1[j in i's lane],  W[i, i] = 0
    x_m,i           = sum_{j != i, sector(i,j) = m} W[i, j]        the channels
    s_i             = sum_{j != i} W[i, j] * cross(h_i, unit(p_j - p_i))
                                                                   which side the
                                                                   traffic is on
    Q_m,i(t)        = rho Q_m,i(t-1) + (1 - rho) x_m,i(t)          PUBLIC, filtered
    psi_i           = [1, (Q_1,i - ref_1)/scale_1, (Q_2,i - ref_2)/scale_2]

    d_i             = (beta*(t) . Q_i) / load_norm                 PRIVATE
                      beta*_m(t) = sigma * L * A(t) * send_m
    e_i             = -sign(s_i)                                   PUBLIC: swerve
                                                                   AWAY from traffic

Three things this buys, each a requirement rather than a convenience:

* The model is EXACTLY linear in psi.  The lane's memory lives in the public
  filtered channels, not in the private disturbance, so the estimator's model
  class is correct rather than approximately correct (P-1.2).
* THE DIRECTION IS PUBLIC AND THE GAIN IS UNKNOWN.  A player can see which side
  his teammate is on; he cannot see how much room the surface will make him
  give today.  That split is what lets a scalar estimate cancel the whole
  disturbance, and it is why the channel is invertible.
* Every sum is strictly over ``j != i`` (P-3.1), so at N = 1 every channel is
  exactly zero, ``psi = [1, 0, 0]`` and the disturbance is exactly zero at any
  severity.  Category C, structurally.

THE OPERATOR IS ASYMMETRIC AND SPREAD (NS-1.2)
--------------------------------------------------------------------------------
``sector(i, j) != sector(j, i)`` in general (I can be in your lane while you are
behind me), ``recv`` differs between the ball carrier and the others, and the
kernel spans an order of magnitude between a teammate 3 m away and one 15 m
away.  A flat proxy -- every teammate equal -- is what NS-1.2 forbids; it was
measured on the URB instance at a fit gain of -0.0045.

numpy only.  No gfootball.
"""

import numpy as np

from .actions import DIR_VEC, N_DIRS

__all__ = ["Coupling"]

_COS45 = np.sqrt(0.5)
_EPS = 1e-12


class Coupling(object):
    """The declared operator, the filtered channels, and the basis.

    Args:
        n_agents:  the controlled squad.  ``r`` does not depend on it (P-1.1).
        p:         ``DialParams``.
        send:      the driver's normalised per-sector gains, ``(r,)``.
    """

    def __init__(self, n_agents, p, send):
        self.n = int(n_agents)
        self.p = p
        self.send = np.asarray(send, dtype=np.float64).reshape(-1)
        self.r = int(self.send.size)
        assert self.r == 2, "this instance declares two sectors (front, flank)"
        self._eye = np.eye(self.n, dtype=bool)

    # ------------------------------------------------------------------ geometry
    def _geometry(self, pos, head):
        pos = np.asarray(pos, dtype=np.float64).reshape(self.n, 2)
        head = np.asarray(head, dtype=np.int64).reshape(self.n)
        has = head >= 0
        h = np.zeros((self.n, 2))
        h[has] = DIR_VEC[head[has]]
        d = pos[None, :, :] - pos[:, None, :]                     # d[i, j] = p_j - p_i
        dist = np.sqrt((d ** 2).sum(-1))
        unit = d / np.maximum(dist, _EPS)[..., None]
        cosang = (unit * h[:, None, :]).sum(-1)                    # (n, n)
        cross = h[:, None, 0] * unit[..., 1] - h[:, None, 1] * unit[..., 0]
        kappa = 1.0 / (1.0 + (dist / float(self.p.kernel_lambda)) ** 2)
        kappa[self._eye] = 0.0                                      # W_ii = 0, ASSERTED
        front = has[:, None] & (cosang >= _COS45) & ~self._eye
        flank = has[:, None] & (cosang < _COS45) & (cosang >= -_COS45) & ~self._eye
        return kappa, cross, front, flank

    def W(self, pos, head, recv=None):
        """``W[i, j]`` -- how much teammate j loads i's lane.  ``(n, n)``, zero
        diagonal.  Declared from geometry, never fitted."""
        recv = np.ones(self.n) if recv is None else np.asarray(recv, dtype=np.float64)
        kappa, _, front, flank = self._geometry(pos, head)
        return recv[:, None] * kappa * (front | flank)

    # ------------------------------------------------------------------ basis
    def channels(self, pos, head, recv=None):
        """``x[i, m]`` and ``s[i]``: the per-sector lane load and the signed
        lateral load, both strictly over ``j != i``.

        ``pos``  ``(n, 2)`` teammates' positions (public)
        ``head`` ``(n,)``   my heading index 0..7, or -1 for 'not under way'
        ``recv`` ``(n,)``   receiver susceptibility (ball carrier), default 1

        A player who is not under way reads exactly zero on every channel: there
        is no lane to load.  Vectorised; ``channels_bruteforce`` is the
        definition and ``verify`` checks this against it at startup.
        """
        recv = np.ones(self.n) if recv is None else np.asarray(recv, dtype=np.float64)
        kappa, cross, front, flank = self._geometry(pos, head)
        w = recv[:, None] * kappa
        x = np.stack([(w * front).sum(1), (w * flank).sum(1)], axis=1)   # (n, r)
        s = (w * (front | flank) * cross).sum(1)                            # (n,)
        return x, s

    def channels_bruteforce(self, pos, head, recv=None):
        """The same thing written straight off the definition (P-3.2).

        Index order and self-exclusion are exactly the kind of wiring bug that
        leaves every diagnostic looking healthy, so the vectorised routine is
        checked against this at startup and the run aborts on mismatch.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(self.n, 2)
        head = np.asarray(head, dtype=np.int64).reshape(self.n)
        recv = np.ones(self.n) if recv is None else np.asarray(recv, dtype=np.float64)
        x = np.zeros((self.n, self.r))
        s = np.zeros(self.n)
        for i in range(self.n):
            if head[i] < 0:
                continue                                   # not under way: no lane
            h = DIR_VEC[head[i]]
            for j in range(self.n):
                if j == i:
                    continue                               # STRICTLY j != i
                dvec = pos[j] - pos[i]
                dist = float(np.sqrt(dvec @ dvec))
                u = dvec / max(dist, _EPS)
                c = float(h @ u)
                if c >= _COS45:
                    m = 0
                elif c >= -_COS45:
                    m = 1
                else:
                    continue                               # behind: not in the lane
                k = 1.0 / (1.0 + (dist / float(self.p.kernel_lambda)) ** 2)
                w = float(recv[i]) * k
                x[i, m] += w
                s[i] += w * (h[0] * u[1] - h[1] * u[0])
        return x, s

    def filter(self, Q, S, x, s):
        """One step of the lane's memory on the PUBLIC channels."""
        rho = float(self.p.rho)
        return rho * Q + (1.0 - rho) * x, rho * S + (1.0 - rho) * s

    def design(self, Q, ref, scale):
        """``psi = [1, (Q - ref) / scale]``, ``(n, 1 + r)``.

        P-3.3: centre and scale on a geometric reference.  Raw channels carry a
        large common mean against an intercept column of 1; on the source
        implementation that gave a design-matrix condition number of ~1.3e5 and an
        unidentifiable per-class split while prediction looked fine.
        """
        z = (np.asarray(Q, dtype=np.float64) - ref[None, :]) / np.maximum(scale[None, :], 1e-9)
        return np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

    # ------------------------------------------------------------------ references
    def _reference_draw(self, pos_ref, rng, jitter):
        k = rng.randint(0, pos_ref.shape[0])
        pos = pos_ref[k].copy()
        if jitter > 0:
            pos = pos + rng.randn(self.n, 2) * float(jitter)
        head = rng.randint(0, N_DIRS, self.n)
        return pos, head

    def geometric_reference(self, pos_ref, samples=512, seed=0, jitter=0.1):
        """``(ref, scale)``, each ``(r,)`` -- P-3.3's centring reference.

        The channel a player would see if the squad stood at its own spawn
        geometry, spread by a declared positional jitter (~5 m, the natural
        spread of a play), every player heading uniformly at random.  Structure
        only -- no run data enters.  An explicit ``RandomState`` so the reference
        cannot consume the run's RNG stream and cannot differ between arms.
        """
        pos_ref = np.asarray(pos_ref, dtype=np.float64).reshape(-1, self.n, 2)
        rng = np.random.RandomState(int(seed))
        vals = []
        for _ in range(int(samples)):
            pos, head = self._reference_draw(pos_ref, rng, jitter)
            x, _ = self.channels(pos, head)
            vals.append(x)
        V = np.concatenate(vals, axis=0)
        ref = V.mean(axis=0)
        scale = np.maximum(V.std(axis=0), 1e-6)
        return ref, scale

    def load_norm(self, pos_ref, samples=512, seed=7):
        """The reference value of ``send . x`` -- what makes sigma MEAN something.

        NS-2.4's statement is *"at sigma = 1, at the driver's peak, with the
        squad at the scenario's own spawn geometry and every player under way,
        the swerve reaching a player is L compass steps per step"*.  Dividing the
        disturbance by this makes that statement true in every scenario at every
        N, so the ladders can be read against each other.  Without it a row that
        looked harsher would only be a row with more teammates.

        Spawn geometry EXACTLY (no jitter): the statement has to be reproducible
        from the scenario file alone.  Headings uniform over the compass.
        """
        pos_ref = np.asarray(pos_ref, dtype=np.float64).reshape(-1, self.n, 2)
        rng = np.random.RandomState(int(seed))
        tot = 0.0
        for _ in range(int(samples)):
            pos, head = self._reference_draw(pos_ref, rng, 0.0)
            x, _ = self.channels(pos, head)
            tot += float((x * self.send[None, :]).sum(1).mean())
        ref = tot / float(samples)
        if not (ref > 0.0):
            raise RuntimeError(
                "the reference lane load came out non-positive at the spawn geometry; "
                "the coupling is inert there and sigma would have no meaning")
        return ref

    # ------------------------------------------------------------------ gates
    def verify(self, pos_ref, seed=0, trials=32, tol=1e-12):
        """Gate 1 (vectorised == definition) and gate 3 (N = 1 reads exactly
        zero).  Both abort; neither warns."""
        pos_ref = np.asarray(pos_ref, dtype=np.float64).reshape(-1, self.n, 2)
        rng = np.random.RandomState(int(seed))
        worst = 0.0
        for _ in range(int(trials)):
            pos, head = self._reference_draw(pos_ref, rng, 0.15)
            head = np.where(rng.rand(self.n) < 0.8, head, -1)
            recv = np.where(rng.rand(self.n) < 0.3, float(self.p.recv_ball), 1.0)
            x, s = self.channels(pos, head, recv)
            xb, sb = self.channels_bruteforce(pos, head, recv)
            worst = max(worst, float(np.max(np.abs(x - xb))), float(np.max(np.abs(s - sb))))
        if worst > tol:
            raise AssertionError(
                "GATE 1 FAILED: vectorised channels differ from the brute-force "
                "definition by %.3e.  Index order or self-exclusion is wrong; every "
                "downstream diagnostic would still look healthy." % worst)
        solo = Coupling(1, self.p, self.send)
        x1, s1 = solo.channels(pos_ref[0][:1], np.array([3]), np.array([self.p.recv_ball]))
        if float(np.max(np.abs(x1))) != 0.0 or float(np.max(np.abs(s1))) != 0.0:
            raise AssertionError(
                "GATE 3 FAILED: a lone player read a non-zero lane load; the sum is "
                "not strictly over j != i and this is category B in disguise")
        return ("channels == definition to %.1e over %d draws; N=1 reads exactly "
                "zero on all %d channels" % (worst, trials, self.r))

    # ------------------------------------------------------------------ report
    def operator_stats(self, pos, head, recv=None):
        """NS-1.2's three properties, measured rather than asserted."""
        W = self.W(pos, head, recv)
        off = W[~self._eye]
        off = off[off > 0]
        den = np.abs(W) + np.abs(W.T)
        mask = (~self._eye) & (den > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            asym = float(np.mean(np.abs(W - W.T)[mask] / den[mask])) if mask.any() else 0.0
        return dict(
            diag_max=float(np.max(np.abs(np.diag(W)))),
            spread=float(off.std() / max(off.mean(), 1e-30)) if off.size > 1 else 0.0,
            ratio=float(off.max() / max(off.min(), 1e-30)) if off.size > 1 else 1.0,
            asymmetry=asym,
            n_links=int(off.size),
        )

    @staticmethod
    def spread(pos):
        """Mean pairwise distance of the squad -- the commons signature for a
        continuous channel (does compensation bunch or spread the players?)."""
        pos = np.asarray(pos, dtype=np.float64)
        n = pos.shape[0]
        if n < 2:
            return float("nan")
        d = np.sqrt(((pos[None, :, :] - pos[:, None, :]) ** 2).sum(-1))
        return float(d[~np.eye(n, dtype=bool)].mean())

    def banner(self, load_norm=None, ref=None, scale=None):
        return ("[GRF-NS] coupling  N=%d  r=%d classes=(front, flank)  lambda=%.3f  rho=%.2f  "
                "recv_ball=%.2f  send(unknown)=%s  load_norm=%s  ref=%s  scale=%s"
                % (self.n, self.r, self.p.kernel_lambda, self.p.rho, self.p.recv_ball,
                   np.round(self.send, 3).tolist(),
                   "n/a" if load_norm is None else "%.4f" % load_norm,
                   "n/a" if ref is None else np.round(ref, 4).tolist(),
                   "n/a" if scale is None else np.round(scale, 4).tolist()))
