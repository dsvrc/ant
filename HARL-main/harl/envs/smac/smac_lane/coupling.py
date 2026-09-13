"""The declared operator for LANE SWERVE UNDER DRIFT -- SMAC's (C, invertible) cell.

THE MEDIUM: MY LANE
--------------------------------------------------------------------------------
A unit moving with a teammate in its path has to go round it: it swerves off the
line it was ordered along.  How far depends on how much room the ground leaves
(the driver).  A unit ALONE never swerves round anyone.

    element       the lane ahead of unit i: the cone within 135 degrees of its
                  ORDERED move direction h_i, weighted by proximity
    kappa(d)      = 1 / (1 + (d / lambda)^2)                     declared
    sector(i, j)  = front  if cos(angle(h_i, p_j - p_i)) >= cos 45
                  = flank  if -cos 45 <= cos(...) < cos 45
                  = not in the lane otherwise (behind)
    x_m,i         = sum_{j != i, alive, sector m} kappa(|p_j - p_i|)     channels
    s_i           = sum_{j != i, in lane} kappa * cross(h_i, unit(p_j - p_i))
                                                    which side the traffic is on
    Q, S          = rho-filtered x, s                             PUBLIC memory
    psi_i         = [1, (Q_i - ref) / scale]

    d_i           = beta*(t) . Q_i / load_norm      swerve angle, radians -- PRIVATE
                    beta*_m(t) = sigma * L * A(t) * send_m
    e_i           = -sign(S_i)                      direction: AWAY from traffic -- PUBLIC

WHY IT IS INVERTIBLE.  SMAC sends every move as a world-space point
(x +- 2, y +- 2), so the executed move can be rotated by any angle, and a rotation
of the plane has an exact inverse.  The direction of the swerve is public (which
side the teammate is on); only its size -- the state of the ground -- is unknown.
A scalar estimate of that size cancels the whole disturbance: with d_hat == d the
executed order IS the commanded order.

Every sum is strictly over j != i, so a lone unit reads exactly zero on every
channel and swerves exactly zero at any severity -- category C, structurally.
The model is exactly linear in psi because the memory lives in the public filtered
channels (P-1.2).

numpy only.
"""

import numpy as np

_COS45 = float(np.sqrt(0.5))
_EPS = 1e-12

#: SMAC's four move actions, in action-index order 2..5: north, south, east, west
MOVE_VEC = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
SECTOR_NAMES = ("front", "flank")


def heading_of(actions_int, n):
    """Unit move vector per agent from SMAC action ids; zeros if not moving."""
    h = np.zeros((n, 2))
    for i, a in enumerate(actions_int):
        if 2 <= int(a) <= 5:
            h[i] = MOVE_VEC[int(a) - 2]
    return h


def rotate(v, phi):
    """Rotate 2-vectors ``v`` (n, 2) counter-clockwise by ``phi`` (n,) radians."""
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


class LaneCoupling(object):
    """Lane geometry, the filtered public channels, and the basis."""

    r = 2

    def __init__(self, n_agents, kernel_lambda=2.0, rho=0.5, send=(1.4, 0.6)):
        self.n = int(n_agents)
        self.lam = float(kernel_lambda)
        self.rho = float(rho)
        self.send = np.asarray(send, dtype=np.float64).reshape(self.r)
        self._eye = np.eye(self.n, dtype=bool)

    def _geometry(self, pos, head, alive):
        pos = np.asarray(pos, dtype=np.float64).reshape(self.n, 2)
        h = np.asarray(head, dtype=np.float64).reshape(self.n, 2)
        al = np.asarray(alive, dtype=np.float64).reshape(self.n) > 0
        moving = (np.abs(h).sum(1) > 0) & al
        d = pos[None, :, :] - pos[:, None, :]                       # d[i, j] = p_j - p_i
        dist = np.sqrt((d ** 2).sum(-1))
        unit = d / np.maximum(dist, _EPS)[..., None]
        cosang = (unit * h[:, None, :]).sum(-1)
        cross = h[:, None, 0] * unit[..., 1] - h[:, None, 1] * unit[..., 0]
        kappa = 1.0 / (1.0 + (dist / self.lam) ** 2)
        kappa[self._eye] = 0.0                                     # W_ii = 0, ASSERTED
        ok = moving[:, None] & al[None, :] & ~self._eye
        front = ok & (cosang >= _COS45)
        flank = ok & (cosang < _COS45) & (cosang >= -_COS45)
        return kappa, cross, front, flank

    def channels(self, pos, head, alive):
        """``x`` (n, r) per-sector lane load and ``s`` (n,) signed lateral load."""
        kappa, cross, front, flank = self._geometry(pos, head, alive)
        x = np.stack([(kappa * front).sum(1), (kappa * flank).sum(1)], axis=1)
        s = (kappa * (front | flank) * cross).sum(1)
        return x, s

    def channels_bruteforce(self, pos, head, alive):
        pos = np.asarray(pos, dtype=np.float64).reshape(self.n, 2)
        h = np.asarray(head, dtype=np.float64).reshape(self.n, 2)
        al = np.asarray(alive, dtype=np.float64).reshape(self.n)
        x = np.zeros((self.n, self.r))
        s = np.zeros(self.n)
        for i in range(self.n):
            if al[i] <= 0 or not np.any(h[i]):
                continue                                     # not under way
            for j in range(self.n):
                if j == i or al[j] <= 0:
                    continue                                 # STRICTLY j != i
                dv = pos[j] - pos[i]
                dist = float(np.sqrt(dv @ dv))
                u = dv / max(dist, _EPS)
                c = float(h[i] @ u)
                if c >= _COS45:
                    m = 0
                elif c >= -_COS45:
                    m = 1
                else:
                    continue                                 # behind
                k = 1.0 / (1.0 + (dist / self.lam) ** 2)
                x[i, m] += k
                s[i] += k * (h[i, 0] * u[1] - h[i, 1] * u[0])
        return x, s

    def filter(self, Q, S, x, s):
        return self.rho * Q + (1.0 - self.rho) * x, self.rho * S + (1.0 - self.rho) * s

    def W(self, pos, head, alive):
        kappa, _, front, flank = self._geometry(pos, head, alive)
        return kappa * (front | flank)

    # ------------------------------------------------------------------ references
    def reference(self, spacing, samples=2048, seed=0, jitter=None):
        """``(ref, scale, load_norm)`` from the DECLARED formation: the squad in a
        line at ``spacing``, jittered by half a spacing, every unit ordered along a
        uniformly random compass direction.  Structure only; its own RNG."""
        rng = np.random.RandomState(int(seed))
        jit = 0.5 * spacing if jitter is None else float(jitter)
        base = np.stack([np.arange(self.n) * float(spacing), np.zeros(self.n)], axis=1)
        X, L = [], []
        for _ in range(int(samples)):
            pos = base + rng.randn(self.n, 2) * jit
            head = MOVE_VEC[rng.randint(0, 4, self.n)]
            x, _ = self.channels(pos, head, np.ones(self.n))
            X.append(x)
            L.append(float((x * self.send[None, :]).sum(1).mean()))
        V = np.concatenate(X, axis=0)
        ref = V.mean(axis=0)
        scale = np.maximum(V.std(axis=0), 1e-6)
        load_norm = float(np.mean(L))
        assert load_norm > 0.0, "the reference lane load is zero: sigma would mean nothing"
        return ref, scale, load_norm

    def design(self, Q, ref, scale):
        z = (np.asarray(Q, dtype=np.float64) - ref[None, :]) / scale[None, :]
        return np.concatenate([np.ones((self.n, 1)), z], axis=1)
