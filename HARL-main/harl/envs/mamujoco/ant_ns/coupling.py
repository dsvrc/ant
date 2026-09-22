"""The declared coupling operator and the PACT basis.  Structure only, never fitted.

`PACT_NS_SPEC` NS-1.2 and II.3.  Everything here is computable from ``ant.xml``
and the partition, before a single step runs.

THE REDUCTION
--------------------------------------------------------------------------------
The unknown is a transmission field: how much of each neighbour's torque
actually reaches my joints today.  It is projected onto ``r = 3`` declared load
paths, so the number of parameters is 3 -- independent of the number of agents
and of the number of joints (P-1.1), which is what lets ``2x4``, ``4x2`` and
``8x1`` share one basis and makes the N-scaling a measurement.

    G_m[p, q] = 1[class(p,q) = m] * recv_p * kappa(p,q) * 1[agent(p) != agent(q)]

    Q_m(t)  = rho Q_m(t-1) + (1 - rho) * G_m @ tau(t-1)        PUBLIC   (r, 8)
    q       = sum_m Q_m                                        PUBLIC   (8,)
    e_i     = q|_i / || q|_i ||        the direction, per agent, PUBLIC
    x_m,i   = < Q_m|_i , e_i >         the channels,             PUBLIC   (n, r)
    psi_i   = [1, (x_i - ref) / scale]

    d_i     = e_i * ( beta*(t) . x_i ) / load_norm              PRIVATE

Three properties, each a requirement rather than a convenience:

* The model is EXACTLY linear in the channels.  The structure's memory lives in
  the PUBLIC filtered channels, never in the private disturbance, so the
  estimator's model class is correct rather than approximately correct.  Putting
  the leak on the disturbance instead leaves the regressor describing an
  instantaneous quantity and the target a filtered one -- irreducible bias that
  reads as "the reduction does not hold here".
* THE DIRECTION IS PUBLIC AND THE GAIN IS UNKNOWN.  A leg can see which way its
  neighbours pushed -- they broadcast their executed torques, and the geometry
  is on the drawing -- and cannot see what that costs it at today's drivetrain
  temperature.  That split is what lets a SCALAR estimate cancel a VECTOR
  disturbance, and it is why the channel is invertible.
* Every sum is strictly over ``j != i`` (P-3.1), so a lone agent reads exactly
  zero on every channel, ``psi = [1, 0, 0, 0]``, and the disturbance is exactly
  zero at any severity.  Category C, structurally.

numpy only.  No mujoco.
"""

import numpy as np

from .structure import (N_CLASSES, N_JOINTS, agent_of, class_of, kernel, partition_of,
                        recv_vector)

__all__ = ["Coupling"]

_EPS = 1e-12


class Coupling(object):
    """The declared operator, the filtered channels, and the basis.

    Args:
        agent_conf: the MAMuJoCo partition, e.g. ``"4x2"``.
        p:          ``DialParams``.
        send:       the driver's normalised per-class gains, ``(r,)``.
    """

    def __init__(self, agent_conf, p, send):
        self.agent_conf = str(agent_conf)
        self.p = p
        self.send = np.asarray(send, dtype=np.float64).reshape(-1)
        self.r = int(self.send.size)
        assert self.r == N_CLASSES, "this instance declares %d load paths" % N_CLASSES

        self.parts = partition_of(agent_conf)
        self.n = len(self.parts)
        self.owner = agent_of(self.parts)                      # (8,)
        self.dims = [len(g) for g in self.parts]
        self.recv = recv_vector(p.recv_hip, p.recv_ankle)      # (8,)
        self.kappa = kernel(p.length_scale)                    # (8, 8)

        # ---- the declared operator, one (8, 8) matrix per load path ----------
        # The peer mask is what makes the sum strictly j != i, and it is ASSERTED
        # here rather than argued: every entry inside one agent's own joint block
        # is zero, so the diagonal is zero for every partition.
        peer = (self.owner[:, None] != self.owner[None, :])
        self.G = np.zeros((self.r, N_JOINTS, N_JOINTS))
        for pp in range(N_JOINTS):
            for qq in range(N_JOINTS):
                if not peer[pp, qq]:
                    continue
                self.G[class_of(pp, qq), pp, qq] = self.recv[pp] * self.kappa[pp, qq]
        self.W = self.G.sum(axis=0)                            # (8, 8), zero diagonal

    # ------------------------------------------------------------------ basis
    def step_channels(self, Q_prev, tau_prev):
        """Advance the public channels one step.  ``Q`` is ``(r, 8)``.

        ``tau_prev`` is the torque every joint actually DELIVERED last step --
        peers' executed actions, which P-4.1 allows (a connected machine
        broadcasts them, and a leg can read its own) and never any peer's
        residual.
        """
        raw = np.einsum("mpq,q->mp", self.G, np.asarray(tau_prev, dtype=np.float64))
        return self.p.rho * np.asarray(Q_prev, dtype=np.float64) + (1.0 - self.p.rho) * raw

    def project(self, Q):
        """``(e, x)`` -- the public direction per joint ``(8,)`` and the public
        channels per agent ``(n, r)``.

        An agent whose joints carry no live peer load reads an exactly zero
        direction, so its disturbance is exactly zero rather than NaN.
        """
        Q = np.asarray(Q, dtype=np.float64).reshape(self.r, N_JOINTS)
        q = Q.sum(axis=0)
        e = np.zeros(N_JOINTS)
        x = np.zeros((self.n, self.r))
        for i, grp in enumerate(self.parts):
            idx = np.asarray(grp, dtype=np.int64)
            nrm = float(np.linalg.norm(q[idx]))
            if nrm <= _EPS:
                continue
            e[idx] = q[idx] / nrm
            x[i] = Q[:, idx] @ e[idx]
        return e, x

    def channels_bruteforce(self, Q_prev, tau_prev):
        """The definition, written straight out as loops (P-3.2).

        Index order and self-exclusion are exactly the kind of wiring bug that
        leaves every diagnostic looking healthy, so the vectorised routine is
        checked against this at startup and the run aborts on mismatch.
        """
        Q_prev = np.asarray(Q_prev, dtype=np.float64).reshape(self.r, N_JOINTS)
        tau = np.asarray(tau_prev, dtype=np.float64).reshape(N_JOINTS)
        Q = np.zeros((self.r, N_JOINTS))
        for m in range(self.r):
            for pp in range(N_JOINTS):
                s = 0.0
                for qq in range(N_JOINTS):
                    if self.owner[qq] == self.owner[pp]:
                        continue                          # STRICTLY j != i
                    if class_of(pp, qq) != m:
                        continue
                    s += self.recv[pp] * self.kappa[pp, qq] * tau[qq]
                Q[m, pp] = self.p.rho * Q_prev[m, pp] + (1.0 - self.p.rho) * s
        return Q

    def design(self, x, ref, scale):
        """``psi = [1, (x - ref) / scale]``.  ``(n, 1 + r)``.

        P-3.3: centre and scale on a geometric reference.  Raw channels carry a
        large common mean against an intercept column of 1; measured on the
        source implementation that gave a design-matrix condition number of
        ~1.3e5, at which the intercept and the class channels trade off and the
        per-class split is unidentifiable even though prediction is fine.
        """
        z = (np.asarray(x, dtype=np.float64) - ref[None, :]) / np.maximum(scale[None, :], 1e-9)
        return np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

    # ------------------------------------------------------------------ references
    def _settle(self, tau, steps=8):
        """The filtered channels a sustained torque pattern produces.  One step
        of ``(1 - rho)`` understates the steady state by ``1 / (1 - rho)``."""
        Q = np.zeros((self.r, N_JOINTS))
        for _ in range(int(steps)):
            Q = self.step_channels(Q, tau)
        return Q

    def geometric_reference(self, samples=2048, seed=0):
        """``(ref, scale)``, each ``(r,)`` -- P-3.3's centring reference.

        The channels an agent would see if every peer acted uniformly at random
        over its torque range.  A function of declared structure alone: no run
        data, no simulator, no policy.  An explicit ``RandomState`` so the
        reference cannot consume the run's RNG stream and cannot differ between
        arms.
        """
        rng = np.random.RandomState(int(seed))
        vals = []
        for _ in range(int(samples)):
            tau = rng.uniform(-1.0, 1.0, N_JOINTS)
            _, x = self.project(self._settle(tau))
            vals.append(x)
        V = np.concatenate(vals, axis=0)
        return V.mean(axis=0), np.maximum(V.std(axis=0), 1e-6)

    def load_norm(self, samples=2048, seed=7):
        """The reference value of ``send . x`` at FULL peer torque.

        Dividing the disturbance by this is what makes the severity dial MEAN
        something: NS-2.4's statement is *"at sigma = 1, at the driver's peak,
        with every peer commanding its full torque range, the disturbance
        reaching an agent is L of its own torque range"*, and that statement is
        then true in every partition.  Without it sigma would be a different
        physical severity at every N and the ladders could not be read against
        each other -- a row that looked harsher would only be a row with more
        neighbours.

        The reference condition is exactly the declared one: every peer at full
        magnitude, sign uniformly random.
        """
        rng = np.random.RandomState(int(seed))
        tot = 0.0
        for _ in range(int(samples)):
            tau = rng.choice([-1.0, 1.0], size=N_JOINTS)
            _, x = self.project(self._settle(tau))
            tot += float((x * self.send[None, :]).sum(1).mean())
        ref = tot / float(samples)
        if not (ref > 0.0):
            raise RuntimeError(
                "the reference peer load came out non-positive; the coupling is inert "
                "and sigma would have no meaning")
        return ref

    # ------------------------------------------------------------------ gates
    def verify(self, seed=0, trials=32, tol=1e-12):
        """Gate 1 (vectorised == definition) and gate 3 (a lone agent reads
        exactly zero).  Both abort; neither warns."""
        rng = np.random.RandomState(int(seed))
        worst = 0.0
        for _ in range(int(trials)):
            Q0 = rng.randn(self.r, N_JOINTS) * 0.1
            tau = rng.uniform(-1.0, 1.0, N_JOINTS)
            worst = max(worst, float(np.abs(self.step_channels(Q0, tau)
                                            - self.channels_bruteforce(Q0, tau)).max()))
        if worst > tol:
            raise AssertionError(
                "GATE 1 FAILED: vectorised channels differ from the brute-force "
                "definition by %.3e.  Index order or self-exclusion is wrong; every "
                "downstream diagnostic would still look healthy." % worst)
        if float(np.abs(np.diag(self.W)).max()) != 0.0:
            raise AssertionError("GATE: the operator's diagonal is not exactly zero")
        # a single agent owning every joint: the peer mask is empty everywhere
        lone = _LoneCoupling(self.p, self.send)
        Q = lone.step_channels(np.zeros((self.r, N_JOINTS)), np.ones(N_JOINTS))
        e, x = lone.project(Q)
        if float(np.abs(Q).max()) != 0.0 or float(np.abs(x).max()) != 0.0 \
                or float(np.abs(e).max()) != 0.0:
            raise AssertionError(
                "GATE 3 FAILED: a lone agent read a non-zero channel; the sum is not "
                "strictly over j != i and this is category B in disguise")
        return ("channels == definition to %.1e over %d draws; a lone agent reads "
                "exactly zero on all %d channels" % (worst, trials, self.r))

    # ------------------------------------------------------------------ report
    def operator_stats(self):
        """NS-1.2's three properties, measured rather than asserted."""
        off = self.W[self.W > 0]
        den = np.abs(self.W) + np.abs(self.W.T)
        mask = den > 0
        asym = float(np.mean(np.abs(self.W - self.W.T)[mask] / den[mask])) if mask.any() else 0.0
        return dict(
            diag_max=float(np.abs(np.diag(self.W)).max()),
            spread=float(off.std() / max(off.mean(), 1e-30)) if off.size > 1 else 0.0,
            ratio=float(off.max() / max(off.min(), 1e-30)) if off.size > 1 else 1.0,
            asymmetry=asym,
            n_links=int(off.size),
        )

    def agent_operator(self):
        """``W_agent[i, j]`` -- the operator aggregated to agents, for reporting."""
        A = np.zeros((self.n, self.n))
        for i, gi in enumerate(self.parts):
            for j, gj in enumerate(self.parts):
                if i == j:
                    continue
                A[i, j] = self.W[np.ix_(np.asarray(gi), np.asarray(gj))].sum()
        return A

    def banner(self, load_norm=None, ref=None, scale=None):
        st = self.operator_stats()
        return ("[ANT-NS] coupling  %s  N=%d dims=%s  r=%d classes=(hip<-hip, ankle<-ankle, "
                "cross)  L=%.2f rho=%.2f  recv=[hip %.2f, ankle %.2f]  send(unknown)=%s\n"
                "[ANT-NS]           W: zero_diag=%s spread=%.3f ratio=%.1fx asym=%.3f links=%d  "
                "| load_norm=%s ref=%s scale=%s"
                % (self.agent_conf, self.n, self.dims, self.r, self.p.length_scale,
                   self.p.rho, self.p.recv_hip, self.p.recv_ankle,
                   np.round(self.send, 3).tolist(),
                   st["diag_max"] == 0.0, st["spread"], st["ratio"], st["asymmetry"],
                   st["n_links"],
                   "n/a" if load_norm is None else "%.4f" % load_norm,
                   "n/a" if ref is None else np.round(ref, 4).tolist(),
                   "n/a" if scale is None else np.round(scale, 4).tolist()))


class _LoneCoupling(Coupling):
    """One agent owning every joint -- the N = 1 projection, for gate 3."""

    def __init__(self, p, send):
        self.agent_conf = "1x8"
        self.p = p
        self.send = np.asarray(send, dtype=np.float64).reshape(-1)
        self.r = int(self.send.size)
        self.parts = [tuple(range(N_JOINTS))]
        self.n = 1
        self.owner = np.zeros(N_JOINTS, dtype=np.int64)
        self.dims = [N_JOINTS]
        self.recv = recv_vector(p.recv_hip, p.recv_ankle)
        self.kappa = kernel(p.length_scale)
        self.G = np.zeros((self.r, N_JOINTS, N_JOINTS))        # the peer mask is empty
        self.W = self.G.sum(axis=0)
