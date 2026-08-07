"""Per-episode dynamics randomization for the (ma)mujoco Ant, used by DORAEMON.

The DORAEMON runner pushes a domain-randomization (DR) *distribution* (a product
of Beta distributions over bounded dynamics factors) to every training env.  On
each ``reset`` this helper samples one factor per DR dimension and multiplies the
corresponding nominal MuJoCo model property by that factor, so each episode runs
under freshly sampled dynamics.  The sampled factor vector (xi) and the episode
return are reported back through the ``info`` dict at episode end, which the
runner harvests to fit the next DORAEMON distribution.

The DR spec is a list of dicts, one per dimension, each with keys:

    param   : which dynamics property to scale -- one of
              {"mass", "damping", "friction", "gravity", "gain"}.
    m, M    : lower / upper bound of the multiplicative factor.
    a, b    : current Beta(a, b) parameters (updated by DORAEMON each iteration).
    indices : (optional) explicit list of model element indices to scale;
              if omitted, a sensible default is used per ``param``.

This module touches only standard MuJoCo model arrays
(``model.body_mass`` / ``body_inertia`` / ``dof_damping`` / ``geom_friction`` /
``opt.gravity`` / ``actuator_gainprm``) via in-place index assignment, which works
for both the legacy ``mujoco_py`` and the new ``mujoco`` python bindings.  Factors
are always applied relative to the once-cached *nominal* values, so randomization
is multiplicative around the original model and fully repeatable.
"""

import numpy as np

_SUPPORTED_PARAMS = {"mass", "damping", "friction", "gravity", "gain"}


class DynamicsRandomizer:
    """Samples and applies per-episode dynamics factors to a MuJoCo model."""

    def __init__(self):
        self.spec = None          # list of dim dicts (param, m, M, a, b, [indices])
        self.version = -1         # distribution version (set by the runner)
        self._nominal = {}        # cached nominal model arrays, keyed by param
        self._cached = False
        self._warned = set()

    # ------------------------------------------------------------------ config
    def set_distribution(self, spec, version):
        """Install a new DR distribution (called once per DORAEMON iteration)."""
        self.spec = [dict(d) for d in spec]
        self.version = int(version)

    @property
    def enabled(self):
        return self.spec is not None and len(self.spec) > 0

    @property
    def ndims(self):
        return 0 if self.spec is None else len(self.spec)

    # ----------------------------------------------------------------- nominal
    def _cache_nominal(self, model):
        if self._cached:
            return
        needed = set(d["param"] for d in self.spec)
        if "mass" in needed:
            self._nominal["mass"] = np.array(model.body_mass, dtype=np.float64).copy()
            try:
                self._nominal["inertia"] = np.array(
                    model.body_inertia, dtype=np.float64
                ).copy()
            except Exception:
                pass
        if "damping" in needed:
            self._nominal["damping"] = np.array(
                model.dof_damping, dtype=np.float64
            ).copy()
        if "friction" in needed:
            self._nominal["friction"] = np.array(
                model.geom_friction, dtype=np.float64
            ).copy()
        if "gravity" in needed:
            self._nominal["gravity"] = np.array(
                model.opt.gravity, dtype=np.float64
            ).copy()
        if "gain" in needed:
            self._nominal["gain"] = np.array(
                model.actuator_gainprm, dtype=np.float64
            ).copy()
        self._cached = True

    # ------------------------------------------------------------------ sample
    def sample(self, np_random=None):
        """Sample one factor per DR dimension: y = x*(M-m)+m, x ~ Beta(a, b)."""
        rng = np_random if np_random is not None else np.random
        ys = np.empty(len(self.spec), dtype=np.float64)
        for i, d in enumerate(self.spec):
            try:
                x = float(rng.beta(d["a"], d["b"]))
            except Exception:
                x = float(np.random.beta(d["a"], d["b"]))
            ys[i] = x * (d["M"] - d["m"]) + d["m"]
        return ys

    # ------------------------------------------------------------------- apply
    def apply(self, model, factors):
        """Scale the nominal model properties by the sampled ``factors``."""
        self._cache_nominal(model)
        for i, d in enumerate(self.spec):
            param = d["param"]
            if param not in _SUPPORTED_PARAMS:
                self._warn_once(param, f"unsupported DR param '{param}'")
                continue
            try:
                self._apply_one(model, param, float(factors[i]), d.get("indices"))
            except Exception as exc:  # never let DR crash the rollout
                self._warn_once(param, f"could not apply DR param '{param}': {exc!r}")

    def _apply_one(self, model, param, factor, indices):
        if param == "mass":
            nominal = self._nominal["mass"]
            idx = indices if indices is not None else range(1, len(nominal))
            for j in idx:
                model.body_mass[j] = nominal[j] * factor
            if "inertia" in self._nominal:
                inertia = self._nominal["inertia"]
                for j in idx:
                    model.body_inertia[j] = inertia[j] * factor
        elif param == "damping":
            nominal = self._nominal["damping"]
            idx = indices if indices is not None else range(len(nominal))
            for j in idx:
                model.dof_damping[j] = nominal[j] * factor
        elif param == "friction":
            nominal = self._nominal["friction"]
            idx = indices if indices is not None else range(nominal.shape[0])
            for j in idx:
                model.geom_friction[j, 0] = nominal[j, 0] * factor
        elif param == "gravity":
            nominal = self._nominal["gravity"]
            model.opt.gravity[2] = nominal[2] * factor
        elif param == "gain":
            nominal = self._nominal["gain"]
            idx = indices if indices is not None else range(nominal.shape[0])
            for j in idx:
                model.actuator_gainprm[j, 0] = nominal[j, 0] * factor

    def _warn_once(self, key, msg):
        if key not in self._warned:
            print(f"[DynamicsRandomizer] {msg}")
            self._warned.add(key)
