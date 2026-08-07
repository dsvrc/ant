"""Runner for the ECHO-R / HASAC backbone.

ECHO-R itself lives entirely in the env wrapper (``EchoRMujocoMulti``): the probe
injection, demodulation and ``c_hat`` conditioning happen inside ``env.step`` /
``env.reset``, and the host observation/action spaces are already augmented when
this runner builds its actor, critic and replay buffer.  The *learner* is
therefore unmodified HASAC (spec P6 / 5.1).

The only reason this thin subclass exists is HARL bookkeeping: the shared
off-policy runner selects the HASAC stochastic-action path and soft actor loss by
testing ``args["algo"] == "hasac"``.  Following the established pattern in this
repo (see ``OffPolicyMbcdRunner``), we temporarily alias the algo name to
``"hasac"`` around ``get_actions`` / ``train`` so those code paths fire while the
experiment still logs and checkpoints under its own name ``echor_hasac``.
"""

import torch

from harl.runners.off_policy_ha_runner import OffPolicyHARunner


class OffPolicyEchoRRunner(OffPolicyHARunner):
    """HASAC runner for ECHO-R (probe/estimator supplied by the env wrapper)."""

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            return super(OffPolicyEchoRRunner, self).get_actions(
                obs, available_actions, add_random
            )
        finally:
            self.args["algo"] = orig

    def train(self):
        orig = self.args["algo"]
        self.args["algo"] = "hasac"
        try:
            super(OffPolicyEchoRRunner, self).train()
        finally:
            self.args["algo"] = orig
