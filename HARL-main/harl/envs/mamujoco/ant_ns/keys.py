"""Every key the layer reads from the task config -- in one mujoco-free place so
``check_plumbing.py`` can check it against ``mamujoco_ns.yaml`` in both
directions without a simulator on the machine.

The dial keys are TASK PHYSICS (NS-3.1) and belong in the yaml; the compensator
keys are set by the PACT-family runners and must not be edited in the yaml.
"""

NS_KWARGS = (
    # the dial (NS-3.1: task physics, read from the task config)
    "ns_on", "ns_severity", "ns_period", "ns_warm_fraction", "ns_loss_at_sigma1",
    "ns_mean_preserving", "ns_phase0",
    # the medium and the declared classes
    "ns_rho", "ns_length_scale", "ns_recv_spread",
    "ns_send_hh", "ns_send_aa", "ns_send_cross",
    "ns_y_clip", "ns_corr_clip", "ns_load_norm",
    # the (B) control
    "ns_direct",
    # the compensator (set by the PACT-family runners, never by a baseline)
    "ns_pact", "ns_trust", "ns_g_fixed", "ns_oracle", "ns_intercept_only",
    "ns_mu", "ns_p0", "ns_warmup", "ns_p_trace_max",
    # ablation: hand the policy its own residual
    "ns_observe_residual",
    # escape hatch for an already-disturbed host; never for a reported run
    "ns_allow_patched_host",
)

#: keys the runner sets; a yaml that sets them non-default is a misconfiguration
RUNNER_KEYS = ("ns_pact", "ns_trust", "ns_oracle", "ns_intercept_only")

#: DialParams field -> yaml key (all are ns_<field>)
DIAL_FIELDS = ("severity", "period", "warm_fraction", "loss_at_sigma1", "mean_preserving",
               "rho", "length_scale", "recv_spread", "send_hh", "send_aa",
               "send_cross", "y_clip", "corr_clip", "direct")

#: PactConfig field -> yaml key
PACT_FIELDS = {"enabled": "ns_pact", "trust": "ns_trust", "g_fixed": "ns_g_fixed",
               "oracle": "ns_oracle", "intercept_only": "ns_intercept_only", "mu": "ns_mu",
               "p0": "ns_p0", "warmup": "ns_warmup", "p_trace_max": "ns_p_trace_max"}
