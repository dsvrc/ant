"""O-MAX — the unfair-advantage ceiling ladder.

Measurement instruments, never a deployable method. Each rung hands the learner a
privileged quantity AND the machinery engineered to exploit it, to find where the
achievable return at sigma=0.45 actually tops out. Everything lives in one env
wrapper (`OmaxMujocoMulti`) + a flag-gated actor std floor + configs; the algo is
plain HAPPO, no custom runner. See `omax_mujoco.py` and `README.md`.
"""
