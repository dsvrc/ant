"""PCR diagnosis-campaign package (spec: PCR_diagnosis_campaign_spec.md).

Contents
--------
* ``ant_pcr_v1``  — frozen golden copy of the deployed PCR ``ant.py`` (SEVERITY=0.9).
                    Never edited. The reference for the V0 equivalence test.
* ``ant_diag``    — the ONLY file ever copied over ``gym/envs/mujoco/ant.py``:
                    ``ant_pcr_v1`` + env-var-gated diagnostic knobs + info keys.
                    With no env vars set it is byte-identical to ``ant_pcr_v1``.
* ``probes``      — ProbeShim + the Tier-0 probe/transform library (pure numpy).
* ``sysid``       — E5 offline ridge system-id + DOB filter export (pure numpy).
* ``diag_mujoco`` — DiagMujocoMulti: the training-arm env shim (recorder + the
                    F2b/F2c privileged-input schema arms + eval clock offset).
* ``report_io``   — shared debug-file / report / bootstrap-CI helpers.

Nothing here is a method component (campaign Prohibition 1): no learned
identifier, no replay shaping, no reward edit, no host-loss change.
"""
