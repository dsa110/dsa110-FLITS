# Controlled joint-fit runner

`run_controlled_joint_fit.py` is the fail-closed manuscript-candidate
entrypoint. It requires `--seed`, `--contract`, and a new `--receipt` path. The
general `run_joint_fit.py` entrypoint remains available for legacy diagnostics.
Controlled component-count runs also require a fixed `--gain-s2`; unsupported
gain likelihoods and fixed residual-dispersion-measure wrappers fail before
preprocessing.

The controlled entrypoint rejects a dirty source tree, a source revision
mismatch, an off-repository or untracked executed source file, command or
working-directory drift, runtime-environment drift, missing or changed
inputs/configuration, changed resolved priors/support, and existing outputs.
Runtime identity verifies every installed NumPy, SciPy, dynesty, PyYAML, and
Matplotlib file against its wheel record and hashes their full contents,
including compiled libraries. It separately records the invoked Python path,
its resolved base-interpreter path, virtual-environment and base prefixes, and
the complete named runtime flags, extended options, and warning options.
Interpreter-only flags that the frozen command cannot replay are rejected.
Resolving a virtual-environment interpreter symlink is not a valid substitute
for the invoked path.

The contract schema is `flits-controlled-joint-fit-contract/v1`:

```json
{
  "schema": "flits-controlled-joint-fit-contract/v1",
  "burst": "zach",
  "source_revision": "<40-hex git revision>",
  "command": {
    "argv": ["<python>", "<repo>/analysis/scattering-refit-2026-06/run_controlled_joint_fit.py", "..."],
    "working_directory": "<absolute working directory>"
  },
  "environment_identity_sha256": "<sha256>",
  "resolved_fit_identity_sha256": "<sha256>",
  "executed_source_files": [
    "controlled_entrypoint",
    "fit_driver",
    "joint_tf_prep_source",
    "burstfit_joint_source",
    "controlled_run_source",
    "model_grid_source",
    "diagnostic_source"
  ],
  "fit_configuration": {
    "burst": "zach",
    "nlive": 1000,
    "nproc": 4,
    "dlogz": 0.5,
    "sample": "rwalk",
    "seed": 20220207,
    "beta_bounds": null,
    "alpha_bounds": [2.0, 6.0],
    "marginalize_gain": false,
    "marginalize_gain_gp": false,
    "mu_degree": 1,
    "components_C": 2,
    "components_D": 4,
    "force_multi": false,
    "gain_s2": 100,
    "fixed_delta_dm_C": null,
    "fixed_delta_dm_D": null,
    "shared_zeta": true
  },
  "files": {
    "chime_input": {"path": "<absolute path>", "sha256": "<sha256>"},
    "dsa_input": {"path": "<absolute path>", "sha256": "<sha256>"},
    "chime_config": {"path": "<absolute path>", "sha256": "<sha256>"},
    "dsa_config": {"path": "<absolute path>", "sha256": "<sha256>"},
    "chime_telescope_config": {"path": "<absolute path>", "sha256": "<sha256>"},
    "dsa_telescope_config": {"path": "<absolute path>", "sha256": "<sha256>"},
    "environment_lock": {"path": "<repo>/uv.lock", "sha256": "<sha256>"},
    "controlled_entrypoint": {"path": "<absolute path>", "sha256": "<sha256>"},
    "fit_driver": {"path": "<absolute path>", "sha256": "<sha256>"},
    "joint_tf_prep_source": {"path": "<absolute path>", "sha256": "<sha256>"},
    "burstfit_joint_source": {"path": "<absolute path>", "sha256": "<sha256>"},
    "controlled_run_source": {"path": "<absolute path>", "sha256": "<sha256>"},
    "model_grid_source": {"path": "<absolute path>", "sha256": "<sha256>"},
    "diagnostic_source": {"path": "<absolute path>", "sha256": "<sha256>"}
  },
  "environment_variables": {
    "FLITS_REPO": "<absolute clean source checkout>",
    "FLITS_RUNS": "<absolute isolated run root>",
    "FLITS_JOINT_AUTO_TF": "1",
    "FLITS_ONPULSE_CROP": "1",
    "FLITS_ONPULSE_PAD": "0.5",
    "FLITS_SNR_TARGET": "10.0",
    "FLITS_MAX_CHANNELS": "64"
  }
}
```

To freeze data-derived priors, use a provisional contract and pass
`--resolved-identity-output <path>`. Preflight still must pass. The runner
writes the resolved likelihood class, every prior endpoint, ordered-component
separation, processed support-array hashes, and sampler settings, then aborts
before constructing the sampler when the provisional hash differs. Put the
reported hash into the contract, remove the incomplete receipt, and rerun the
exact same command.

The receipt is written before preprocessing. Immediately before sampler
construction, the controlled callback verifies the resolved-fit identity and
re-hashes the contract, source, inputs, configurations, environment lock, and
runtime environment. Finalization repeats those checks. After sampling it
binds the fit-summary bytes and both the container and canonical
scientific-array hashes for the weighted posterior.
The weighted-posterior artifact includes the original log-weight, evidence,
evidence-error, and likelihood-call histories. Finalization recomputes weights,
the final evidence, its error, and the call total from those histories.

The fixed completion set is fit summary, weighted samples, model grid,
residual diagnostics, and panel. Callers cannot weaken it. The receipt remains
`outputs_complete: false` until all five are appended and their internal roles,
dimensions, burst identity, source revision, contract hash, and resolved-fit
hash agree. The model grid uses the same finite Gaussian gain prior as the
multi-component likelihood, including its ridge and rank-one fallback; it does
not substitute an ordinary least-squares model. Downstream independent
regeneration inside finalization must match posterior summaries, model grids,
residual diagnostics, and rendered SVG bytes before review. The beta prior-edge
diagnostic uses weighted posterior mass, not a summary-only approximation. A
receipt alone never approves a fit or panel.

The runner surfaces the deprecated-Zach component guards in both the fit
summary and receipt. The downstream diagnostic step must evaluate them before
review admission.
