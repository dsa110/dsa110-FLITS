export const meta = {
  name: "fit-verify",
  description:
    "Adversarial pre-publication verification of FLITS scattering fit results. Fans out one verifier agent per *_fit_results.json, each told to TRY TO REFUTE the fit's PASS claim against the runtime 3-level fit-quality gate (burstfit.py classify_fit_quality), then aggregates to confirmed-PASS / MARGINAL / FAIL with reasons. A separate agent does the judging so the fitter's own self-preferential bias cannot pass its own work.",
  phases: [
    "discover: glob the results tree for *_fit_results.json fit artifacts",
    "verify (parallel, adversarial): one verifier per fit, each attempts to REFUTE the PASS claim against the exact 3-level gate cut points",
    "aggregate: synthesize verdicts into confirmed-PASS / MARGINAL / FAIL with per-fit reasons and a publication-readiness summary",
  ],
};

// Results path or glob to verify. Edit this constant to retarget the run.
// Default matches the canonical per-burst fit artifact produced by the
// scattering pipeline anywhere under the repo.
const TARGET = "**/*_fit_results.json";

// The FLITS fit-quality contract, verbatim from the runtime classifier
// (scattering/scat_analysis/burstfit.py classify_fit_quality, ~L1342-1386;
// constants L67-70) plus the Level-1 physical gates and Level-3 physics
// consistency. This is the AUTHORITATIVE rubric — the doc/VT files disagree on
// several cuts and the runtime code intentionally overrides them. Every verifier
// is judged ONLY against these runtime cut points.
const GATE_CONTRACT = `
AUTHORITATIVE FLITS FIT-QUALITY CONTRACT (runtime classify_fit_quality in
scattering/scat_analysis/burstfit.py — this is what actually fires; the
AGENT_CONFIGURATION doc and VALIDATION_THRESHOLDS.py disagree and are SUPERSEDED).

LEVEL 1 — physical / convergence gates (ANY failure => FAIL, regardless of chi2):
  - fit must have CONVERGED (no optimizer failure / non-finite parameters).
  - tau (scattering timescale, ms): valid only if 0.0001 < tau < 100.
      FAIL if tau <= 0, tau < 0.0001 (below time resolution), or tau > 100 (never observed).
  - alpha (scattering index): Level-1 hard gate valid only if 1.5 < alpha < 6.0.
  - covariance / Jacobian must be non-singular (condition number < ~1e6).

LEVEL 2 — goodness of fit (drives the PASS/MARGINAL/FAIL flag):
  Reduced chi-square (chi2_red), constants CHI_SQ_RED_SUSPICIOUSLY_LOW=0.3,
  CHI_SQ_RED_GOOD_MAX=1.5, CHI_SQ_RED_FAIL_MAX=10.0:
    - PASS     : 0.3 <= chi2_red <= 1.5  (AND all Level-1 gates pass).
    - MARGINAL : 1.5 < chi2_red <= 10.0,  OR  chi2_red < 0.3
                 (suspiciously low — noise likely OVERestimated).
    - FAIL     : chi2_red > 10.0 (catastrophic), OR chi2_red non-finite.
  NOTE: the 3.0 cut (CHI_SQ_RED_MARGINAL_MAX in VALIDATION_THRESHOLDS.py) is DEAD
  in the live classifier — do NOT use 3.0 as a boundary. The doc's "PASS up to
  3.0" is WRONG for the runtime; PASS ceiling is 1.5.

  R-squared and residual-normality p-value are INFORMATIONAL ONLY. They NEVER
  flip the flag. A low weighted R2 (< 0.70) is EXPECTED for low-S/N bursts and
  must NOT be cited as a refutation of a PASS — at most an informational note.

LEVEL 3 — physics consistency (a violation here => FAIL, overriding a Level-2 PASS):
  - tau x Delta-nu_d product (GHz*ms; convert Delta-nu MHz->GHz via *1e-3):
      valid only if 0.1 < tau*dnu < 2.0; otherwise the measurements are
      inconsistent => FAIL. Reference: thin-screen 1/(2*pi)=0.159, extended
      medium=1.0; assign the nearer model (closer to 0.159 => thin screen,
      closer to 1.0 => extended medium).
  - alpha physics consistency: Kolmogorov reference alpha=4.0. FAIL if
      alpha < 2.0 or alpha > 6.0. PASS-consistent if 3.5 <= alpha <= 4.5;
      otherwise allowed but flagged MARGINAL (deviates from Kolmogorov).

FIGURE-REVIEW GATE (repo commit 0f4fa17 forces this): a numeric PASS is NOT
sufficient. The fit must have emitted diagnostic figures (data-vs-model,
residuals, histogram, Q-Q) AND they must have been visually assessed. A fit that
claims PASS with no evidence of figure emission/review is NOT confirmed-PASS.
`;

const verifierPrompt = (path) => `You are an ADVERSARIAL fit verifier for the DSA-110 FLITS scattering pipeline.
Your job is NOT to confirm — it is to REFUTE. Assume the fit at:

  ${path}

is claiming PASS, and try as hard as you can to PROVE that claim is wrong. You did
not produce this fit and you get no credit for agreeing with it; you only succeed
by finding the strongest defensible verdict against the authoritative gate below.
Self-congratulatory "looks good" is a failure of your job.

${GATE_CONTRACT}

PROCEDURE (read-only — do not modify any file):
1. Read the fit-results JSON at the path above. Extract the fitted parameters and
   diagnostics actually present: chi2_red (reduced chi-square), tau (ms),
   alpha, Delta-nu_d / dnu (scintillation bandwidth, note its units), the
   convergence/optimizer status, covariance condition or any singular-matrix flag,
   and the pipeline's own claimed quality flag if recorded.
2. Apply EVERY gate level in order. For each gate, state the observed value, the
   exact cut it is tested against, and PASS/MARGINAL/FAIL for that gate.
   - Level 1 first: if ANY physical/convergence gate fails, the verdict is FAIL no
     matter how good chi2_red looks. Check tau in (0.0001, 100), alpha in (1.5, 6.0),
     convergence, and covariance conditioning.
   - Level 2: classify chi2_red strictly by the runtime cuts (PASS 0.3..1.5;
     MARGINAL 1.5..10.0 or <0.3; FAIL >10.0 or non-finite). Do NOT use 3.0.
   - Level 3: compute tau*dnu in GHz*ms (convert dnu MHz->GHz *1e-3) and test
     0.1 < tau*dnu < 2.0; test alpha physics (FAIL <2.0 or >6.0; deviation from
     3.5..4.5 is MARGINAL). A Level-3 violation downgrades a Level-2 PASS to FAIL.
3. Treat R2 and residual-normality p ONLY as informational notes. If your only
   complaint is a low R2 on a low-S/N burst, that is NOT grounds to refute a PASS.
4. Check the figure-review gate: is there evidence the required diagnostic figures
   were emitted AND assessed (sibling figure files, a figures.review.json, or a
   review flag near the result)? If not, a numeric PASS cannot be confirmed-PASS.
5. If any required field is missing or non-finite, that is itself a refutation —
   you cannot confirm a PASS on absent evidence. Say so explicitly.

Then commit to the single most defensible verdict. Return the result JSON.`;

const verifierSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "path",
    "verdict",
    "chi2_red",
    "tau_ms",
    "alpha",
    "tau_dnu",
    "level1_pass",
    "level2",
    "level3_pass",
    "figures_assessed",
    "refutation_attempt",
    "reasons",
  ],
  properties: {
    path: { type: "string" },
    verdict: {
      type: "string",
      enum: ["confirmed-PASS", "MARGINAL", "FAIL"],
      description:
        "confirmed-PASS only if the refutation attempt FAILED on every gate AND figures were assessed.",
    },
    chi2_red: {
      type: ["number", "null"],
      description: "Observed reduced chi-square, or null if missing/non-finite.",
    },
    tau_ms: { type: ["number", "null"] },
    alpha: { type: ["number", "null"] },
    tau_dnu: {
      type: ["number", "null"],
      description: "tau*Delta-nu_d in GHz*ms (dnu converted MHz->GHz), or null.",
    },
    level1_pass: {
      type: "boolean",
      description: "True only if all physical/convergence gates pass.",
    },
    level2: {
      type: "string",
      enum: ["PASS", "MARGINAL", "FAIL"],
      description: "chi2_red classification by runtime cuts (PASS 0.3..1.5).",
    },
    level3_pass: {
      type: "boolean",
      description: "True only if tau*dnu in (0.1,2.0) and alpha in (2.0,6.0).",
    },
    figures_assessed: {
      type: "boolean",
      description: "Evidence the diagnostic figures were emitted AND assessed.",
    },
    refutation_attempt: {
      type: "string",
      description:
        "The strongest case you could build AGAINST a PASS, and whether it held.",
    },
    reasons: {
      type: "array",
      items: { type: "string" },
      description:
        "Per-gate findings: observed value, exact cut tested, gate verdict.",
    },
  },
};

const aggregateSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "total",
    "confirmed_pass",
    "marginal",
    "fail",
    "publication_ready",
    "per_fit",
    "summary",
  ],
  properties: {
    total: { type: "integer" },
    confirmed_pass: { type: "integer" },
    marginal: { type: "integer" },
    fail: { type: "integer" },
    publication_ready: {
      type: "boolean",
      description: "True only if every fit is confirmed-PASS (no MARGINAL, no FAIL).",
    },
    per_fit: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["path", "verdict", "headline_reason"],
        properties: {
          path: { type: "string" },
          verdict: {
            type: "string",
            enum: ["confirmed-PASS", "MARGINAL", "FAIL"],
          },
          headline_reason: { type: "string" },
        },
      },
    },
    summary: {
      type: "string",
      description:
        "Publication-readiness narrative: which fits block publication and why.",
    },
  },
};

export default async function ({ agent, parallel }) {
  // PHASE 1 — discover the fit artifacts.
  const discovery = await agent(
    `List every FLITS fit-result file under the repo matching the glob:

  ${TARGET}

These are JSON artifacts written by the scattering pipeline (typically named
*_fit_results.json). Search the repository working tree (read-only). Return the
absolute path of each matching file. If none match, return an empty array.`,
    {
      model: "haiku",
      schema: {
        type: "object",
        additionalProperties: false,
        required: ["paths"],
        properties: {
          paths: { type: "array", items: { type: "string" } },
        },
      },
    },
  );

  const paths = (discovery && discovery.paths) || [];
  if (paths.length === 0) {
    return {
      total: 0,
      confirmed_pass: 0,
      marginal: 0,
      fail: 0,
      publication_ready: false,
      per_fit: [],
      summary: `No fit-result files matched ${TARGET}. Nothing to verify.`,
    };
  }

  // PHASE 2 — adversarial verification. One verifier per fit, all in parallel.
  // The verifier is a DIFFERENT agent from whatever produced the fit, which is
  // what kills self-preferential bias (Workflow pattern 3: adversarial verification).
  const verdicts = await parallel(
    paths.map((path) => () =>
      agent(verifierPrompt(path), { schema: verifierSchema }),
    ),
  );

  // PHASE 3 — aggregate. A fresh agent synthesizes the publication verdict so the
  // roll-up reasoning is also independent of the per-fit verifiers.
  const aggregate = await agent(
    `Aggregate these adversarial FLITS fit verdicts into a single publication-readiness
report. Each verdict is the output of a verifier that TRIED to refute a PASS claim
against the authoritative runtime gate.

Rules for the roll-up:
- Count confirmed-PASS, MARGINAL, and FAIL.
- publication_ready is true ONLY if every fit is confirmed-PASS.
- For each fit give a one-line headline_reason (the binding gate that set its verdict).
- In the summary, call out exactly which fits block publication and the specific gate
  each failed (e.g. "fit X: chi2_red=14.2 > 10.0 FAIL", "fit Y: tau*dnu=0.04 < 0.1
  physics-inconsistent FAIL", "fit Z: figures not assessed, PASS unconfirmed").

Verdicts:
${JSON.stringify(verdicts, null, 2)}`,
    { schema: aggregateSchema },
  );

  return aggregate;
}
