"""Maistro provenance sidecar for the FLITS DM/scattering budget.

Companion to ``maistro.py`` (galaxy-search provenance). Where that one records
the foreground-galaxy search, this records the downstream **per-sightline DM &
scattering budget** as queryable orchestrator state, so a later session/agent can
``render_context`` the current picture ("what's locked in / what's pending")
without re-deriving it.

Mapping to Maistro's trust model:
- **Canonical state** (``setState``): the per-sightline budget verdicts + the
  scattering attribution. A burst-measured tau enters canonical state only when
  the fit passed the quality gate (``tau_obs_quality == PASS``) — i.e. the
  promotion gate IS the scattering-fit quality gate.
- **Staging** (``stage``): sightlines whose scattering fit is present but
  *withheld* by the gate (FAIL/MARGINAL) are staged as ``refit_needed``
  candidates — they need a re-fit or human review before they can be trusted.

Pure ``build_*`` helpers + an injectable client keep the tests network-free.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Mapping, Sequence

from flits.orchestration.maistro import MaistroClient, git_metadata

STATE_RUN_ID = "flits-dsa110-scattering"
AGENT_ID = "flits:scattering-budget"
BUDGET_KIND = "flits.sightline.budget"
REFIT_KIND = "flits.scattering.refit_needed"

# Quality flags under which a measured tau is trusted into canonical state.
_TRUSTED_TAU_QUALITY = {"PASS", "INJECTED"}


def _num(value: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row_budget(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "z_frb", "l_deg", "b_deg",
        "dm_obs", "dm_mw_ism", "dm_mw_halo", "dm_cosmic",
        "dm_intervening", "dm_intervening_capped", "dm_intervening_regime",
        "dm_host", "dm_host_capped",
        "tau_obs_ms", "tau_obs_err_minus", "tau_obs_err_plus", "tau_obs_quality",
        "tau_mw_ms", "tau_intervening_ms", "tau_intervening_lo", "tau_intervening_hi",
        "n_foreground", "n_intersecting",
        "intervening_mass_source", "intervening_mass_confidence",
        "z_is_placeholder", "verdict_scattering", "verdict_dm",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in row:
            out[k] = _num(row[k])
    return out


def _tau_locked(row: Mapping[str, Any]) -> bool:
    q = row.get("tau_obs_quality")
    tau = _num(row.get("tau_obs_ms"))
    return q in _TRUSTED_TAU_QUALITY and tau is not None


def build_budget_payloads(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str = STATE_RUN_ID,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    """Build the write_batch (canonical state) + stage (refit candidates) payloads."""
    state_items: list[tuple[str, Any]] = [
        ("pipeline.kind", "flits-dm-scattering-budget"),
        ("pipeline.status", "passed"),
        ("pipeline.git_sha", git_sha),
        ("pipeline.git_dirty", git_dirty),
    ]

    n_locked = 0
    n_intervening_candidate = 0
    refit_stage: list[dict[str, Any]] = []

    for row in budget_rows:
        name = str(row.get("name"))
        slug = name.lower()
        state_items.append((f"budget.{slug}", _row_budget(row)))

        if _tau_locked(row):
            n_locked += 1
        verdict = str(row.get("verdict_scattering") or "")
        if "intervening galaxy" in verdict and ("dominates" in verdict or "contribute" in verdict):
            n_intervening_candidate += 1

        # A fit present but withheld by the gate -> stage for refit/review.
        q = row.get("tau_obs_quality")
        if q in ("FAIL", "MARGINAL", "UNKNOWN"):
            refit_stage.append(
                {
                    "run_id": run_id,
                    "kind": REFIT_KIND,
                    "payload": {
                        "sightline": name,
                        "tau_obs_quality": q,
                        "tau_obs_chi2_reduced": _num(row.get("tau_obs_chi2_reduced")),
                        "review_note": "scattering fit present but failed the quality gate; re-fit or review before trusting tau",
                    },
                    "derived_from": [],
                }
            )

    state_items.append(("summary.n_sightlines", len(budget_rows)))
    state_items.append(("summary.n_tau_locked_in", n_locked))
    state_items.append(("summary.n_intervening_scattering_candidates", n_intervening_candidate))

    write_batch = {
        "run_id": run_id,
        "items": [{"op": "setState", "key": key, "value": value} for key, value in state_items],
        "agent_id": AGENT_ID,
    }
    return {"run_id": run_id, "write_batch": write_batch, "stage": refit_stage}


def ingest_budget(
    budget_df,
    *,
    run_id: str = STATE_RUN_ID,
    client: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest a budget DataFrame into Maistro (or return payloads on dry_run)."""
    rows = [dict(r) for _, r in budget_df.iterrows()] if hasattr(budget_df, "iterrows") else list(budget_df)
    git_sha, git_dirty = git_metadata()
    payloads = build_budget_payloads(rows, run_id=run_id, git_sha=git_sha, git_dirty=git_dirty)

    if dry_run:
        return payloads

    rpc = client or MaistroClient()
    rpc.ensure_ready()
    rpc.write_batch(payloads["write_batch"])
    for stage in payloads["stage"]:
        rpc.stage(stage)
    return payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the FLITS DM/scattering budget into Maistro.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default=STATE_RUN_ID)
    parser.add_argument("--orch-rpc-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from galaxies.v2_0.sightline_budget import build_all_budgets

    budget_df = build_all_budgets(results_dir=args.results_dir, enrich=False)
    client = None if args.dry_run else MaistroClient(base_url=args.orch_rpc_url)
    payloads = ingest_budget(budget_df, run_id=args.run_id, client=client, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(payloads, indent=2, sort_keys=True, default=str))
    else:
        print(f"Ingested {len(payloads['write_batch']['items'])} state items, "
              f"{len(payloads['stage'])} refit candidates into run {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
