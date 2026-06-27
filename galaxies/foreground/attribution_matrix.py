"""Per-burst sightline attribution matrix (two-screen × foreground cross-check)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.foreground.census_registry import build_intervening_census_registry
from galaxies.foreground.tau_consistency import (
    ALPHA_CONSISTENCY,
    CHIME_REF_MHZ,
    DSA_REF_MHZ,
    build_tau_consistency_row,
    co_detected_nicknames,
    consistency_implied_c,
    consistency_status,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# Legacy ACF measurements (casey, freya, wilhelm). Others: not_attempted until scint campaign.
_LEGACY_DNU_MHZ: dict[str, dict[str, float]] = {
    "casey": {"chime": np.nan, "dsa": np.nan},  # measured but values not in gate csv — placeholder
    "freya": {"chime": 12.9, "dsa": 7.0},
    "wilhelm": {"chime": 9.95, "dsa": 2.72},
}


def _dnu_cells(burst: str) -> dict:
    burst = burst.lower()
    if burst not in _LEGACY_DNU_MHZ:
        return {
            "dnu_chime_mhz": "N/A — not_attempted",
            "dnu_dsa_mhz": "N/A — not_attempted",
            "dnu_status": "not_attempted",
        }
    vals = _LEGACY_DNU_MHZ[burst]
    chime = vals.get("chime")
    dsa = vals.get("dsa")
    if not np.isfinite(chime) and not np.isfinite(dsa):
        return {
            "dnu_chime_mhz": "N/A — not_attempted",
            "dnu_dsa_mhz": "N/A — not_attempted",
            "dnu_status": "not_attempted",
        }
    status = "measured"
    if np.isfinite(chime) and np.isfinite(dsa) and chime > dsa:
        status = "inverse_scaling"
    return {
        "dnu_chime_mhz": chime if np.isfinite(chime) else "N/A — not_attempted",
        "dnu_dsa_mhz": dsa if np.isfinite(dsa) else "N/A — not_attempted",
        "dnu_status": status,
    }


def _registry_foreground_summary(registry: pd.DataFrame, nickname: str) -> dict:
    sub = registry[registry.nickname == nickname]
    confirmed = sub[sub.final_verdict == "confirmed"]
    eligible = confirmed[confirmed.budget_eligible]
    dom = ""
    if len(eligible):
        dom_row = eligible.sort_values("impact_kpc", na_position="last").iloc[0]
        dom = f"{dom_row.type}:{dom_row.obj}"
    return {
        "n_confirmed": int(len(confirmed)),
        "n_budget_eligible": int(len(eligible)),
        "dominant_interv_obj": dom or "N/A — none eligible",
    }


def _multi_screen_triggers(
    tau_row: dict,
    dnu: dict,
) -> dict:
    triggers: list[str] = []
    tau_c = tau_row.get("tau_consistency_chime_ms")
    tau_d = tau_row.get("tau_consistency_dsa_ms")
    dnu_c = dnu.get("dnu_chime_mhz")
    dnu_d = dnu.get("dnu_dsa_mhz")

    c_chime = (
        consistency_status(float(tau_c), float(dnu_c))
        if isinstance(dnu_c, (int, float)) and np.isfinite(tau_c)
        else "N/A — no dnu_chime"
    )
    c_dsa = (
        consistency_status(float(tau_d), float(dnu_d))
        if isinstance(dnu_d, (int, float)) and np.isfinite(tau_d)
        else "N/A — no dnu_dsa"
    )

    if c_chime == "inconsistent":
        triggers.append("consistency_chime")
    if c_dsa == "inconsistent":
        triggers.append("consistency_dsa")
    if dnu.get("dnu_status") == "inverse_scaling":
        triggers.append("inverse_dnu_scaling")
    if tau_row.get("refit_status", "").startswith("pending"):
        triggers.append("tau_consistency_pending")

    return {
        "consistency_chime": c_chime,
        "consistency_dsa": c_dsa,
        "C_implied_chime": (
            consistency_implied_c(float(tau_c), float(dnu_c))
            if isinstance(dnu_c, (int, float)) and np.isfinite(tau_c)
            else np.nan
        ),
        "C_implied_dsa": (
            consistency_implied_c(float(tau_d), float(dnu_d))
            if isinstance(dnu_d, (int, float)) and np.isfinite(tau_d)
            else np.nan
        ),
        "multi_screen_triggers": "|".join(triggers) if triggers else "",
        "two_screen_coherence": "N/A — multi-scale dnu not wired",
        "constraint_level": _constraint_level(triggers, dnu),
    }


def _constraint_level(triggers: list[str], dnu: dict) -> str:
    if "tau_consistency_pending" in triggers:
        return "SILVER — tau_consistency pending"
    if dnu.get("dnu_status") == "not_attempted":
        return "SILVER — dnu not_attempted"
    if triggers:
        return "GOLD — multi-screen trigger"
    if dnu.get("dnu_status") == "measured":
        return "GOLD — tau+dnu"
    return "SILVER"


def build_attribution_matrix(
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    registry = registry if registry is not None else build_intervening_census_registry()
    rows: list[dict] = []
    for burst in co_detected_nicknames():
        tau_row = build_tau_consistency_row(burst)
        dnu = _dnu_cells(burst)
        fg = _registry_foreground_summary(registry, burst)
        screen = _multi_screen_triggers(tau_row, dnu)
        rows.append(
            {
                "nickname": burst,
                "tau_joint_1ghz_ms": tau_row.get("tau_joint_1ghz_ms"),
                "alpha_joint_free": tau_row.get("alpha_joint_free"),
                "joint_gate_final": tau_row.get("joint_gate_final"),
                "tau_consistency_1ghz_ms": tau_row.get("tau_consistency_1ghz_ms"),
                "tau_consistency_chime_ms": tau_row.get("tau_consistency_chime_ms"),
                "tau_consistency_dsa_ms": tau_row.get("tau_consistency_dsa_ms"),
                "alpha_consistency_fixed": ALPHA_CONSISTENCY,
                "refit_status": tau_row.get("refit_status"),
                "pbf_alpha_tension": tau_row.get("pbf_alpha_tension"),
                "chime_ref_mhz": CHIME_REF_MHZ,
                "dsa_ref_mhz": DSA_REF_MHZ,
                **dnu,
                **fg,
                **screen,
                "scattering_screen": "undetermined",
                "dm_budget_verdict": "N/A — sightline_budget not wired to registry",
                "cross_check_notes": "",
            }
        )
    return pd.DataFrame(rows)


def write_attribution_matrix(path: Path | str | None = None) -> Path:
    out = Path(path) if path is not None else DATA_DIR / "sightline_attribution_matrix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_attribution_matrix().to_csv(out, index=False)
    return out
