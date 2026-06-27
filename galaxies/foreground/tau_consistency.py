"""α-fixed (α=4) joint τ for scintillation consistency pairing (dual-τ policy)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
REPO_ROOT = PACKAGE_DIR.parents[1]

ALPHA_CONSISTENCY = 4.0
TAU_REF_MHZ = 1000.0
CHIME_REF_MHZ = 600.0
DSA_REF_MHZ = 1400.0

JOINT_GATE_CSV = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "joint_gate_verdicts.csv"
REFIT_DIR = REPO_ROOT / "analysis" / "scattering-refit-2026-06"
ALLEXP_FITS_DIR = REFIT_DIR / "_a1_fits"
CITABLE_ROSTER_JSON = REFIT_DIR / "citable_alpha_roster.json"
TAU_CONSISTENCY_DIR = DATA_DIR / "tau_consistency"


def co_detected_nicknames() -> list[str]:
    from galaxies.foreground.config import TARGETS

    return [name.lower() for name, *_ in TARGETS]


def scale_tau_1ghz_ms(
    tau_1ghz_ms: float,
    nu_mhz: float,
    alpha: float = ALPHA_CONSISTENCY,
    nu_ref_mhz: float = TAU_REF_MHZ,
) -> float:
    if not np.isfinite(tau_1ghz_ms) or not np.isfinite(nu_mhz) or tau_1ghz_ms <= 0:
        return np.nan
    return float(tau_1ghz_ms * (nu_mhz / nu_ref_mhz) ** (-alpha))


def _normalize_burst(name: str) -> str:
    return str(name).lower().replace("johndoeii", "johndoeii")


def _posterior_median(block: object, nested_key: str = "median") -> float:
    """Median from dynesty-style dict, scalar PPC export, or missing."""
    if block is None:
        return np.nan
    if isinstance(block, dict):
        val = block.get(nested_key, np.nan)
        try:
            out = float(val)
        except (TypeError, ValueError):
            return np.nan
        return out if np.isfinite(out) else np.nan
    try:
        out = float(block)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _joint_fit_scalar(payload: dict, param: str) -> float:
    """Read τ or α median from joint-fit JSON (posterior dict or scalar PPC row)."""
    percentiles = payload.get("percentiles") or {}
    block = payload.get(param)
    if block is None and isinstance(percentiles, dict):
        block = percentiles.get(param)
    return _posterior_median(block)


def load_joint_gate_table(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or JOINT_GATE_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=["burst"])
    df = pd.read_csv(csv_path)
    df["burst"] = df["burst"].map(_normalize_burst)
    return df


def find_allexp_joint_json(burst: str, search_dir: Path | None = None) -> Path | None:
    burst = _normalize_burst(burst)
    root = search_dir or ALLEXP_FITS_DIR
    if not root.is_dir():
        return None
    matches = [
        p
        for p in sorted(root.glob("*_joint*pbf-exp-exp.json"))
        if p.name.lower().startswith(f"{burst}_joint") and "ppc" not in p.name.lower()
    ]
    if not matches:
        matches = [
            p
            for p in sorted(root.glob("*joint*pbf-exp-exp.json"))
            if burst in p.name.lower() and "ppc" not in p.name.lower()
        ]
    return matches[0] if matches else None


def load_citable_budget_nicknames() -> frozenset[str]:
    """Nicknames eligible for all-exp τ on fig:budget (ADR-0005 Tier A/B + whitney)."""
    if not CITABLE_ROSTER_JSON.exists():
        return frozenset()
    with open(CITABLE_ROSTER_JSON) as fh:
        roster = json.load(fh)
    names = {str(e["nickname"]).lower() for e in roster.get("tier_a_fully_adjudicated", [])}
    names |= {str(e["nickname"]).lower() for e in roster.get("tier_b_provisional_pending_s2", [])}
    exemplar = roster.get("multiplicity_exemplar") or {}
    if exemplar.get("nickname"):
        names.add(str(exemplar["nickname"]).lower())
    return frozenset(names)


def find_citable_joint_json(burst: str) -> Path | None:
    """Canonical all-exp joint fit JSON for a citable-roster burst."""
    burst = _normalize_burst(burst)
    if CITABLE_ROSTER_JSON.exists():
        with open(CITABLE_ROSTER_JSON) as fh:
            roster = json.load(fh)
        exemplar = roster.get("multiplicity_exemplar") or {}
        if str(exemplar.get("nickname", "")).lower() == burst:
            override = exemplar.get("fit_json")
            if override:
                path = REPO_ROOT / str(override)
                if path.exists():
                    return path
    return find_allexp_joint_json(burst)


def _find_joint_ppc_json(fit_path: Path) -> Path | None:
    if "_joint_fit" not in fit_path.name:
        return None
    burst, tag = fit_path.name.split("_joint_fit", 1)
    ppc_path = fit_path.parent / f"{burst}_joint_ppc_multi{tag}"
    if ppc_path.exists():
        return ppc_path
    prefix = burst.lower()
    for candidate in sorted(fit_path.parent.glob("*_joint_ppc_multi*.json")):
        if candidate.name.lower().startswith(f"{prefix}_joint_ppc_multi"):
            return candidate
    return None


def _joint_tau_posterior(fit: dict) -> tuple[float, float, float]:
    pct = fit.get("percentiles") or {}
    block = fit.get("tau_1ghz")
    if not isinstance(block, dict):
        block = pct.get("tau_1ghz")
    tau = _posterior_median(block)
    err_minus = _posterior_median(block, "err_minus") if isinstance(block, dict) else np.nan
    err_plus = _posterior_median(block, "err_plus") if isinstance(block, dict) else np.nan
    return float(tau), float(err_minus), float(err_plus)


def _import_gate_one():
    import sys

    refit_str = str(REFIT_DIR)
    if refit_str not in sys.path:
        sys.path.insert(0, refit_str)
    from gate_joint_committed import gate_one

    return gate_one


def load_allexp_joint_tau_for_budget(burst: str) -> dict | None:
    """All-exp joint τ + ADR-0004 gate for fig:budget overlay (citable roster only)."""
    burst = _normalize_burst(burst)
    if burst not in load_citable_budget_nicknames():
        return None
    fit_path = find_citable_joint_json(burst)
    if fit_path is None:
        return None
    with open(fit_path) as fh:
        fit = json.load(fh)
    ppc_path = _find_joint_ppc_json(fit_path)
    ppc = json.loads(ppc_path.read_text()) if ppc_path is not None else None
    verdict = _import_gate_one()(burst, fit, ppc)
    tau, err_minus, err_plus = _joint_tau_posterior(fit)
    if not np.isfinite(tau) or tau <= 0.0:
        return None
    cc, cd = verdict.get("chi2_chime"), verdict.get("chi2_dsa")
    chi2_red = (
        max(float(cc), float(cd))
        if cc is not None and cd is not None and np.isfinite(cc) and np.isfinite(cd)
        else np.nan
    )
    return {
        "tau": tau,
        "err_minus": err_minus,
        "err_plus": err_plus,
        "quality_flag": verdict["final"],
        "chi2_reduced": chi2_red,
        "source": "allexp_joint",
    }


def load_joint_free_alpha(burst: str) -> dict:
    """Free-α all-exp joint fit posteriors (provisional-citable α roster track)."""
    burst = _normalize_burst(burst)
    gate = load_joint_gate_table()
    row = gate[gate.burst == burst]
    if len(row):
        r = row.iloc[0]
        return {
            "tau_joint_1ghz_ms": float(r.tau) if pd.notna(r.tau) else np.nan,
            "alpha_joint_free": float(r.alpha) if pd.notna(r.alpha) else np.nan,
            "joint_gate_final": str(r.final),
            "joint_gate_source": str(JOINT_GATE_CSV),
        }
    path = find_allexp_joint_json(burst)
    if path is None:
        return {
            "tau_joint_1ghz_ms": np.nan,
            "alpha_joint_free": np.nan,
            "joint_gate_final": "N/A — no joint fit",
            "joint_gate_source": "",
        }
    with open(path) as fh:
        payload = json.load(fh)
    return {
        "tau_joint_1ghz_ms": _joint_fit_scalar(payload, "tau_1ghz"),
        "alpha_joint_free": _joint_fit_scalar(payload, "alpha"),
        "joint_gate_final": "from_json",
        "joint_gate_source": str(path),
    }


def load_alpha4_refit(burst: str) -> dict | None:
    burst = _normalize_burst(burst)
    path = TAU_CONSISTENCY_DIR / f"{burst}_joint_alpha4_pbf-exp-exp.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def tau_consistency_from_refit(payload: dict) -> dict:
    tau_1 = _joint_fit_scalar(payload, "tau_1ghz")
    return {
        "tau_consistency_1ghz_ms": tau_1,
        "tau_consistency_chime_ms": scale_tau_1ghz_ms(tau_1, CHIME_REF_MHZ),
        "tau_consistency_dsa_ms": scale_tau_1ghz_ms(tau_1, DSA_REF_MHZ),
        "alpha_consistency_fixed": ALPHA_CONSISTENCY,
        "refit_status": "alpha4_joint_complete",
        "refit_source": payload.get("_source_path", ""),
    }


def build_tau_consistency_row(burst: str) -> dict:
    burst = _normalize_burst(burst)
    free = load_joint_free_alpha(burst)
    alpha4 = load_alpha4_refit(burst)
    row = {"nickname": burst, **free}
    if alpha4 is not None:
        alpha4["_source_path"] = str(TAU_CONSISTENCY_DIR / f"{burst}_joint_alpha4_pbf-exp-exp.json")
        row.update(tau_consistency_from_refit(alpha4))
    else:
        row.update(
            {
                "tau_consistency_1ghz_ms": np.nan,
                "tau_consistency_chime_ms": np.nan,
                "tau_consistency_dsa_ms": np.nan,
                "alpha_consistency_fixed": ALPHA_CONSISTENCY,
                "refit_status": "pending — run build_tau_consistency_refits",
                "refit_source": "",
            }
        )
    tau_j = row.get("tau_joint_1ghz_ms")
    alpha_j = row.get("alpha_joint_free")
    tau_c = row.get("tau_consistency_1ghz_ms")
    if np.isfinite(tau_j) and np.isfinite(tau_c) and tau_j > 0:
        row["pbf_alpha_tension"] = abs(alpha_j - ALPHA_CONSISTENCY) > 0.5 or abs(tau_c - tau_j) / tau_j > 0.2
    else:
        row["pbf_alpha_tension"] = False
    return row


def build_tau_consistency_catalog() -> pd.DataFrame:
    return pd.DataFrame([build_tau_consistency_row(b) for b in co_detected_nicknames()])


def write_tau_consistency_catalog(path: Path | str | None = None) -> Path:
    out = Path(path) if path is not None else DATA_DIR / "tau_consistency_catalog.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_tau_consistency_catalog().to_csv(out, index=False)
    return out


def consistency_implied_c(tau_ms: float, dnu_mhz: float) -> float:
    if not np.isfinite(tau_ms) or not np.isfinite(dnu_mhz) or tau_ms <= 0 or dnu_mhz <= 0:
        return np.nan
    return float(2 * np.pi * (tau_ms * 1e-3) * (dnu_mhz * 1e6))


def consistency_status(tau_ms: float, dnu_mhz: float) -> str:
    c = consistency_implied_c(tau_ms, dnu_mhz)
    if not np.isfinite(c):
        return "N/A — missing tau or dnu"
    c_lo = 2 * np.pi * 0.1
    c_hi = 2 * np.pi * 2.0
    if c_lo < c < c_hi:
        return "consistent"
    return "inconsistent"
