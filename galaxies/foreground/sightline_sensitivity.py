"""Prior-predictive sensitivity runs for FRB sightline DM/scattering budgets."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from galaxies.foreground import sightline_budget as sb


@dataclass(frozen=True)
class PriorFamily:
    """Named literature-anchored nuisance-prior family for robustness checks."""

    name: str
    dm_mw_halo_mean: float
    dm_mw_halo_sigma: float
    dm_mw_halo_min: float
    dm_mw_halo_max: float
    f_igm_min: float
    f_igm_max: float
    measured_mass_sigma_dex: float
    assumed_logmstar_min: float
    assumed_logmstar_max: float
    f_hot_min: float
    f_hot_max: float
    cool_dm_factor_min: float
    cool_dm_factor_max: float
    cool_boost_min: float
    cool_boost_max: float
    cosmic_scatter_sigma: float
    placeholder_z_min: float
    placeholder_z_max: float


def default_prior_families() -> dict[str, PriorFamily]:
    """Return named prior families used by the sensitivity runner."""
    families = [
        PriorFamily(
            name="fiducial_literature",
            dm_mw_halo_mean=40.0,
            dm_mw_halo_sigma=15.0,
            dm_mw_halo_min=10.0,
            dm_mw_halo_max=100.0,
            f_igm_min=0.75,
            f_igm_max=0.90,
            measured_mass_sigma_dex=0.25,
            assumed_logmstar_min=9.0,
            assumed_logmstar_max=11.0,
            f_hot_min=0.40,
            f_hot_max=0.90,
            cool_dm_factor_min=0.0,
            cool_dm_factor_max=0.60,
            cool_boost_min=1.0,
            cool_boost_max=30.0,
            cosmic_scatter_sigma=0.20,
            placeholder_z_min=0.20,
            placeholder_z_max=1.50,
        ),
        PriorFamily(
            name="conservative_low_cgm",
            dm_mw_halo_mean=30.0,
            dm_mw_halo_sigma=10.0,
            dm_mw_halo_min=10.0,
            dm_mw_halo_max=80.0,
            f_igm_min=0.70,
            f_igm_max=0.84,
            measured_mass_sigma_dex=0.25,
            assumed_logmstar_min=8.8,
            assumed_logmstar_max=10.8,
            f_hot_min=0.20,
            f_hot_max=0.60,
            cool_dm_factor_min=0.0,
            cool_dm_factor_max=0.30,
            cool_boost_min=1.0,
            cool_boost_max=10.0,
            cosmic_scatter_sigma=0.25,
            placeholder_z_min=0.10,
            placeholder_z_max=1.20,
        ),
        PriorFamily(
            name="aggressive_cgm_scattering",
            dm_mw_halo_mean=60.0,
            dm_mw_halo_sigma=20.0,
            dm_mw_halo_min=20.0,
            dm_mw_halo_max=120.0,
            f_igm_min=0.84,
            f_igm_max=0.95,
            measured_mass_sigma_dex=0.35,
            assumed_logmstar_min=9.2,
            assumed_logmstar_max=11.5,
            f_hot_min=0.60,
            f_hot_max=1.00,
            cool_dm_factor_min=0.10,
            cool_dm_factor_max=0.80,
            cool_boost_min=3.0,
            cool_boost_max=50.0,
            cosmic_scatter_sigma=0.30,
            placeholder_z_min=0.30,
            placeholder_z_max=2.00,
        ),
    ]
    return {family.name: family for family in families}


def _truncated_normal(rng: np.random.Generator, mean: float, sigma: float, lo: float, hi: float, n: int) -> np.ndarray:
    values = rng.normal(mean, sigma, size=n)
    return np.clip(values, lo, hi)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))


def sample_nuisance_draws(family: PriorFamily, n: int, seed: int | None = None) -> pd.DataFrame:
    """Draw nuisance astrophysics parameters for one prior family."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(
        {
            "prior_family": family.name,
            "draw_id": np.arange(n, dtype=int),
            "dm_mw_halo": _truncated_normal(
                rng, family.dm_mw_halo_mean, family.dm_mw_halo_sigma,
                family.dm_mw_halo_min, family.dm_mw_halo_max, n,
            ),
            "f_igm": rng.uniform(family.f_igm_min, family.f_igm_max, size=n),
            "mass_shift_measured_dex": rng.normal(0.0, family.measured_mass_sigma_dex, size=n),
            "mass_shift_assumed_dex": rng.uniform(
                family.assumed_logmstar_min - 10.0,
                family.assumed_logmstar_max - 10.0,
                size=n,
            ),
            "f_hot": rng.uniform(family.f_hot_min, family.f_hot_max, size=n),
            "cool_dm_factor": rng.uniform(family.cool_dm_factor_min, family.cool_dm_factor_max, size=n),
            "cool_boost": _log_uniform(rng, family.cool_boost_min, family.cool_boost_max, n),
            "cosmic_scatter": rng.lognormal(mean=0.0, sigma=family.cosmic_scatter_sigma, size=n),
            "placeholder_z": rng.uniform(family.placeholder_z_min, family.placeholder_z_max, size=n),
        }
    )
    return out


def scaled_dm_cosmic(z: float, f_igm: float, cosmic_scatter: float) -> float:
    """Return a prior-drawn cosmic DM mean scaled from the deterministic Macquart helper."""
    if not math.isfinite(float(z)) or float(z) <= 0.0:
        return math.nan
    scale = float(f_igm) / sb.F_IGM
    return float(sb.dm_cosmic_macquart(float(z), f_igm=sb.F_IGM) * scale * float(cosmic_scatter))


def _finite(value: object) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def apply_draw_to_budget(budget: Mapping[str, object], draw: Mapping[str, object]) -> dict[str, object]:
    """Apply one nuisance draw to one deterministic budget row.

    This is a prior-predictive robustness calculation. It does not refit burst
    data or infer posterior distributions.
    """
    z_placeholder = bool(budget.get("z_is_placeholder", False))
    z_used = float(draw["placeholder_z"]) if z_placeholder else float(budget["z_frb"])
    z_status = "placeholder_hypothetical" if z_placeholder else "measured"

    dm_obs = float(budget["dm_obs"]) if _finite(budget.get("dm_obs")) else math.nan
    dm_mw_ism = float(budget["dm_mw_ism"]) if _finite(budget.get("dm_mw_ism")) else 0.0
    dm_mw_halo = float(draw["dm_mw_halo"])
    dm_cosmic = scaled_dm_cosmic(z_used, float(draw["f_igm"]), float(draw["cosmic_scatter"]))

    mass_conf = str(budget.get("intervening_mass_confidence", "none"))
    shift = (
        float(draw["mass_shift_assumed_dex"])
        if mass_conf == "assumed"
        else float(draw["mass_shift_measured_dex"])
    )
    mass_factor = 10.0 ** shift if mass_conf in {"assumed", "measured"} else 1.0
    hot_factor = float(draw["f_hot"]) / 0.75
    cool_factor = float(draw["cool_dm_factor"]) / 0.30 if float(draw["cool_dm_factor"]) > 0.0 else 0.0
    dm_interv_base = float(budget.get("dm_intervening_capped", 0.0) or 0.0)
    dm_interv_capped = max(0.0, dm_interv_base * mass_factor * hot_factor * (1.0 + cool_factor) / 2.0)

    tau_base = float(budget.get("tau_intervening_ms", 0.0) or 0.0)
    tau_interv = max(
        0.0,
        tau_base
        * mass_factor
        * hot_factor
        * (float(draw["cool_boost"]) / 10.0),
    )

    if math.isfinite(dm_obs):
        dm_host_capped = dm_obs - dm_mw_ism - dm_mw_halo - dm_cosmic - dm_interv_capped
    else:
        dm_host_capped = math.nan

    tau_obs = float(budget["tau_obs_ms"]) if _finite(budget.get("tau_obs_ms")) else math.nan
    tau_threshold = 0.1
    if math.isfinite(tau_obs) and tau_obs > 0.0:
        tau_fraction_threshold = 0.1 * tau_obs
    else:
        tau_fraction_threshold = math.nan

    prior_dominated = mass_conf == "assumed"
    return {
        "prior_family": draw["prior_family"],
        "draw_id": int(draw["draw_id"]),
        "name": budget["name"],
        "z_used": z_used,
        "z_status": z_status,
        "hypothetical_placeholder_z": z_placeholder,
        "dm_obs": dm_obs,
        "dm_mw_ism": dm_mw_ism,
        "dm_mw_halo": dm_mw_halo,
        "dm_cosmic": dm_cosmic,
        "dm_intervening_capped": dm_interv_capped,
        "dm_host_capped": dm_host_capped,
        "tau_intervening_ms": tau_interv,
        "tau_obs_ms": tau_obs,
        "tau_threshold_ms": tau_threshold,
        "tau_fraction_threshold_ms": tau_fraction_threshold,
        "host_negative": bool(math.isfinite(dm_host_capped) and dm_host_capped < 0.0),
        "interv_dm_gt_100": bool(dm_interv_capped > 100.0),
        "tau_gt_0p1ms": bool(tau_interv > tau_threshold),
        "tau_gt_obs_over_10": bool(math.isfinite(tau_fraction_threshold) and tau_interv > tau_fraction_threshold),
        "dm_intervening_regime": budget.get("dm_intervening_regime", "none"),
        "intervening_mass_confidence": mass_conf,
        "prior_dominated": prior_dominated,
    }


def _host_budget_label(p_host_negative: float) -> str:
    if p_host_negative >= 0.90:
        return "robust_negative_host"
    if p_host_negative >= 0.70:
        return "likely_negative_host"
    if p_host_negative > 0.30:
        return "ambiguous_host_budget"
    return "likely_positive_host"


def run_sensitivity(
    budgets: list[Mapping[str, object]],
    n_per_family: int = 1000,
    seed: int | None = 20260619,
    families: Mapping[str, PriorFamily] | None = None,
) -> pd.DataFrame:
    """Run prior-predictive sensitivity draws for deterministic budget rows."""
    if families is None:
        families = default_prior_families()
    rows: list[dict[str, object]] = []
    base_seed = 0 if seed is None else int(seed)
    for family_index, family in enumerate(families.values()):
        draws = sample_nuisance_draws(family, n=n_per_family, seed=base_seed + family_index)
        for _, draw in draws.iterrows():
            for budget in budgets:
                rows.append(apply_draw_to_budget(budget, draw))
    return pd.DataFrame(rows)


def summarize_sensitivity(draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize prior-predictive robustness probabilities by sightline."""
    if draws.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for name, group in draws.groupby("name", sort=False):
        p_host = float(group["host_negative"].mean())
        p_interv = float(group["interv_dm_gt_100"].mean())
        p_tau = float(group["tau_gt_0p1ms"].mean())
        p_tau_frac = float(group["tau_gt_obs_over_10"].mean())
        host_label = _host_budget_label(p_host)
        prior_dominated = bool(group["prior_dominated"].any())
        placeholder = bool(group["hypothetical_placeholder_z"].any())
        robust_flags = []
        if host_label in {"robust_negative_host", "likely_negative_host", "ambiguous_host_budget"}:
            robust_flags.append(host_label)
        if p_interv >= 0.80:
            robust_flags.append("robust_intervening_dm")
        if p_tau >= 0.80 or p_tau_frac >= 0.80:
            robust_flags.append("robust_tau_relevant")
        if placeholder:
            robust_flags.append("placeholder_z_hypothetical")
        if prior_dominated:
            robust_flags.append("prior_dominated")
        if not robust_flags:
            robust_flags.append("no_robust_flag")

        rows.append(
            {
                "name": name,
                "n_draws": int(len(group)),
                "p_host_negative": p_host,
                "p_interv_dm_gt_100": p_interv,
                "p_tau_gt_0p1ms": p_tau,
                "p_tau_gt_obs_over_10": p_tau_frac,
                "dm_host_cap_median": float(group["dm_host_capped"].median()),
                "dm_host_cap_p16": float(group["dm_host_capped"].quantile(0.16)),
                "dm_host_cap_p84": float(group["dm_host_capped"].quantile(0.84)),
                "dm_interv_cap_median": (
                    float(group["dm_intervening_capped"].median())
                    if "dm_intervening_capped" in group else math.nan
                ),
                "tau_interv_median_ms": (
                    float(group["tau_intervening_ms"].median())
                    if "tau_intervening_ms" in group else math.nan
                ),
                "host_budget_label": host_label,
                "placeholder_z_hypothetical": placeholder,
                "prior_dominated": prior_dominated,
                "robustness_label": ";".join(robust_flags),
                "dm_intervening_regime": str(group["dm_intervening_regime"].iloc[0]),
                "intervening_mass_confidence": str(group["intervening_mass_confidence"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def format_summary_markdown(summary: pd.DataFrame) -> str:
    """Render a compact prior-predictive robustness table."""
    lines = [
        "# Sightline Budget Prior-Predictive Sensitivity Summary",
        "",
        "These probabilities are prior-predictive robustness checks, not posterior evidence.",
        "",
        "| Sightline | P(host<0) | P(DM_interv>100) | P(tau>0.1ms) | Host label | Robustness flags |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {name} | {p_host_negative:.2f} | {p_interv_dm_gt_100:.2f} | "
            "{p_tau_gt_0p1ms:.2f} | {host_budget_label} | {robustness_label} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _priors_yaml_text(families: Mapping[str, PriorFamily] | None = None) -> str:
    if families is None:
        families = default_prior_families()
    lines = ["# Prior families for sightline budget prior-predictive sensitivity runs"]
    for name, family in families.items():
        lines.append(f"{name}:")
        for key, value in asdict(family).items():
            lines.append(f"  {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_sensitivity_artifacts(
    draws: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path = "results",
    families: Mapping[str, PriorFamily] | None = None,
) -> dict[str, Path]:
    """Write draw-level and summary sensitivity artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "draws_csv": out / "sightline_budget_sensitivity_draws.csv",
        "summary_csv": out / "sightline_budget_sensitivity_summary.csv",
        "summary_md": out / "sightline_budget_sensitivity_summary.md",
        "priors_yaml": out / "sightline_budget_sensitivity_priors.yaml",
    }
    draws.to_csv(paths["draws_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_md"].write_text(format_summary_markdown(summary))
    paths["priors_yaml"].write_text(_priors_yaml_text(families))
    return paths


def load_current_budgets(
    results_dir: str = "results",
    configs_dir: str | None = None,
    bursts_dir: str | None = None,
) -> list[dict[str, object]]:
    """Build current deterministic sightline budgets as dictionaries."""
    df = sb.build_all_budgets(results_dir=results_dir, configs_dir=configs_dir, bursts_dir=bursts_dir, enrich=False)
    return df.to_dict(orient="records")


def _baseline_draw(family: PriorFamily) -> dict[str, object]:
    return {
        "prior_family": family.name,
        "draw_id": 0,
        "dm_mw_halo": family.dm_mw_halo_mean,
        "f_igm": 0.5 * (family.f_igm_min + family.f_igm_max),
        "mass_shift_measured_dex": 0.0,
        "mass_shift_assumed_dex": 0.0,
        "f_hot": 0.5 * (family.f_hot_min + family.f_hot_max),
        "cool_dm_factor": 0.5 * (family.cool_dm_factor_min + family.cool_dm_factor_max),
        "cool_boost": math.sqrt(family.cool_boost_min * family.cool_boost_max),
        "cosmic_scatter": 1.0,
        "placeholder_z": 0.5 * (family.placeholder_z_min + family.placeholder_z_max),
    }


def one_parameter_sweep(
    budget: Mapping[str, object],
    parameter: str,
    values: list[float],
    family: PriorFamily | None = None,
) -> pd.DataFrame:
    """Vary one nuisance parameter around a fiducial draw for one sightline."""
    if family is None:
        family = default_prior_families()["fiducial_literature"]
    rows = []
    for value in values:
        draw = _baseline_draw(family)
        if parameter not in draw:
            raise KeyError(f"unknown sweep parameter: {parameter}")
        draw[parameter] = float(value)
        row = apply_draw_to_budget(budget, draw)
        row["parameter"] = parameter
        row["parameter_value"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def make_knob_plot(sweeps: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for name, group in sweeps.groupby("name", sort=False):
        ax.plot(group["parameter_value"], group["dm_host_capped"], marker="o", label=name)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel(str(sweeps["parameter"].iloc[0]))
    ax.set_ylabel("DM_host_cap (pc cm^-3)")
    ax.set_title("Prior-predictive one-parameter sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-family", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    budgets = load_current_budgets(results_dir=args.results_dir)
    families = default_prior_families()
    draws = run_sensitivity(budgets, n_per_family=args.n_per_family, seed=args.seed, families=families)
    summary = summarize_sensitivity(draws)
    paths = write_sensitivity_artifacts(draws, summary, output_dir=args.output_dir, families=families)
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")

    iconic = {"Zach", "Whitney", "Phineas"}
    sweeps = []
    for budget in budgets:
        if budget.get("name") in iconic:
            sweeps.append(one_parameter_sweep(budget, "dm_mw_halo", [20.0, 40.0, 60.0, 80.0]))
    if sweeps:
        sweep_df = pd.concat(sweeps, ignore_index=True)
        fig = make_knob_plot(sweep_df)
        knob_path = Path(args.output_dir) / "sightline_budget_sensitivity_knobs.png"
        fig.savefig(knob_path, dpi=200, bbox_inches="tight")
        print(f"Wrote knobs_png: {knob_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
