# Sightline Budget Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a literature-anchored prior-predictive sensitivity runner for the CHIME/DSA FRB sightline DM and scattering budget.

**Architecture:** Keep the deterministic budget in `galaxies/foreground/sightline_budget.py` unchanged and add a focused sensitivity layer in `galaxies/foreground/sightline_sensitivity.py`. The sensitivity layer samples nuisance astrophysics, recomputes derived DM/scattering quantities from the existing per-sightline budget and foreground catalogs, writes draw-level and summary artifacts, and labels results as prior-predictive robustness checks rather than posterior inference.

**Tech Stack:** Python 3.12, NumPy, pandas, scipy/astropy via existing modules, pytest, matplotlib Agg, project package `galaxies.foreground`.

---

## File Structure

- Create `galaxies/foreground/sightline_sensitivity.py`
  - Owns prior family definitions, correlated draw generation, deterministic recomputation for one budget record, summary metrics, markdown formatting, plots, and CLI entrypoint.
  - Does not query network catalogs or run scattering fits.
  - Calls existing helpers from `sightline_budget.py`, `build_unified.py`, and `scattering_predict.py`.
- Create `galaxies/foreground/test_sightline_sensitivity.py`
  - Network-free unit tests with small temporary CSV fixtures and injected deterministic budgets.
- Modify `galaxies/foreground/sightline_budget.py`
  - Only if needed, extract one pure helper for recomputing host residuals from component values.
  - Do not alter current deterministic budget outputs unless a test shows a bug.
- Modify `pyproject.toml`
  - Optional: add a console script only if the CLI is useful enough as `flits-sightline-sensitivity`.
  - Otherwise use `python -m galaxies.foreground.sightline_sensitivity`.
- Generated outputs from the final runner:
  - `results/sightline_budget_sensitivity_draws.csv`
  - `results/sightline_budget_sensitivity_summary.csv`
  - `results/sightline_budget_sensitivity_summary.md`
  - `results/sightline_budget_sensitivity_knobs.png`
  - `results/sightline_budget_sensitivity_priors.yaml`

## Scientific Contract

The new runner must use the phrase `prior-predictive` in summary artifacts and must not call the draw summaries posteriors.

Predefined claim labels:

```text
robust_negative_host       P(DM_host_cap < 0) >= 0.90
likely_negative_host       0.70 <= P(DM_host_cap < 0) < 0.90
ambiguous_host_budget      0.30 < P(DM_host_cap < 0) < 0.70
likely_positive_host       P(DM_host_cap < 0) <= 0.30
robust_intervening_dm      P(DM_interv_cap > 100 pc cm^-3) >= 0.80
robust_tau_relevant        P(tau_interv > tau_threshold_ms) >= 0.80
placeholder_z_hypothetical z_is_placeholder == true
prior_dominated            dominant screen has assumed mass or label changes across prior families
```

Default thresholds:

```text
dm_interv_threshold = 100.0 pc cm^-3
tau_threshold_ms = 0.1 ms
tau_fraction_threshold = 0.1 * tau_obs only when tau_obs is finite and quality accepted
```

Prior families:

```text
fiducial_literature
conservative_low_cgm
aggressive_cgm_scattering
```

The first version should support reproducible seeded draws and one-parameter knob sweeps, but it does not need a full hierarchical likelihood or cosmology inference.

### Task 1: Prior Family Data Model

**Files:**
- Create: `galaxies/foreground/sightline_sensitivity.py`
- Test: `galaxies/foreground/test_sightline_sensitivity.py`

- [ ] **Step 1: Write failing tests for prior families**

Add this test file with the imports and first tests:

```python
import math

import numpy as np
import pytest

from galaxies.foreground import sightline_sensitivity as ss


def test_default_prior_families_have_expected_names_and_bounds():
    families = ss.default_prior_families()
    assert set(families) == {
        "fiducial_literature",
        "conservative_low_cgm",
        "aggressive_cgm_scattering",
    }
    for family in families.values():
        assert family.dm_mw_halo_mean > 0.0
        assert family.dm_mw_halo_sigma > 0.0
        assert 0.0 < family.f_igm_min < family.f_igm_max < 1.0
        assert family.measured_mass_sigma_dex > 0.0
        assert family.assumed_logmstar_min < family.assumed_logmstar_max
        assert family.cool_boost_min > 0.0
        assert family.cool_boost_min < family.cool_boost_max


def test_sample_nuisance_draws_are_reproducible_and_finite():
    family = ss.default_prior_families()["fiducial_literature"]
    a = ss.sample_nuisance_draws(family, n=5, seed=123)
    b = ss.sample_nuisance_draws(family, n=5, seed=123)
    assert a.equals(b)
    assert len(a) == 5
    for col in (
        "dm_mw_halo",
        "f_igm",
        "mass_shift_measured_dex",
        "mass_shift_assumed_dex",
        "f_hot",
        "cool_dm_factor",
        "cool_boost",
        "cosmic_scatter",
        "placeholder_z",
    ):
        assert col in a.columns
        assert np.isfinite(a[col]).all()
    assert ((a["f_igm"] > 0.0) & (a["f_igm"] < 1.0)).all()
    assert (a["cool_boost"] > 0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `sightline_sensitivity.py` and its functions do not exist.

- [ ] **Step 3: Implement prior dataclasses and sampler**

Create `galaxies/foreground/sightline_sensitivity.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
```

Expected: PASS for the two tests.

- [ ] **Step 5: Commit**

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py
git commit -m "feat(galaxies): add sightline sensitivity prior families"
```

### Task 2: Draw-Level Budget Recompute

**Files:**
- Modify: `galaxies/foreground/sightline_sensitivity.py`
- Modify: `galaxies/foreground/test_sightline_sensitivity.py`

- [ ] **Step 1: Add failing tests for recompute semantics**

Append:

```python
def test_apply_draw_to_budget_with_measured_redshift_keeps_real_z():
    budget = {
        "name": "Aaa",
        "z_frb": 0.30,
        "z_is_placeholder": False,
        "dm_obs": 400.0,
        "dm_mw_ism": 80.0,
        "dm_intervening_capped": 50.0,
        "dm_intervening": 60.0,
        "tau_intervening_ms": 0.02,
        "tau_intervening_hi": 0.06,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": None,
        "dm_intervening_regime": "CGM",
        "intervening_mass_confidence": "measured",
    }
    draw = ss.sample_nuisance_draws(ss.default_prior_families()["fiducial_literature"], n=1, seed=1).iloc[0]
    row = ss.apply_draw_to_budget(budget, draw)
    assert row["name"] == "Aaa"
    assert row["z_used"] == pytest.approx(0.30)
    assert row["z_status"] == "measured"
    expected_cosmic = ss.scaled_dm_cosmic(0.30, draw["f_igm"], draw["cosmic_scatter"])
    assert row["dm_cosmic"] == pytest.approx(expected_cosmic)
    expected_host = 400.0 - 80.0 - draw["dm_mw_halo"] - expected_cosmic - row["dm_intervening_capped"]
    assert row["dm_host_capped"] == pytest.approx(expected_host)


def test_apply_draw_to_budget_with_placeholder_redshift_is_hypothetical():
    budget = {
        "name": "Freya",
        "z_frb": 1.0,
        "z_is_placeholder": True,
        "dm_obs": 912.0,
        "dm_mw_ism": 68.0,
        "dm_intervening_capped": 4.0,
        "dm_intervening": 5.0,
        "tau_intervening_ms": 0.001,
        "tau_intervening_hi": 0.003,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": "FAIL",
        "dm_intervening_regime": "CGM",
        "intervening_mass_confidence": "assumed",
    }
    draw = ss.sample_nuisance_draws(ss.default_prior_families()["fiducial_literature"], n=1, seed=2).iloc[0]
    row = ss.apply_draw_to_budget(budget, draw)
    assert row["z_status"] == "placeholder_hypothetical"
    assert row["z_used"] == pytest.approx(draw["placeholder_z"])
    assert math.isfinite(row["dm_cosmic"])
    assert row["hypothetical_placeholder_z"] is True
    assert row["prior_dominated"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py::test_apply_draw_to_budget_with_measured_redshift_keeps_real_z galaxies/foreground/test_sightline_sensitivity.py::test_apply_draw_to_budget_with_placeholder_redshift_is_hypothetical -q
```

Expected: FAIL with missing `apply_draw_to_budget` and `scaled_dm_cosmic`.

- [ ] **Step 3: Implement scaled cosmic DM and draw application**

Add to `sightline_sensitivity.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py
git commit -m "feat(galaxies): apply sensitivity draws to sightline budgets"
```

### Task 3: Multi-Family Runner And Summary Labels

**Files:**
- Modify: `galaxies/foreground/sightline_sensitivity.py`
- Modify: `galaxies/foreground/test_sightline_sensitivity.py`

- [ ] **Step 1: Add failing summary tests**

Append:

```python
def test_run_sensitivity_returns_draws_for_each_family():
    budgets = [
        {
            "name": "Aaa",
            "z_frb": 0.30,
            "z_is_placeholder": False,
            "dm_obs": 400.0,
            "dm_mw_ism": 80.0,
            "dm_intervening_capped": 50.0,
            "dm_intervening": 60.0,
            "tau_intervening_ms": 0.02,
            "tau_intervening_hi": 0.06,
            "tau_obs_ms": math.nan,
            "tau_obs_quality": None,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ]
    draws = ss.run_sensitivity(budgets, n_per_family=4, seed=99)
    assert len(draws) == 12
    assert set(draws["prior_family"]) == set(ss.default_prior_families())
    assert set(draws["name"]) == {"Aaa"}


def test_summarize_sensitivity_labels_placeholder_and_prior_dominated():
    rows = []
    for i in range(10):
        rows.append({
            "name": "Freya",
            "prior_family": "fiducial_literature",
            "dm_host_capped": -10.0 if i < 9 else 20.0,
            "host_negative": i < 9,
            "interv_dm_gt_100": False,
            "tau_gt_0p1ms": False,
            "tau_gt_obs_over_10": False,
            "hypothetical_placeholder_z": True,
            "prior_dominated": True,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "assumed",
        })
    summary = ss.summarize_sensitivity(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["name"] == "Freya"
    assert row["p_host_negative"] == pytest.approx(0.9)
    assert row["host_budget_label"] == "robust_negative_host"
    assert row["placeholder_z_hypothetical"] is True
    assert row["prior_dominated"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py::test_run_sensitivity_returns_draws_for_each_family galaxies/foreground/test_sightline_sensitivity.py::test_summarize_sensitivity_labels_placeholder_and_prior_dominated -q
```

Expected: FAIL with missing `run_sensitivity` and `summarize_sensitivity`.

- [ ] **Step 3: Implement runner and summary**

Add:

```python
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
                "dm_interv_cap_median": float(group["dm_intervening_capped"].median()),
                "tau_interv_median_ms": float(group["tau_intervening_ms"].median()),
                "host_budget_label": host_label,
                "placeholder_z_hypothetical": placeholder,
                "prior_dominated": prior_dominated,
                "robustness_label": ";".join(robust_flags),
                "dm_intervening_regime": str(group["dm_intervening_regime"].iloc[0]),
                "intervening_mass_confidence": str(group["intervening_mass_confidence"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py
git commit -m "feat(galaxies): summarize sightline sensitivity robustness"
```

### Task 4: Real Budget Integration And Artifacts

**Files:**
- Modify: `galaxies/foreground/sightline_sensitivity.py`
- Modify: `galaxies/foreground/test_sightline_sensitivity.py`
- Optional modify: `pyproject.toml`

- [ ] **Step 1: Add failing artifact tests**

Append:

```python
def test_format_summary_markdown_declares_prior_predictive():
    summary = pd.DataFrame([
        {
            "name": "Aaa",
            "n_draws": 30,
            "p_host_negative": 0.91,
            "p_interv_dm_gt_100": 0.2,
            "p_tau_gt_0p1ms": 0.0,
            "p_tau_gt_obs_over_10": 0.0,
            "dm_host_cap_median": -12.0,
            "dm_host_cap_p16": -30.0,
            "dm_host_cap_p84": 5.0,
            "dm_interv_cap_median": 70.0,
            "tau_interv_median_ms": 0.02,
            "host_budget_label": "robust_negative_host",
            "placeholder_z_hypothetical": False,
            "prior_dominated": False,
            "robustness_label": "robust_negative_host",
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ])
    md = ss.format_summary_markdown(summary)
    assert "prior-predictive" in md.lower()
    assert "Aaa" in md
    assert "robust_negative_host" in md


def test_write_sensitivity_artifacts_creates_expected_files(tmp_path):
    draws = pd.DataFrame([
        {
            "name": "Aaa",
            "prior_family": "fiducial_literature",
            "draw_id": 0,
            "dm_host_capped": -1.0,
            "host_negative": True,
            "interv_dm_gt_100": False,
            "tau_gt_0p1ms": False,
            "tau_gt_obs_over_10": False,
            "dm_intervening_capped": 10.0,
            "tau_intervening_ms": 0.001,
            "hypothetical_placeholder_z": False,
            "prior_dominated": False,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ])
    summary = ss.summarize_sensitivity(draws)
    paths = ss.write_sensitivity_artifacts(draws, summary, output_dir=tmp_path)
    for key in ("draws_csv", "summary_csv", "summary_md", "priors_yaml"):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py::test_format_summary_markdown_declares_prior_predictive galaxies/foreground/test_sightline_sensitivity.py::test_write_sensitivity_artifacts_creates_expected_files -q
```

Expected: FAIL with missing formatting/writer functions.

- [ ] **Step 3: Implement artifact formatting and writer**

Add:

```python
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
```

- [ ] **Step 4: Add real-budget loader and CLI**

Add:

```python
def load_current_budgets(
    results_dir: str = "results",
    configs_dir: str | None = None,
    bursts_dir: str | None = None,
) -> list[dict[str, object]]:
    """Build current deterministic sightline budgets as dictionaries."""
    df = sb.build_all_budgets(results_dir=results_dir, configs_dir=configs_dir, bursts_dir=bursts_dir, enrich=False)
    return df.to_dict(orient="records")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-run the module with small draws**

Run:

```bash
conda run -n flits python -m galaxies.foreground.sightline_sensitivity --n-per-family 5 --seed 7 --output-dir /tmp/flits-sensitivity-smoke
```

Expected output includes:

```text
Wrote draws_csv: /tmp/flits-sensitivity-smoke/sightline_budget_sensitivity_draws.csv
Wrote summary_csv: /tmp/flits-sensitivity-smoke/sightline_budget_sensitivity_summary.csv
Wrote summary_md: /tmp/flits-sensitivity-smoke/sightline_budget_sensitivity_summary.md
Wrote priors_yaml: /tmp/flits-sensitivity-smoke/sightline_budget_sensitivity_priors.yaml
```

- [ ] **Step 7: Commit**

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py
git commit -m "feat(galaxies): write sightline sensitivity artifacts"
```

### Task 5: Knob Sweeps And Plot

**Files:**
- Modify: `galaxies/foreground/sightline_sensitivity.py`
- Modify: `galaxies/foreground/test_sightline_sensitivity.py`

- [ ] **Step 1: Add failing tests for knob sweeps**

Append:

```python
def test_knob_sweep_varies_one_parameter_for_iconic_sightline():
    budget = {
        "name": "Whitney",
        "z_frb": 0.479,
        "z_is_placeholder": False,
        "dm_obs": 462.0,
        "dm_mw_ism": 46.0,
        "dm_intervening_capped": 200.0,
        "dm_intervening": 364.0,
        "tau_intervening_ms": 0.27,
        "tau_intervening_hi": 0.96,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": None,
        "dm_intervening_regime": "GALAXY_INTERIOR",
        "intervening_mass_confidence": "assumed",
    }
    sweep = ss.one_parameter_sweep(budget, parameter="dm_mw_halo", values=[20.0, 40.0, 80.0])
    assert list(sweep["parameter_value"]) == [20.0, 40.0, 80.0]
    assert sweep["dm_host_capped"].iloc[0] > sweep["dm_host_capped"].iloc[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py::test_knob_sweep_varies_one_parameter_for_iconic_sightline -q
```

Expected: FAIL with missing `one_parameter_sweep`.

- [ ] **Step 3: Implement one-parameter sweep and plot writer**

Add:

```python
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
```

- [ ] **Step 4: Extend CLI to write `sightline_budget_sensitivity_knobs.png`**

After writing artifacts in `main`, add:

```python
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
```

- [ ] **Step 5: Run tests and smoke run**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py -q
conda run -n flits python -m galaxies.foreground.sightline_sensitivity --n-per-family 5 --seed 7 --output-dir /tmp/flits-sensitivity-smoke
```

Expected: tests PASS and smoke output includes `Wrote knobs_png`.

- [ ] **Step 6: Commit**

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py
git commit -m "feat(galaxies): add sightline sensitivity knob plots"
```

### Task 6: Full Verification And Current Results

**Files:**
- Generated: `results/sightline_budget_sensitivity_draws.csv`
- Generated: `results/sightline_budget_sensitivity_summary.csv`
- Generated: `results/sightline_budget_sensitivity_summary.md`
- Generated: `results/sightline_budget_sensitivity_knobs.png`
- Generated: `results/sightline_budget_sensitivity_priors.yaml`

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n flits pytest galaxies/foreground/test_sightline_sensitivity.py galaxies/foreground/test_sightline_budget.py galaxies/foreground/test_scattering_predict.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader galaxy tests**

Run:

```bash
conda run -n flits pytest galaxies/foreground -q
```

Expected: PASS.

- [ ] **Step 3: Generate final sensitivity artifacts**

Run:

```bash
conda run -n flits python -m galaxies.foreground.sightline_sensitivity --n-per-family 1000 --seed 20260619 --output-dir results
```

Expected output includes all five generated result files.

- [ ] **Step 4: Inspect summary for scientific labels**

Run:

```bash
conda run -n flits python - <<'PY'
import pandas as pd
df = pd.read_csv("results/sightline_budget_sensitivity_summary.csv")
print(df[["name", "p_host_negative", "p_interv_dm_gt_100", "p_tau_gt_0p1ms", "host_budget_label", "robustness_label"]].to_string(index=False))
PY
```

Expected: a 12-row table with no missing `host_budget_label` or `robustness_label` values.

- [ ] **Step 5: Run closeout check**

Run:

```bash
mskill tool agent-closeout-check --repo /Users/jakobfaber/Developer/repos/github.com/dsa110/dsa110-FLITS --touched galaxies/foreground/sightline_sensitivity.py --touched galaxies/foreground/test_sightline_sensitivity.py --touched results/sightline_budget_sensitivity_summary.md
```

Expected: PASS or a concrete dirty-state/restart packet request. If it requests a dirty-state handoff because the repo already has unrelated modified DSA YAMLs, prepare that packet and keep those YAMLs out of the sensitivity commit.

- [ ] **Step 6: Final commit**

If generated artifacts are intended to be versioned, commit them with the code:

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py results/sightline_budget_sensitivity_draws.csv results/sightline_budget_sensitivity_summary.csv results/sightline_budget_sensitivity_summary.md results/sightline_budget_sensitivity_knobs.png results/sightline_budget_sensitivity_priors.yaml
git commit -m "feat(galaxies): add prior-predictive sightline sensitivity results"
```

If draw-level artifacts are too large or should remain regenerated products, commit only code, tests, summary markdown, plot, and priors YAML:

```bash
git add galaxies/foreground/sightline_sensitivity.py galaxies/foreground/test_sightline_sensitivity.py results/sightline_budget_sensitivity_summary.csv results/sightline_budget_sensitivity_summary.md results/sightline_budget_sensitivity_knobs.png results/sightline_budget_sensitivity_priors.yaml
git commit -m "feat(galaxies): add prior-predictive sightline sensitivity results"
```

## Self-Review

- Spec coverage: The plan implements predefined claims, one-parameter sweeps, multi-family Monte Carlo draws, measured-vs-assumed mass handling, placeholder-redshift hypothetical labels, prior-dominated labels, artifacts, tests, and verification.
- Placeholder scan: No implementation step contains `TBD`, `TODO`, or unspecified tests. Generated result commit choice is intentionally explicit because versioning draw-level CSVs is a repository policy decision.
- Type consistency: `PriorFamily`, `sample_nuisance_draws`, `apply_draw_to_budget`, `run_sensitivity`, `summarize_sensitivity`, artifact writers, and knob sweeps use consistent column names across tasks.

