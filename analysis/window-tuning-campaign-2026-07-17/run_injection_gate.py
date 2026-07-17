"""Fixed-seed known-truth injection gate for the objective-window fitter."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scintillation.scint_analysis import figure_manifest
from scintillation.scint_analysis import window_refit as wr

SEED = 20260717
N_TRIALS = 100
GAMMAS = (0.02, 0.05, 0.1, 0.3, 1.0, 3.0)
OUT = Path(__file__).parent / "results"
CRITERIA = {
    "max_median_gamma_relative_error": 0.10,
    "min_isolated_resolved_fraction": 0.95,
    "min_two_component_adoption_fraction": 0.95,
    "max_false_two_component_fraction": 0.05,
    "max_median_modulation_relative_error": 0.05,
}


def _lorentz(lags, amplitude, gamma):
    return amplitude / (1.0 + (lags / gamma) ** 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lags = np.arange(0, wr.LAG_MAX + 1e-9, 0.0061)

    def simulate(gamma, modulation, broad_gamma=None, broad_modulation=0.0):
        acf = _lorentz(lags, modulation**2, gamma)
        if broad_gamma is not None:
            acf += _lorentz(lags, broad_modulation**2, broad_gamma)
        return acf + rng.normal(0.0, 0.01, lags.size)

    isolated = []
    for truth in GAMMAS:
        fits = [wr._fit_subband(lags, simulate(truth, 0.7)) for _ in range(N_TRIALS)]
        recovered = [f["gamma"] for f in fits if f.get("resolved")]
        median = float(np.median(recovered))
        isolated.append(
            {
                "truth_gamma_mhz": truth,
                "median_gamma_mhz": median,
                "median_relative_error": abs(median / truth - 1.0),
                "resolved_fraction": len(recovered) / N_TRIALS,
            }
        )

    decompositions = []
    for broad in (1.0, 2.0, 5.0):
        fits = [
            wr._fit_subband(lags, simulate(0.08, 0.6, broad, 0.7))
            for _ in range(N_TRIALS)
        ]
        adopted = [f for f in fits if f.get("model_sel") == "2L"]
        median = float(np.median([f["gamma"] for f in adopted]))
        decompositions.append(
            {
                "broad_gamma_mhz": broad,
                "adoption_fraction": len(adopted) / N_TRIALS,
                "median_narrow_gamma_mhz": median,
                "median_relative_error": abs(median / 0.08 - 1.0),
            }
        )

    envelope_false = sum(
        wr._fit_subband(lags, simulate(3.0, 0.7)).get("model_sel") == "2L"
        for _ in range(N_TRIALS)
    )
    noise_false = sum(
        wr._fit_subband(lags, rng.normal(0.0, 0.01, lags.size)).get("model_sel") == "2L"
        for _ in range(N_TRIALS)
    )
    false_splits = {
        "envelope_only_fraction": envelope_false / N_TRIALS,
        "noise_only_fraction": noise_false / N_TRIALS,
    }

    modulation = []
    for truth in (0.6, 0.9):
        fits = [
            wr._fit_subband(lags, simulate(0.08, truth, 2.0, 0.7))
            for _ in range(N_TRIALS)
        ]
        values = [f["m"] for f in fits if f.get("model_sel") == "2L"]
        median = float(np.median(values))
        modulation.append(
            {
                "truth_modulation": truth,
                "median_modulation": median,
                "median_relative_error": abs(median / truth - 1.0),
            }
        )

    checks = {
        "isolated_gamma": all(
            row["median_relative_error"] <= CRITERIA["max_median_gamma_relative_error"]
            and row["resolved_fraction"] >= CRITERIA["min_isolated_resolved_fraction"]
            for row in isolated
        ),
        "two_component_recovery": all(
            row["adoption_fraction"] >= CRITERIA["min_two_component_adoption_fraction"]
            and row["median_relative_error"] <= CRITERIA["max_median_gamma_relative_error"]
            for row in decompositions
        ),
        "false_split_control": max(false_splits.values())
        <= CRITERIA["max_false_two_component_fraction"],
        "modulation_recovery": all(
            row["median_relative_error"]
            <= CRITERIA["max_median_modulation_relative_error"]
            for row in modulation
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].loglog(
        [row["truth_gamma_mhz"] for row in isolated],
        [row["median_gamma_mhz"] for row in isolated],
        "o-",
        label="isolated",
    )
    axes[0].plot(GAMMAS, GAMMAS, "k--", lw=1, label="identity")
    axes[0].set(xlabel="injected gamma (MHz)", ylabel="median recovered gamma (MHz)")
    axes[0].legend()
    axes[1].bar(
        ["broad=1", "broad=2", "broad=5", "envelope null", "noise null"],
        [row["adoption_fraction"] for row in decompositions]
        + [false_splits["envelope_only_fraction"], false_splits["noise_only_fraction"]],
    )
    axes[1].axhline(CRITERIA["min_two_component_adoption_fraction"], color="k", ls="--")
    axes[1].set(ylabel="fraction", xlabel="injection or null case", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    figure = OUT / "injection_recovery_gate.png"
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_manifest.register_figure(
        OUT,
        figure.name,
        "Recovered isolated widths follow the identity line; the 2L adoption bars exceed "
        "the 0.95 guide for injected two-scale ACFs while both null false-split bars remain low.",
        campaign="CHIME objective-window scintillation diagnostics",
    )

    code_path = Path(wr.__file__)
    payload = {
        "gate_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "criteria": CRITERIA,
        "seed": SEED,
        "n_trials_per_point": N_TRIALS,
        "isolated": isolated,
        "two_component": decompositions,
        "false_splits": false_splits,
        "modulation": modulation,
        "provenance": {
            "command": "FLITS_ROOT=$PWD uv run python analysis/window-tuning-campaign-2026-07-17/run_injection_gate.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "window_refit_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        },
    }
    (OUT / "injection_recovery.json").write_text(json.dumps(payload, indent=2) + "\n")
    if payload["gate_status"] != "pass":
        raise SystemExit("injection gate failed")
    print(json.dumps({"gate_status": payload["gate_status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
