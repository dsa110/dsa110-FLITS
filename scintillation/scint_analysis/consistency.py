import json
import logging
from pathlib import Path

import numpy as np

from . import plotting
from .analysis import scattering_scintillation_consistency

log = logging.getLogger(__name__)


def run_consistency_check(
    scat_results_path: str,
    scint_results_path: str,
    burst_id: str = None,
    c_factor: float = 1.16,
    output_dir: str = None,
):
    """
    Load results from both scattering and scintillation pipelines and
    generate a consistency plot.

    Parameters
    ----------
    scat_results_path : str
        Path to scattering _fit_results.json
    scint_results_path : str
        Path to scintillation _analysis_results.json
    burst_id : str, optional
        ID for the burst (default derived from filenames)
    c_factor : float
        Proportionality constant C.
    output_dir : str, optional
        Directory to save results
    """
    # 1. Load Results
    try:
        with open(scat_results_path) as f:
            scat_results = json.load(f)
    except Exception as e:
        log.error(f"Failed to load scattering results: {e}")
        return

    try:
        with open(scint_results_path) as f:
            scint_results = json.load(f)
    except Exception as e:
        log.error(f"Failed to load scintillation results: {e}")
        return

    # 2. Extract burst ID if not provided
    if burst_id is None:
        burst_id = Path(scat_results_path).stem.split("_fit_results")[0]

    # 3. Prepare data for plotter
    # The scintillation JSON might have slightly different structure than expected by the plotter
    # We need to maps subband_measurements to subband_gamma correctly
    # Checking Scintillation JSON structure from previous view:
    # components -> component_1 -> subband_measurements -> [freq_mhz, bw, bw_err, ...]

    scint_plot_data = {"subband_center_freqs_mhz": [], "subband_gamma": [], "subband_gamma_err": []}

    # Try to find sub-band measurements in the best model/component
    # This assumes we want component_1 for now or a flattened list
    components = scint_results.get("components", {})
    for comp_id, comp_data in components.items():
        measurements = comp_data.get("subband_measurements", [])
        for m in measurements:
            scint_plot_data["subband_center_freqs_mhz"].append(m.get("freq_mhz"))
            scint_plot_data["subband_gamma"].append(m.get("bw"))
            scint_plot_data["subband_gamma_err"].append(
                m.get("bw_err", 0) if m.get("bw_err") is not None else 0
            )

    # Convert to arrays and sort by frequency
    nu = np.array(scint_plot_data["subband_center_freqs_mhz"])
    gamma = np.array(scint_plot_data["subband_gamma"])
    err = np.array(scint_plot_data["subband_gamma_err"])

    sort_idx = np.argsort(nu)
    scint_plot_data["subband_center_freqs_mhz"] = nu[sort_idx]
    scint_plot_data["subband_gamma"] = gamma[sort_idx]
    scint_plot_data["subband_gamma_err"] = err[sort_idx]

    # 4. Generate Plot
    if output_dir:
        save_path = Path(output_dir) / f"{burst_id}_scat_scint_consistency.png"
    else:
        save_path = f"{burst_id}_scat_scint_consistency.png"

    log.info(f"Generating consistency plot for {burst_id}...")
    fig = plotting.plot_scat_scint_consistency(
        scint_plot_data, scat_results, c_factor=c_factor, save_path=str(save_path)
    )

    return fig


def band_consistency(tau_1ghz_ms, alpha, nu0_ghz, dnu_mhz, C=1.16):
    """C_implied = 2*pi*tau*Dnu and the screen-count verdict at one band.

    Scales tau(1GHz) to the band centre nu0 (tau ~ nu^-alpha), then applies the
    canonical scattering_scintillation_consistency relation (references in that
    func's docstring). C_implied ~ 1 => one screen plausibly does both scattering
    and scintillation; C_implied far from 1 => >=2 screens, or a systematic in
    one measurement. C is the reference geometry constant (NE2025 thin-screen
    ~1.16); the verdict uses C_implied, not C.
    """
    tau_band_ms = tau_1ghz_ms * nu0_ghz ** (-alpha)
    r = scattering_scintillation_consistency(tau_band_ms, dnu_mhz, C=C)
    r["tau_band_ms"] = tau_band_ms
    return r


def consistency_from_multiscale(path, C=1.16):
    """Per-band consistency verdict for one *_multiscale_results.json burst.

    The multiscale file already carries tau_1ghz, alpha, the per-band centres
    (bands.<B>.nu0_GHz) and decorrelation bandwidths (nu_scaling.dnu_<B>_MHz),
    so no cross-pipeline join is needed.
    """
    d = json.loads(Path(path).read_text())
    tau_1ghz, alpha = d["tau_1ghz"], d["alpha"]
    burst = d.get("burst") or Path(path).stem.replace("_multiscale_results", "")
    row = {"burst": burst, "tau_1ghz_ms": tau_1ghz, "alpha": alpha}
    for band, bd in d["bands"].items():
        dnu = d.get("nu_scaling", {}).get(f"dnu_{band}_MHz")
        if dnu is None or not np.isfinite(dnu):
            continue
        r = band_consistency(tau_1ghz, alpha, bd["nu0_GHz"], dnu, C=C)
        row[f"dnu_{band}_MHz"] = dnu
        row[f"C_implied_{band}"] = r["C_implied"]
        row[f"consistent_{band}"] = r["consistent"]
    return row


def consistency_table(
    results_glob="analysis/scattering-refit-2026-06/*_multiscale_results.json", C=1.16
):
    """Screen-count verdict table over every burst with measured scintillation."""
    import glob

    import pandas as pd

    rows = [consistency_from_multiscale(p, C=C) for p in sorted(glob.glob(results_glob))]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Scintillation-Scattering consistency check.")
    parser.add_argument(
        "scat_json", nargs="?", type=str, help="Path to scattering fit results JSON."
    )
    parser.add_argument(
        "scint_json", nargs="?", type=str, help="Path to scintillation analysis results JSON."
    )
    parser.add_argument("--burst_id", type=str, help="Burst ID.")
    parser.add_argument("--c_factor", type=float, default=1.16, help="Proportionality constant C.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    parser.add_argument(
        "--table",
        action="store_true",
        help="Emit per-burst screen-count verdict table from *_multiscale_results.json",
    )
    parser.add_argument(
        "--out", type=str, help="CSV output path for --table (default consistency.csv)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.table:
        df = consistency_table(C=args.c_factor)
        out = args.out or "consistency.csv"
        df.to_csv(out, index=False)
        print(f"[ok] {len(df)} bursts -> '{out}' (scat-scint consistency / screen count)")
    else:
        if not args.scat_json or not args.scint_json:
            parser.error("scat_json and scint_json are required unless --table is given")
        run_consistency_check(
            args.scat_json,
            args.scint_json,
            burst_id=args.burst_id,
            c_factor=args.c_factor,
            output_dir=args.outdir,
        )
