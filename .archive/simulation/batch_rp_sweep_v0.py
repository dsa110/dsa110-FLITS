# batch_rp_sweep.py – Monte‑Carlo driver for two‑screen RP regimes
"""
Generate synthetic FRB bursts for the three Resolution‑Power regimes in
Pradeep et al. (2025) and dump summary statistics to CSV.

Key features
------------
* **paper‑like RP presets** (host ≈200 Mpc, tiny θ)
* **power / coherent** propagation toggle (`--power`)
* **anisotropic host** for RP = 9.5 (`--anisotropic`)
* **sample‑rate flag** `--fs` (default 20 MHz)
* **spectral channels** flag `--nchan` (default 512)
* **NaN guard** – resimulate burst if `fit_acf` fails
* timestamped, mode‑annotated CSV filenames
"""
from __future__ import annotations
import argparse, pathlib, time
from typing import List

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frb_scintillator import Screen, Scintillator, fit_acf

# ------------------------------------------------------------------ I/O --
CSV_DIR = pathlib.Path("csv"); CSV_DIR.mkdir(exist_ok=True, parents=True)

# paper‑like RP presets ----------------------------------------------------
RP_DEFAULTS = {
    0.20: {"theta_L_mw": 5e-6, "theta_L_host": 3e-9,  "D_host": 2e24},
    0.96: {"theta_L_mw": 5e-6, "theta_L_host": 1e-8,  "D_host": 2e24},
    9.50: {"theta_L_mw": 5e-6, "theta_L_host": 3e-8,  "D_host": 2e24},
}

# ---------------------------------------------------------------- worker --
def run_regime(*, RP: float, n_real: int, wavelength_m: float,
               fs_Hz: float, nchan: int,
               power_mode: bool, anisotropic: bool,
               rng_seed: int | None = None) -> pd.DataFrame:
    """Simulate `n_real` bursts for a given RP and return a DataFrame."""
    cfg = RP_DEFAULTS[RP]
    rng = np.random.default_rng(rng_seed)
    rows: List[dict] = []

    D_MW, D_HOST = 3e19, cfg["D_host"]        # 1 kpc, 200 Mpc
    dt = 1 / fs_Hz
    ntap = 4
    min_len = nchan * ntap

    for bid in range(n_real):
        # Milky‑Way screen
        scr_mw = Screen(dist_m=D_MW,
                        theta_L_rad=cfg["theta_L_mw"], rng=rng)

        # Host screen
        if anisotropic and RP == 9.50:
            scr_host = Screen(dist_m=D_HOST,
                              theta_L_x_rad=2*cfg["theta_L_host"],
                              theta_L_y_rad=cfg["theta_L_host"],
                              pa_deg=30, rng=rng)
        else:
            scr_host = Screen(dist_m=D_HOST,
                              theta_L_rad=cfg["theta_L_host"], rng=rng)

        scint = Scintillator(scr_mw, scr_host,
                             wavelength_m=wavelength_m,
                             combine_in_power=power_mode)

        # simulate one burst ------------------------------------------------
        pulse = np.zeros(2**12, complex); pulse[0] = 1
        # pad to ensure at least one PFB frame
        if pulse.size < min_len:
            pulse = np.pad(pulse, (0, min_len - pulse.size), constant_values=0)
        try:
            f, I = scint.dynamic_spectrum(pulse, dt, fs_Hz=fs_Hz, nchan=nchan)
            
            fig = plt.figure()
            plt.imshow(I, aspect='auto')
            fig.savefig('/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/scattering/simulation/figs/dynspec.png')
        
        except ValueError as e:
            if 'negative dimensions' in str(e):
                raise RuntimeError(
                    f"Requested nchan={nchan} (ntap={ntap}) is too large "
                    f"for pulse length {pulse.size}. Reduce --nchan or increase pulse size."
                ) from e
            else:
                raise
        except RuntimeError as e:          # FFT cap reached
            raise RuntimeError(
                f"FFT length exceeds cap at fs={fs_Hz:.1e} Hz; "
                f"consider lowering --fs or --nchan."
            ) from e

        spec = I[:, 0]
        fit  = fit_acf(spec, dnu=f[1]-f[0], corr_thresh=0.05)
        if np.isnan(fit["m2"]):
            continue                       # redo burst

        fit.update(dict(
            burst            = bid,
            RP_nominal       = RP,
            RP_effective     = scint.RP,
            wavelength_m     = wavelength_m,
            fs_Hz            = fs_Hz,
            nchan            = nchan,
            mode             = "pow" if power_mode else "coh",
            anisotropic_host = bool(anisotropic and RP == 9.50),
            theta_L_mw       = cfg["theta_L_mw"],
            theta_L_host_x   = getattr(scr_host, "theta_L_x_rad", cfg["theta_L_host"]),
            theta_L_host_y   = getattr(scr_host, "theta_L_y_rad", cfg["theta_L_host"]),
        ))
        rows.append(fit)

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ CLI ----
def main():
    p = argparse.ArgumentParser(description="Monte‑Carlo RP sweep → CSV")
    p.add_argument("--rp",  type=float, choices=list(RP_DEFAULTS),
                   help="simulate only this RP value")
    p.add_argument("--n",   type=int,   default=100, help="bursts per RP")
    p.add_argument("--lam", type=float, default=0.21, help="wavelength [m]")
    p.add_argument("--fs",  type=float, default=20e6, help="sample rate [Hz]")
    p.add_argument("--nchan", type=int, default=512,
                   help="number of spectral channels (default 512)")
    p.add_argument("--power",       action="store_true",
                   help="use power‑multiplication propagation")
    p.add_argument("--anisotropic", action="store_true",
                   help="elliptical host when RP=9.5")
    p.add_argument("--self-test",   action="store_true", help="quick CI test")
    args = p.parse_args()

    if args.self_test:
        df = run_regime(RP=0.20, n_real=2, wavelength_m=args.lam,
                        fs_Hz=args.fs, nchan=args.nchan,
                        power_mode=True,
                        anisotropic=False, rng_seed=0)
        assert len(df)==2 and df["m2"].notna().all()
        print("batch_rp_sweep self‑test passed ✓"); return

    rps = [args.rp] if args.rp else RP_DEFAULTS
    for rp in rps:
        df = run_regime(RP=rp, n_real=args.n, wavelength_m=args.lam,
                        fs_Hz=args.fs, nchan=args.nchan,
                        power_mode=args.power,
                        anisotropic=args.anisotropic)
        mode  = "pow" if args.power else "coh"
        stamp = time.strftime("%Y%m%dT%H%M%S")
        fn    = f"summary_RP={rp:.2f}_{mode}_lam={args.lam:.3f}_fs={int(args.fs)}_nchan={args.nchan}_{stamp}.csv"
        #basepath = "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/scattering/simulation/csv"
        #df.to_csv(os.path.join(basepath, fn), index=False)
        df.to_csv(CSV_DIR/fn, index=False)
        print(f"[saved] {fn}  ({len(df)} rows)")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
