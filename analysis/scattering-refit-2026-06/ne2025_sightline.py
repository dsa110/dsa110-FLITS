#!/usr/bin/env python
"""Milky-Way scattering/scintillation prediction along a sightline (NE2025).

NE2025 Galactic electron-density model [Ocker & Cordes, ADS:2026ApJ..1002....3O]
via `mwprop.nemod.NE2025.ne2025`. We integrate the model to the Galaxy edge (large
dmax, ndir<0 = d->DM) and return the *total* Galactic dispersion + scattering for
an extragalactic source (FRB). The model reports pulse-broadening TAU and
scintillation bandwidth SBW at 1 GHz; both scale as nu^-4.4 / nu^+4.4 respectively
(scattering_functions2020.py:90, tauiss = 1000*(sm/292)^1.2 * d * nu^-4.4), with
NE2025's C1 = 1.16.

For a co-detected FRB this fixes the MW screen: the predicted SBW(nu) is the MW
scintillation bandwidth (compare to a *resolved* scintle to identify the MW screen),
and the predicted TAU(nu) is the MW pulse-broadening floor (a measured tau far
above it is extragalactic). DEFFSM2 is NE2025's effective MW screen distance.

CLI:  python ne2025_sightline.py            # wilhelm sightline + measured comparison
      python ne2025_sightline.py L B        # any (l,b) in deg
"""

import sys
import warnings

FREQ_EXP = 4.4  # tau ~ nu^-4.4, sbw ~ nu^+4.4 (Kolmogorov thin screen; NE2025 C1=1.16)


def mw_scattering(l_deg, b_deg, dmax_kpc=30.0):
    """Integrate NE2025 to the Galaxy edge; return the MW prediction dict (@1 GHz)."""
    warnings.filterwarnings("ignore")
    from mwprop.nemod.NE2025 import ne2025

    Dk, Dv, Du, Dd = ne2025(
        ldeg=l_deg, bdeg=b_deg, dmd=dmax_kpc, ndir=-1, classic=False, dmd_only=False
    )
    return {
        "dm_mw": Dv["DM"],  # pc/cm^3, integrated to dmax
        "sm": Dv["SM"],  # kpc m^-20/3
        "tau_1ghz_ms": Dv["TAU"],  # ms @ 1 GHz
        "sbw_1ghz_mhz": Dv["SBW"],  # MHz @ 1 GHz
        "scintime_1ghz_s": Dv["SCINTIME"],  # s @ 1 GHz @ 100 km/s
        "theta_g_mas": Dv["THETA_G"],  # mas @ 1 GHz
        "d_eff_kpc": Dv["DEFFSM2"],  # effective MW screen distance
        "nu_t_ghz": Dv["NU_T"],  # transition frequency
    }


def at_freq(val_1ghz, freq_ghz, kind):
    """Scale a 1-GHz scattering quantity to freq_ghz. kind: 'tau' (down) | 'sbw' (up)."""
    e = -FREQ_EXP if kind == "tau" else FREQ_EXP
    return val_1ghz * freq_ghz**e


def _report(l_deg, b_deg, measured=None):
    p = mw_scattering(l_deg, b_deg)
    print(f"=== NE2025 MW prediction  (l={l_deg:.3f}, b={b_deg:.3f}) ===")
    print(f"  DM_MW         = {p['dm_mw']:8.2f} pc/cm^3   (integrated to Galaxy edge)")
    print(f"  SM            = {p['sm']:.3e} kpc m^-20/3")
    print(f"  d_eff (screen)= {p['d_eff_kpc']:8.2f} kpc")
    print(f"  tau   @1 GHz  = {p['tau_1ghz_ms'] * 1e3:8.3f} us")
    print(f"  Dnu_d @1 GHz  = {p['sbw_1ghz_mhz']:8.4f} MHz")
    print(f"  theta_G@1 GHz = {p['theta_g_mas']:8.3f} mas")
    if measured:
        print("\n  band-scaled MW prediction vs measured:")
        for name, fghz in measured["bands"].items():
            tau = at_freq(p["tau_1ghz_ms"], fghz, "tau") * 1e3  # us
            sbw = at_freq(p["sbw_1ghz_mhz"], fghz, "sbw")  # MHz
            print(f"    {name:5s} ({fghz:.3f} GHz):  tau_MW={tau:8.3f} us   Dnu_MW={sbw:8.4f} MHz")
        tmeas = measured["tau_1ghz_ms"]
        excess = tmeas / p["tau_1ghz_ms"]
        print(
            f"\n  measured tau_1GHz = {tmeas * 1e3:.1f} us  ->  {excess:.0f}x the MW floor"
            f" ({p['tau_1ghz_ms'] * 1e3:.2f} us)  =>  tau-screen is EXTRAGALACTIC"
        )
        for label, dnu in measured["scintles"].items():
            print(f"  measured scintle '{label}': {dnu:.3f} MHz")
    return p


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _report(float(sys.argv[1]), float(sys.argv[2]))
    else:
        # wilhelm (FRB 20221203A): l,b from astropy; measured from the joint fit + DSA ACF
        _report(
            107.135,
            16.691,
            measured={
                "bands": {"CHIME": 0.684, "DSA": 1.405},
                "tau_1ghz_ms": 0.26,  # joint M3 fit
                "scintles": {"DSA broad": 5.3, "DSA narrow": 0.12, "CHIME (unres)": 0.06},
            },
        )
