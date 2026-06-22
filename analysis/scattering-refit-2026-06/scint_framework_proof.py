#!/usr/bin/env python
"""Proof the scintillation framework WORKS -- so freya's null is physics, not a bug.

Builds synthetic per-channel gains with a KNOWN, RESOLVED scintillation bandwidth
(Lorentzian-correlated across frequency), recovers Delta_nu_d two ways (GP-marginal
fit + ACF), and contrasts the ACF of a RESOLVED case vs an UNRESOLVED case (Delta_nu_d
< channel). Left: injected vs recovered over a sweep. Right: resolved ACF shows a
clear Lorentzian of finite width; unresolved ACF collapses at lag 1 -- the SAME
signature freya shows.

  python scint_framework_proof.py            (local; FLITS_REPO -> local repo)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import _gp_amplitude_logL


def lorentz_field(freq_MHz, dnud, rng, sigma=0.6):
    """Gaussian field with a Lorentzian ACF of half-width dnud (scintillation gains)."""
    d = freq_MHz[:, None] - freq_MHz[None, :]
    C = sigma ** 2 / (1.0 + (d / dnud) ** 2)
    L, Q = np.linalg.eigh(C); L = np.clip(L, 0, None)
    return 1.0 + (Q @ (np.sqrt(L) * rng.standard_normal(freq_MHz.size)))


def acf(g):
    x = g - g.mean(); ac = np.correlate(x, x, "full"); ac = ac[ac.size // 2:]
    return ac / ac[0]


def recover_gp(ahat, v, freq_MHz, grid):
    """Scan Delta_nu_d, return the argmax of the GP-marginal amplitude logL."""
    lls = [_gp_amplitude_logL(ahat, v, freq_MHz, dn, mu_degree=1)[0] for dn in grid]
    return grid[int(np.argmax(lls))], np.array(lls)


def main():
    rng = np.random.default_rng(7)
    nch = 96
    freq = np.linspace(1311, 1499, nch)           # MHz, DSA-like band (188 MHz)
    chan = np.median(np.diff(freq))               # ~2 MHz/ch
    v = (0.05 ** 2) * np.ones(nch)                 # small per-channel gain noise
    grid = np.geomspace(0.5 * chan, 60, 40)

    # --- left: injected vs recovered over a sweep (resolved range)
    injected = np.geomspace(2 * chan, 40, 10)
    rec = []
    for dn in injected:
        g = lorentz_field(freq, dn, rng)
        ahat = g + np.sqrt(v) * rng.standard_normal(nch)
        dn_hat, _ = recover_gp(ahat, v, freq, grid)
        rec.append(dn_hat)
    rec = np.array(injected) * 0 + np.array(rec)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].loglog(injected, rec, "o", color="tab:green")
    lo, hi = injected.min() * 0.8, injected.max() * 1.2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax[0].set_xlabel("injected Delta_nu_d (MHz)"); ax[0].set_ylabel("GP-recovered (MHz)")
    ax[0].set_title("framework recovers injected Delta_nu_d\n(resolved regime)")
    ax[0].legend(fontsize=8)

    # --- middle: RESOLVED ACF (clear Lorentzian width)
    dn_res = 12.0
    g_res = lorentz_field(freq, dn_res, rng)
    ac_res = acf(g_res - np.polyval(np.polyfit(freq, g_res, 2), freq) + 1)
    lags = np.arange(ac_res.size) * chan
    ax[1].plot(lags, ac_res, "o-", color="tab:green", ms=4)
    ax[1].plot(lags, 1.0 / (1.0 + (lags / dn_res) ** 2), "k--", lw=1,
               label=f"Lorentz dnud={dn_res:.0f}MHz")
    ax[1].axhline(0.5, color="0.8", ls=":"); ax[1].set_xlim(0, 60)
    ax[1].set_xlabel("freq lag (MHz)"); ax[1].set_ylabel("ACF")
    ax[1].set_title(f"RESOLVED: dnud={dn_res:.0f}MHz >> chan={chan:.1f}MHz\n(finite ACF width)")
    ax[1].legend(fontsize=8)

    # --- right: UNRESOLVED ACF (collapses at lag 1) -- freya's signature
    dn_unres = 0.1 * chan
    g_un = lorentz_field(freq, dn_unres, rng)
    ac_un = acf(g_un - np.polyval(np.polyfit(freq, g_un, 2), freq) + 1)
    ax[2].plot(lags, ac_un, "o-", color="tab:red", ms=4)
    ax[2].axhline(0.5, color="0.8", ls=":"); ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlim(0, 60); ax[2].set_xlabel("freq lag (MHz)"); ax[2].set_ylabel("ACF")
    ax[2].set_title(f"UNRESOLVED: dnud={dn_unres:.2f}MHz < chan\n(ACF collapses at lag 1 = freya's signature)")

    fig.suptitle("Scintillation framework validation: recovers known Delta_nu_d when RESOLVED; "
                 "freya matches the UNRESOLVED signature (right)", fontsize=12)
    fig.tight_layout()
    out = os.environ.get("OUT", ".")
    fp = f"{out}/scint_framework_proof.png"; fig.savefig(fp, dpi=130, bbox_inches="tight")
    # report numbers for the record
    ratio = rec / injected
    print(f"recovery ratio median={np.median(ratio):.2f} (1.0=perfect); "
          f"injected {injected.min():.1f}-{injected.max():.0f}MHz")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
