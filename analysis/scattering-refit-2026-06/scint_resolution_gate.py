#!/usr/bin/env python
"""Cheap scintillation resolution gate -- run BEFORE trusting the GP fit.

Uses ONLY the per-channel gains saved by the gain-marginalized fit (npz: gain_C,
gain_D, freq_C, freq_D). No model, no refit, no scat_analysis import -- pure numpy.

For unresolved diffractive scintillation each channel averages Delta_nu_chan/Delta_nu_d
independent scintles, so the observed modulation index obeys
    m_obs^2(Delta_nu_chan) = Delta_nu_d / Delta_nu_chan + m_noise^2
i.e. LINEAR in 1/Delta_nu_chan with slope = Delta_nu_d (m_intrinsic ~ 1 for diffractive)
and an intercept that absorbs the (uncorrelated) gain noise. Re-binning the saved
gains COARSER samples several Delta_nu_chan, so a line fit gives a NOISE-CORRECTED
Delta_nu_d -- the value the 460-line GP returns for unresolved data, in ~10 lines.

Verdict:
  RESOLVED   if Delta_nu_d >~ 3 * native channel width (GP fit justified -- it adds
             a real posterior + alpha coupling)
  UNRESOLVED if Delta_nu_d < native channel width (report m^2*chan as Delta_nu_d
             estimate / upper limit; the GP is dead weight until finer channels)

  python scint_resolution_gate.py <npz_path>
"""
import sys
import numpy as np


def modulation_index(freq_GHz, gain, fbin):
    """Detrended modulation index after averaging `fbin` adjacent channels."""
    ok = np.isfinite(gain) & (gain > 0)
    f, g = np.asarray(freq_GHz)[ok], np.asarray(gain)[ok]
    if fbin > 1:
        n = (f.size // fbin) * fbin
        f = f[:n].reshape(-1, fbin).mean(1)
        g = g[:n].reshape(-1, fbin).mean(1)
    if g.size < 4:
        return None, None
    # detrend the smooth intrinsic spectrum (quadratic), modulation about 1
    trend = np.polyval(np.polyfit(f, g, 2), f)
    gn = g / np.where(trend > 0, trend, np.nan)
    gn = gn[np.isfinite(gn)]
    chan_MHz = float(np.median(np.abs(np.diff(f)))) * 1e3
    return float(np.std(gn)), chan_MHz


def gate_band(freq_GHz, gain, name):
    nat = float(np.median(np.abs(np.diff(freq_GHz)))) * 1e3
    pts = []
    for fbin in (1, 2, 4):
        m, cw = modulation_index(freq_GHz, gain, fbin)
        if m is not None:
            pts.append((cw, m))
    if len(pts) < 2:
        print(f"  {name}: too few channels to gate ({len(pts)} binnings)")
        return
    cw = np.array([p[0] for p in pts]); m2 = np.array([p[1] ** 2 for p in pts])
    # m^2 = dnud * (1/cw) + noise^2  -> slope = Delta_nu_d (noise-corrected)
    slope, intercept = np.polyfit(1.0 / cw, m2, 1)
    dnud = max(slope, 0.0)                        # MHz
    m_native = pts[0][1]
    naive = m_native ** 2 * nat                   # single-binning estimate (noise-inflated)
    resolved = dnud >= 3 * nat
    verdict = "RESOLVED" if resolved else ("UNRESOLVED" if dnud < nat else "MARGINAL")
    print(f"  {name}: chan={nat:.2f}MHz  m_native={m_native:.2f}  "
          f"Delta_nu_d(slope)={dnud:.2f}MHz  (naive m^2*chan={naive:.2f}MHz, "
          f"noise^2={max(intercept,0):.3f})  -> {verdict}")
    return dict(dnud=dnud, native_chan=nat, resolved=resolved, verdict=verdict)


def main():
    npz = np.load(sys.argv[1], allow_pickle=True)
    if "gain_C" not in npz:
        sys.exit("npz has no saved gains (not a gain-marginalized fit)")
    print(f"scintillation resolution gate: {sys.argv[1].split('/')[-1]}")
    rC = gate_band(npz["freq_C"], npz["gain_C"], "CHIME")
    rD = gate_band(npz["freq_D"], npz["gain_D"], "DSA")
    any_res = (rC and rC["resolved"]) or (rD and rD["resolved"])
    print(f"\n  GP-in-fit justified: {'YES (a band resolves)' if any_res else 'NO -- report Delta_nu_d = m^2*chan as estimate/upper limit; GP adds nothing until finer channels'}")


if __name__ == "__main__":
    # self-check: synthetic unresolved scintillation must read back as UNRESOLVED
    if len(sys.argv) == 1:
        rng = np.random.default_rng(0)
        f = np.linspace(1.311, 1.499, 16)
        # point-scintle averaging: m ~ sqrt(dnud/chan); chan=12.5MHz, inject dnud=1MHz -> m~0.28
        g = 1.0 + 0.28 * rng.standard_normal(16)
        r = gate_band(f, g, "SELFTEST")
        assert r and not r["resolved"], "gate misread sub-channel scintillation as resolved"
        print("  self-check OK (sub-channel -> UNRESOLVED)")
    else:
        main()
