"""P5: coherent trial-DM likelihood envelope (expert-endorsed). See .agents/audit-chime-side-dm.md.

For each burst, coherently re-dedisperse the BASEBAND at a grid of trial DMs (full band, no zero-fill,
intra-channel correct) and fit a shared-arrival scattering template at each trial; the chi^2(DM)
envelope -> DM + honest sigma. Bright bursts give a narrow envelope at the DSA DM (control); faint
bursts give a wide envelope -> exclusion interval, not a spurious tight point.

Validate on a bright control (DM_TARGETS=zach) before the faint set. dm_envelope.py is vendored in
scripts/ (docker has no flits).
Run: DM_TARGETS=zach DM_WINDOW=12 DM_STEP=0.5 bin/baseband_analysis_python.sh \
       scripts/extract_chime_dm_v4_envelope.py
"""

import json
import os
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dm_envelope import K_DM, bootstrap_dm, fit_waterfall

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
DIAG = ROOT + "/diagnostics/chime_dm_v4"
os.makedirs(DIAG, exist_ok=True)
TDS = 16
N_SB = 16  # sub-bands for the bootstrap (the independent timing units that set sigma_DM)
WINDOW = float(os.environ.get("DM_WINDOW", "12"))
STEP = float(os.environ.get("DM_STEP", "0.5"))
TARGETS = os.environ.get("DM_TARGETS", "zach").split(",")


def _downsample_t(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


def _bin_to(arr, nb):
    """Block-average axis 0 of (n_ch, n_t) down to ~nb rows."""
    k = max(arr.shape[0] // nb, 1)
    m = (arr.shape[0] // k) * k
    return arr[:m].reshape(m // k, k, arr.shape[1]).mean(1)


def extract_one(path, dm_dsa):
    """Fast coherent-once envelope: ONE coherent_dedisp at dm_dsa, then every trial DM is a per-channel
    fractional time-shift of the (zero-padded) downsampled intensity via a cached FFT phase ramp -- a
    pure inter-channel delay, which commutes with magnitude-squaring, so it is exact except for
    intra-channel smear (sub-resolution near the minimum where sigma_DM is set). The burst SHAPE
    (sigma, tau) is fit once at the reference and frozen; each trial optimises only the shared arrival.
    """
    bb = BBData.from_file(path)
    dt = float(bb.attrs["delta_time"]) * TDS
    freq = np.asarray(bb.index_map["freq"]["centre"], float)

    bbref = coherent_dedisp(bb, dm_dsa)  # the ONE expensive call
    ref = _downsample_t(
        np.nan_to_num(np.abs(bbref[:, 0, :]) ** 2 + np.abs(bbref[:, 1, :]) ** 2), TDS
    )
    csd = ref.std(1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    I0 = ref[good]
    fw = freq[good]
    nu_ref = float(fw.max())
    prof = (I0 - np.median(I0, 1, keepdims=True)).sum(0)
    pk = int(np.argmax(np.convolve(prof, np.ones(max(int(1.5e-3 / dt), 1)), "same")))
    lo = max(pk - int(15e-3 / dt), 0)
    hi = min(pk + int(45e-3 / dt), I0.shape[1])

    # per-channel residual-delay coefficient [samples per pc/cm^3]; zero-pad to keep the shift linear
    cj = K_DM * (1.0 / fw**2 - 1.0 / nu_ref**2) / dt
    pad = int(np.ceil(np.abs(cj).max() * WINDOW)) + 8
    F0 = np.fft.rfft(np.pad(I0, ((0, 0), (pad, pad))), axis=1)
    fb = np.fft.rfftfreq(I0.shape[1] + 2 * pad)
    cl, ch = lo + pad, hi + pad

    def shifted_crop(ddm):  # shift at FINE resolution (intra-band dispersion resolved), then crop
        ik = np.fft.irfft(
            F0 * np.exp(2j * np.pi * fb[None, :] * (cj * ddm)[:, None]),
            n=I0.shape[1] + 2 * pad,
            axis=1,
        )
        return ik[:, cl:ch]

    # reference full fit on the fine band: freeze burst shape (sigma, tau) for the t0-only envelope
    _c, _dof, p_ref, r_ref = fit_waterfall(I0[:, lo:hi], fw, dt)
    sigma_ref, tau_ref = np.exp(p_ref[1]), np.exp(p_ref[2])

    n_sb = N_SB
    fw16 = _bin_to(fw[:, None], n_sb)[:, 0]
    trials = dm_dsa + np.arange(-WINDOW, WINDOW + STEP / 2, STEP)
    crops = np.stack(
        [_bin_to(shifted_crop(tdm - dm_dsa), n_sb) for tdm in trials]
    )  # (n_tr, n_sb, n_t)
    res = bootstrap_dm(trials, crops, fw16, dt, sigma_ref, tau_ref, n_boot=120)
    res.update(
        dm_dsa=dm_dsa,
        n_chan_ok=int(good.sum()),
        crop_ms=[round(lo * dt * 1e3, 1), round(hi * dt * 1e3, 1)],
        sigma_ref_ms=round(float(sigma_ref) * 1e3, 3),
        tau_ref_ms=round(float(tau_ref) * 1e3, 3),
        ref_chi2_red=round(float(r_ref), 2),
    )
    fig = _figure(I0[:, lo:hi], fw, dt, dm_dsa, res)
    return res, fig


def _figure(wf, freq, dt, dm_dsa, res):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    nf = (wf.shape[0] // 4) * 4
    m = wf[:nf].reshape(nf // 4, 4, wf.shape[1]).mean(1)
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    ax[0].imshow(
        (m - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=[0, wf.shape[1] * dt * 1e3, freq.min(), freq.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(title=f"burst @ DSA DM={dm_dsa:g}", xlabel="t (ms)", ylabel="freq MHz")
    dm_grid = np.asarray(res["dm_grid"])
    chi2 = np.asarray(res["chi2_full"], float)
    ax[1].plot(dm_grid, chi2 - chi2.min(), ".-", ms=3)
    ax[1].axvline(dm_dsa, color="k", ls=":", label=f"DSA={dm_dsa:g}")
    s = "None" if res["sigma"] is None else f"{res['sigma']:.3f}"
    e = res["excl95_pc"]
    es = "None" if e is None else f"{e:.2f}"
    ax[1].axvline(res["dm"], color="r", label=f"DM={res['dm']:.2f}+/-{s}")
    if res["sigma"] is not None:
        ax[1].axvspan(res["dm"] - res["sigma"], res["dm"] + res["sigma"], color="r", alpha=0.15)
    ax[1].set(
        title=f"chi2 envelope: constrains={res['constrains_dm']} "
        f"(chi2_red={res['chi2_red_min']:.2f}, excl95={es}, nboot={res['n_boot_ok']})",
        xlabel="trial DM",
        ylabel="chi^2 - min (full sub-bands)",
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    bursts = {b["name"]: b for b in json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]}
    out, manifest = [], []
    for name in TARGETS:
        meta = bursts.get(name)
        if meta is None:
            continue
        cid, dm = meta["chime_id"], float(meta["dm"])
        path = f"{SB}/{name.lower()}/singlebeam_{cid}.h5"
        if not os.path.exists(path):
            continue
        try:
            res, fig = extract_one(path, dm)
            png = f"{DIAG}/{name}_dm.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            res.update(name=name, chime_id=cid)
            out.append(res)
            s = "None" if res["sigma"] is None else f"{res['sigma']:.2f}"
            manifest.append(
                {
                    "path": os.path.basename(png),
                    "expectation": f"{name}: chi2(DM) envelope, DSA DM={dm:g}; min DM={res['dm']:.2f} "
                    f"+/-{s}; constrains={res['constrains_dm']} chi2_red={res['chi2_red_min']:.2f} "
                    f"excl95={res['excl95_pc']} (bright->narrow min at DSA; faint->wide)",
                }
            )
            print(
                f"[OK] {name:11s} DSA={dm:8.2f} env_DM={res['dm']:8.2f} sigma={s:>6} "
                f"constrains={res['constrains_dm']!s:5} chi2red={res['chi2_red_min']:.2f} "
                f"excl95={res['excl95_pc']}",
                flush=True,
            )
        except Exception as exc:
            out.append({"name": name, "chime_id": cid, "status": f"error: {exc}"})
            print(f"[ERR] {name}: {exc}\n{traceback.format_exc()}", flush=True)
        json.dump(out, open(ROOT + "/results/chime_dm_v4.json", "w"), indent=2)
        json.dump({"figures": manifest}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote results/chime_dm_v4.json ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
