"""P4: measure independent CHIME DMs for the 12 co-detections with the custom DM tool.

Per .agents/audit-chime-side-dm.md (structure-max retracted): for each burst, coherently dedisperse
the CHIME singlebeam at the DSA DM, then run dispersion/chime_dm.measure_dm (wide incoherent search +
scatter-corrected arrival-time regression). Records DM, sigma, constrains_dm, S/N and writes a 3-panel
diagnostic. chime_dm.py is copied fresh from the repo by the runner (no vendored-copy drift).

Run: bin/baseband_analysis_python.sh scripts/extract_chime_dm_v2.py
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
from chime_dm import K_DM, measure_dm

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
DIAG = ROOT + "/diagnostics/chime_dm_v2"
os.makedirs(DIAG, exist_ok=True)
TDS = 16  # time-downsample (2.56us -> 41us); bursts are ms-scale, keeps the coarse search cheap


def extract_one(path, dm_dsa):
    bb = BBData.from_file(path)
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)  # 871 descending
    ra = float(np.asarray(bb["tiedbeam_locations"]["ra"]).ravel()[0])
    dec = float(np.asarray(bb["tiedbeam_locations"]["dec"]).ravel()[0])

    bbdd = coherent_dedisp(bb, dm_dsa)  # proper dedispersion (time_shift=True default)
    intensity = np.nan_to_num(
        np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2
    )  # (871, ntime)
    # RFI / dead-channel mask from per-channel variance (drop dead AND RFI-loud channels)
    csd = intensity.std(axis=1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    iw, fw = intensity[good], freq[good]
    nt = (iw.shape[1] // TDS) * TDS
    wf = iw[:, :nt].reshape(iw.shape[0], nt // TDS, TDS).mean(2)
    dt_ds = dt * TDS

    res = measure_dm(
        wf, fw, dt_ds, dm_dsa, n_subband=8, dm_window=5.0, dm_step=0.1, dm_err_max=20.0
    )
    res["n_chan_ok"] = int(good.sum())
    res["chime_ra_deg"] = ra
    res["chime_dec_deg"] = dec

    fig = _figure(wf, fw, dt_ds, dm_dsa, res)
    return res, fig


def _figure(wf, freq, dt, dm_dsa, res):
    from chime_dm import _dedisperse  # noqa: PLC0415 (diagnostic only)

    order = np.argsort(freq)
    wfo, fro = wf[order], freq[order]
    ddm_c = res["coarse_dm"] - dm_dsa
    dd = _dedisperse(wfo, fro, dt, ddm_c, fro.max())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    nf = (dd.shape[0] // 4) * 4
    m = dd[:nf].reshape(nf // 4, 4, dd.shape[1]).mean(1)
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    ax[0].imshow(
        (m - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=[0, dd.shape[1] * dt * 1e3, fro.min(), fro.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(title=f"dedisp @coarse DM={res['coarse_dm']:.2f}", xlabel="t (ms)", ylabel="freq MHz")
    cc = res["coarse_curve"]
    ax[1].plot(cc["dm"], cc["snr"], ".-", ms=3)
    ax[1].axvline(dm_dsa, color="k", ls=":", label=f"DSA={dm_dsa:g}")
    ax[1].axvline(res["coarse_dm"], color="r", label=f"coarse={res['coarse_dm']:.2f}")
    ax[1].set(title=f"coarse S/N(DM) peak~{res['peak_snr']:.0f}", xlabel="trial DM")
    ax[1].legend(fontsize=8)
    sb = res["subbands"]
    if sb:
        nu = np.array([s["freq_mhz"] for s in sb])
        t0 = np.array([s["t0_s"] for s in sb]) * 1e3
        e = np.array([s["t0_err_s"] for s in sb]) * 1e3
        x = K_DM * (1.0 / nu**2 - 1.0 / fro.max() ** 2)
        ax[2].errorbar(x, t0, yerr=e, fmt="o", ms=4)
        if res["dm"] is not None:
            xs = np.linspace(x.min(), x.max(), 50)
            ax[2].plot(
                xs,
                (res["dm"] - res["coarse_dm"]) * xs * 1e3
                + np.median(t0 - (res["dm"] - res["coarse_dm"]) * x * 1e3),
                "r-",
            )
    dmtxt = "None" if res["dm"] is None else f"{res['dm']:.2f}±{res['dm_err']:.2f}"
    ax[2].set(
        title=f"arrival regression DM={dmtxt} ({res['constrains_dm']})",
        xlabel="K_DM(nu^-2-ref^-2)",
        ylabel="t0 (ms)",
    )
    fig.tight_layout()
    return fig


def main():
    bursts = json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]
    out, manifest = [], []
    for meta in bursts:
        cid, name, dm = meta["chime_id"], meta["name"], float(meta["dm"])
        path = f"{SB}/{name.lower()}/singlebeam_{cid}.h5"
        if not os.path.exists(path):
            continue
        try:
            res, fig = extract_one(path, dm)
            png = f"{DIAG}/{name}_dm.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            res.update(name=name, chime_id=cid, dm_dsa=dm)
            out.append(res)
            dm_s = "None" if res["dm"] is None else f"{res['dm']:.2f}"
            err_s = 0.0 if res["dm_err"] is None else res["dm_err"]
            manifest.append(
                {
                    "path": os.path.basename(png),
                    "expectation": f"{name}: burst dedispersed near DSA DM={dm:g}; coarse S/N peak; "
                    f"arrival-time regression DM={dm_s} constrains={res['constrains_dm']} "
                    f"(S/N~{res['peak_snr']:.0f}, {res['n_good_subbands']} sub-bands)",
                }
            )
            print(
                f"[OK] {name:11s} DSA={dm:8.2f} CHIME={dm_s:>8} +/- {err_s:5.2f} "
                f"constrains={res['constrains_dm']!s:5} S/N={res['peak_snr']:5.0f} "
                f"nsub={res['n_good_subbands']}",
                flush=True,
            )
        except Exception as exc:
            out.append({"name": name, "chime_id": cid, "status": f"error: {exc}"})
            print(f"[ERR] {name}: {exc}\n{traceback.format_exc()}", flush=True)
        json.dump(out, open(ROOT + "/results/chime_dm_v2.json", "w"), indent=2)
        json.dump({"figures": manifest}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote results/chime_dm_v2.json ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
