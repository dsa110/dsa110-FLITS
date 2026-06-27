"""CHIME + DSA joint model-vs-data figure for one burst.

  python plot_jointmodel_prototypes.py <npz> <out-dir> [tag]

Writes {tag}_jointmodel_vC.{pdf,svg,png}
"""
from __future__ import annotations

import json
import os
import sys
import importlib.util
from pathlib import Path

import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOA_JSON = REPO / "crossmatching" / "toa_crossmatch_results.json"
TEL_CFG = REPO / "scattering/configs/telescopes.yaml"
RUNS = Path(os.environ.get("FLITS_RUNS", Path.home() / "Developer/dsa110-local-data/flits-runs"))

# DSA native voltage sample (telescopes.yaml dsa.dt_ms_raw)
TARGET_DT_MS = float(yaml.safe_load(TEL_CFG.read_text())["dsa"]["dt_ms_raw"])

# Observed band edges (GHz); gap between CHIME high and DSA low
F_CHIME_HI = 0.800
F_DSA_LO = 1.311
GAP_LABEL = rf"unobserved {F_CHIME_HI:.2f}–{F_DSA_LO:.2f}\,GHz"


def _load(npz_fp: Path):
    z = np.load(npz_fp, allow_pickle=True)
    return z, dict(
        burst=str(z["burst"]),
        alpha=float(z["alpha"]),
        tau=float(z["tau_1ghz"]),
        chiC=float(z["chi2C"]),
        chiD=float(z["chi2D"]),
    )


def _fit_json_for(npz_fp: Path) -> Path | None:
    if "_jointmodel" not in npz_fp.name:
        return None
    head, tail = npz_fp.name.split("_jointmodel", 1)
    fp = npz_fp.parent / f"{head}_joint_fit{tail.rsplit('.npz', 1)[0]}.json"
    return fp if fp.exists() else None


def _fit_t0_delta(npz_fp: Path) -> float:
    """t0_C - t0_D in each band's crop-relative ms frame."""
    fit_fp = _fit_json_for(npz_fp)
    if not fit_fp:
        return 0.0
    med = {
        k: v["median"]
        for k, v in json.loads(fit_fp.read_text())["percentiles"].items()
    }
    if "t0_C" not in med or "t0_D" not in med:
        return 0.0
    return float(med["t0_C"]) - float(med["t0_D"])


def _toa_offset_ms(burst: str) -> float | None:
    """Measured CHIME−DSA offset at 400 MHz (toa_crossmatch_results.json)."""
    if not TOA_JSON.exists():
        return None
    row = json.loads(TOA_JSON.read_text()).get(burst.lower())
    if not row:
        return None
    return float(row["measured_offset_ms"])


def _chime_time_shift(npz_fp: Path, burst: str) -> tuple[float, float, str]:
    """Return (CHIME shift ms, displayed offset ms, alignment note).

    Crop-relative fit t0 can misalign bands by ~10 ms; measured TOA at 400 MHz
    is authoritative. Shift CHIME so peaks differ by measured_offset_ms while
    DSA stays in its native crop frame:
      t_C' = t_C - (t0_C - t0_D) + measured_offset_ms
    """
    delta_fit = _fit_t0_delta(npz_fp)
    delta_toa = _toa_offset_ms(burst)
    if delta_toa is None:
        return -delta_fit, delta_fit, "fit t0 (no TOA row)"
    shift = -delta_fit + delta_toa
    return shift, delta_toa, "TOA @400 MHz"


def _t_factor_for(telescope: str) -> int:
    dt_raw = float(yaml.safe_load(TEL_CFG.read_text())[telescope]["dt_ms_raw"])
    return max(1, int(round(TARGET_DT_MS / dt_raw)))


def _dump_module():
    spec = importlib.util.spec_from_file_location("dump_jointmodel", HERE / "dump_jointmodel.py")
    mod = importlib.util.module_from_spec(spec)
    os.environ.setdefault("FLITS_REPO", str(REPO))
    os.environ.setdefault("FLITS_RUNS", str(RUNS))
    sys.path.insert(0, str(REPO / "scattering"))
    spec.loader.exec_module(mod)
    return mod


def _profile_peak_t(data, valid, time) -> float:
    pd = np.nansum(data[valid], axis=0)
    return float(np.asarray(time, float)[int(np.nanargmax(pd))])


def _crop_frame_delta(z, band: str, model) -> float:
    """Align fine-reload crop to npz frame (center_burst shifts with t_factor)."""
    valid = z[f"valid{band}"].astype(bool)
    t_npz = np.asarray(z[f"time{band}"], float)
    t_fine = np.asarray(model.time, float)
    peak_npz = _profile_peak_t(z[f"data{band}"], valid, t_npz)
    peak_fine = _profile_peak_t(model.data, np.ones(model.data.shape[0], dtype=bool), t_fine)
    return peak_npz - peak_fine


def _band_from_rec(rec: dict, *, t_shift: float = 0.0) -> dict:
    d, m = rec["data"], rec["model"]
    t = np.asarray(rec["time"], float) + t_shift
    f = np.asarray(rec["freq"], float)
    sig = np.asarray(rec["noise"], float).reshape(-1)
    valid = np.asarray(rec["valid"]).astype(bool)
    finite = d[np.isfinite(d)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    resid = (d - m) / sig[:, None]
    rr = (
        np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99)
        if np.isfinite(resid).any()
        else 1.0
    )
    pd = np.nansum(d[valid], axis=0)
    pm = np.nansum(m[valid], axis=0)
    dt = float(np.median(np.diff(t))) if t.size > 1 else TARGET_DT_MS
    return dict(d=d, m=m, f=f, t=t, resid=resid, vmin=vmin, vmax=vmax, rr=rr, pd=pd, pm=pm, dt=dt, noise=sig)


def _reload_fine_bands(z, npz_fp: Path, burst: str, shift_c: float) -> dict[str, dict] | None:
    """Reload cubes at native freq (f_factor=1) and ~32 µs time. None if configs missing."""
    fit_fp = _fit_json_for(npz_fp)
    cfg_d = RUNS / "configs" / f"{burst}_dsa_run.yaml"
    cfg_c = RUNS / "configs" / f"{burst}_chime_run.yaml"
    if not (fit_fp and fit_fp.exists() and cfg_d.exists() and cfg_c.exists()):
        return None
    djm = _dump_module()
    fit = json.loads(fit_fp.read_text())
    p = {k: v["median"] for k, v in fit["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    nC, nD = int(fit.get("components_C", 1)), int(fit.get("components_D", 1))
    outdir = str(RUNS / "data/joint")

    def prep(cfg_path, name, tel_key, pbf, beta):
        cfg = yaml.safe_load(open(cfg_path))
        cfg = dict(cfg)
        cfg["f_factor"] = 1
        cfg["t_factor"] = _t_factor_for(tel_key)
        tmp = Path(outdir) / f".plot_{burst}_{name}_run.yaml"
        tmp.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=True))
        return djm.prepare(str(tmp), name, outdir, pbf, beta)

    mC = prep(cfg_c, f"{burst}_chime", "chime", fit.get("pbf_C", "exp"), fit.get("beta_C"))
    mD = prep(cfg_d, f"{burst}_dsa", "dsa", fit.get("pbf_D", "exp"), fit.get("beta_D"))
    dC = _crop_frame_delta(z, "C", mC)
    dD = _crop_frame_delta(z, "D", mD)
    p = dict(p)
    p["t0_C"] = p["t0_C"] - dC
    p["t0_D"] = p["t0_D"] - dD
    if fit.get("shared_zeta"):
        from scat_analysis.burstfit import FRBParams

        zC = p["zeta_1ghz"] * np.asarray(mC.freq, float) ** p["x_zeta"]
        zD = p["zeta_1ghz"] * np.asarray(mD.freq, float) ** p["x_zeta"]
        psC = [FRBParams(c0=1.0, t0=p["t0_C"], gamma=0.0, zeta=zC, tau_1ghz=tau, alpha=al, delta_dm=p.get("delta_dm_C", 0.0))]
        psD = [FRBParams(c0=1.0, t0=p["t0_D"], gamma=0.0, zeta=zD, tau_1ghz=tau, alpha=al, delta_dm=p.get("delta_dm_D", 0.0))]
    else:
        psC = djm.band_params(p, "C", nC, tau, al)
        psD = djm.band_params(p, "D", nD, tau, al)
    C, _ = djm.recover(mC, psC)
    D, _ = djm.recover(mD, psD)
    return {
        "D": _band_from_rec(D, t_shift=dD),
        "C": _band_from_rec(C, t_shift=shift_c + dC),
    }


def _band(z, band: str, *, t_shift: float = 0.0):
    d = z[f"data{band}"]
    m = z[f"model{band}"]
    f = z[f"freq{band}"]
    t = np.asarray(z[f"time{band}"], float) + t_shift
    sig = z[f"noise{band}"]
    valid = z[f"valid{band}"].astype(bool)
    finite = d[np.isfinite(d)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    resid = (d - m) / sig[:, None]
    rr = (
        np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99)
        if np.isfinite(resid).any()
        else 1.0
    )
    pd = np.nansum(d[valid], axis=0)
    pm = np.nansum(m[valid], axis=0)
    return dict(d=d, m=m, f=f, t=t, resid=resid, vmin=vmin, vmax=vmax, rr=rr, pd=pd, pm=pm)


def _bands(z, npz_fp: Path, burst: str):
    shift_c, offset_ms, align_note = _chime_time_shift(npz_fp, burst)
    fine = _reload_fine_bands(z, npz_fp, burst, shift_c)
    if fine is not None:
        out = fine
        dt_note = (
            f"dt D={out['D']['dt']*1e3:.1f} µs C={out['C']['dt']*1e3:.1f} µs; "
            f"n_f D={out['D']['d'].shape[0]} C={out['C']['d'].shape[0]} (native)"
        )
        align_note = f"{align_note}; {dt_note}" if align_note else dt_note
    else:
        shifts = {"D": 0.0, "C": shift_c}
        out = {band: _band(z, band, t_shift=shifts[band]) for band in "DC"}
    out["C"] = _resample_band_to_dsa_df(out["C"], out["D"])  # native CHIME → DSA Δf grid
    xlim = _tight_xlim(out)
    return out, xlim, offset_ms, align_note


def _dsa_df_ghz(b_d: dict) -> float:
    f = np.asarray(b_d["f"], float)
    return float(np.median(np.diff(f)))


def _freq_grid(lo: float, hi: float, df: float) -> np.ndarray:
    n = max(2, int(round((hi - lo) / df)) + 1)
    f = lo + np.arange(n) * df
    if f[-1] > hi + 0.5 * df:
        f = f[f <= hi + 1e-9]
    if f.size < 2 or f[-1] < hi - 0.25 * df:
        f = np.linspace(lo, hi, n)
    return f


def _interp_along_freq(f_out, f_in, arr):
    f_in = np.asarray(f_in, float)
    f_out = np.asarray(f_out, float)
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        return np.interp(f_out, f_in, arr)
    out = np.empty((len(f_out), arr.shape[1]))
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(f_out, f_in, arr[:, j], left=np.nan, right=np.nan)
    return out


def _resample_band_to_dsa_df(b_chime: dict, b_dsa: dict) -> dict:
    """Interpolate CHIME cubes onto DSA channel spacing (uniform GHz step)."""
    df = _dsa_df_ghz(b_dsa)
    f_new = _freq_grid(float(b_chime["f"][0]), float(b_chime["f"][-1]), df)
    d = _interp_along_freq(f_new, b_chime["f"], b_chime["d"])
    m = _interp_along_freq(f_new, b_chime["f"], b_chime["m"])
    resid = _interp_along_freq(f_new, b_chime["f"], b_chime["resid"])
    noise = _interp_along_freq(f_new, b_chime["f"], np.asarray(b_chime["noise"], float))
    finite = d[np.isfinite(d)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (b_chime["vmin"], b_chime["vmax"])
    rr = (
        np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99)
        if np.isfinite(resid).any()
        else b_chime["rr"]
    )
    valid = np.isfinite(d) & np.isfinite(m)
    pd = np.nansum(np.where(valid, d, np.nan), axis=0)
    pm = np.nansum(np.where(valid, m, np.nan), axis=0)
    out = dict(b_chime)
    out.update(dict(d=d, m=m, f=f_new, resid=resid, noise=noise, vmin=vmin, vmax=vmax, rr=rr, pd=pd, pm=pm))
    return out


GAP_ROW_RATIO = 0.10  # fixed visual strip (not proportional to native channel count)


def _tight_xlim(bands: dict, *, pad_ms: float = 1.0, floor_frac: float = 0.08) -> tuple[float, float]:
    """Union of on-pulse windows across bands (minimize off-pulse)."""
    spans: list[tuple[float, float]] = []
    for b in bands.values():
        t = np.asarray(b["t"], float)
        pd = np.asarray(b["pd"], float)
        if t.size == 0 or pd.size == 0:
            continue
        peak = float(np.max(pd))
        thr = max(floor_frac * peak, float(np.median(pd)))
        on = pd >= thr
        if not np.any(on):
            pk = int(np.argmax(pd))
            spans.append((float(t[max(0, pk - 8)]), float(t[min(len(t) - 1, pk + 8)])))
            continue
        idx = np.where(on)[0]
        spans.append((float(t[idx[0]]), float(t[idx[-1]])))
    if not spans:
        return 0.0, 1.0
    t0 = min(s[0] for s in spans) - pad_ms
    t1 = max(s[1] for s in spans) + pad_ms
    return t0, t1


def _style_gap_ax(ax, xlim):
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticklabels([])
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.add_patch(
        mpatches.Rectangle(
            (xlim[0], 0),
            xlim[1] - xlim[0],
            1,
            facecolor="0.88",
            edgecolor="0.55",
            hatch="///",
            linewidth=0,
            zorder=0,
        )
    )
    ax.text(
        0.5,
        0.5,
        GAP_LABEL,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="0.35",
        zorder=1,
    )


def _add_row_gap(fig, gs, row: int, col_slice, xlim):
    ax = fig.add_subplot(gs[row, col_slice])
    _style_gap_ax(ax, xlim)
    return ax


def _wf(ax, t, f, img, *, vmin, vmax, cmap="magma", xlim=None):
    ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    if xlim is not None:
        ax.set_xlim(xlim)


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (
        ("png", dict(dpi=150)),
        ("pdf", dict(bbox_inches="tight")),
        ("svg", dict(bbox_inches="tight")),
    ):
        fp = out_dir / f"{stem}.{ext}"
        fig.savefig(fp, **kw)
        print(f"wrote {fp}")


def _suptitle(meta, subtitle: str = "", *, offset_ms: float = 0.0, align_note: str = ""):
    b, al, tau, cC, cD = meta["burst"], meta["alpha"], meta["tau"], meta["chiC"], meta["chiD"]
    base = (
        f"{b.capitalize()} — joint fit  "
        rf"$\alpha={al:.3f}$, $\tau_{{1\,\mathrm{{GHz}}}}={tau:.4f}\,\mathrm{{ms}}$  "
        rf"(DSA $\chi^2_\nu={cD:.2f}$, CHIME $\chi^2_\nu={cC:.2f}$)"
    )
    if align_note.startswith("TOA"):
        base += rf"; bands aligned to measured TOA ($\Delta t_{{400}}={offset_ms:.2f}\,\mathrm{{ms}}$)"
    elif abs(offset_ms) > 1e-6:
        base += rf"; CHIME shifted by $\Delta t={offset_ms:.2f}\,\mathrm{{ms}}$ ({align_note})"
    return f"{base}\n{subtitle}".strip()


def plot_jointmodel(meta, out_dir: Path, tag: str, bands, xlim, offset_ms: float, align_note: str):
    """Compact 2×3 — DSA above CHIME; data | model | whitened resid."""
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, GAP_ROW_RATIO, 1], hspace=0.02, wspace=0.2)
    _add_row_gap(fig, gs, 1, slice(None), xlim)
    col_labels = ["data", "model", r"whitened resid"]
    for row, band in zip((0, 2), "DC"):
        b = bands[band]
        panels = [b["d"], b["m"], b["resid"]]
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            if col < 2:
                _wf(ax, b["t"], b["f"], panels[col], vmin=b["vmin"], vmax=b["vmax"], xlim=xlim)
            else:
                _wf(ax, b["t"], b["f"], panels[col], vmin=-b["rr"], vmax=b["rr"], cmap="coolwarm", xlim=xlim)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if row == 2:
                ax.set_xlabel("time (ms)", fontsize=7)
            else:
                ax.set_xticklabels([])
    fig.supylabel("Frequency (GHz)", fontsize=9)
    fig.suptitle(_suptitle(meta, offset_ms=offset_ms, align_note=align_note), fontsize=10)
    fig.subplots_adjust(top=0.88, left=0.07)
    _save(fig, out_dir, f"{tag}_jointmodel_vC")
    plt.close(fig)


def main():
    npz_fp = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else npz_fp.name.split("_jointmodel")[0]
    z, meta = _load(npz_fp)
    bands, xlim, offset_ms, align_note = _bands(z, npz_fp, meta["burst"])
    plot_jointmodel(meta, out_dir, tag, bands, xlim, offset_ms, align_note)


if __name__ == "__main__":
    main()
