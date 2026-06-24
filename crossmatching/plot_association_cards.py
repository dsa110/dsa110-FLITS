"""Per-burst CHIME/DSA association-card figures."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import RegularGridInterpolator

plt.style.use("default")
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from analysis.chime_beam import FWHM_EW_400, FWHM_NS_400
from analysis.dsa_beam import DEFAULT_BEAM, load_power_beam
from analysis.flux_cal import dsa_pointing_dec

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "association_cards"
MANUSCRIPT_OUTDIR = Path("/Users/jakobfaber/Developer/overleaf/Faber2026/figures/association_cards")


def _load_json(name: str):
    return json.loads((HERE / name).read_text())


def _tns_label(nickname: str) -> str:
    try:
        from scattering.scat_analysis.burst_metadata import load_tns_name

        return load_tns_name(nickname)
    except Exception:
        return nickname.capitalize()


def _coord_from_fixture(row: dict) -> SkyCoord:
    return SkyCoord(row["source_coord"], unit=(u.hourangle, u.deg), frame="icrs")


def _to_minutes_grid(center: SkyCoord, span_deg: float, n: int = 180):
    x_arcmin = np.linspace(-span_deg * 30.0, span_deg * 30.0, n)
    y_arcmin = np.linspace(-span_deg * 30.0, span_deg * 30.0, n)
    x_deg = x_arcmin / 60.0
    y_deg = y_arcmin / 60.0
    ra = center.ra.deg + x_deg / math.cos(center.dec.radian)
    dec = center.dec.deg + y_deg
    return x_arcmin, y_arcmin, ra[None, :], dec[:, None]


def _chime_gain_grid(ra_deg, dec_deg, *, ra0_deg: float, dec0_deg: float, freq_mhz: float = 600.0):
    fwhm_ew = FWHM_EW_400 * 400.0 / freq_mhz
    fwhm_ns = FWHM_NS_400 * 400.0 / freq_mhz
    d_ew = (ra_deg - ra0_deg) * np.cos(np.radians(dec0_deg))
    d_ns = dec_deg - dec0_deg
    k = 4.0 * np.log(2.0)
    return np.exp(-k * (d_ew / fwhm_ew) ** 2) * np.exp(-k * (d_ns / fwhm_ns) ** 2)


def _dsa_beam_interpolator():
    if not Path(DEFAULT_BEAM).exists():
        return None
    fz, theta, phi, power = load_power_beam(DEFAULT_BEAM)
    return RegularGridInterpolator((fz, theta, phi), power, bounds_error=False, fill_value=np.nan)


def _dsa_gain_grid(interp, x_arcmin, y_arcmin, *, source_dec_deg: float, pointing_dec_deg: float):
    if interp is None:
        return None
    x_deg = x_arcmin[None, :] / 60.0
    y_deg = y_arcmin[:, None] / 60.0
    ns_offset = source_dec_deg + y_deg - pointing_dec_deg
    ew_offset = x_deg
    theta = np.hypot(ew_offset, ns_offset)
    phi = np.degrees(np.arctan2(ew_offset, ns_offset)) % 360.0
    freq = np.full(theta.shape, 1.4)
    return interp(np.column_stack([freq.ravel(), theta.ravel(), phi.ravel()])).reshape(theta.shape)


def _dm_inset(ax, row: dict, toa: dict):
    if row.get("dm_chime") is None:
        ax.text(
            0.98,
            0.95,
            "DM not constraining",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.85",
                "alpha": 0.92,
            },
        )
        return
    inset = ax.inset_axes([0.60, 0.66, 0.34, 0.25])
    dm_dsa = float(toa["dm"])
    dm_dsa_err = float(toa.get("dm_uncertainty") or 0.1)
    dm_chime = float(row["dm_chime"])
    dm_chime_err = float(row.get("dm_chime_err") or 0.0)
    lo = min(dm_dsa - dm_dsa_err, dm_chime - dm_chime_err)
    hi = max(dm_dsa + dm_dsa_err, dm_chime + dm_chime_err)
    pad = max(0.15, 0.15 * (hi - lo))
    inset.errorbar(dm_chime, 1, xerr=max(dm_chime_err, 0.02), fmt="o", color="#2563eb", ms=4)
    inset.errorbar(dm_dsa, 0, xerr=dm_dsa_err, fmt="o", color="#dc2626", ms=4)
    inset.set_xlim(lo - pad, hi + pad)
    inset.set_ylim(-0.7, 1.7)
    inset.set_yticks([0, 1], ["DSA", "CHIME"], fontsize=6)
    inset.tick_params(axis="x", labelsize=6, length=2)
    inset.set_xlabel(r"DM", fontsize=7, labelpad=1)
    inset.set_title("DM agreement", fontsize=8, pad=1)
    inset.grid(axis="x", alpha=0.25)


def _plot_timing_panel(ax, nickname: str, toa: dict, chime_row: dict):
    residual = float(toa["measured_offset_ms"] - toa["geometric_delay_ms"])
    err = float(np.hypot(toa["combined_dm_uncertainty_ms"], toa.get("fwhm_ms") or 0.0))
    xlim = max(10.0, math.ceil((abs(residual) + err) / 5.0) * 5.0)
    ax.axvspan(-err, err, color="#2563eb", alpha=0.10, label=r"$1\sigma$ budget")
    ax.axvline(0, color="0.05", lw=1.1)
    ax.errorbar(0, 1.0, xerr=err, fmt="o", color="#2563eb", capsize=3, label="CHIME reference")
    ax.errorbar(residual, 0.0, xerr=err, fmt="o", color="#dc2626", capsize=3, label="DSA residual")
    ax.annotate(
        f"DSA - CHIME = {residual:+.2f} ms",
        xy=(residual, 0.0),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=7,
        color="0.15",
    )
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1], ["DSA", "CHIME"])
    ax.set_xlabel("Residual at 400 MHz reference (ms)")
    ax.text(
        0.02,
        0.96,
        _tns_label(nickname),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.85",
            "alpha": 0.92,
        },
    )
    ax.grid(axis="x", alpha=0.25)
    _dm_inset(ax, chime_row, toa)


def _plot_position_panel(
    ax, nickname: str, fixture_row: dict, chime_row: dict, assoc_row: dict, dsa_interp
):
    dsa = _coord_from_fixture(fixture_row)
    chime = SkyCoord(
        chime_row["chime_ra_deg"], chime_row["chime_dec_deg"], unit=u.deg, frame="icrs"
    )
    sep_arcmin = dsa.separation(chime).arcmin
    span_deg = max(0.55, min(2.0, 2.4 * float(assoc_row["position"]["radius_deg"])))

    chime_dx = (chime.ra.deg - dsa.ra.deg) * math.cos(dsa.dec.radian) * 60.0
    chime_dy = (chime.dec.deg - dsa.dec.deg) * 60.0
    loc_radius = float(assoc_row["position"]["radius_deg"]) * 60.0
    ax.add_patch(
        plt.Circle(
            (chime_dx, chime_dy),
            loc_radius,
            facecolor="#2563eb10",
            edgecolor="#2563eb",
            linewidth=1.4,
            label="CHIME localization",
        )
    )
    ax.plot(0, 0, "o", color="#dc2626", ms=5, label="DSA position")
    ax.plot(chime_dx, chime_dy, "o", color="#2563eb", ms=4, label="CHIME tied beam")
    ax.plot([0, chime_dx], [0, chime_dy], color="0.15", lw=0.8)
    ax.text(
        0.03,
        0.04,
        f"separation = {sep_arcmin:.2f} arcmin",
        transform=ax.transAxes,
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.85",
            "alpha": 0.92,
        },
    )
    ax.set_aspect("equal", adjustable="box")
    lim = span_deg * 30.0
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta$RA cos Dec (arcmin)")
    ax.set_ylabel(r"$\Delta$Dec (arcmin)")
    ax.grid(alpha=0.20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=5.6, framealpha=0.92, borderpad=0.3)

    beam_span_deg = 2.6
    bx, by, bra, bdec = _to_minutes_grid(dsa, beam_span_deg, n=150)
    bxx, byy = np.meshgrid(bx, by)
    chime_beam = _chime_gain_grid(
        bra,
        bdec,
        ra0_deg=float(chime.ra.deg),
        dec0_deg=float(chime.dec.deg),
    )
    dsa_beam = _dsa_gain_grid(
        dsa_interp,
        bx,
        by,
        source_dec_deg=float(dsa.dec.deg),
        pointing_dec_deg=dsa_pointing_dec(nickname.lower().replace("ii", "ii")),
    )
    inset = ax.inset_axes([0.06, 0.58, 0.38, 0.34])
    inset.contour(bxx, byy, chime_beam, levels=[0.5], colors=["0.45"], linewidths=1.0)
    if dsa_beam is not None and np.nanmin(dsa_beam) < 0.5 < np.nanmax(dsa_beam):
        inset.contour(
            bxx, byy, dsa_beam, levels=[0.5], colors=["#0e7490"], linestyles="--", linewidths=1.0
        )
    inset.plot(0, 0, "o", color="#dc2626", ms=2.5)
    inset.plot(chime_dx, chime_dy, "o", color="#2563eb", ms=2.2)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xlim(beam_span_deg * 30.0, -beam_span_deg * 30.0)
    inset.set_ylim(-beam_span_deg * 30.0, beam_span_deg * 30.0)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("beam 50%", fontsize=6, pad=1)


def plot_card(
    nickname: str, toa: dict, chime_row: dict, fixture_row: dict, assoc_row: dict, dsa_interp
):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    _plot_timing_panel(axes[0], nickname, toa, chime_row)
    _plot_position_panel(axes[1], nickname, fixture_row, chime_row, assoc_row, dsa_interp)
    return fig


def main() -> None:
    toa = _load_json("toa_crossmatch_results.json")
    assoc = {row["name"]: row for row in _load_json("association_report.json")["bursts"]}
    chime = {row["name"]: row for row in _load_json("chime_side_inputs.json")}
    fixture = {
        row["name"]: row for row in _load_json("notebook_reproduction_fixture.json")["bursts"]
    }
    dsa_interp = _dsa_beam_interpolator()

    OUTDIR.mkdir(exist_ok=True)
    MANUSCRIPT_OUTDIR.mkdir(parents=True, exist_ok=True)
    names = list(fixture)
    for name in names:
        fig = plot_card(name, toa[name], chime[name], fixture[name], assoc[name], dsa_interp)
        stem = f"association_card_{name.lower()}"
        for ext in ("pdf", "png"):
            out = OUTDIR / f"{stem}.{ext}"
            fig.savefig(out, dpi=300)
            if ext == "pdf":
                shutil.copy2(out, MANUSCRIPT_OUTDIR / out.name)
        plt.close(fig)
    print(f"wrote {len(names)} cards to {OUTDIR} and copied PDFs to {MANUSCRIPT_OUTDIR}")


if __name__ == "__main__":
    main()
