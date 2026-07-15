"""Forward-model convolution for per-sightline P(DM_host).

STATUS: SUPERSEDED (2026-07-15). Do NOT use for the manuscript. The published
DM_host forward model (the fig:dm_host_posteriors figure and tab:host-forward-model
table in Faber2026) is produced by the super-repository script
``scripts/dm_budget_uncertainty.py``, which uses the TNG-300 IGM log-normal
calibration (Walker/Connor reproduction package) with an f_IGM marginalization.
This module is the earlier EXPLORATORY convolution built on the Macquart deviate
PDF with the fixed sigma_DM = F z^{-1/2} scaling, and it also carries stale
DM_int inputs (the pre-2026-07-15 census values, e.g. 70/41/84/41 for
20220207C/20221113A/20221203A/20230913A, which the census remediation zeroed).
It was never V-validated and is retained only for provenance of the approach that
the TNG-calibrated script replaced. Nothing produced here is manuscript-quotable.

Model
-----
Observer frame:  DM_obs = DM_MW,disk + DM_MW,halo + DM_cosmic + DM_int
                          + DM_host,rest / (1 + z).

The cosmic term follows the Macquart et al. (2020, Nature 581, 391) deviate
PDF: with Delta = DM_cosmic / <DM_cosmic>(z),

    p(Delta) = A * Delta^-beta * exp( -(Delta^-alpha - C0)^2
                                       / (2 alpha^2 sigma^2) ),

alpha = beta = 3, sigma = F / sqrt(z) (James et al. 2022, MNRAS 509, 4775),
with C0 fixed per sigma by the mean constraint E[Delta] = 1 and A by
normalization; both are computed numerically here (no scipy dependency).

Nuisance parameters are marginalized by Monte Carlo, with priors matched to
the `fiducial_literature` family of sightline_sensitivity.py:

    DM_MW,halo  ~ TruncNormal(40, 15, [10, 100])   pc/cm^3
    disk scale  ~ TruncNormal(1.0, 0.2, [0.5, 1.5])  (multiplies NE2001 disk)
    f_IGM       ~ U(0.75, 0.90)  (Macquart mean scales linearly; table value
                                  was computed at f_IGM = 0.84)
    DM_int      ~ table value x 10^N(0, 0.3 dex)   (only where > 0)
    F           ~ U(0.20, 0.50)  (cosmic fluctuation parameter)

For each draw the host term is DM_host,obs = DM_ext - DM_cosmic with
DM_ext = DM_obs - DM_MW - DM_int, so the draw-conditional likelihood of
DM_host,obs on a grid is p_Delta((DM_ext - DM_host)/mu_c)/mu_c. Averaging
over draws gives the marginal likelihood curve; we report posteriors under
(a) a flat prior on DM_host,obs >= 0 and (b) a rest-frame log-normal host
prior (median 68 pc/cm^3, sigma_ln = 1.0; form per Macquart et al. 2020 --
the specific numbers are an assumption knob to pin at validation time).

Inputs are the V5-cleared tab:budget rows (embedded below); the three
placeholder-redshift sightlines are excluded by construction.

Outputs (results/dmhost_posterior/): per-sightline posterior curves (CSV +
PNG), a summary table (CSV + markdown). Run:  python -m
galaxies.foreground.dm_host_posterior  (or with --n-draws).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import zlib

from galaxies.foreground.sightline_sensitivity import (
    default_prior_families,
    _truncated_normal,
)

try:  # NumPy >= 2.0
    _trapz = np.trapezoid
except AttributeError:  # NumPy 1.x
    _trapz = np.trapz

# --- V5-cleared tab:budget inputs (DM in pc/cm^3, observer frame) ----------
# name, z_host, DM_obs, DM_MW_total (NE2001 disk + 40 halo prior), DM_int
# Fields: name, z, DM_obs, DM_MW_total, DM_int, mass_conf, dm_int_status.
# mass_conf mirrors tab:budget's "mass" column and widens the DM_int scatter
# for assumed-mass halos (0.6 dex vs 0.3 dex measured). dm_int_status records
# why a zero is a zero: "unconstrained" = outside the deep-imaging footprints
# (tab:budget note u) -- DM_int = 0 is a floor, so the DM_host posterior is an
# UPPER BOUND; "model-zero" = confirmed halos whose modeled column vanishes
# under assumed masses (note m) -- same caveat, model-conditional.
SIGHTLINES = [
    ("FRB20220207C", 0.043, 262.0, 116.0, 70.0, "measured", "constrained"),
    ("FRB20220310F", 0.479, 462.0, 86.0, 11.0, "assumed", "constrained"),
    ("FRB20220506D", 0.300, 397.0, 125.0, 0.0, None, "unconstrained"),
    ("FRB20221113A", 0.251, 411.0, 132.0, 41.0, "measured", "constrained"),
    ("FRB20221203A", 0.510, 602.0, 123.0, 84.0, "assumed", "constrained"),
    ("FRB20230307A", 0.271, 610.0, 78.0, 241.0, "assumed", "constrained"),
    ("FRB20230913A", 0.302, 518.0, 115.0, 41.0, "assumed", "constrained"),
    ("FRB20240203A", 0.074, 272.0, 116.0, 0.0, None, "unconstrained+model-zero"),
    ("FRB20240229A", 0.287, 491.0, 78.0, 0.0, None, "model-zero"),
]
DMINT_SIGMA_DEX = {"measured": 0.3, "assumed": 0.6}
# Macquart mean at the host z as tabulated (f_IGM = 0.84 baseline).
MACQUART_MEAN = {
    "FRB20220207C": 36.0,
    "FRB20220310F": 427.0,
    "FRB20220506D": 262.0,
    "FRB20221113A": 217.0,
    "FRB20221203A": 456.0,
    "FRB20230307A": 235.0,
    "FRB20230913A": 264.0,
    "FRB20240203A": 62.0,
    "FRB20240229A": 250.0,
}

FIDUCIAL_HALO = 40.0  # pc/cm^3, the halo prior already inside DM_MW_total
ALPHA = 3.0
BETA = 3.0
HOST_PRIOR_MEDIAN_REST = 68.0  # pc/cm^3 (assumption knob; Macquart+20 form)
HOST_PRIOR_SIGMA_LN = 1.0


# --- Macquart deviate PDF ---------------------------------------------------

def _delta_pdf_unnorm(delta: np.ndarray, c0: float, sigma: float) -> np.ndarray:
    """Unnormalized Macquart deviate PDF, alpha = beta = 3."""
    out = np.zeros_like(delta)
    pos = delta > 0
    d = delta[pos]
    out[pos] = d ** -BETA * np.exp(
        -((d ** -ALPHA - c0) ** 2) / (2.0 * ALPHA ** 2 * sigma ** 2)
    )
    return out


def _mean_of_delta(c0: float, sigma: float, grid: np.ndarray) -> float:
    p = _delta_pdf_unnorm(grid, c0, sigma)
    norm = _trapz(p, grid)
    if norm <= 0:
        return np.inf
    return float(_trapz(grid * p, grid) / norm)


def solve_c0(sigma: float, grid: np.ndarray | None = None) -> float:
    """C0 such that E[Delta] = 1, by bisection (E[Delta] decreases with C0)."""
    if grid is None:
        grid = np.linspace(1e-4, 12.0, 6000)
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _mean_of_delta(mid, sigma, grid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DeltaPdf:
    """Normalized deviate PDF with a per-sigma C0/normalization cache."""

    def __init__(self) -> None:
        self._cache: dict[float, tuple[float, float, np.ndarray]] = {}
        self._grid = np.linspace(1e-4, 12.0, 6000)

    def params(self, sigma: float) -> tuple[float, float]:
        key = round(float(sigma), 2)
        if key not in self._cache:
            c0 = solve_c0(key, self._grid)
            norm = float(_trapz(_delta_pdf_unnorm(self._grid, c0, key), self._grid))
            self._cache[key] = (c0, norm, self._grid)
        c0, norm, _ = self._cache[key]
        return c0, norm

    def __call__(self, delta: np.ndarray, sigma: float) -> np.ndarray:
        c0, norm = self.params(sigma)
        return _delta_pdf_unnorm(delta, c0, sigma) / norm


@dataclass
class SightlineResult:
    name: str
    z: float
    point_residual: float
    flat_median: float
    flat_lo68: float
    flat_hi68: float
    lognorm_median: float
    lognorm_lo68: float
    lognorm_hi68: float
    tension_mass: float  # marginal-likelihood mass at DM_host,obs < 0


def _quantiles(grid: np.ndarray, pdf: np.ndarray, qs=(0.16, 0.5, 0.84)):
    cdf = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    return [float(np.interp(q, cdf, grid)) for q in qs]


def run_sightline(name, z, dm_obs, dm_mw_total, dm_int, mu_c_table,
                  mass_conf=None,
                  n_draws=2000, seed=20260707, pdf: DeltaPdf | None = None):
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 10_000)
    pdf = pdf or DeltaPdf()

    fam = default_prior_families()["fiducial_literature"]
    disk0 = dm_mw_total - FIDUCIAL_HALO
    halo = _truncated_normal(rng, fam.dm_mw_halo_mean, fam.dm_mw_halo_sigma,
                             fam.dm_mw_halo_min, fam.dm_mw_halo_max, n_draws)
    # Disk-scale and DM_int-dex knobs are NOT in PriorFamily; local to this
    # module by design (candidates for promotion into the family at V7).
    disk = disk0 * _truncated_normal(rng, 1.0, 0.2, 0.5, 1.5, n_draws)
    f_igm = rng.uniform(fam.f_igm_min, fam.f_igm_max, n_draws)
    F = rng.uniform(0.20, 0.50, n_draws)
    if dm_int > 0:
        sigma_dex = DMINT_SIGMA_DEX.get(mass_conf, 0.6)
        dmint = dm_int * 10 ** rng.normal(0.0, sigma_dex, n_draws)
    else:
        dmint = np.zeros(n_draws)

    # Extended grid: allow negative host values so the likelihood retains the
    # tension mass instead of silently truncating it. The lower bound is
    # adaptive: DM_host = DM_ext - Delta*mu_c, and the deviate PDF is
    # normalized on Delta in (0, 12], so lo = min(DM_ext) - 12*max(mu_c)
    # captures the entire negative tail (review fix: a fixed -200 floor
    # underestimated the reported P(DM_host<0)).
    hi = max(dm_obs, 300.0)
    dm_ext_all = dm_obs - disk - halo - dmint
    mu_c_all = mu_c_table * (f_igm / 0.84)
    lo = float(dm_ext_all.min() - 12.0 * mu_c_all.max())
    n_grid = int(min(9000, max(2200, (hi - lo) / 0.75)))
    grid = np.linspace(lo, hi, n_grid)

    like = np.zeros_like(grid)
    for i in range(n_draws):
        dm_ext = dm_obs - disk[i] - halo[i] - dmint[i]
        mu_c = mu_c_table * (f_igm[i] / 0.84)
        sigma = min(F[i] / np.sqrt(z), 3.0)
        delta = (dm_ext - grid) / mu_c
        like += pdf(delta, sigma) / mu_c
    like /= n_draws

    tension = float(
        _trapz(like[grid < 0], grid[grid < 0]) / _trapz(like, grid)
    )

    # (a) flat prior on DM_host,obs >= 0
    pos = grid >= 0
    flat_post = np.where(pos, like, 0.0)
    flat_q = _quantiles(grid[pos], flat_post[pos])

    # (b) rest-frame log-normal prior, observer frame via x_rest = x*(1+z)
    mu_ln = np.log(HOST_PRIOR_MEDIAN_REST)
    x = np.where(grid > 0, grid * (1.0 + z), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        prior = np.where(
            grid > 0,
            np.exp(-((np.log(x) - mu_ln) ** 2) / (2 * HOST_PRIOR_SIGMA_LN ** 2))
            / x,
            0.0,
        )
    ln_post = np.nan_to_num(like * prior)
    ln_q = _quantiles(grid[pos], ln_post[pos])

    point = dm_obs - dm_mw_total - mu_c_table - dm_int
    return SightlineResult(name, z, point, flat_q[1], flat_q[0], flat_q[2],
                           ln_q[1], ln_q[0], ln_q[2], tension), grid, like, \
        flat_post, ln_post


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir or os.path.join(repo, "results", "dmhost_posterior")
    os.makedirs(out_dir, exist_ok=True)

    pdf = DeltaPdf()
    rows = []
    curves = {}
    status_by_name = {}
    for name, z, dm_obs, dm_mw, dm_int, mass_conf, dm_int_status in SIGHTLINES:
        status_by_name[name] = dm_int_status
        res, grid, like, flat_post, ln_post = run_sightline(
            name, z, dm_obs, dm_mw, dm_int, MACQUART_MEAN[name],
            mass_conf=mass_conf, n_draws=args.n_draws, pdf=pdf)
        rows.append(res)
        curves[name] = (grid, like, flat_post, ln_post)
        print(f"{name}: point={res.point_residual:+7.1f}  "
              f"flat={res.flat_median:6.1f} [{res.flat_lo68:.1f},{res.flat_hi68:.1f}]  "
              f"logn={res.lognorm_median:6.1f} [{res.lognorm_lo68:.1f},{res.lognorm_hi68:.1f}]  "
              f"P(<0)={res.tension_mass:.2f}")

    # summary CSV + markdown
    import csv
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "z", "point_residual", "flat_median", "flat_lo68",
                    "flat_hi68", "lognorm_median", "lognorm_lo68",
                    "lognorm_hi68", "tension_mass", "dm_int_status"])
        for r in rows:
            w.writerow([r.name, r.z, f"{r.point_residual:.1f}",
                        f"{r.flat_median:.1f}", f"{r.flat_lo68:.1f}",
                        f"{r.flat_hi68:.1f}", f"{r.lognorm_median:.1f}",
                        f"{r.lognorm_lo68:.1f}", f"{r.lognorm_hi68:.1f}",
                        f"{r.tension_mass:.3f}", status_by_name[r.name]])

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# DM_host posterior summary (EXPLORATORY - not V-validated)\n\n")
        f.write("Observer-frame DM_host in pc/cm^3; multiply by (1+z) for rest frame.\n"
                "Rows with DM_int status unconstrained/model-zero treat DM_int=0 as a\n"
                "floor, so their DM_host posteriors are UPPER BOUNDS.\n\n")
        f.write("| sightline | z | point residual | flat-prior median [68%] | "
                "lognormal-prior median [68%] | P(DM_host<0) | DM_int status |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r.name} | {r.z:.3f} | {r.point_residual:+.1f} | "
                    f"{r.flat_median:.1f} [{r.flat_lo68:.1f}, {r.flat_hi68:.1f}] | "
                    f"{r.lognorm_median:.1f} [{r.lognorm_lo68:.1f}, {r.lognorm_hi68:.1f}] | "
                    f"{r.tension_mass:.3f} | {status_by_name[r.name]} |\n")

    # per-sightline curve CSVs + figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False)
    for ax, (name, (grid, like, flat_post, ln_post)) in zip(
            axes.ravel(), curves.items()):
        np.savetxt(os.path.join(out_dir, f"{name}_curves.csv"),
                   np.column_stack([grid, like, flat_post, ln_post]),
                   delimiter=",",
                   header="dm_host_obs,marginal_like,flat_posterior,lognormal_posterior",
                   comments="")
        for y, lab, sty in [(like / like.max(), "marginal likelihood", "-"),
                            (ln_post / ln_post.max(), "lognormal-prior posterior", "--")]:
            ax.plot(grid, y, sty, lw=1.2, label=lab)
        ax.axvline(0, color="0.6", lw=0.7)
        r = next(r for r in rows if r.name == name)
        ax.axvline(r.point_residual, color="tab:red", lw=0.9, ls=":",
                   label="point residual")
        ax.set_title(f"{name} (z={r.z})", fontsize=9)
        ax.set_xlim(-150, max(300.0, r.flat_hi68 * 2.5))
        ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel(r"DM$_{\rm host}$ (observer frame, pc cm$^{-3}$)")
    fig.suptitle("P(DM_host) forward-model convolution -- EXPLORATORY, not V-validated",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dmhost_posteriors.png"), dpi=170)
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
