# DIAGNOSTIC 2 (outside frozen spec): no-demean variant hardening —
# (a) end-to-end multiplicative-injection SNR at the floor cell,
# (b) SNR vs low-delay-bin exclusion (intrinsic-envelope robustness),
# (c) null a_hat mean/sigma calibration.
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("/Users/jakobfaber/Developer/scratch/worktrees/flits-p3-optimal-estimator/analysis/chime-scintillation/experiments/p3-optimal-estimator")
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent / "p2-routeb-voltage"))
sys.path.insert(0, str(HERE))

import routeb_calibration as p2  # noqa: E402
from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402
import gate0b_forecast as g0b  # noqa: E402


class Args:
    pol0 = p2.DEFAULT_POL0
    pol1 = p2.DEFAULT_POL1
    frequencies = p2.DEFAULT_FREQUENCIES
    time0_metadata = p2.DEFAULT_METADATA


products = p2.Products(Args())
pool = products.off_pool
bad = ~products.good_channels


def transform(field):
    x = np.asarray(field, float)
    x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.fft.rfft(x)


def cross(f1, f2):
    return np.real(transform(f1) * np.conj(transform(f2)))[1:]


stack = []
for i in range(g0b.N_NULL):
    rng = np.random.default_rng(g0b.NULL_SEED_BASE + i)
    perm = rng.permutation(pool)
    on = np.sort(perm[: p2.N_ON])
    off = np.sort(perm[p2.N_ON :])
    f1, f2 = g0b.split_ratio_fields(products, on, off)
    stack.append(cross(f1, f2))
stack = np.asarray(stack)
var = g0b.smooth_variance(stack.var(axis=0, ddof=1))

# templates for the two window cells
templates = {}
for di, dnu in enumerate(g0b.DNU_KHZ):
    if dnu not in (127.0, 213.0, 352.0):
        continue
    acc = None
    for j in range(g0b.N_TEMPLATE):
        rng = np.random.default_rng(g0b.TEMPLATE_SEED_BASE + 1000 * di + j)
        d = rb.lorentzian_gain_field(
            rng, n_channels=products.n_band_channels, width_channels=p2._width_channels(dnu)
        ).astype(float)
        d[bad] = np.nan
        pw = cross(d, d)
        acc = pw if acc is None else acc + pw
    templates[dnu] = acc / g0b.N_TEMPLATE

# (b) SNR vs low-k exclusion
excl = {}
for kmin in (1, 6, 11, 21, 51, 101):
    row = {}
    for dnu, T in templates.items():
        sig = float(np.sum(T[kmin - 1 :] ** 2 / var[kmin - 1 :]) ** -0.5)
        row[str(dnu)] = round((g0b.F_B * 0.17) ** 2 / sig, 3)
    excl[str(kmin)] = row

# (c) null calibration + (a) injections at floor cell, kmin=1 and kmin=11
results = {}
for kmin in (1, 11):
    T = templates[213.0][kmin - 1 :]
    w = 1.0 / var[kmin - 1 :]
    denom = float(np.sum(w * T**2))

    def est(power):
        return float(np.sum(w * power[kmin - 1 :] * T) / denom)

    a_null = np.array([est(p) for p in stack])
    injected = []
    for r in range(g0b.N_INJECTION_CHECK):
        rng = np.random.default_rng(g0b.INJECTION_CELL_SEED + r)
        perm = rng.permutation(pool)
        on = np.sort(perm[: p2.N_ON])
        off = np.sort(perm[p2.N_ON :])
        delta = rb.lorentzian_gain_field(
            rng, n_channels=products.n_band_channels, width_channels=p2._width_channels(213.0)
        )
        gain = 1.0 + g0b.F_B * (1.0 + 0.17 * delta)
        f1, f2 = g0b.split_ratio_fields(products, on, off, on_gain=gain)
        injected.append(est(cross(f1, f2)))
    injected = np.array(injected)
    sigma = float(a_null.std(ddof=1))
    results[f"kmin={kmin}"] = {
        "null_mean_over_sigma": round(float(a_null.mean()) / sigma, 3),
        "sigma_empirical": sigma,
        "a_true": (g0b.F_B * 0.17) ** 2,
        "a_hat_median": float(np.median(injected)),
        "empirical_snr_mean_sub": round(float((np.median(injected) - a_null.mean()) / sigma), 3),
    }

out = {"diagnostic": "nodemean_hardening", "snr_vs_kmin_m017": excl, "floor_cell": results}
print(json.dumps(out, indent=1))
Path(HERE / "gate0b_nodemean_diagnostic2.json").write_text(json.dumps(out, indent=2) + "\n")
