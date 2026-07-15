# DIAGNOSTIC (outside the frozen Gate-0b spec): full-band transform WITHOUT
# block demeaning -- answers "did demeaning kill the wide-scintle window?"
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


def fullband_nodemean(field):
    x = np.asarray(field, float)
    x = x - np.nanmean(x)  # remove global mean only (DC)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.fft.rfft(x)


def cross(f1, f2):
    return np.real(fullband_nodemean(f1) * np.conj(fullband_nodemean(f2)))[1:]


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
out = {}
for di, dnu in enumerate(g0b.DNU_KHZ):
    acc = None
    for j in range(g0b.N_TEMPLATE):
        rng = np.random.default_rng(g0b.TEMPLATE_SEED_BASE + 1000 * di + j)
        d = rb.lorentzian_gain_field(
            rng, n_channels=products.n_band_channels, width_channels=p2._width_channels(dnu)
    ).astype(float)
        d[bad] = np.nan
        pw = cross(d, d)
        acc = pw if acc is None else acc + pw
    T = acc / g0b.N_TEMPLATE
    sig = float(np.sum(T**2 / var) ** -0.5)
    out[dnu] = {m: (g0b.F_B * m) ** 2 / sig for m in g0b.MODULATIONS}
result = {
    "diagnostic": "fullband_no_block_demean",
    "snr_m017": {str(k): round(v[0.17], 3) for k, v in out.items()},
}
print(json.dumps(result))
Path(HERE / "gate0b_nodemean_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
