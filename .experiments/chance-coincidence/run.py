"""Run the chance-coincidence experiment: analytic vs Monte-Carlo, + sweeps."""

from __future__ import annotations

import estimator_analytic as A
import estimator_mc as B
import inputs as I
import numpy as np

bursts = I.load_bursts()
BASE = dict(
    rate_per_day=I.R_SKY_PER_DAY_CONSERVATIVE,
    omega_win_deg2=I.OMEGA_WIN_BASELINE_DEG2,
    dt_s=I.DT_BASELINE_S,
    ddm=I.DDM_BASELINE,
)

print("=" * 78)
print(
    "INPUTS (conservative): rate=%.0f/sky/day  Omega=%.3f deg^2  dt=+/-%.0fs  ddm=+/-%.0f"
    % (BASE["rate_per_day"], BASE["omega_win_deg2"], BASE["dt_s"], BASE["ddm"])
)
print("=" * 78)

# --- 1. Baseline analytic per burst ----------------------------------------
res = A.run(bursts, **BASE)
print("\n[A] Analytic per-burst false-alarm probability:")
print(f"  {'name':11s} {'DM':>7s} {'mu':>12s} {'P_chance':>12s}")
mus = []
for r in res:
    mus.append(r["mu"])
    print(f"  {r['name']:11s} {r['dm']:7.1f} {r['mu']:12.3e} {r['P']:12.3e}")
sum_mu = sum(mus)
print(f"\n  Expected # chance associations across all 12 (sum mu) = {sum_mu:.3e}")
print(f"  P(>=1 chance assoc in 12) = {1 - np.exp(-sum_mu):.3e}")
print(f"  max single-burst P = {max(r['P'] for r in res):.3e}")

# --- 2. MC <-> analytic cross-validation in a MEASURABLE regime -------------
# Baseline mu ~1e-9 is far below MC sensitivity (need ~1/realisations to see a
# hit). Inflate the window so lambda is measurable, confirm A==B there, then
# trust the closed form in the tiny-mu regime.
infl = dict(rate_per_day=1000.0, omega_win_deg2=200.0, dt_s=3600.0, ddm=50.0)
mu_infl = A.mu_analytic(500.0, **infl)
print("\n" + "=" * 78)
print(f"[A vs B] Cross-validation at inflated window (mu~{mu_infl:.3e}, DM=500):")
one = [{"name": "probe", "dm": 500.0}]
pa = A.run(one, **infl)[0]["P"]
# repeat MC with several seeds -> report mean +/- std (variance, not one run)
ps = []
for s in (1, 2, 3, 4, 5):
    pb = B.run(one, **infl, realisations=2_000_000, seed=s)[0]["P"]
    ps.append(pb)
ps = np.array(ps)
print(f"  Analytic P = {pa:.4e}")
print(f"  MC P       = {ps.mean():.4e} +/- {ps.std():.1e}  (5 seeds, 2e6 each)")
print(f"  ratio MC/analytic = {ps.mean() / pa:.3f}")

# --- 3. Robustness sweeps (analytic; closed form is the trusted estimator) --
print("\n" + "=" * 78)
print("[A] Robustness: max single-burst P and sum-mu vs each window (others at baseline)")


def sweep(label, key, values, fmt="%g"):
    print(f"\n  {label}:")
    for v in values:
        kw = dict(BASE)
        kw[key] = v
        rr = A.run(bursts, **kw)
        print(
            f"    {key}={fmt % v:>10s}  sum_mu={sum(x['mu'] for x in rr):.2e}  "
            f"maxP={max(x['P'] for x in rr):.2e}"
        )


sweep("temporal window dt (s)", "dt_s", [0.001, 0.1, 1.0, 60.0, 3600.0, 86400.0])
sweep("positional window (deg^2)", "omega_win_deg2", [1e-4, 1e-2, 0.785, 9.1, 200.0])
sweep("DM match (pc/cm^3)", "ddm", [0.1, 1.0, 5.0, 50.0])
sweep("CHIME rate (/sky/day)", "rate_per_day", [I.R_SKY_PER_DAY_CENTRAL, 1000.0, 1e4])

print("\nDONE")
