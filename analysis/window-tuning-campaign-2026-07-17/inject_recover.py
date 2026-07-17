"""Independent injection-recovery test of PR-192's _fit_subband.
Builds known-truth synthetic ACFs (Lorentzian scintle +/- broad envelope + noise),
runs the PR fitter, and checks: gamma recovery, 2-comp vs 1-comp under envelope,
and the dBIC>=6 false-positive rate on envelope-only/noise-only ACFs."""
import sys, os, numpy as np
sys.path.insert(0, "scint_analysis")
import window_refit as wr

rng = np.random.default_rng(20260717)
LAGS = np.arange(0, wr.LAG_MAX + 1e-9, 0.0061)   # 6.1 kHz-like lag grid (finely upchan)
def lor(l, A, g): return A / (1 + (l/g)**2)

def make_acf(gamma_n, m_n, gamma_b=None, m_b=0.0, noise=0.01):
    """ACF amplitude = m^2 (matches pipeline norm). Optional broad envelope + white noise."""
    a = lor(LAGS, m_n**2, gamma_n)
    if gamma_b: a = a + lor(LAGS, m_b**2, gamma_b)
    a = a + rng.normal(0, noise, LAGS.size)
    return a

def fit(acf): return wr._fit_subband(LAGS, acf)

# ---- TEST 1: gamma recovery, isolated scintle (no envelope) ----
print("=== T1: isolated scintle, gamma recovery (m=0.7, noise=0.01, 200 trials each) ===")
print(f"{'gamma_true':>10} {'gamma_med':>10} {'ratio':>7} {'resolved%':>9} {'model':>6}")
for gt in [0.02, 0.05, 0.1, 0.3, 1.0, 3.0]:
    gs, res, mdl = [], 0, []
    for _ in range(200):
        f = fit(make_acf(gt, 0.7, noise=0.01))
        if f["ok"]:
            if f["resolved"]: res += 1; gs.append(f["gamma"])
            mdl.append(f["model_sel"])
    gm = np.median(gs) if gs else np.nan
    frac2 = 100*sum(x=="2L" for x in mdl)/max(len(mdl),1)
    print(f"{gt:>10.3f} {gm:>10.4f} {gm/gt if gs else np.nan:>7.2f} {100*res/200:>8.0f}% {frac2:>5.0f}%2L")

# ---- TEST 2: scintle UNDER a broad envelope — the case single-Lorentzian censors ----
print("\n=== T2: narrow scintle (gamma=0.08, m=0.6) under broad envelope (gamma_b, m_b=0.7) ===")
print(f"{'gamma_b':>8} {'1L_gamma':>9} {'1L_ratio':>9} {'2L_gamma':>9} {'2L_ratio':>9} {'2L_adopted%':>11}")
for gb in [1.0, 2.0, 5.0]:
    g1,g2,adopt = [],[],0
    for _ in range(200):
        acf = make_acf(0.08, 0.6, gamma_b=gb, m_b=0.7, noise=0.01)
        f = fit(acf)
        if not f["ok"]: continue
        if f["model_sel"]=="2L": adopt+=1; g2.append(f["gamma"])
        else: g1.append(f["gamma"])
    m1=np.median(g1) if g1 else np.nan; m2=np.median(g2) if g2 else np.nan
    print(f"{gb:>8.1f} {m1:>9.4f} {m1/0.08 if g1 else np.nan:>9.2f} {m2:>9.4f} {m2/0.08 if g2 else np.nan:>9.2f} {100*adopt/200:>10.0f}%")

# ---- TEST 3: FALSE-POSITIVE rate of the dBIC>=6 2-comp gate on NON-scintle ACFs ----
print("\n=== T3: false 2L-split rate on envelope-only + noise-only (150 trials each) ===")
# envelope only (broad Lorentzian, no narrow scintle)
fp_env = sum(1 for _ in range(150) if (r:=fit(make_acf(3.0, 0.7, noise=0.01)))["ok"] and r["model_sel"]=="2L")
# pure noise
fp_noise = sum(1 for _ in range(150) if (r:=fit(rng.normal(0,0.01,LAGS.size)))["ok"] and r["model_sel"]=="2L")
print(f"envelope-only false 2L: {fp_env}/150 ({100*fp_env/150:.1f}%)")
print(f"noise-only false 2L:    {fp_noise}/150 ({100*fp_noise/150:.1f}%)")

# ---- TEST 4: modulation-index recovery (PR claims m from A_n unbiased where 1L inflates) ----
print("\n=== T4: m recovery for scintle under envelope (truth m_n) ===")
print(f"{'m_true':>7} {'m_2L_med':>9} {'ratio':>7}")
for mt in [0.6, 0.9]:
    ms=[]
    for _ in range(200):
        f = fit(make_acf(0.08, mt, gamma_b=2.0, m_b=0.7, noise=0.01))
        if f["ok"] and f["model_sel"]=="2L": ms.append(f["m"])
    mm=np.median(ms) if ms else np.nan
    print(f"{mt:>7.2f} {mm:>9.3f} {mm/mt if ms else np.nan:>7.2f}")
