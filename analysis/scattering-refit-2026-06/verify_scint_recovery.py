"""Independent adversarial verify: scint GP recovery + tau/alpha unbiasedness.

Different seed (20260619) and Delta_nu_d values than the implementer used
(they injected 25.20 / 5.90 MHz; here 12.3 MHz CHIME, 3.1 MHz DSA).

Checks:
  (1) Per-band: GP likelihood profiled over a Delta_nu_d grid peaks near the
      injected value, for BOTH bands independently.
  (2) The 10-D joint GP loglike, profiled over (tau, alpha), recovers the
      injected tau/alpha and does NOT bias them relative to the no-scint truth
      (i.e. the (tau,alpha) MLE under the scintillated data sits at the truth).
  (3) Cross-check: scattering goodness still discriminates wrong tau/alpha.
"""
import sys
import numpy as np
from dataclasses import replace

sys.path.insert(0, "/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams, _gp_amplitude_logL
from scat_analysis.burstfit_joint import _JointLogLikelihoodGainGP

RNG = np.random.default_rng(20260619)

# ---- injected truth (NEW values, different from implementer) ----
TAU_TRUE = 0.85      # ms at 1 GHz
ALPHA_TRUE = 4.0
DNU_C_TRUE = 12.3    # MHz  (CHIME band)
DNU_D_TRUE = 3.1     # MHz  (DSA band)
SIGMA_G_C = 0.45     # rms fractional scint modulation about envelope
SIGMA_G_D = 0.55

# ---- axes (coarse but resolvable for both injected dnu_d) ----
# CHIME-like band 600-700 MHz, DSA-like band 1300-1500 MHz.
def make_band(f_lo_GHz, f_hi_GHz, n_freq, n_time, dt_ms, df_native_MHz):
    freq = np.linspace(f_lo_GHz, f_hi_GHz, n_freq)   # ascending
    time = np.arange(n_time) * dt_ms
    return freq, time, df_native_MHz

freq_C, time_C, df_C = make_band(0.600, 0.700, 200, 256, 0.05, 0.390625)
freq_D, time_D, df_D = make_band(1.300, 1.500, 240, 256, 0.05, 0.0305)

def lorentz_gains(freq_GHz, dnu_d_MHz, sigma_g, envelope_idx, rng):
    """Draw a frequency-correlated gain spectrum: smooth power-law envelope *
    (1 + Lorentzian-correlated zero-mean modulation)."""
    nu = freq_GHz * 1e3  # MHz
    d = nu[:, None] - nu[None, :]
    C = 1.0 / (1.0 + (d / dnu_d_MHz) ** 2)
    # PSD draw of correlated modulation
    L, Q = np.linalg.eigh(0.5 * (C + C.T))
    L = np.clip(L, 0, None)
    z = rng.standard_normal(nu.size)
    mod = (Q * np.sqrt(L)) @ z
    mod = mod / (np.std(mod) + 1e-30) * sigma_g    # set rms = sigma_g
    env = (freq_GHz / np.median(freq_GHz)) ** envelope_idx
    return env * (1.0 + mod)

def build_data(freq, time, df_native, dt, dm_init, tau, alpha, t0, zeta,
               dnu_d, sigma_g, env_idx, noise_amp, rng):
    # unit-amplitude model with c0=1, gamma=0 -> per-channel kernel K_f(t)
    base = FRBModel(time, freq, data=None, dm_init=dm_init, df_MHz=df_native)
    p = FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=zeta, tau_1ghz=tau,
                  alpha=alpha, delta_dm=0.0)
    K = base(p, "M3")                       # (n_freq, n_time), unit per channel
    g = lorentz_gains(freq, dnu_d, sigma_g, env_idx, rng)   # injected gains
    clean = g[:, None] * K
    noise = noise_amp * rng.standard_normal(clean.shape)
    data = clean + noise
    off = np.r_[0:40, time.size - 40:time.size]
    m = FRBModel(time, freq, data=data, dm_init=dm_init, df_MHz=df_native,
                 noise_std=np.full(freq.size, noise_amp), off_pulse=off)
    return m, g, p

# Common injected kinematics
T0_C, ZETA_C = time_C[128], 0.08
T0_D, ZETA_D = time_D[128], 0.08

mC, gC, pC_true = build_data(freq_C, time_C, df_C, 0.05, dm_init=0.0,
                             tau=TAU_TRUE, alpha=ALPHA_TRUE, t0=T0_C, zeta=ZETA_C,
                             dnu_d=DNU_C_TRUE, sigma_g=SIGMA_G_C, env_idx=-1.5,
                             noise_amp=0.30, rng=RNG)
mD, gD, pD_true = build_data(freq_D, time_D, df_D, 0.05, dm_init=20.0,
                             tau=TAU_TRUE, alpha=ALPHA_TRUE, t0=T0_D, zeta=ZETA_D,
                             dnu_d=DNU_D_TRUE, sigma_g=SIGMA_G_D, env_idx=-1.5,
                             noise_amp=0.30, rng=RNG)

print(f"INJECTED: tau={TAU_TRUE} alpha={ALPHA_TRUE} "
      f"dnu_C={DNU_C_TRUE} dnu_D={DNU_D_TRUE} MHz (seed 20260619)")
print(f"CHIME n_freq={freq_C.size} DSA n_freq={freq_D.size}")

# ---- (1) per-band Delta_nu_d recovery via the GP likelihood at TRUE tau/alpha ----
def recover_dnu(model, p_true, grid):
    lls = np.array([model.log_likelihood_gain_marginal_gp(
        p_true, "M3", delta_nu_d_MHz=float(g), mu_degree=1) for g in grid])
    return grid[np.argmax(lls)], lls

grid_C = np.geomspace(2.0, 40.0, 60)
grid_D = np.geomspace(0.6, 20.0, 60)
dnu_C_hat, llC = recover_dnu(mC, pC_true, grid_C)
dnu_D_hat, llD = recover_dnu(mD, pD_true, grid_D)
print(f"(1) dnu_C recovered={dnu_C_hat:.2f} (inj {DNU_C_TRUE}, ratio {dnu_C_hat/DNU_C_TRUE:.2f})")
print(f"(1) dnu_D recovered={dnu_D_hat:.2f} (inj {DNU_D_TRUE}, ratio {dnu_D_hat/DNU_D_TRUE:.2f})")

# ---- (2) tau/alpha unbiasedness via the 10-D joint GP loglike ----
joint = _JointLogLikelihoodGainGP(mC, mD, mu_degree=1)

def joint_ll(tau, alpha, dnuC, dnuD):
    theta = np.array([tau, alpha, T0_C, ZETA_C, 0.0, dnuC,
                      T0_D, ZETA_D, 20.0, dnuD])
    return joint(theta)

# Profile tau over a grid at injected alpha + recovered dnu (scint folded in)
tau_grid = np.linspace(0.40, 1.40, 41)
ll_tau = np.array([joint_ll(t, ALPHA_TRUE, dnu_C_hat, dnu_D_hat) for t in tau_grid])
tau_hat = tau_grid[np.argmax(ll_tau)]

# Profile alpha at injected tau
alpha_grid = np.linspace(2.5, 5.5, 31)
ll_al = np.array([joint_ll(TAU_TRUE, a, dnu_C_hat, dnu_D_hat) for a in alpha_grid])
alpha_hat = alpha_grid[np.argmax(ll_al)]

print(f"(2) tau MLE (scint GP)={tau_hat:.3f}  (inj {TAU_TRUE}, "
      f"bias {tau_hat-TAU_TRUE:+.3f})")
print(f"(2) alpha MLE (scint GP)={alpha_hat:.3f}  (inj {ALPHA_TRUE}, "
      f"bias {alpha_hat-ALPHA_TRUE:+.3f})")

# Compare to FLAT (no-scint) path: does ignoring scint bias tau/alpha?
def joint_flat_ll(tau, alpha):
    pC = FRBParams(c0=1.0, t0=T0_C, gamma=0.0, zeta=ZETA_C, tau_1ghz=tau,
                   alpha=alpha, delta_dm=0.0)
    pD = FRBParams(c0=1.0, t0=T0_D, gamma=0.0, zeta=ZETA_D, tau_1ghz=tau,
                   alpha=alpha, delta_dm=20.0)
    return (mC.log_likelihood_gain_marginal(pC, "M3")
            + mD.log_likelihood_gain_marginal(pD, "M3"))

ll_tau_flat = np.array([joint_flat_ll(t, ALPHA_TRUE) for t in tau_grid])
tau_hat_flat = tau_grid[np.argmax(ll_tau_flat)]
ll_al_flat = np.array([joint_flat_ll(TAU_TRUE, a) for a in alpha_grid])
alpha_hat_flat = alpha_grid[np.argmax(ll_al_flat)]
print(f"(2) tau MLE (flat) ={tau_hat_flat:.3f}   alpha MLE (flat) ={alpha_hat_flat:.3f}")

# ---- (3) discrimination: GP loglike must reject wrong tau/alpha ----
ll_true = joint_ll(TAU_TRUE, ALPHA_TRUE, dnu_C_hat, dnu_D_hat)
ll_wtau = joint_ll(TAU_TRUE * 2.0, ALPHA_TRUE, dnu_C_hat, dnu_D_hat)
ll_wal = joint_ll(TAU_TRUE, ALPHA_TRUE + 1.5, dnu_C_hat, dnu_D_hat)
print(f"(3) GP joint logL: true={ll_true:.1f}  wrong_tau(2x)={ll_wtau:.1f}  "
      f"wrong_alpha(+1.5)={ll_wal:.1f}")

# ---- verdict ----
tol_dnu = 0.5      # factor (recovered within sqrt(2.5) of injected)
tol_tau = 0.075    # ms (~ 1.5 grid steps)
tol_al = 0.30      # ~ 3 grid steps
ok = True
ok &= 1/(1+tol_dnu*5) < dnu_C_hat/DNU_C_TRUE < (1+tol_dnu*5)  # loose factor
ok &= 0.5 < dnu_C_hat/DNU_C_TRUE < 2.0
ok &= 0.5 < dnu_D_hat/DNU_D_TRUE < 2.0
ok &= abs(tau_hat - TAU_TRUE) <= tol_tau
ok &= abs(alpha_hat - ALPHA_TRUE) <= tol_al
ok &= ll_true > ll_wtau and ll_true > ll_wal
print("VERDICT:", "PASS" if ok else "FAIL")
