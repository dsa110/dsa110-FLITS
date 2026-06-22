import sys
import numpy as np
from dataclasses import replace

sys.path.insert(0, "/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams

rng = np.random.default_rng(7)

# --- build a realistic small FRBModel with injected scattered burst ---
nf, nt = 48, 256
freq = np.linspace(1.30, 1.50, nf)          # GHz, ascending
time = np.linspace(-5.0, 20.0, nt)          # ms
chan_w_MHz = (freq[1] - freq[0]) * 1e3

p_true = FRBParams(c0=1.0, t0=2.0, gamma=0.0, zeta=0.4, tau_1ghz=1.5, alpha=4.0, delta_dm=0.0)

# unit kernel from the model itself, then impose a spectrum + scintillation gain
m0 = FRBModel(time=time, freq=freq, data=np.zeros((nf, nt)), noise_std=np.ones(nf))
K = m0(replace(p_true, c0=1.0, gamma=0.0), "M3")     # (nf, nt) unit kernels
# smooth spectrum * scintillation modulation
spec = 3.0 * (freq / 1.4) ** -1.5
nu_MHz = freq * 1e3
# correlated gain with ~6 MHz scale
dd = nu_MHz[:, None] - nu_MHz[None, :]
Cg = 1.0 / (1.0 + (dd / 6.0) ** 2)
Lg = np.linalg.cholesky(Cg + 1e-9 * np.eye(nf))
g = spec * (1.0 + 0.3 * (Lg @ rng.standard_normal(nf)))
sigma = 0.25
data = g[:, None] * K + sigma * rng.standard_normal((nf, nt))
model = FRBModel(time=time, freq=freq, data=data, noise_std=np.full(nf, sigma))

print(f"nf={nf} n_valid={int(np.sum(model.valid))} chan_w={chan_w_MHz:.3f} MHz")

# evaluate both likelihoods over a grid of theta (vary tau, alpha, t0, zeta)
def make_params(i):
    return FRBParams(
        c0=1.0, t0=2.0 + 0.4 * np.sin(i), gamma=0.0,
        zeta=0.4 + 0.05 * np.cos(i),
        tau_1ghz=1.5 * (1.0 + 0.25 * np.sin(2 * i)),
        alpha=4.0 + 0.3 * np.cos(2 * i), delta_dm=0.0,
    )

thetas = [make_params(i) for i in np.linspace(0, 6, 25)]

# ===== (b1) None-dispatch must be EXACT-equal to the flat marginal =====
d_none = []
for th in thetas:
    a = model.log_likelihood_gain_marginal(th, "M3")
    b = model.log_likelihood_gain_marginal_gp(th, "M3", delta_nu_d_MHz=None)
    d_none.append(abs(a - b))
print(f"(b1) None-dispatch max|diff| = {max(d_none):.3e}  (expect 0 exactly)")

# ===== (b2) flat-prior limit: C->I (dnu_d << chan width => decorrelated) AND
#          sigma_g^2 -> inf. Then the GP gain prior becomes independent + flat per
#          channel, so logL_gp should equal flat + a THETA-INDEPENDENT constant. =====
dnu_small = 1e-4 * chan_w_MHz        # << channel width: off-diagonal C ~ 0 => C ~ I
sg2_big = 1e10                       # flat (improper) prior amplitude

flat = np.array([model.log_likelihood_gain_marginal(th, "M3") for th in thetas])
gp = np.array([model.log_likelihood_gain_marginal_gp(
        th, "M3", delta_nu_d_MHz=dnu_small, mu_degree=1, sigma_g2=sg2_big)
        for th in thetas])

diff = gp - flat
c = np.mean(diff)                    # theta-independent additive constant
resid = diff - c
swing = flat.max() - flat.min()
print(f"(b2) flat swing over grid          = {swing:.4f}")
print(f"(b2) additive const (mean gp-flat) = {c:.6f}")
print(f"(b2) max|resid| after removing const = {np.max(np.abs(resid)):.6e}")
print(f"(b2) std(resid)                      = {np.std(resid):.6e}")
print(f"(b2) rel = max|resid|/swing          = {100*np.max(np.abs(resid))/swing:.4f}%")

# tighter: push sigma_g2 even larger and dnu even smaller, check resid shrinks
gp2 = np.array([model.log_likelihood_gain_marginal_gp(
        th, "M3", delta_nu_d_MHz=1e-6*chan_w_MHz, mu_degree=1, sigma_g2=1e14)
        for th in thetas])
resid2 = (gp2 - flat) - np.mean(gp2 - flat)
print(f"(b2-tight) max|resid| (sg2=1e14, dnu=1e-6 chanW) = {np.max(np.abs(resid2)):.6e}")

# print a few raw rows
print("\n idx     flat            gp(sg2=1e10)     gp-flat-c")
for i in range(0, len(thetas), 5):
    print(f" {i:3d}  {flat[i]:14.5f}  {gp[i]:14.5f}  {resid[i]:+.3e}")
