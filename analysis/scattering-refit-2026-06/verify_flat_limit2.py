import sys, numpy as np
from dataclasses import replace
sys.path.insert(0, "/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams, _gp_amplitude_logL

rng = np.random.default_rng(7)
nf, nt = 48, 256
freq = np.linspace(1.30, 1.50, nf); time = np.linspace(-5.0, 20.0, nt)
chan_w_MHz = (freq[1]-freq[0])*1e3
p_true = FRBParams(c0=1.0,t0=2.0,gamma=0.0,zeta=0.4,tau_1ghz=1.5,alpha=4.0,delta_dm=0.0)
m0 = FRBModel(time=time,freq=freq,data=np.zeros((nf,nt)),noise_std=np.ones(nf))
K = m0(replace(p_true,c0=1.0,gamma=0.0),"M3")
spec = 3.0*(freq/1.4)**-1.5; nu_MHz=freq*1e3
dd=nu_MHz[:,None]-nu_MHz[None,:]; Cg=1.0/(1.0+(dd/6.0)**2)
Lg=np.linalg.cholesky(Cg+1e-9*np.eye(nf))
g=spec*(1.0+0.3*(Lg@rng.standard_normal(nf)))
sigma=0.25
data=g[:,None]*K+sigma*rng.standard_normal((nf,nt))
model=FRBModel(time=time,freq=freq,data=data,noise_std=np.full(nf,sigma))

def thetas(i):
    return FRBParams(c0=1.0,t0=2.0+0.4*np.sin(i),gamma=0.0,zeta=0.4+0.05*np.cos(i),
        tau_1ghz=1.5*(1.0+0.25*np.sin(2*i)),alpha=4.0+0.3*np.cos(2*i),delta_dm=0.0)
ths=[thetas(i) for i in np.linspace(0,6,25)]

# Hypothesis: in the C->I, sigma_g2->inf limit, the GP spectral block equals the
# flat path EXCEPT for the GLS mu-profile (-0.5 (ahat-mu)^T diag(1/v)(ahat-mu) replaces
# the implicit mu=0 / per-channel-free flat amplitude) + Jacobian. Quantify how the
# residual scales with mu_degree: a higher-degree poly should make resid theta-dep
# differently (the flat path has NO mu at all -> they CANNOT match to float noise).
for deg in (1, 3, 5):
    flat=np.array([model.log_likelihood_gain_marginal(t,"M3") for t in ths])
    gp=np.array([model.log_likelihood_gain_marginal_gp(t,"M3",
        delta_nu_d_MHz=1e-6*chan_w_MHz,mu_degree=deg,sigma_g2=1e14) for t in ths])
    r=(gp-flat)-np.mean(gp-flat)
    print(f"mu_degree={deg}: max|resid|={np.max(np.abs(r)):.4e}  std={np.std(r):.4e}")

# Direct decomposition at ONE theta: build ahat, v and compare _gp_amplitude_logL
# (C=I, sigma_g2 huge) to the flat spectral block sum.
t=ths[10]
K2=model(replace(t,c0=1.0,gamma=0.0),"M3",freq_subset=model.valid)
d=model.data[model.valid]; sig=np.clip(model.noise_std[model.valid],1e-9,None); var=sig**2
S_dd=np.einsum("ij,ij->i",d,d); S_dk=np.einsum("ij,ij->i",d,K2); S_kk=np.einsum("ij,ij->i",K2,K2)
ok=S_kk>1e-30
ahat=S_dk[ok]/S_kk[ok]; v=var[ok]/S_kk[ok]; nu=model.freq[model.valid][ok]*1e3
# flat spectral block per channel: occam -0.5 ln S_kk  (const +0.5 ln 2pi var handled separately)
flat_spec = float(np.sum(-0.5*np.log(S_kk[ok])))
for sg2 in (1e8,1e12,1e16,1e20):
    lA,_,_,_=_gp_amplitude_logL(ahat,v,nu,1e-6*chan_w_MHz,mu_degree=1,sigma_g2=sg2)
    # subtract the white-noise normalizer the GP adds but flat folds into 'const'
    wn = float(np.sum(-0.5*np.log(2.0*np.pi*v)))
    print(f"sg2={sg2:.0e}: gp_amp={lA:.4f}  gp_amp-wn={lA-wn:.4f}  flat_spec(-0.5 ln Skk)={flat_spec:.4f}  diff={lA-wn-flat_spec:.4f}")
