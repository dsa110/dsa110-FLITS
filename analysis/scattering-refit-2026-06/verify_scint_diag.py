"""Diagnose the GP tau bias seen in verify_scint_recovery.

Same injection. Questions:
  A) Is the tau bias a real GP defect or a tau<->dnu_d degeneracy? Re-profile tau
     while RE-PROFILING dnu_d at each tau (joint), not at fixed dnu_hat.
  B) Decompose the GP joint logL(tau) into temporal (shared w/ flat) + spectral
     (GP-only) pieces to localize which block pulls tau down.
  C) Finer tau grid + check the flat path's own tau curvature for reference.
"""
import sys
import numpy as np
from dataclasses import replace

sys.path.insert(0, "/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams, _gp_amplitude_logL

RNG = np.random.default_rng(20260619)
TAU_TRUE, ALPHA_TRUE = 0.85, 4.0
DNU_C_TRUE, DNU_D_TRUE = 12.3, 3.1
SIGMA_G_C, SIGMA_G_D = 0.45, 0.55

def make_band(lo, hi, nf, nt, dt, dfn):
    return np.linspace(lo, hi, nf), np.arange(nt)*dt, dfn
freq_C, time_C, df_C = make_band(0.600, 0.700, 200, 256, 0.05, 0.390625)
freq_D, time_D, df_D = make_band(1.300, 1.500, 240, 256, 0.05, 0.0305)

def lorentz_gains(freq, dnu, sg, ei, rng):
    nu = freq*1e3; d = nu[:,None]-nu[None,:]
    C = 1.0/(1.0+(d/dnu)**2); L,Q = np.linalg.eigh(0.5*(C+C.T)); L=np.clip(L,0,None)
    mod = (Q*np.sqrt(L))@rng.standard_normal(nu.size); mod=mod/(np.std(mod)+1e-30)*sg
    return (freq/np.median(freq))**ei*(1.0+mod)

def build(freq,time,dfn,dm,tau,al,t0,ze,dnu,sg,ei,na,rng):
    base=FRBModel(time,freq,data=None,dm_init=dm,df_MHz=dfn)
    p=FRBParams(c0=1.0,t0=t0,gamma=0.0,zeta=ze,tau_1ghz=tau,alpha=al,delta_dm=0.0)
    K=base(p,"M3"); g=lorentz_gains(freq,dnu,sg,ei,rng)
    data=g[:,None]*K+na*rng.standard_normal(K.shape)
    off=np.r_[0:40,time.size-40:time.size]
    return FRBModel(time,freq,data=data,dm_init=dm,df_MHz=dfn,
                    noise_std=np.full(freq.size,na),off_pulse=off)

T0_C,ZE_C=time_C[128],0.08; T0_D,ZE_D=time_D[128],0.08
mC=build(freq_C,time_C,df_C,0.0,TAU_TRUE,ALPHA_TRUE,T0_C,ZE_C,DNU_C_TRUE,SIGMA_G_C,-1.5,0.30,RNG)
mD=build(freq_D,time_D,df_D,20.0,TAU_TRUE,ALPHA_TRUE,T0_D,ZE_D,DNU_D_TRUE,SIGMA_G_D,-1.5,0.30,RNG)

def pC(tau,al): return FRBParams(c0=1.0,t0=T0_C,gamma=0.0,zeta=ZE_C,tau_1ghz=tau,alpha=al,delta_dm=0.0)
def pD(tau,al): return FRBParams(c0=1.0,t0=T0_D,gamma=0.0,zeta=ZE_D,tau_1ghz=tau,alpha=al,delta_dm=20.0)

dnu_grid_C=np.geomspace(2.0,40.0,40)
dnu_grid_D=np.geomspace(0.6,20.0,40)

def band_gp_profiled(m,p,grid):
    """max over dnu of the GP loglike at fixed p; return (best_ll, best_dnu)."""
    lls=np.array([m.log_likelihood_gain_marginal_gp(p,"M3",delta_nu_d_MHz=float(g),mu_degree=1) for g in grid])
    i=np.argmax(lls); return lls[i], grid[i]

# --- (A) tau profile with dnu RE-PROFILED at each tau (joint GP) ---
tau_grid=np.linspace(0.40,1.40,21)
print("tau   GPjoint(dnu-prof)   flat       dnuC*  dnuD*")
best=(None,-1e99)
for t in tau_grid:
    llC,dC=band_gp_profiled(mC,pC(t,ALPHA_TRUE),dnu_grid_C)
    llD,dD=band_gp_profiled(mD,pD(t,ALPHA_TRUE),dnu_grid_D)
    gp=llC+llD
    flat=(mC.log_likelihood_gain_marginal(pC(t,ALPHA_TRUE),"M3")
          +mD.log_likelihood_gain_marginal(pD(t,ALPHA_TRUE),"M3"))
    if gp>best[1]: best=(t,gp)
    print(f"{t:.3f}  {gp:12.1f}  {flat:11.1f}  {dC:5.2f} {dD:5.2f}")
print(f"(A) tau MLE GP(dnu-profiled)={best[0]:.3f}  bias {best[0]-TAU_TRUE:+.3f}")

# --- (B) decompose at injected tau vs biased tau: temporal vs spectral ---
def decompose(m,p,dnu):
    K=m(replace(p,c0=1.0,gamma=0.0),"M3",freq_subset=m.valid)
    d=m.data[m.valid]; sig=np.clip(m.noise_std[m.valid],1e-9,None); var=sig**2
    Sdd=np.einsum("ij,ij->i",d,d); Sdk=np.einsum("ij,ij->i",d,K); Skk=np.einsum("ij,ij->i",K,K)
    ok=Skk>1e-30; Skk_s=np.where(ok,Skk,1.0)
    chi2=np.where(ok,(Sdd-Sdk**2/Skk_s)/var,Sdd/var)
    temporal=float(np.sum(-0.5*chi2)); const=float(np.sum(0.5*np.log(2*np.pi*var)))
    ahat=Sdk[ok]/Skk[ok]; v=var[ok]/Skk[ok]; nu=m.freq[m.valid][ok]*1e3
    lamp,_,_,_=_gp_amplitude_logL(ahat,v,nu,dnu,mu_degree=1)
    return temporal, const, lamp

for label,t in [("true_tau",TAU_TRUE),("biased_tau",0.60)]:
    tC,cC,sC=decompose(mC,pC(t,ALPHA_TRUE),DNU_C_TRUE)
    tD,cD,sD=decompose(mD,pD(t,ALPHA_TRUE),DNU_D_TRUE)
    print(f"(B) {label:10s} tau={t}: temporal={tC+tD:12.1f}  spectral_GP={sC+sD:12.1f}  total={tC+cC+sC+tD+cD+sD:12.1f}")
