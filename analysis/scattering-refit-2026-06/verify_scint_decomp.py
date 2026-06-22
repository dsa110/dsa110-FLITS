import sys
import numpy as np
from dataclasses import replace
sys.path.insert(0, "/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams, _gp_amplitude_logL

RNG = np.random.default_rng(20260619)
TAU_TRUE, ALPHA_TRUE = 0.85, 4.0
DNU_C_TRUE, DNU_D_TRUE = 12.3, 3.1
SIGMA_G_C, SIGMA_G_D = 0.45, 0.55
def mb(lo,hi,nf,nt,dt,dfn): return np.linspace(lo,hi,nf),np.arange(nt)*dt,dfn
freq_C,time_C,df_C=mb(0.600,0.700,200,256,0.05,0.390625)
freq_D,time_D,df_D=mb(1.300,1.500,240,256,0.05,0.0305)
def lg(freq,dnu,sg,ei,rng):
    nu=freq*1e3;d=nu[:,None]-nu[None,:];C=1.0/(1.0+(d/dnu)**2)
    L,Q=np.linalg.eigh(0.5*(C+C.T));L=np.clip(L,0,None)
    mod=(Q*np.sqrt(L))@rng.standard_normal(nu.size);mod=mod/(np.std(mod)+1e-30)*sg
    return (freq/np.median(freq))**ei*(1.0+mod)
def bd(freq,time,dfn,dm,tau,al,t0,ze,dnu,sg,ei,na,rng):
    base=FRBModel(time,freq,data=None,dm_init=dm,df_MHz=dfn)
    p=FRBParams(c0=1.0,t0=t0,gamma=0.0,zeta=ze,tau_1ghz=tau,alpha=al,delta_dm=0.0)
    K=base(p,"M3");g=lg(freq,dnu,sg,ei,rng);data=g[:,None]*K+na*rng.standard_normal(K.shape)
    off=np.r_[0:40,time.size-40:time.size]
    return FRBModel(time,freq,data=data,dm_init=dm,df_MHz=dfn,noise_std=np.full(freq.size,na),off_pulse=off)
T0_C,ZE_C=time_C[128],0.08;T0_D,ZE_D=time_D[128],0.08
mC=bd(freq_C,time_C,df_C,0.0,TAU_TRUE,ALPHA_TRUE,T0_C,ZE_C,DNU_C_TRUE,SIGMA_G_C,-1.5,0.30,RNG)
mD=bd(freq_D,time_D,df_D,20.0,TAU_TRUE,ALPHA_TRUE,T0_D,ZE_D,DNU_D_TRUE,SIGMA_G_D,-1.5,0.30,RNG)
def pC(t,a):return FRBParams(c0=1.0,t0=T0_C,gamma=0.0,zeta=ZE_C,tau_1ghz=t,alpha=a,delta_dm=0.0)
def pD(t,a):return FRBParams(c0=1.0,t0=T0_D,gamma=0.0,zeta=ZE_D,tau_1ghz=t,alpha=a,delta_dm=20.0)
def dec(m,p,dnu):
    K=m(replace(p,c0=1.0,gamma=0.0),"M3",freq_subset=m.valid)
    d=m.data[m.valid];sig=np.clip(m.noise_std[m.valid],1e-9,None);var=sig**2
    Sdd=np.einsum("ij,ij->i",d,d);Sdk=np.einsum("ij,ij->i",d,K);Skk=np.einsum("ij,ij->i",K,K)
    ok=Skk>1e-30;Skk_s=np.where(ok,Skk,1.0)
    chi2=np.where(ok,(Sdd-Sdk**2/Skk_s)/var,Sdd/var);temporal=float(np.sum(-0.5*chi2))
    ahat=Sdk[ok]/Skk[ok];v=var[ok]/Skk[ok];nu=m.freq[m.valid][ok]*1e3
    lamp,s2,mu,mod=_gp_amplitude_logL(ahat,v,nu,dnu,mu_degree=1)
    return temporal,lamp,s2,mod
# decompose temporal vs spectral GP at injected dnu, true vs biased tau
for lab,t in [("true_tau_0.85",0.85),("biased_tau_0.55",0.55)]:
    tC,sC,s2C,mC_=dec(mC,pC(t,ALPHA_TRUE),DNU_C_TRUE)
    tD,sD,s2D,mD_=dec(mD,pD(t,ALPHA_TRUE),DNU_D_TRUE)
    print(f"{lab}: temporal={tC+tD:11.1f} spectralGP={sC+sD:11.1f} total={tC+sC+tD+sD:11.1f}"
          f"  | C(s2={s2C:.3f},m={mC_:.3f}) D(s2={s2D:.3f},m={mD_:.3f})")
# Also: spectral GP alone as a function of tau (does it favor low tau?)
print("tau   temporal      spectralGP    sum")
for t in [0.5,0.65,0.75,0.85,0.95]:
    tC,sC,_,_=dec(mC,pC(t,ALPHA_TRUE),DNU_C_TRUE);tD,sD,_,_=dec(mD,pD(t,ALPHA_TRUE),DNU_D_TRUE)
    print(f"{t:.2f}  {tC+tD:11.1f}  {sC+sD:11.1f}  {tC+sC+tD+sD:11.1f}")
