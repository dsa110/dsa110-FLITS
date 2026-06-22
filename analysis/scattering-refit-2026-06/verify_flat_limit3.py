import sys, numpy as np
from dataclasses import replace
from scipy.linalg import eigh
sys.path.insert(0,"/home/jfaber/flits/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams

rng=np.random.default_rng(7)
nf,nt=48,256; freq=np.linspace(1.30,1.50,nf); time=np.linspace(-5.0,20.0,nt)
chan_w=(freq[1]-freq[0])*1e3
pt=FRBParams(c0=1.0,t0=2.0,gamma=0.0,zeta=0.4,tau_1ghz=1.5,alpha=4.0,delta_dm=0.0)
m0=FRBModel(time=time,freq=freq,data=np.zeros((nf,nt)),noise_std=np.ones(nf))
K=m0(replace(pt,c0=1.0,gamma=0.0),"M3"); spec=3.0*(freq/1.4)**-1.5; nu=freq*1e3
dd=nu[:,None]-nu[None,:]; Cg=1.0/(1.0+(dd/6.0)**2); Lg=np.linalg.cholesky(Cg+1e-9*np.eye(nf))
g=spec*(1.0+0.3*(Lg@rng.standard_normal(nf))); sg=0.25
data=g[:,None]*K+sg*rng.standard_normal((nf,nt))
model=FRBModel(time=time,freq=freq,data=data,noise_std=np.full(nf,sg))
def th(i): return FRBParams(c0=1.0,t0=2.0+0.4*np.sin(i),gamma=0.0,zeta=0.4+0.05*np.cos(i),
    tau_1ghz=1.5*(1.0+0.25*np.sin(2*i)),alpha=4.0+0.3*np.cos(2*i),delta_dm=0.0)
ths=[th(i) for i in np.linspace(0,6,25)]

# Compute, per theta, the GLS-Jacobian +0.5 ln|X^T diag(1/v) X| in the C->I, sg2->inf
# limit (Sigma -> sg2*I, so Sigma^-1 -> (1/sg2) I; X^T Sigma^-1 X = (1/sg2) X^T X => the
# 1/sg2 is a const; the theta-dep part is 0.5 ln|X^T diag(1/v) X| from the v-weighting).
# This term is PRESENT in GP, ABSENT in flat. Quantify its theta-swing.
jac=[]
for t in ths:
    K2=model(replace(t,c0=1.0,gamma=0.0),"M3",freq_subset=model.valid)
    d=model.data[model.valid]; var=np.clip(model.noise_std[model.valid],1e-9,None)**2
    S_dk=np.einsum("ij,ij->i",d,K2); S_kk=np.einsum("ij,ij->i",K2,K2); ok=S_kk>1e-30
    v=var[ok]/S_kk[ok]; nuu=model.freq[model.valid][ok]*1e3
    span=nuu.max()-nuu.min(); nc=(nuu-nuu.mean())/span
    X=np.vander(nc,N=2,increasing=True)
    XtSiX=(X/v[:,None]).T@X   # X^T diag(1/v) X  (the sg2->inf weighting -> diag(1/v))
    s,ld=np.linalg.slogdet(XtSiX); jac.append(0.5*ld)
jac=np.array(jac); jr=jac-jac.mean()
print(f"GLS-Jacobian 0.5 ln|X^T diag(1/v) X| theta-swing: max|dev|={np.max(np.abs(jr)):.4f} std={np.std(jr):.4f}")

# also the mu-profile residual term -0.5 (ahat-mu)^T diag(1/v)(ahat-mu) vs the flat
# (which effectively has the per-channel free amp => that quadratic is 0). Quantify.
muq=[]
for t in ths:
    K2=model(replace(t,c0=1.0,gamma=0.0),"M3",freq_subset=model.valid)
    d=model.data[model.valid]; var=np.clip(model.noise_std[model.valid],1e-9,None)**2
    S_dk=np.einsum("ij,ij->i",d,K2); S_kk=np.einsum("ij,ij->i",K2,K2); ok=S_kk>1e-30
    ahat=S_dk[ok]/S_kk[ok]; v=var[ok]/S_kk[ok]; nuu=model.freq[model.valid][ok]*1e3
    span=nuu.max()-nuu.min(); nc=(nuu-nuu.mean())/span; X=np.vander(nc,N=2,increasing=True)
    W=1.0/v; XtWX=(X*W[:,None]).T@X; XtWa=(X*W[:,None]).T@(W*ahat*0+ahat)  # weighted
    XtWa=(X*W[:,None]).T@ahat; beta=np.linalg.solve(XtWX,XtWa); r=ahat-X@beta
    muq.append(-0.5*np.sum(W*r*r))
muq=np.array(muq); mr=muq-muq.mean()
print(f"mu-profile quad -0.5 r^T diag(1/v) r theta-swing: max|dev|={np.max(np.abs(mr)):.4f} std={np.std(mr):.4f}")
print(f"sum of the two terms' theta-swing: max|dev|={np.max(np.abs(jr+mr)):.4f}  (compare measured resid 8.0876)")
