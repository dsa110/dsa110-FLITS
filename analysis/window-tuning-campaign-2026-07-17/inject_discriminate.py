import sys, numpy as np
sys.path.insert(0,"scint_analysis")
import window_refit as wr
rng=np.random.default_rng(31337)
NCH=4096; NT=400; DF=0.01
LAGS=np.arange(0,wr.LAG_MAX+1e-9,DF)
def corr_field(gamma):
    psd=1.0/(1.0+(np.fft.rfftfreq(NCH,DF)*(2*np.pi*gamma))**2)
    w=np.fft.irfft(np.sqrt(psd)*(rng.normal(size=psd.size)+1j*rng.normal(size=psd.size)),n=NCH)
    return w/np.std(w)
def make_ds(gamma_n, gamma_b, m_n, m_b, tau, snr=30.0):
    """core columns modulated by NARROW scintle (gamma_n); tail columns by BROAD env (gamma_b)."""
    t=np.arange(NT); c=120
    prof=np.where(t>=c, np.exp(-(t-c)/tau), 0.0); prof[c-1:c+1]=1.0; prof=np.clip(prof,0,None)
    g_n=1.0+m_n*corr_field(gamma_n); g_b=1.0+m_b*corr_field(gamma_b)
    ds=np.zeros((NCH,NT))
    core_end=c+max(2,tau//3)
    for j in range(NT):
        if j<c: continue
        gain = g_n if j<core_end else g_b     # tail carries broad structure
        ds[:,j]=gain*prof[j]*snr
    ds=ds+rng.normal(0,1.0,size=ds.shape)
    return ds, prof, c, core_end
def acf_of(s):
    x=s-np.nanmean(s); ac=np.correlate(x,x,"full")[x.size-1:]; return ac/ac[0] if ac[0]!=0 else ac
def fit_win(ds,prof,burst,off,weighted):
    b0,b1=burst;o0,o1=off
    if weighted:
        w=np.clip(prof[b0:b1]-np.median(prof[o0:o1]),0,None)
        spec=(ds[:,b0:b1]*w).sum(1)/(w.sum() if w.sum()>0 else 1)
    else: spec=ds[:,b0:b1].mean(1)
    spec=spec-ds[:,o0:o1].mean(1).mean()
    ac=acf_of(spec); lags=np.arange(ac.size)*DF
    return wr._fit_subband(lags[:LAGS.size],ac[:LAGS.size])
gn,gb=0.08,2.0; off=(300,395)
print(f"TRUTH: narrow scintle gamma_n={gn} (core), broad env gamma_b={gb} (tail). Good estimator -> ~{gn}")
print(f"{'tau':>4} | {'core':>16} {'tail':>16} {'wtail':>16}")
for tau in [8,20,40]:
    c0=120; core_end=c0+max(2,tau//3)
    core=(c0-1,core_end); tail=(c0-1,c0+int(3*tau))
    rr={}
    for lbl,(win,wt) in {"core":(core,False),"tail":(tail,False),"wtail":(tail,True)}.items():
        gs=[]
        for _ in range(80):
            ds,prof,c,ce=make_ds(gn,gb,0.6,0.7,tau)
            f=fit_win(ds,prof,win,off,wt)
            if f["ok"] and f["resolved"]: gs.append(f["gamma"])
        gs=np.array(gs); rr[lbl]=(np.median(gs) if gs.size else np.nan, gs.size)
    def cell(v): m,n=v; return f"{m:.3f}({m/gn:.2f}x,n{n})" if not np.isnan(m) else "nan"
    print(f"{tau:>4} | {cell(rr['core']):>16} {cell(rr['tail']):>16} {cell(rr['wtail']):>16}")
