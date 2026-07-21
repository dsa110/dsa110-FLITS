import time, numpy as np, sys
sys.path.insert(0,"/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dm_envelope import fit_waterfall
import json
b=json.load(open("scripts/burst_inputs.json"))["bursts"]
m={x["name"]:x for x in b}["zach"]; dm=float(m["dm"])
bb=BBData.from_file(f"/data/Faber2026/data/chime-frb/zach/singlebeam_{m['chime_id']}.h5")
dt=float(bb.attrs["delta_time"]); freq=np.asarray(bb.index_map["freq"]["centre"],float)
t=time.time(); bbdd=coherent_dedisp(bb,dm); print("coherent_dedisp: %.1fs"%(time.time()-t),flush=True)
TDS=16
inten=np.nan_to_num(np.abs(bbdd[:,0,:])**2+np.abs(bbdd[:,1,:])**2)
csd=inten.std(1); med=np.median(csd[csd>0]); good=np.isfinite(csd)&(csd>0.2*med)&(csd<8*med)
iw=inten[good]; nt=(iw.shape[1]//TDS)*TDS; wf=iw[:,:nt].reshape(iw.shape[0],nt//TDS,TDS).mean(2)
pk=int(np.argmax(wf.sum(0))); lo=max(pk-230,0); hi=pk+690; wfc=wf[:,lo:hi]; fw=freq[good]
print("crop shape",wfc.shape,flush=True)
t=time.time(); c,dof,p,r=fit_waterfall(wfc,fw,dt*TDS); print("fit_waterfall: %.1fs chi2red=%.2f"%(time.time()-t,r),flush=True)
