#!/usr/bin/env python
"""Dump per-band data + joint-fit recovered model (+axes) to .npz for LOCAL
plotting. Same OLS gain recovery as joint_ppc_multi.py; no matplotlib here so it
runs on HPCC despite the broken matplotlibrc. Plot locally with plot_jointmodel.py.

  python dump_jointmodel.py <burst> <fit_suffix>
"""
import json, os, sys
from dataclasses import replace
import numpy as np
import yaml

REPO = os.environ.get('FLITS_REPO', '/home/jfaber/flits/dsa110-FLITS')
RUNS = os.environ.get('FLITS_RUNS', '/central/scratch/jfaber/flits-runs')
sys.path.insert(0, f'{REPO}/scattering')
from scat_analysis.burstfit import FRBParams
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset


def prepare(cfg_path, name, outdir, pbf, beta):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg['telcfg_path'], cfg['telescope'])
    ds = BurstDataset(cfg['path'], outdir, name=name, telescope=tel,
                      f_factor=int(cfg['f_factor']), t_factor=int(cfg['t_factor']),
                      outer_trim=float(cfg.get('outer_trim', 0.15)),
                      onpulse_crop=os.environ.get('FLITS_ONPULSE_CROP', '1') == '1',
                      onpulse_pad_factor=float(os.environ.get('FLITS_ONPULSE_PAD', '0.5')))
    m = ds.model
    m.dm_init = float(cfg.get('dm_init', 0.0))
    m.pbf = pbf
    if beta is not None:
        m.pbf_beta = float(beta)
    return m


def recover(model, params_list):
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), 'M3') for p in params_list])  # (N,F,T)
    d = np.asarray(model.data, float)
    sig = np.clip(np.asarray(model.noise_std, float).reshape(-1), 1e-9, None)
    M = np.einsum('nft,mft->fnm', Ks, Ks)
    b = np.einsum('nft,ft->fn', Ks, d)
    N = len(params_list)
    jit = 1e-9 * max(float(np.einsum('fnn->f', M).mean()), 1e-30)
    g = np.linalg.solve(M + jit * np.eye(N), b[..., None])[..., 0]
    mod = np.einsum('fn,nft->ft', g, Ks)
    valid = model.valid
    v = np.ones(d.shape[0], bool) if valid is None else np.asarray(valid).reshape(-1).astype(bool)
    r = ((d - mod) / sig[:, None])[v]
    r = r[np.isfinite(r)]
    chi2 = float(np.sum(r ** 2)) / max(int(r.size) - 7, 1)
    return dict(data=d, model=mod, freq=np.asarray(model.freq, float),
                time=np.asarray(model.time, float), noise=sig, valid=v), chi2


def band_params(p, X, n, tau, al):
    ddm = float(p.get(f'delta_dm_{X}', 0.0))
    out = []
    for i in range(1, n + 1):
        t0 = p.get(f't0_{X}{i}', p.get(f't0_{X}'))
        ze = p.get(f'zeta_{X}{i}', p.get(f'zeta_{X}'))
        out.append(FRBParams(c0=1.0, t0=float(t0), gamma=0.0, zeta=float(ze),
                             tau_1ghz=tau, alpha=al, delta_dm=ddm))
    return out


def main():
    b = sys.argv[1]
    suf = sys.argv[2] if len(sys.argv) > 2 else ''
    out = f'{RUNS}/data/joint'
    d = json.load(open(f'{out}/{b}_joint_fit{suf}.json'))
    p = {k: v['median'] for k, v in d['percentiles'].items()}
    tau, al = p['tau_1ghz'], p['alpha']
    nC, nD = int(d.get('components_C', 1)), int(d.get('components_D', 1))
    mC = prepare(f'{RUNS}/configs/{b}_chime_run.yaml', f'{b}_chime', out, d.get('pbf_C', 'exp'), d.get('beta_C'))
    mD = prepare(f'{RUNS}/configs/{b}_dsa_run.yaml', f'{b}_dsa', out, d.get('pbf_D', 'exp'), d.get('beta_D'))
    if d.get('shared_zeta'):
        zC = p['zeta_1ghz'] * np.asarray(mC.freq, float) ** p['x_zeta']
        zD = p['zeta_1ghz'] * np.asarray(mD.freq, float) ** p['x_zeta']
        psC = [FRBParams(c0=1.0, t0=p['t0_C'], gamma=0.0, zeta=zC, tau_1ghz=tau, alpha=al, delta_dm=p['delta_dm_C'])]
        psD = [FRBParams(c0=1.0, t0=p['t0_D'], gamma=0.0, zeta=zD, tau_1ghz=tau, alpha=al, delta_dm=p['delta_dm_D'])]
    else:
        psC = band_params(p, 'C', nC, tau, al)
        psD = band_params(p, 'D', nD, tau, al)
    C, chiC = recover(mC, psC)
    D, chiD = recover(mD, psD)
    fp = f'{out}/{b}_jointmodel{suf}.npz'
    np.savez_compressed(fp,
                        dataC=C['data'], modelC=C['model'], freqC=C['freq'], timeC=C['time'], noiseC=C['noise'], validC=C['valid'],
                        dataD=D['data'], modelD=D['model'], freqD=D['freq'], timeD=D['time'], noiseD=D['noise'], validD=D['valid'],
                        alpha=al, tau_1ghz=tau, chi2C=chiC, chi2D=chiD, nC=nC, nD=nD, burst=b)
    print(f'{b}: wrote {fp}  alpha={al:.3f} tau={tau:.4f} chiC={chiC:.2f} chiD={chiD:.2f}')


if __name__ == '__main__':
    main()
