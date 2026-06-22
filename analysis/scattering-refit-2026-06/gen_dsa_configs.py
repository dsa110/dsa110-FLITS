#!/usr/bin/env python
"""Generate HPCC run-configs for the 12 DSA bursts.

Mirrors gen_chime_configs but for DSA: keeps per-burst f_factor/t_factor/dm_init,
repoints `path` to the local scratch DSA copy (resolved by GLOB on burst name, so
the arc-path typos in the repo configs don't matter), telescope=dsa (so the
freq_descending flip applies), and the corrected sampler knobs.
"""
import glob, os, yaml

REPO = "/home/jfaber/flits/dsa110-FLITS"
RUNS = "/central/scratch/jfaber/flits-runs"
DATA = f"{RUNS}/data/dsa"
CFG = f"{RUNS}/configs"
TEL = f"{REPO}/scattering/configs/telescopes.yaml"
SAMP = f"{REPO}/scattering/configs/sampler.yaml"

KNOBS = dict(
    telcfg_path=TEL, sampcfg_path=SAMP, telescope="dsa",
    fitting_method="nested", outer_trim=0.15,
    nlive=400, dlogz=0.5, nlive_walks=15,
    alpha_fixed=4.0, nproc=8,
)

os.makedirs(CFG, exist_ok=True)
made, skipped = [], []
for src in sorted(glob.glob(f"{REPO}/scattering/configs/bursts/dsa/*_dsa.yaml")):
    burst = os.path.basename(src)[:-len("_dsa.yaml")]
    cfg = yaml.safe_load(open(src)) or {}
    hits = glob.glob(f"{DATA}/{burst}_dsa_*.npy")            # resolve by burst, not path
    if not hits:
        skipped.append(burst); continue
    cfg["path"] = hits[0]
    cfg.update(KNOBS)
    # DSA is incoherently dedispersed -> intra-channel smearing at catalog DM must be
    # modelled. DM is encoded in the filename (<b>_dsa_I_<DMint>_<DMfrac>_...). CHIME
    # stays dm_init=0 (coherently dedispersed). See burstfit.py:_smearing_sigma.
    toks = os.path.basename(hits[0]).split("_")
    j = toks.index("I")
    cfg["dm_init"] = float(f"{toks[j+1]}.{toks[j+2]}")
    with open(f"{CFG}/{burst}_dsa_run.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
    made.append(burst)

print(f"generated {len(made)} DSA configs: {', '.join(sorted(made))}")
if skipped:
    print(f"SKIPPED (no local data): {', '.join(skipped)}")
