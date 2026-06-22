#!/usr/bin/env python
"""Validate every co-detected burst's CHIME+DSA run-config before joint fitting:
data path exists, DSA dm_init set, f/t factors. Flags anything that can't run."""
import os
import yaml

CFG = os.environ.get("FLITS_CFG", "/central/scratch/jfaber/flits-runs/configs")
DONE = set("johndoeII wilhelm phineas oran".split())
BURSTS = "casey chromatica freya hamilton isha johndoeII mahi oran phineas whitney wilhelm zach".split()

hdr = f"{'burst':12s} {'run':5s} {'CHIME':8s} {'DSA':8s} {'dm_init_D':10s} {'fC/tC':8s} {'fD/tD':8s}"
print(hdr)
print("-" * len(hdr))
problems = []
for b in BURSTS:
    cc_p, dc_p = f"{CFG}/{b}_chime_run.yaml", f"{CFG}/{b}_dsa_run.yaml"
    if not (os.path.exists(cc_p) and os.path.exists(dc_p)):
        print(f"{b:12s} MISSING CONFIG")
        problems.append((b, "missing config"))
        continue
    cc, dc = yaml.safe_load(open(cc_p)), yaml.safe_load(open(dc_p))
    pc, pd = cc.get("path", ""), dc.get("path", "")
    okc = "OK" if pc and os.path.exists(pc) else "MISSING"
    okd = "OK" if pd and os.path.exists(pd) else "MISSING"
    dmi = dc.get("dm_init", "NONE")
    run = "DONE" if b in DONE else "todo"
    print(f"{b:12s} {run:5s} {okc:8s} {okd:8s} {str(dmi):10s} "
          f"{cc.get('f_factor')}/{cc.get('t_factor'):<5} {dc.get('f_factor')}/{dc.get('t_factor')}")
    if okc != "OK":
        problems.append((b, f"CHIME data missing: {pc}"))
    if okd != "OK":
        problems.append((b, f"DSA data missing: {pd}"))
    if dmi in ("NONE", None, 0, 0.0):
        problems.append((b, f"DSA dm_init unset/zero: {dmi}"))

print()
if problems:
    print(f"PROBLEMS ({len(problems)}):")
    for b, m in problems:
        print(f"  {b}: {m}")
else:
    print("ALL 12 CONFIGS VALID — clear to fit.")
