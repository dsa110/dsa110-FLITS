"""Guardrail-3 prep smoke for the zach binning-drop (option b).

Runs joint_tf_prep.prepare_pair for zach at the production 30 ms trailing cap
(baseline) and at the fine 12 ms cap, prints both bands' AUTO-TF captions, and
asserts the CHIME t-factor is UNCHANGED (guardrail 3: if CHIME binning shifts,
stop + flag). No sampler; probing + masked downsample only.
"""
import os, sys, tempfile

WT = "/home/ubuntu/worktrees/joint-tf-fits"
os.environ.setdefault("FLITS_REPO", WT)
os.environ.setdefault("FLITS_RUNS", "/home/ubuntu/flits-runs")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, f"{WT}/scattering")
sys.path.insert(0, f"{WT}/analysis/scattering-refit-2026-06")

import joint_tf_prep as jtp

RUNS = os.environ["FLITS_RUNS"]
cC = f"{RUNS}/configs/zach_chime_run.yaml"
cD = f"{RUNS}/configs/zach_dsa_run.yaml"
td = tempfile.mkdtemp(prefix="zachfine_")


def probe():
    (mC, pC), (mD, pD) = jtp.prepare_pair(cC, cD, "zach_smoke", td, auto=True)
    return pC, pD


print(f"=== BASELINE trail_cap = {jtp.WIN_TRAIL_CAP_MS:.1f} ms ===", flush=True)
b_C, b_D = probe()
print("  CHIME:", b_C.caption())
print("  DSA  :", b_D.caption())

NEW = 12.0
jtp.WIN_TRAIL_CAP_MS = NEW
for n, o in list(vars(jtp).items()):
    kd = getattr(o, "__kwdefaults__", None)
    if kd and "trail_cap_ms" in kd:
        kd["trail_cap_ms"] = NEW
        print(f"  patched {n}.trail_cap_ms -> {NEW}")

print(f"=== FINE trail_cap = {NEW:.1f} ms ===", flush=True)
f_C, f_D = probe()
print("  CHIME:", f_C.caption())
print("  DSA  :", f_D.caption())

print()
print(f"GUARDRAIL-3 CHIME t_factor: baseline={b_C.t_factor} fine={f_C.t_factor} "
      f"dt {b_C.dt_ms*1e3:.1f}->{f_C.dt_ms*1e3:.1f} us  "
      f"{'UNCHANGED-OK' if b_C.t_factor == f_C.t_factor else 'SHIFTED-STOP+FLAG'}")
print(f"DSA t_factor: baseline={b_D.t_factor} (dt {b_D.dt_ms*1e3:.1f} us) -> "
      f"fine={f_D.t_factor} (dt {f_D.dt_ms*1e3:.1f} us)  "
      f"{'REACHED-t2-OK' if f_D.dt_ms*1e3 < 100 else 'STILL-COARSE'}")
