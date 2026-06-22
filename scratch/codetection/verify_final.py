"""Independent reproduce-check of foreground_final.csv.

Re-derives the merged final_verdict for every object straight from the two upstream
files (foreground_validated.csv, ps1_strm_resolution.csv) with from-scratch logic and
diffs against foreground_final.csv. Also checks no row was dropped/duplicated.
Exit non-zero on any mismatch.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
val = pd.read_csv(os.path.join(HERE, "foreground_validated.csv"))
strm = pd.read_csv(os.path.join(HERE, "ps1_strm_resolution.csv"))
fin = pd.read_csv(os.path.join(HERE, "foreground_final.csv"))
for df in (val, strm, fin):
    df["obj"] = df.obj.astype(str)

strm_v = strm.set_index("obj").strm_verdict.to_dict()
val_v = val.set_index("obj").foreground_verdict.to_dict()


def collapse(s):
    s = str(s)
    return (
        "confirmed"
        if s.startswith("confirmed")
        else ("refuted" if s.startswith("refuted") else "inconclusive")
    )


fails = []
# row set identity
if set(fin.obj) != set(val.obj):
    fails.append("obj set differs between final and validated")
if len(fin) != len(val):
    fails.append(f"row count {len(fin)} != {len(val)}")
if fin.obj.duplicated().any():
    fails.append("duplicate obj in final")

for _, r in fin.iterrows():
    expect = collapse(strm_v[r.obj]) if r.obj in strm_v else collapse(val_v[r.obj])
    if r.final_verdict != expect:
        fails.append(f"{r.obj}: final={r.final_verdict} expected={expect}")

# every STRM obj must be sourced from the STRM file (carry strm_class) and be inconclusive
for o in strm_v:
    row = fin[fin.obj == o]
    if row.empty or pd.isna(row.iloc[0].strm_class) or row.iloc[0].final_verdict != "inconclusive":
        fails.append(f"STRM {o}: not properly merged from PS1-STRM")

vc = fin.final_verdict.value_counts().to_dict()
print(f"rows={len(fin)}  verdicts={vc}")
print(f"independent re-derivation checks: {len(fin)}; STRM-source checks: {len(strm_v)}")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL FINAL-MERGE CHECKS PASS")
