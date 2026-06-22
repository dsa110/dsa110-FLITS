"""Independent fidelity check: the published tables (CSV/MkDocs) must match the
verified foreground_final.csv exactly — same objects, same verdicts, no row dropped.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
fin = pd.read_csv(os.path.join(HERE, "foreground_final.csv"))
cat = pd.read_csv(os.path.join(HERE, "foreground_catalog.csv"))
fin["obj"] = fin.obj.astype(str)
cat["obj_id"] = cat.obj_id.astype(str)
md = open(os.path.join(REPO, "docs-analysis", "foreground.md")).read()
fails = []

# row counts + verdict tallies
if len(cat) != len(fin):
    fails.append(f"catalog rows {len(cat)} != final {len(fin)}")
exp = fin.final_verdict.value_counts().to_dict()
got = cat.verdict.value_counts().to_dict()
if exp != got:
    fails.append(f"verdict tallies differ: final={exp} catalog={got}")

# every object present with matching verdict
fmap = dict(zip(fin.obj, fin.final_verdict))
for _, r in cat.iterrows():
    if r.obj_id not in fmap:
        fails.append(f"catalog obj {r.obj_id} not in final")
    elif fmap[r.obj_id] != r.verdict:
        fails.append(f"obj {r.obj_id}: catalog verdict {r.verdict} != final {fmap[r.obj_id]}")

# every objID appears in md
for oid in cat.obj_id:
    if oid not in md:
        fails.append(f"obj {oid} missing from foreground.md")

# Markdown table data rows == 49 (lines starting '| ' minus header; separator starts '|---')
mdrows = [ln for ln in md.splitlines() if ln.startswith("| ")]
n_md = len(mdrows) - 1  # minus header row
if n_md != len(fin):
    fails.append(f"md data rows {n_md} != {len(fin)}")

print(f"catalog rows: {len(cat)}  verdicts: {got}")
print(f"md: {n_md} data rows")
print(f"objID presence checks: {len(cat)} x md")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails[:20]:
        print("  " + x)
    sys.exit(1)
print("\nALL CATALOG-TABLE FIDELITY CHECKS PASS")
