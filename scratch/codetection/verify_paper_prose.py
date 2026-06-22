"""Cross-check: every falsifiable number in docs-analysis/foreground.md matches
the verified data files. Exit non-zero on any mismatch."""
import os
import re
import sys

import pandas as pd

HERE = "scratch/codetection"
REPO = "."
fin = pd.read_csv(os.path.join(HERE, "foreground_final.csv"))
fg  = pd.read_csv(os.path.join(HERE, "foreground.csv"))
doc = open(os.path.join(REPO, "docs-analysis", "foreground.md")).read()
fails = []

# 1) verdict tallies stated in prose
vc = fin.final_verdict.value_counts().to_dict()
claims = {
    "confirmed": (29, vc.get("confirmed")),
    "background/refuted": (7, vc.get("refuted")),
    "inconclusive": (13, vc.get("inconclusive")),
    "total": (49, len(fin)),
}
for label, (stated, actual) in claims.items():
    if stated != actual:
        fails.append(f"{label}: expected {stated}, data has {actual}")
for n in ("29", "7", "13", "49"):
    if not re.search(rf"\b{n}\b", doc):
        fails.append(f"docs foreground page missing the number {n}")

# 2) "14 of the 15 clusters lie outside their own R500"
clusters = fg[fg["type"].str.contains("cluster", case=False, na=False)]
n_clusters = len(clusters)
# outside R500 == impact > R500 == b_over_r500 > 1 (or flagged IMPACT_GT_R500)
outside = clusters[(clusters["b_over_r500"] > 1.0) | clusters["flags"].fillna("").str.contains("IMPACT_GT_R500")]
n_outside = len(outside)
if n_clusters != 15:
    fails.append(f"cluster count: prose says 15, data has {n_clusters}")
if n_outside != 14:
    fails.append(f"clusters outside R500: prose says 14, data has {n_outside}")

# 3) generated docs table has one data row per object
mdrows = [ln for ln in doc.splitlines() if ln.startswith("| ")]
n_md = len(mdrows) - 1
if n_md != len(fin):
    fails.append(f"docs table rows: expected {len(fin)}, found {n_md}")

print(f"verdicts: {vc}")
print(f"clusters: {n_clusters} total, {n_outside} outside R500 (b/R500>1 or flagged)")
print(f"docs table rows: {n_md}")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL FOREGROUND-DOCS PROSE CHECKS PASS")
