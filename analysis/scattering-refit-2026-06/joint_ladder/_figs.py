"""Synthesis figures for the joint CHIME-DSA scattering ladder.
Reads *_joint_fit*.json in this dir; writes PNGs to figs_ladder/.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(__file__)
OUT = os.path.join(D, "figs_ladder")
os.makedirs(OUT, exist_ok=True)


def load():
    rows = {}
    for f in glob.glob(os.path.join(D, "*_joint_fit*.json")):
        d = json.load(open(f))
        m = re.match(r"(.+?)_joint_fit_?(.*)\.json", os.path.basename(f))
        burst, tag = m.group(1), (m.group(2) or "base")
        a = d.get("alpha", {})
        rows[(burst, tag)] = dict(
            a=a.get("median"),
            ep=a.get("err_plus"),
            em=a.get("err_minus"),
            tau=(d.get("tau_1ghz") or {}).get("median"),
            lnZ=d.get("log_evidence"),
            C=d.get("components_C"),
            D=d.get("components_D"),
            s2=d.get("gain_s2"),
        )
    return rows


rows = load()

# ---- Figure 1: final-alpha gate dot plot --------------------------------
# chosen model per burst (matches LADDER_SUMMARY)
chosen = {
    "freya": "sharedzeta",
    "mahi": "C1D1",
    "phineas": "C3D3",
    "chromatica": "sharedzeta",
    "oran": "C2D1",
    "wilhelm": "sharedzeta",
    "zach": "C2D3",
    "casey": "sharedzeta",
    "isha": "C2D1",
    "hamilton": "C4D1",
    "whitney": "base",
    "johndoeII": "C2D2",
}
flag = {  # PASS / MARGINAL / FAIL
    "freya": "P",
    "mahi": "P",
    "phineas": "P",
    "chromatica": "P",
    "oran": "P",
    "wilhelm": "P",
    "zach": "P",
    "casey": "P",
    "isha": "M",
    "hamilton": "F",
    "whitney": "F",
    "johndoeII": "F",
}
col = {"P": "#1b7837", "M": "#d9a300", "F": "#b2182b"}
order = sorted(chosen, key=lambda b: -(rows[(b, chosen[b])]["a"] or 0))

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.axvspan(1.5, 2.0, color="#b2182b", alpha=0.07, lw=0)
ax.axvline(4.0, color="0.5", ls=":", lw=1)
ax.text(4.0, len(order) - 0.3, " Kolmogorov α=4", color="0.4", fontsize=8, va="top")
ax.axvline(1.5, color="#b2182b", ls="--", lw=1)
ax.text(1.5, -0.7, "prior floor", color="#b2182b", fontsize=8, ha="left")
for i, b in enumerate(order):
    r = rows[(b, chosen[b])]
    ax.errorbar(
        r["a"],
        i,
        xerr=[[r["em"]], [r["ep"]]],
        fmt="o",
        ms=7,
        color=col[flag[b]],
        ecolor=col[flag[b]],
        capsize=3,
        lw=1.6,
    )
    lbl = f"{b}  ({chosen[b]})"
    ax.text(6.15, i, lbl, va="center", fontsize=9)
ax.set_yticks([])
ax.set_xlim(1.3, 6.05)
ax.set_ylim(-1, len(order))
ax.set_xlabel("scattering index  α  (shared CHIME–DSA)")
ax.set_title(
    "Joint scattering index per burst — final model\n"
    "green PASS · amber MARGINAL · red FAIL (railed/unphysical)",
    fontsize=10,
)
import matplotlib.lines as ml

ax.legend(
    handles=[
        ml.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=col[k],
            label={"P": "PASS", "M": "MARGINAL", "F": "FAIL"}[k],
        )
        for k in "PMF"
    ],
    loc="lower right",
    fontsize=8,
    frameon=False,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alpha_gate.png"), dpi=140)
plt.close(fig)

# ---- Figure 2: profiled alpha vs component count (profile-bias story) -----
ladder_bursts = ["zach", "phineas", "hamilton", "oran", "isha", "mahi"]
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for b in ladder_bursts:
    pts = []
    for (bb, tag), r in rows.items():
        if bb != b or r["s2"] is not None or r["C"] is None:
            continue
        if tag in ("base", "sharedzeta"):
            continue
        pts.append((r["C"] + r["D"], r["a"], r["em"], r["ep"], f"C{r['C']}D{r['D']}"))
    pts.sort()
    if not pts:
        continue
    n = [p[0] for p in pts]
    al = [p[1] for p in pts]
    ax.errorbar(
        n,
        al,
        yerr=[[p[2] for p in pts], [p[3] for p in pts]],
        marker="o",
        capsize=3,
        lw=1.6,
        label=b,
    )
ax.axhspan(1.5, 2.0, color="#b2182b", alpha=0.07, lw=0)
ax.axhline(4.0, color="0.5", ls=":", lw=1)
ax.axhline(1.5, color="#b2182b", ls="--", lw=1)
ax.set_xlabel("total temporal components (C + D)")
ax.set_ylabel("profiled α")
ax.set_title(
    "Profile-bias: α vs component count (profiled ladder)\n"
    "zach 3.32→2.41 as hidden sub-pulses are modeled",
    fontsize=10,
)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alpha_vs_components.png"), dpi=140)
plt.close(fig)

# ---- Figure 3: fixed-s2 cross-N sign-flip --------------------------------
s2vals = [1, 10, 100]


def g(b, tag, s2):
    r = rows.get((b, f"{tag}_s2-{s2}"))
    return r["lnZ"] if r else None


tests = [  # (burst, hi, lo, label)
    ("whitney", "C2D1", "C1D1", "whitney  C2D1−C1D1"),
    ("johndoeII", "C2D2", "C2D1", "johndoeII  C2D2−C2D1"),
    ("phineas", "C3D1", "C2D1", "phineas  C3D1−C2D1"),
    ("zach", "C2D2", "C2D1", "zach  C2D2−C2D1"),
    ("zach", "C2D3", "C2D2", "zach  C2D3−C2D2"),
]
fig, ax = plt.subplots(figsize=(8.0, 4.8))
x = np.arange(len(tests))
w = 0.26
for j, s2 in enumerate(s2vals):
    vals = []
    for b, hi, lo, _ in tests:
        a, c = g(b, hi, s2), g(b, lo, s2)
        vals.append((a - c) if (a is not None and c is not None) else np.nan)
    ax.bar(x + (j - 1) * w, vals, w, label=f"s2={s2}")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(5, color="0.5", ls=":", lw=1)
ax.axhline(-5, color="0.5", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([t[3] for t in tests], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("ΔlnZ (higher-N − lower-N)")
ax.set_title(
    "Fixed-s2 cross-N Bayes factor — component real only if consistently >0\n"
    "sign flip across s2 ⇒ prior-driven, NOT real",
    fontsize=10,
)
ax.set_yscale("symlog", linthresh=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s2_signflip.png"), dpi=140)
plt.close(fig)

print("wrote:", os.listdir(OUT))
