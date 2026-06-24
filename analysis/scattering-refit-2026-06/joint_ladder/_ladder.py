import glob
import json
import os
import re
from collections import defaultdict

rows = []
for f in glob.glob(os.path.join(os.path.dirname(__file__), "*_joint_fit*.json")):
    d = json.load(open(f))
    name = os.path.basename(f)[: -len("_joint_fit*.json") + len("*.json") - 5]
    b = d.get("burst", "?")
    # parse tag from filename
    m = re.match(r"(.+?)_joint_fit(.*)\.json", os.path.basename(f))
    tag = m.group(2).lstrip("_") if m else ""
    a = d.get("alpha", {})
    t = d.get("tau_1ghz", {})
    lo, hi = d.get("alpha_bounds") or [None, None]
    amed = a.get("median")
    aep, aem = a.get("err_plus"), a.get("err_minus")
    railed = ""
    if amed is not None and lo is not None:
        if amed - lo < 0.02:
            railed = "RAIL_LO"
        elif hi - amed < 0.02:
            railed = "RAIL_HI"
    rows.append(
        dict(
            burst=b,
            tag=tag or "(base)",
            C=d.get("components_C"),
            D=d.get("components_D"),
            s2=d.get("gain_s2"),
            alpha=amed,
            aep=aep,
            aem=aem,
            tau=t.get("median"),
            tep=t.get("err_plus"),
            tem=t.get("err_minus"),
            lnZ=d.get("log_evidence"),
            lnZe=d.get("log_evidence_err"),
            railed=railed,
            # ADR-0003: lnZ is incomparable across PBF families; legacy mixed files omit pbf_*
            pbf="exp/exp" if d.get("pbf_C") == "exp" and d.get("pbf_D") == "exp" else "mix",
        )
    )

bursts = defaultdict(list)
for r in rows:
    bursts[r["burst"]].append(r)


def fz(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


for b in sorted(bursts):
    rs = bursts[b]
    rs.sort(key=lambda r: -(r["lnZ"] or -9e9))
    best = rs[0]["lnZ"]
    print(f"\n===== {b}  (best lnZ ranked; ΔlnZ vs best) =====")
    print(
        f"{'tag':24} {'C/D':5} {'s2':5} {'pbf':7} {'alpha':>16} {'tau_ms':>14} {'lnZ':>12} {'ΔlnZ':>9}  flag"
    )
    for r in rs:
        cd = f"{r['C']}/{r['D']}" if r["C"] else "-"
        s2 = "" if r["s2"] is None else str(r["s2"])
        al = (
            f"{fz(r['alpha'])}+{fz(r['aep'], 3)}/-{fz(r['aem'], 3)}"
            if r["alpha"] is not None
            else "-"
        )
        ta = f"{fz(r['tau'], 4)}" if r["tau"] is not None else "-"
        dz = (r["lnZ"] - best) if r["lnZ"] is not None else None
        print(
            f"{r['tag']:24} {cd:5} {s2:5} {r['pbf']:7} {al:>16} {ta:>14} {fz(r['lnZ'], 1):>12} {fz(dz, 1):>9}  {r['railed']}"
        )
