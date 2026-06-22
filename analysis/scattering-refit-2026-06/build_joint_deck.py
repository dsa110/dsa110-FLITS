#!/usr/bin/env python
"""Build a self-contained joint-scattering slide deck (docs/joint_scattering_deck.html).

Data-driven + idempotent: scans joint_json/*_joint_fit.json for the per-sightline
shared (alpha, tau_1ghz) + lnZ (summary table, ALL available bursts), joint_ppc.json
for per-band chi2, and dsa_figs/{b}_{joint_ppc,corner,fullband_waterfall}.png for the
detail slides. Re-run as more fits/figures land to auto-extend toward all 12.

Slides: title -> method -> summary table -> tau(nu) ladder -> per-sightline detail
(only bursts with >=1 figure) -> caveats. Self-contained (base64 images), prev/next
+ arrow-key nav. Output is iframe-embedded by the 'Joint Scattering' tab in
docs/index.html.

  python build_joint_deck.py
"""
import os
import json
import glob
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "dsa_figs")
JSN = os.path.join(HERE, "joint_json")
OUT = "/Users/jakobfaber/Developer/repos/github.com/dsa110/dsa110-FLITS/docs/joint_scattering_deck.html"

ALO, AHI = 1.0, 6.0
# co-detected sample order (CHIME alphabetical-ish); casey last (non-standard binning)
ORDER = ["johndoeII", "wilhelm", "phineas", "oran", "chromatica", "freya",
         "hamilton", "isha", "mahi", "whitney", "zach", "casey"]
NOTES = {
    "johndoeII": "Headline: sub-Kolmogorov alpha, clean both bands. Strongest measurement.",
    "wilhelm": "Good fit; resolved scint band-width implies a DIFFERENT screen than the broadening tau.",
    "phineas": "Steep slope, but DSA band is the poorest fit - treat cautiously.",
    "oran": "alpha rails to the 1.0 floor; CHIME nuisance unconstrained -> NOT a measurement, drop.",
}


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_fit(b):
    f = os.path.join(JSN, f"{b}_joint_fit.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    p = d["percentiles"]
    a, t = p["alpha"], p["tau_1ghz"]
    zc, zd = p["zeta_C"]["median"], p["zeta_D"]["median"]
    lnz = d.get("log_evidence", float("nan"))
    # status, worst-first: prior rails and unphysical intrinsic width are
    # disqualifying; a wide posterior or an anomalous evidence is a warning.
    if a["lower"] <= ALO + 0.05:
        status, scls = "rails alpha=1 floor", "bad"
    elif a["upper"] >= AHI - 0.05:
        status, scls = "rails alpha=6 ceiling", "bad"
    elif max(zc, zd) > 3.0:
        status, scls = "zeta runaway (unphysical)", "bad"
    elif (a["upper"] - a["lower"]) > 2.0:
        status, scls = "alpha unconstrained", "warn"
    elif lnz < -1e5:
        status, scls = "anomalous lnZ", "warn"
    else:
        status, scls = "clean", "ok"
    return dict(
        burst=b, a=a["median"], alo=a["lower"], ahi=a["upper"],
        tau=t["median"], tlo=t["lower"], thi=t["upper"],
        ddmC=p["delta_dm_C"]["median"], ddmD=p["delta_dm_D"]["median"],
        zc=zc, zd=zd, lnz=lnz, status=status, scls=scls,
    )


def load_chi2(b):
    f = os.path.join(JSN, f"{b}_joint_ppc.json")
    if not os.path.exists(f):
        return None, None
    d = json.load(open(f))
    return d.get("chi2_chime"), d.get("chi2_dsa")


def figs_for(b):
    out = {}
    for key, suff in [("waterfall", "fullband_waterfall"), ("ppc", "joint_ppc"), ("corner", "corner")]:
        p = os.path.join(FIG, f"{b}_{suff}.png")
        if os.path.exists(p):
            out[key] = b64(p)
    return out


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- assemble data ----
fits = {b: load_fit(b) for b in ORDER}
fits = {b: v for b, v in fits.items() if v}
ladder = os.path.join(FIG, "tau_nu_ladder.png")
ladder_b64 = b64(ladder) if os.path.exists(ladder) else None

slides = []

# S1 title
n_fit = len(fits)
cleans = sorted(v["a"] for v in fits.values() if v["scls"] == "ok")
n_clean = len(cleans)
crange = f"{cleans[0]:.2f}&ndash;{cleans[-1]:.2f}" if cleans else "none"
slides.append(f"""
<section class="slide title">
  <h1>CHIME &times; DSA Joint Scattering Fits</h1>
  <p class="sub">Measuring the scattering index &alpha; (&tau;&prop;&nu;<sup>&minus;&alpha;</sup>)
     directly from the ~1&nbsp;GHz cross-band &tau; lever</p>
  <p class="big">{n_fit} / 12 sightlines fit &middot; {n_clean} give a clean &alpha; ({crange}); the rest rail the prior or show unphysical widths</p>
  <p class="foot">shared (&tau;<sub>1GHz</sub>, &alpha;) + per-telescope (c0, t0, &gamma;, &zeta;, &delta;DM) &middot; dynesty nested sampling &middot; &alpha;~U[1,6]</p>
</section>""")

# S2 method
slides.append("""
<section class="slide">
  <h2>Why a joint fit</h2>
  <ul>
    <li>A <b>single band</b> fixes only the product &tau;&middot;&nu;<sup>&minus;&alpha;</sup> at its own frequency &mdash; &alpha; and &tau; are degenerate.</li>
    <li>Fitting CHIME (~0.6&nbsp;GHz) and DSA (~1.4&nbsp;GHz) <b>together</b> with a shared (&tau;<sub>1GHz</sub>, &alpha;) breaks the degeneracy: the ratio &tau;<sub>C</sub>/&tau;<sub>D</sub> pins the slope.</li>
    <li>Intrinsic / timing params (c0, t0, &gamma;, &zeta;, &delta;DM) stay <b>per telescope</b> &mdash; they absorb band-specific structure; only scattering + dispersion are shared.</li>
    <li>Caveat: with 16 ch/band the lever is effectively <b>2-point</b> &mdash; &alpha; is driven by the cross-band &tau; ratio, so per-band fit quality matters.</li>
  </ul>
</section>""")

# S3 summary table (all available)
rows = []
for b, v in fits.items():
    cC, cD = load_chi2(b)
    chi = f"{cC:.2f}/{cD:.2f}" if cC else "&mdash;"
    flag = f'<span class="{v["scls"]}">{esc(v["status"])}</span>'
    rows.append(
        f"<tr><td>{esc(b)}</td><td>{v['a']:.2f} "
        f"<span class='pm'>[{v['alo']:.2f},{v['ahi']:.2f}]</span></td>"
        f"<td>{v['tau']:.3f}</td><td>{v['zc']:.2f}/{v['zd']:.2f}</td>"
        f"<td>{chi}</td><td>{v['lnz']:.0f}</td><td>{flag}</td></tr>"
    )
slides.append(f"""
<section class="slide">
  <h2>All fitted sightlines &mdash; shared (&alpha;, &tau;<sub>1GHz</sub>)</h2>
  <table class="summary">
    <tr><th>burst</th><th>&alpha; [p16,p84]</th><th>&tau;<sub>1GHz</sub> (ms)</th>
        <th>&zeta; C/D</th><th>&chi;&sup2; C/D</th><th>lnZ</th><th>status</th></tr>
    {''.join(rows)}
  </table>
  <p class="foot">Only <span class="ok">clean</span> rows are usable. A rail (&alpha; pinned to a prior edge) or &zeta; runaway (intrinsic width &gt; 3 ms &mdash; the model absorbing un-fittable CHIME structure) is not a measurement. casey: job timed out.</p>
</section>""")

# S4 ladder
if ladder_b64:
    slides.append(f"""
<section class="slide">
  <h2>&tau;(&nu;) ladder &mdash; the cross-band lever</h2>
  <img src="{ladder_b64}" class="wide"/>
  <p class="foot">Curves pinch in the CHIME band (where &tau;<sub>1GHz</sub> normalization is pinned) and fan out by DSA; the fan-out slope IS &alpha;. Tight band = well-constrained.</p>
</section>""")

# S5.. per-sightline detail (only those with >=1 figure)
for b, v in fits.items():
    fg = figs_for(b)
    if not fg:
        continue
    cC, cD = load_chi2(b)
    chi = f"&chi;&sup2; = {cC:.2f} (C) / {cD:.2f} (D)" if cC else ""
    note = NOTES.get(b) or f'status: {esc(v["status"])}.'
    imgs = ""
    if "waterfall" in fg:
        imgs += f'<img src="{fg["waterfall"]}" class="wide"/>'
    pair = ""
    if "ppc" in fg:
        pair += f'<img src="{fg["ppc"]}" class="half"/>'
    if "corner" in fg:
        pair += f'<img src="{fg["corner"]}" class="half"/>'
    if pair:
        imgs += f'<div class="pair">{pair}</div>'
    slides.append(f"""
<section class="slide">
  <h2>{esc(b)} &mdash; &alpha; = {v['a']:.2f} <span class="pm">[{v['alo']:.2f}, {v['ahi']:.2f}]</span>,
      &tau;<sub>1GHz</sub> = {v['tau']:.3f} ms</h2>
  <p class="note">{chi} &nbsp; {note}</p>
  {imgs}
</section>""")

# caveats
pending = [b for b in ORDER if b not in fits]
slides.append(f"""
<section class="slide">
  <h2>Caveats &amp; status</h2>
  <ul>
    <li><b>2-point lever:</b> 16 ch/band &rarr; &alpha; from the cross-band &tau; ratio; multi-screen sightlines can bias it.</li>
    <li><b>oran dropped:</b> CHIME nuisance unconstrained, &alpha; rails the floor &mdash; not a measurement.</li>
    <li><b>Scint cross-check (wilhelm):</b> resolved &Delta;&nu;<sub>d</sub> &rarr; a different (nearby) screen than the broadening &tau; &mdash; &alpha;<sub>&tau;</sub> need not equal &alpha;<sub>&Delta;&nu;</sub>.</li>
    <li><b>Sample reality:</b> only ~4 sightlines give a clean &alpha;. Several <b>rail the &alpha;=6 ceiling</b> (chromatica, freya, hamilton, mahi) or carry <b>unphysical &zeta;</b> (isha &zeta;&asymp;87, oran &zeta;&asymp;63) &mdash; the model absorbing un-fittable CHIME structure, not measuring &alpha;&gt;6.</li>
    <li><b>casey:</b> job <b>timed out</b> (non-standard f32/t4 binning, too slow for 30&nbsp;min) &mdash; re-run with a longer limit, and re-bin before comparing.</li>
    <li><b>Next:</b> diagnose the railed CHIME bands (quality / multi-component / the &alpha;-ceiling &times; &zeta; degeneracy) before trusting any steep &alpha;.</li>
    <li><b>Pending figures:</b> {esc(', '.join(pending)) if pending else 'none'}.</li>
  </ul>
</section>""")

n = len(slides)
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Joint Scattering Deck</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0b0f1a; color:#e6edf3; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
  .deck {{ position:relative; height:100vh; width:100vw; overflow:hidden; }}
  .slide {{ display:none; position:absolute; inset:0; padding:3.2vh 4vw 7vh; overflow-y:auto;
            flex-direction:column; }}
  .slide.active {{ display:flex; }}
  .slide.title {{ justify-content:center; align-items:center; text-align:center; }}
  h1 {{ font-size:2.6rem; margin:.2em 0; color:#7fb3ff; }}
  h2 {{ font-size:1.55rem; margin:.1em 0 .5em; color:#7fb3ff; border-bottom:1px solid #1d2840; padding-bottom:.3em; }}
  .sub {{ font-size:1.2rem; color:#9fb0c8; max-width:46em; }}
  .big {{ font-size:1.35rem; color:#ffd479; margin-top:1em; }}
  .foot {{ font-size:.82rem; color:#7b8aa3; margin-top:auto; }}
  .note {{ font-size:.95rem; color:#cfe0ff; margin:.2em 0 .6em; }}
  ul {{ font-size:1.12rem; line-height:1.7; max-width:54em; }}
  img.wide {{ max-width:96%; max-height:64vh; object-fit:contain; align-self:center; margin:.3em 0; }}
  .pair {{ display:flex; gap:1%; justify-content:center; }}
  img.half {{ max-width:49%; max-height:40vh; object-fit:contain; }}
  table.summary {{ border-collapse:collapse; font-size:1rem; margin:.4em 0; }}
  table.summary th, table.summary td {{ border:1px solid #1d2840; padding:.34em .7em; text-align:left; }}
  table.summary th {{ background:#13203a; color:#9fb3d6; }}
  .pm {{ color:#8595af; font-size:.85em; }}
  .ok {{ color:#5fd28a; }} .warn {{ color:#ffd479; }} .bad {{ color:#ff8a8a; }}
  .nav {{ position:fixed; bottom:1.4vh; right:2vw; display:flex; gap:.5em; align-items:center; z-index:10; }}
  .nav button {{ background:#16223c; color:#cfe0ff; border:1px solid #2a3a5c; border-radius:6px;
                 font-size:1.1rem; padding:.25em .7em; cursor:pointer; }}
  .nav button:hover {{ background:#22345a; }}
  #count {{ font-size:.9rem; color:#7b8aa3; min-width:4em; text-align:center; }}
  .barwrap {{ position:fixed; top:0; left:0; height:3px; width:100%; background:#13203a; z-index:10; }}
  #bar {{ height:100%; width:0; background:#7fb3ff; transition:width .15s; }}
</style></head>
<body>
<div class="barwrap"><div id="bar"></div></div>
<div class="deck" id="deck">
{''.join(slides)}
</div>
<div class="nav">
  <button onclick="go(-1)">Prev</button>
  <span id="count"></span>
  <button onclick="go(1)">Next</button>
</div>
<script>
  const slides = document.querySelectorAll('.slide');
  let i = 0;
  function show(n) {{
    i = Math.max(0, Math.min(slides.length-1, n));
    slides.forEach((s,k)=>s.classList.toggle('active', k===i));
    document.getElementById('count').textContent = (i+1)+' / '+slides.length;
    document.getElementById('bar').style.width = ((i+1)/slides.length*100)+'%';
  }}
  function go(d) {{ show(i+d); }}
  document.addEventListener('keydown', e=>{{
    if(e.key==='ArrowRight'||e.key===' ') go(1);
    else if(e.key==='ArrowLeft') go(-1);
    else if(e.key==='Home') show(0);
    else if(e.key==='End') show(slides.length-1);
  }});
  show(0);
</script>
</body></html>
"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT}  ({n} slides; {n_fit} sightlines fit, "
      f"{sum(1 for b in fits if figs_for(b))} with figures)")
