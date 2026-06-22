#!/usr/bin/env python
"""Self-contained HTML slide deck of the multi-component visual analysis.
One slide per burst (CHIME + DSA both shown), framed by context + summary slides.
Embeds the hi-fi profile PNGs as base64 so the deck is a single portable file.
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HI = os.path.join(HERE, "profiles_hi")


def img64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def burst_img(burst):
    p = os.path.join(HI, f"{burst}_hi.png")
    if not os.path.exists(p):
        return f"<p style='color:#f55'>[missing {burst}_hi.png]</p>"
    return f"<img class=burstimg src='{img64(p)}'/>"


# ---- per-burst: (verdict, css class, note, CHIME sub-bursts, DSA sub-bursts).
# "?" = under-resolved (count not determinable by eye at the available sampling).
# Ordered: controls, confirmed, under-resolved ----
BURSTS = [
    ("freya", "CONTROL — clean PASS", "g",
     "Single sharp peak + scattering tail in BOTH bands (find_peaks 83σ / 92σ, α=4.48). Calibrates the count: a clean burst is one peak.",
     "1", "1"),
    ("johndoeII", "CONTROL — rail ≠ pulse", "g",
     "Single in both bands (find_peaks 7σ / 6.5σ). α-railed LOW but ONE peak each → its rail is NOT a hidden pulse (α↔ΔDM degeneracy or genuine sub-Kolmogorov).",
     "1", "1"),
    ("hamilton", "MULTI-COMPONENT", "b",
     "CHIME: close DOUBLET 11.61 + 11.92 ms (find_peaks 22σ + 82σ) — rails α LOW; a 3rd peak sits right at the 4σ floor (marginal). DSA: clean single (15σ).",
     "2", "1"),
    ("whitney", "MULTI-COMPONENT (DSA)", "b",
     "DSA: clear DOUBLE — 28.64 + 28.97 ms (find_peaks 53σ + 22σ). CHIME: single at 4σ (6σ peak; any 2nd is sub-threshold). [CORRECTED: my earlier visual call had this backwards as CHIME-2 / DSA-1 — wrong in BOTH bands.]",
     "1", "2"),
    ("phineas", "MULTI-COMPONENT (DSA)", "b",
     "DSA (145t): sharp peak 2.56 ms + DISTINCT 2nd peak 3.87 ms (find_peaks 79σ + 8.9σ). CHIME: broad single.",
     "1", "2"),
    ("zach", "MIXED — DSA 2nd pulse, CHIME tail-shape", "y",
     "DSA (577t): main 9.63 ms + 2nd peak 12.12 ms (find_peaks 47σ + 15σ). CHIME: SINGLE peak (63σ) + long FREQUENCY-DEPENDENT scattering tail — the flagged CHIME residual is tail-shape, NOT a 2nd pulse. A 2-comp fit is the wrong fix for CHIME.",
     "1", "2"),
    ("oran", "UNDER-RESOLVED", "y",
     "DSA (only 21t): NO peak clears 4σ — band-integrated profile too coarse / noise-estimate breaks. CHIME: one marginal peak (4.1σ), noise-dominated. α-railed HIGH and multimodal (α=1.44 vs 5.96).",
     "1", "?"),
    ("chromatica", "UNDER-RESOLVED (drift?)", "y",
     "DSA (24t): one peak at 7.3σ but sub-bands peak at DIFFERENT times → frequency drift / smear as much as a discrete pulse; tentative single. CHIME: clean single (62σ).",
     "1", "1?"),
    ("mahi", "UNDER-RESOLVED", "y",
     "DSA (12t): NO peak clears 4σ — too coarse to count. CHIME: single (7σ, low S/N).",
     "1", "?"),
    ("isha", "UNDER-RESOLVED", "y",
     "DSA (only 9t): NO peak clears 4σ; τ unresolved (smallest in the sample). CHIME: single (8.6σ).",
     "1", "?"),
]

slides = []

slides.append(("Multi-component structure in the co-detected FRB sample",
    """<p class=sub>One slide per burst — CHIME + DSA on-pulse profiles (the exact window the joint fit sees).</p>
    <div class=box><b>Bottom line (10 bursts).</b>
    <ul>
      <li><b>4 show a discrete 2nd pulse</b> — hamilton, whitney, phineas, zach&nbsp;DSA → multi-component fit is the right fix.</li>
      <li><b>1 is a frequency-dependent scattering tail, not a pulse</b> — zach&nbsp;CHIME → a 2nd component is the <i>wrong</i> fix.</li>
      <li><b>4 under-resolved</b> — oran, chromatica, mahi, isha: DSA windows of 9–24 samples can't separate pulse vs drift vs tail.</li>
      <li><b>2 clean controls</b> — freya, johndoeII (single) → the read is calibrated.</li>
    </ul>
    Hypothesis <b>partly confirmed, partly complicated</b>: "add a 2nd component" is not a universal fix.</div>"""))

slides.append(("The problem &amp; the hypothesis",
    """<p>9 of 12 co-detected bursts were <b>excluded</b> from the joint scintillation analysis by a fit-quality gate: 3 α-railed (hamilton, johndoeII, oran), 6 temporal-fail (non-white residual: chromatica, isha, mahi, phineas, whitney, zach).</p>
    <div class=box><b>Hypothesis.</b> An <i>unmodeled</i> 2nd pulse forces the one-component fit to absorb it by distorting the tail and railing α. A 2-component fit should whiten the residual <i>and</i> un-rail α — making the gate a model limitation, not a data property. "Hidden" = hidden from the <i>model</i>, not the eye.</div>
    <p class=note>This deck tests the premise by eye: do the flagged bursts actually show a 2nd pulse?</p>"""))

slides.append(("The capability — built &amp; verified (synthetic only)",
    """<ul>
      <li>Multi-component gain-marginal joint likelihood: proper finite-variance gain prior (valid lnZ), min-separation prior, per-component per-channel matched filter.</li>
      <li><b>N=1 reduces exactly</b> to the existing likelihood (3.5e-11); <b>merge singularity killed</b> (pure-noise ΔlnZ 0.014, was +324 nats) — caught by an adversarial pass.</li>
      <li><b>Injection-recovery (real sampler):</b> a hidden 2nd pulse rails 1-comp α to the 1.50 floor; 2-comp recovers <b>α=3.45±0.07</b> (truth 3.5). ΔlnZ=+10267.</li>
    </ul>
    <p class=note>Real-burst un-rail is unproven — oran N=2 pilot running on HPCC (job 64440466).</p>"""))

slides.append(("How to read each burst slide",
    """<p>Per band: band-integrated profile (black) + 3 frequency sub-band profiles + waterfall, zoomed to the on-pulse window. Any non-white residual <i>must</i> originate inside this window.</p>
    <ul>
      <li><b>Single peak + monotonic decay</b> → scattering tail (1 component).</li>
      <li><b>A distinct bump after a dip</b> → a 2nd component.</li>
      <li><b>Sub-bands peaking at different times</b> → frequency drift / smear (not a discrete pulse).</li>
    </ul>
    <p class=note>The original 2×2 thumbnails crushed a 0.3 ms doublet into one spike — these full-width panels are what the calls rest on.</p>
    <div class=box><b>Confidence key.</b> Every burst slide carries an explicit tag:
    <span class=confhi>CONFIDENCE: HIGH</span> = countable by eye ·
    <span class=conflo>CONFIDENCE: LOW</span> = DSA too coarsely sampled (9–24 samples) to count.
    The OUTCOME slide tabulates this; counts there are confident for 6/10 bursts.</div>"""))

# one slide per burst
for burst, label, cls, note, cN, dN in BURSTS:
    low = ("?" in cN) or ("?" in dN)
    pill = ("<span class=conflo>CONFIDENCE: LOW — DSA under-sampled, count not reliable</span>"
            if low else "<span class=confhi>CONFIDENCE: HIGH</span>")
    badge = (f"<div class=counts>sub-bursts &nbsp;→&nbsp; "
             f"<span class=cnt>CHIME: <b>{cN}</b></span> &nbsp;·&nbsp; "
             f"<span class=cnt>DSA: <b>{dN}</b></span> &nbsp;&nbsp; {pill}</div>")
    slides.append((f"{burst} — <span class={cls}>{label}</span>",
                   f"{badge}<p class=bnote>{note}</p>{burst_img(burst)}", "burst"))

# ---- outcome assessment: sub-burst counts ----
def conf(cN, dN):
    return "indeterminate" if ("?" in cN or "?" in dN) else "confident"
rows = "".join(
    f"<tr><td>{b}</td><td class=ctr><b>{cN}</b></td><td class=ctr><b>{dN}</b></td>"
    f"<td class={'y' if conf(cN,dN)=='indeterminate' else 'g'}>{conf(cN,dN)}</td></tr>"
    for b, _l, _c, _n, cN, dN in BURSTS)
n_known = sum(1 for *_x, cN, dN in BURSTS if "?" not in cN and "?" not in dN)
slides.append(("OUTCOME — sub-burst counts (deterministic, find_peaks ≥4σ)", f"""
    <p class=sub>Counts from scipy find_peaks (prominence ≥4σ of MAD noise) on the band-integrated profile — objective, after visual counts disagreed. "?" = no peak clears 4σ / sampling too coarse.</p>
    <table>
    <tr><th>burst</th><th>CHIME N</th><th>DSA N</th><th>confidence</th></tr>
    {rows}
    </table>
    <div class=box><b>Assessment.</b> Confident for {n_known}/10. <b>CHIME multi only in hamilton (2, +1 marginal); single elsewhere.</b> <b>DSA multi in whitney, phineas, zach (2 each)</b> — strong (2nd-peak prominence 9–22σ). The coarse-DSA trio oran/mahi/isha returns <b>no ≥4σ peak</b> (12–9–21t; even the algorithm's noise estimate breaks) → genuinely indeterminate, the actionable gap (re-extract at finer DSA time resolution). <b>Correction:</b> my earlier visual calls were wrong on 3 bursts (whitney both bands, phineas DSA, johndoeII DSA) — deterministic peak-finding supersedes the eyeball.</div>"""))

slides.append(("Per-burst summary", """
    <table>
    <tr><th>burst</th><th>CHIME</th><th>DSA</th><th>flagged</th><th>verdict</th></tr>
    <tr><td>freya</td><td>single</td><td>single</td><td>— (PASS)</td><td class=g>clean control</td></tr>
    <tr><td>johndoeII</td><td>single</td><td>single</td><td>α-rail</td><td class=g>rail ≠ pulse</td></tr>
    <tr><td>hamilton</td><td><b>double</b> 0.3ms</td><td>single</td><td>CHIME</td><td class=b>multi-component</td></tr>
    <tr><td>whitney</td><td>single</td><td><b>double</b></td><td>CHIME</td><td class=b>multi (DSA); note: flagged band is CHIME but its double is in DSA</td></tr>
    <tr><td>phineas</td><td>broad</td><td><b>double</b> 3.8ms</td><td>DSA</td><td class=b>multi-component</td></tr>
    <tr><td>zach</td><td>single + ν-tail</td><td><b>double</b> 11.5ms</td><td>CHIME</td><td class=y>mixed: CHIME tail-shape, DSA 2nd comp</td></tr>
    <tr><td>oran</td><td>noise-dom.</td><td>struct., 21t</td><td>DSA</td><td class=y>under-resolved</td></tr>
    <tr><td>chromatica</td><td>single</td><td>drift?, 24t</td><td>DSA</td><td class=y>under-resolved (drift/smear)</td></tr>
    <tr><td>mahi</td><td>single</td><td>broad, 12t</td><td>DSA</td><td class=y>under-resolved</td></tr>
    <tr><td>isha</td><td>struct./noisy</td><td>9t</td><td>both</td><td class=y>under-resolved</td></tr>
    </table>
    <p class=cap>4 confirmed multi · 1 ν-dependent-tail · 4 under-resolved · 2 clean controls</p>"""))

slides.append(("Implications &amp; next steps", """
    <ul>
      <li><b>Confirmed doubles</b> (hamilton, whitney, phineas, zach-DSA): 2-component joint fit is the right fix; expect α un-rail + whitening.</li>
      <li><b>ν-dependent tail</b> (zach-CHIME): NOT multi-component — needs a tail-shape/frequency model.</li>
      <li><b>Under-resolved DSA</b> (oran, chromatica, mahi, isha): <b>re-extract at finer DSA time resolution</b> before fitting; multi-comp weakly constrained at 9–24 samples.</li>
      <li><b>Methodology:</b> replace the hard exclusion gate with <b>marginalize-not-gate</b> — propagate (α, ΔDM) posterior into Δν_d so a burst stays in with honest error bars.</li>
      <li><b>In flight:</b> oran N=2 pilot (HPCC job 64440466) — but oran is coarse-DSA, so treat its result as weakly constrained.</li>
    </ul>"""))

slides.append(("Honest caveats", """
    <ul>
      <li>Multi-component likelihood verified on <b>synthetic</b> injections only; real-burst α un-rail unproven (pilot running).</li>
      <li>Initial visual calls off downscaled thumbnails were <b>wrong</b> (missed hamilton's 0.3 ms doublet) — conclusions use full-width renders.</li>
      <li>Local <code>joint_json</code> posteriors are <b>stale</b> pre-fix junk — all numbers use HPCC post-fix fits.</li>
      <li>oran α is <b>multimodal</b> (1.44 vs 5.96 via the α↔ΔDM ridge) — run multimodal-aware.</li>
      <li>"2nd pulse" is the leading explanation for a non-white residual, <b>not the only one</b> — drift, DM smearing, non-exponential tails mimic it; the evidence test arbitrates.</li>
    </ul>"""))

body = ""
for i, (title, html, *rest) in enumerate(slides):
    kind = rest[0] if rest else ""
    body += f"<section class='slide {kind}'><div class=num>{i+1} / {len(slides)}</div><h1>{title}</h1>{html}</section>\n"

doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Multi-component analysis</title>
<style>
  :root{{--bg:#0e1116;--fg:#e6edf3;--accent:#58a6ff;--g:#3fb950;--y:#d29922;--b:#58a6ff;}}
  *{{box-sizing:border-box}} body{{margin:0;background:#000;color:var(--fg);font:18px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
  .slide{{display:none;min-height:100vh;padding:3vh 5vw;background:var(--bg);flex-direction:column}}
  .slide.active{{display:flex}}
  .slide.burst{{align-items:center}}
  h1{{color:var(--accent);font-size:28px;margin:0 0 12px;border-bottom:1px solid #30363d;padding-bottom:8px;width:100%}}
  .sub{{font-size:21px;color:#8b949e;margin-top:-4px}}
  ul{{margin:8px 0}} li{{margin:6px 0}}
  .box{{background:#161b22;border-left:3px solid var(--accent);padding:12px 16px;border-radius:4px;margin:12px 0}}
  .note{{color:#8b949e;font-size:15px;font-style:italic}}
  .bnote{{font-size:17px;max-width:1100px;margin:0 0 10px;text-align:center}}
  .counts{{font-size:20px;color:#e6edf3;margin:0 0 8px}} .cnt{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:4px 14px}} .cnt b{{color:var(--accent);font-size:22px}}
  .confhi{{background:#11341c;color:var(--g);border:1px solid var(--g);border-radius:14px;padding:4px 12px;font-size:14px;font-weight:600}}
  .conflo{{background:#3a2d0a;color:var(--y);border:1px solid var(--y);border-radius:14px;padding:4px 12px;font-size:14px;font-weight:600}}
  .ctr{{text-align:center}}
  .cap{{color:#8b949e;font-size:14px;margin:6px 0 0;text-align:center}}
  img.burstimg{{max-height:72vh;width:auto;max-width:62%;border:1px solid #30363d;border-radius:6px;background:#fff}}
  table{{border-collapse:collapse;width:100%;font-size:15px}} th,td{{border:1px solid #30363d;padding:6px 10px;text-align:left}}
  th{{background:#161b22;color:var(--accent)}} .g{{color:var(--g)}} .y{{color:var(--y)}} .b{{color:var(--b);font-weight:600}}
  code{{background:#161b22;padding:1px 5px;border-radius:3px}}
  .num{{position:fixed;top:14px;right:20px;color:#8b949e;font-size:13px}}
  .hint{{position:fixed;bottom:12px;left:20px;color:#484f58;font-size:12px}}
</style></head><body>
{body}
<div class=hint>← → or space to navigate · {len(slides)} slides</div>
<script>
let i=0;const s=document.querySelectorAll('.slide');
function show(n){{i=Math.max(0,Math.min(s.length-1,n));s.forEach((x,k)=>x.classList.toggle('active',k===i));location.hash=i+1}}
document.addEventListener('keydown',e=>{{if(['ArrowRight',' ','PageDown'].includes(e.key))show(i+1);
 else if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);else if(e.key==='Home')show(0);else if(e.key==='End')show(s.length-1)}});
show((parseInt(location.hash.slice(1))||1)-1);
</script></body></html>"""

out = os.path.join(HERE, "multicomponent_analysis_deck.html")
with open(out, "w") as f:
    f.write(doc)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, {len(slides)} slides)")
