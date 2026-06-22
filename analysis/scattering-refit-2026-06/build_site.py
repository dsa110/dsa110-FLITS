"""Emit a static GitHub Pages gallery (index.html + assets/) for the CHIME
scattering deck. One section per burst, best->worst by residual sigma."""
import os, shutil, html

FQ = "/tmp/fq_local"; PDF = "/tmp/scattering_deck.pdf"
SITE = "/tmp/faber_pages"; ASSETS = f"{SITE}/assets"

# burst, model, tau_str, zeta_str, chi2, resid_sigma, r2, verdict, tier(css)
# ORIENTATION-CORRECTED: the CHIME+DSA freq axis was stored descending; these
# tau values supersede the earlier backwards-freq fits.
B = [
 ("mahi","M2","0.212 (+0.015/−0.014)","— (M2)","1.053","1.03","0.045","detection (M2: scattering-only)","ok"),
 ("oran","M3","0.540 (+0.063/−0.059)","1.95","1.058","1.03","0.060","detection","ok"),
 ("johndoeII","M3","0.143 (+0.0055/−0.0048)","0.12","1.120","1.06","0.188","detection","ok"),
 ("whitney","M2","0.117 (+0.0057/−0.0053)","— (M2)","1.164","1.08","0.099","detection (M2: scattering-only)","ok"),
 ("phineas","M3","0.274 (+0.011/−0.011)","1.26","1.199","1.10","0.323","detection","ok"),
 ("isha","M3","0.290 (+0.016/−0.014)","0.78","1.216","1.10","0.152","detection","ok"),
 ("wilhelm","M3","0.144 (+0.0055/−0.0054)","0.17","1.704","1.31","0.281","marginal","mg"),
 ("hamilton","M2","0.020 (+0.0002/−0.0002)","— (M2)","3.362","1.83","0.338","marginal","mg"),
 ("chromatica","M3","0.114 (+0.0009/−0.0008)","0.46","5.113","2.26","0.663","poor fit","bad"),
 ("freya","M2","0.150 (+0.0006/−0.0007)","— (M2)","9.560","3.09","0.504","FAIL — under-dedispersed","bad"),
 ("zach","M3","0.262 (+0.0009/−0.0009)","0.16","22.000","4.68","0.614","FAIL — under-dedispersed","bad"),
]
PENDING = [("casey","refit re-running orientation-corrected (M3 stalls at noise floor; M1 strongly preferred). Slide updates when it lands.")]

os.makedirs(ASSETS, exist_ok=True)
for b, *_ in B:
    shutil.copy(f"{FQ}/{b}_fq.png", f"{ASSETS}/{b}_fq.png")
shutil.copy(PDF, f"{ASSETS}/scattering_deck.pdf")

COL = {"ok":"#1a7a1a","nd":"#1a5a9a","mg":"#b8860b","bad":"#b00020"}

rows = "\n".join(
    f'<tr class="{t}"><td>{html.escape(b)}</td><td>{m}</td><td>{tau}</td>'
    f'<td>{z}</td><td>{c}</td><td><b>{rs}</b></td><td>{html.escape(v)}</td></tr>'
    for (b,m,tau,z,c,rs,r2,v,t) in B)

cards = "\n".join(
    f'<section class="card {t}"><h2>{html.escape(b.upper())} '
    f'<span class="badge" style="background:{COL[t]}">{html.escape(v)}</span></h2>'
    f'<p class="meta">model {m} · τ₁GHz = {tau} ms · ζ = {z} ms · χ²ᵣ = {c} · '
    f'residual σ = <b>{rs}</b> (target 1.0) · R² = {r2}</p>'
    f'<img src="assets/{b}_fq.png" alt="{b} fit-quality figure" loading="lazy"></section>'
    for (b,m,tau,z,c,rs,r2,v,t) in B)

pending = "\n".join(
    f'<section class="card mg"><h2>{b.upper()} <span class="badge" '
    f'style="background:#777">pending</span></h2><p class="meta">{html.escape(note)}</p></section>'
    for b,note in PENDING)

DOC = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>DSA-110 × CHIME — CHIME-band scattering fits</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fafafa}}
 header{{background:#11243f;color:#fff;padding:28px 6vw}}
 header h1{{margin:0 0 4px;font-size:26px}} header p{{margin:2px 0;color:#b9c7da}}
 main{{max-width:1100px;margin:0 auto;padding:24px 5vw 60px}}
 a.dl{{display:inline-block;margin:14px 0;padding:9px 16px;background:#11243f;color:#fff;
   text-decoration:none;border-radius:6px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;background:#fff}}
 th,td{{padding:6px 9px;border-bottom:1px solid #e5e5e5;text-align:left}}
 th{{background:#f0f2f5}} td:first-child{{font-weight:600}}
 tr.ok td:first-child{{color:#1a7a1a}} tr.nd td:first-child{{color:#1a5a9a}}
 tr.mg td:first-child{{color:#b8860b}} tr.bad td:first-child{{color:#b00020}}
 .card{{background:#fff;border:1px solid #e5e5e5;border-left:5px solid #999;
   border-radius:8px;padding:14px 18px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 .card.ok{{border-left-color:#1a7a1a}} .card.nd{{border-left-color:#1a5a9a}}
 .card.mg{{border-left-color:#b8860b}} .card.bad{{border-left-color:#b00020}}
 .card h2{{margin:0 0 4px;font-size:19px}}
 .badge{{font-size:12px;color:#fff;padding:2px 9px;border-radius:11px;vertical-align:middle;font-weight:600}}
 .meta{{color:#555;font-size:13px;margin:0 0 10px}}
 .card img{{width:100%;height:auto;border:1px solid #eee;border-radius:4px}}
 .note{{color:#777;font-size:12.5px;margin-top:30px;border-top:1px solid #ddd;padding-top:12px}}
</style></head><body>
<header>
 <h1>DSA-110 × CHIME co-detected FRBs — CHIME-band scattering fits</h1>
 <p>Nested sampling + Bayesian evidence (M0–M3) · init-independent priors · α fixed = 4.0 · 16 chan, 0.4–0.8 GHz</p>
 <p><b>Orientation-corrected</b> — the CHIME/DSA freq axis was stored descending; these τ supersede the earlier backwards-freq values. Fit quality judged by residual σ (target = 1.0). Unpublished — Faber et&nbsp;al. 2026 (in prep).</p>
</header>
<main>
 <a class="dl" href="assets/scattering_deck.pdf">⬇ Download deck (PDF)</a>
 <p><b>6 clean detections</b> (σ≈1): mahi, oran, johndoeII, whitney, phineas, isha ·
    <b>2 marginal</b>: wilhelm, hamilton ·
    <b>3 fail</b>: chromatica, freya, zach (freya/zach under-dedispersed) · casey pending.</p>
 <table><thead><tr><th>burst</th><th>model</th><th>τ₁GHz (ms)</th><th>ζ (ms)</th>
   <th>χ²/dof</th><th>resid σ</th><th>verdict</th></tr></thead><tbody>
 {rows}
 </tbody></table>
 {cards}
 {pending}
 <p class="note">Residual σ is the standard deviation of the per-channel, noise-normalized
  residuals; a correct fit gives σ≈1 with a white (structureless) residual map. χ²/dof and
  R² alone are ambiguous for faint bursts (low R² with χ²≈1). Generated by the FLITS
  scattering pipeline (plot_fit_quality). Pre-publication; do not redistribute.</p>
</main></body></html>"""

open(f"{SITE}/index.html","w").write(DOC)
open(f"{SITE}/.nojekyll","w").write("")
print(f"wrote {SITE}/index.html + {len(B)+1} assets")
