"""One-slide-per-burst PDF deck of the CHIME scattering fits (16:9).
Each slide = fit-quality figure + parameters + resid-sigma verdict.
No new deps (matplotlib PdfPages)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FQ = "/tmp/fq_local"
OUT = "/tmp/scattering_deck.pdf"

# stats: model, tau(med,-,+) ms, zeta(med,-,+) ms, chi2_red, resid_sigma, r2, verdict
# ordered best -> worst by resid_sigma.  ORIENTATION-CORRECTED (CHIME+DSA freq
# axis was stored descending; fits now pair tau(nu) with the right frequency).
B = [
 ("mahi","M2",(0.212,0.014,0.015),None,1.053,1.026,0.045,"detection (M2: scattering-only)"),
 ("oran","M3",(0.540,0.059,0.063),(1.949,0.172,0.176),1.058,1.028,0.060,"detection"),
 ("johndoeII","M3",(0.143,0.0048,0.0055),(0.116,0.012,0.013),1.120,1.058,0.188,"detection"),
 ("whitney","M2",(0.117,0.0053,0.0057),None,1.164,1.079,0.099,"detection (M2: scattering-only)"),
 ("phineas","M3",(0.274,0.011,0.011),(1.255,0.046,0.045),1.199,1.095,0.323,"detection"),
 ("isha","M3",(0.290,0.014,0.016),(0.781,0.045,0.047),1.216,1.102,0.152,"detection"),
 ("wilhelm","M3",(0.144,0.0054,0.0055),(0.168,0.009,0.009),1.704,1.305,0.281,"marginal"),
 ("hamilton","M2",(0.020,0.0002,0.0002),None,3.362,1.831,0.338,"marginal (M2)"),
 ("chromatica","M3",(0.114,0.0008,0.0009),(0.459,0.004,0.004),5.113,2.256,0.663,"poor fit"),
 ("freya","M2",(0.150,0.0007,0.0006),None,9.560,3.090,0.504,"FAIL — under-dedispersed (DM sweep)"),
 ("zach","M3",(0.262,0.0009,0.0009),(0.159,0.002,0.002),22.000,4.683,0.614,"FAIL — under-dedispersed (DM sweep)"),
]
PENDING = [("casey","refit re-running orientation-corrected (M3 stalls at noise floor; M1 strongly preferred). Slide updates when it lands.")]

VC = {"detection":"#1a7a1a", "non-detection (no scattering term)":"#1a5a9a",
      "marginal":"#b8860b", "marginal (M2: no intrinsic width)":"#b8860b",
      "poor fit":"#cc5500", "FAIL — under-dedispersed (DM sweep)":"#b00020"}

def fmt(t):
    if t is None: return "—"
    m, lo, hi = t
    return f"{m:.4g}  (+{hi:.2g}/−{lo:.2g})"

with PdfPages(OUT) as pdf:
    # title slide
    fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.70, "DSA-110 × CHIME Co-detected FRBs", ha="center", size=30, weight="bold")
    fig.text(0.5, 0.62, "CHIME-band scattering fits  (orientation-corrected)", ha="center", size=20, color="0.3")
    fig.text(0.5, 0.50,
             "Nested sampling + Bayesian evidence (M0–M3)  ·  init-independent priors  ·  α fixed = 4.0\n"
             "16 chan, 0.4–0.8 GHz  ·  outer_trim 0.15, nlive 400  ·  fit quality judged by residual σ (target = 1)\n"
             "freq axis was stored descending — every τ here supersedes the earlier (backwards-freq) values",
             ha="center", size=13, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.32,
             "6 clean detections (σ≈1):  mahi · oran · johndoeII · whitney · phineas · isha\n"
             "2 marginal:  wilhelm · hamilton        3 fail:  chromatica · freya · zach\n"
             "(freya/zach under-dedispersed — DM, not the fit)        casey: refit in progress",
             ha="center", size=13, color="0.15", linespacing=1.7)
    fig.text(0.5, 0.06, "1 slide / burst — fit-quality figure + parameters + verdict", ha="center", size=11, color="0.5")
    pdf.savefig(fig); plt.close(fig)

    for burst, model, tau, zeta, chi2, rsig, r2, verdict in B:
        fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")
        col = VC.get(verdict, "0.2")
        # header
        fig.text(0.04, 0.93, burst.upper(), size=26, weight="bold")
        fig.text(0.04, 0.88, f"best model: {model}", size=14, color="0.3")
        fig.text(0.96, 0.93, verdict, ha="right", size=15, weight="bold", color=col)
        fig.text(0.96, 0.885, f"residual σ = {rsig:.2f}   (target 1.0)", ha="right", size=13, color=col)
        # figure image
        ax = fig.add_axes([0.03, 0.10, 0.74, 0.74]); ax.axis("off")
        ax.imshow(plt.imread(f"{FQ}/{burst}_fq.png"))
        # stats panel
        sx = 0.79
        lines = [
            ("τ₁GHz (ms)", fmt(tau)),
            ("ζ intrinsic (ms)", fmt(zeta)),
            ("χ²/dof", f"{chi2:.3f}"),
            ("residual σ", f"{rsig:.3f}"),
            ("R²", f"{r2:.3f}"),
        ]
        y = 0.74
        for k, v in lines:
            fig.text(sx, y, k, size=12, color="0.45"); fig.text(sx, y-0.035, v, size=14, weight="bold")
            y -= 0.10
        pdf.savefig(fig); plt.close(fig)

    for burst, note in PENDING:
        fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")
        fig.text(0.04, 0.93, burst.upper(), size=26, weight="bold")
        fig.text(0.5, 0.5, note, ha="center", size=16, color="0.3", wrap=True)
        pdf.savefig(fig); plt.close(fig)

print(f"wrote {OUT}  ({len(B)+1+len(PENDING)} slides)")
