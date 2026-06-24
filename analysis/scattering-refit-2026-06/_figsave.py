"""Save a figure in publication-vector form (PDF for LaTeX + SVG) and keep a PNG.

The Faber2026 manuscript builds with pdflatex, which embeds the PDF; SVG is the
archival vector form; the PNG stays for the HTML deck / figure-review gate. Pass
a path *stem* (extension optional/ignored). ``dpi`` applies to the PNG only —
vector formats ignore it. Returns the .pdf path (the manuscript-bound artifact).
"""

VECTOR = ("pdf", "svg")


def save_fig(fig, stem, *, png=True, dpi=110, **kw):
    stem = str(stem)
    if stem.lower().endswith((".png", ".pdf", ".svg")):
        stem = stem.rsplit(".", 1)[0]
    kw.setdefault("bbox_inches", "tight")
    for ext in VECTOR:
        fig.savefig(f"{stem}.{ext}", **kw)  # vector: dpi-independent
    if png:
        fig.savefig(f"{stem}.png", dpi=dpi, **kw)
    return f"{stem}.pdf"


def _selftest():
    import tempfile
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as d:
        pdf = save_fig(fig, f"{d}/x.png", dpi=80)  # pass a .png stem on purpose
        assert pdf == f"{d}/x.pdf"
        for ext in ("pdf", "svg", "png"):
            assert Path(f"{d}/x.{ext}").exists(), ext
        plt.close(fig)
        # png=False skips the raster
        save_fig(plt.subplots()[0], f"{d}/y", png=False)
        assert Path(f"{d}/y.pdf").exists() and not Path(f"{d}/y.png").exists()
    print("ok")


if __name__ == "__main__":
    _selftest()
