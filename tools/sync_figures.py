"""Sync manuscript-bound figures from the pipeline into the Faber2026 manuscript.

Single source of truth is the figure manifest (default results/figures.manifest.json):
a JSON object keyed by figure filename. An entry is *manuscript-bound* when it
carries both ``source`` (path to the produced figure, relative to the repo root)
and ``dest`` (filename under the manuscript's ``figures/``). Optional ``section``
and ``caption`` drive the emitted LaTeX stub. Entries lacking ``source``/``dest``
are review-only and skipped.

Dry-run by default: resolves sources and prints what would copy + the
``\\includegraphics`` stub. Pass --apply to actually copy into the manuscript.

    python tools/sync_figures.py                 # dry-run, all bound entries
    python tools/sync_figures.py --stubs         # also print LaTeX stubs
    python tools/sync_figures.py --apply          # copy into <manuscript>/figures/
    python tools/sync_figures.py --selftest       # offline self-check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "results" / "figures.manifest.json"
DEFAULT_MANUSCRIPT = Path("~/Developer/overleaf/Faber2026").expanduser()


def bound_entries(manifest: dict) -> list[tuple[str, dict]]:
    """Return (key, entry) pairs that are manuscript-bound (have source+dest)."""
    return [
        (k, v)
        for k, v in manifest.items()
        if isinstance(v, dict) and v.get("source") and v.get("dest")
    ]


def stub(key: str, entry: dict) -> str:
    """LaTeX \\includegraphics block for one figure, matching the manuscript style."""
    label = key.rsplit(".", 1)[0]
    dest = entry["dest"].rsplit(".", 1)[0]  # graphicspath resolves the bare name
    caption = entry.get("caption", entry.get("title", label))
    return (
        f"% --- {entry.get('section', '<section>')} ---\n"
        f"\\begin{{figure}}\n  \\centering\n"
        f"  \\includegraphics[width=0.95\\columnwidth]{{{dest}}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:{label}}}\n\\end{{figure}}"
    )


def sync(
    manifest_path: Path,
    manuscript: Path,
    apply: bool,
    stubs: bool,
    repo: Path = REPO,
    out=sys.stdout,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    entries = bound_entries(manifest)
    if not entries:
        print("no manuscript-bound figures (none carry source+dest)", file=out)
        return 0
    figdir = manuscript / "figures"
    missing = 0
    for key, entry in entries:
        src = repo / entry["source"]
        dst = figdir / entry["dest"]
        if not src.exists():
            print(f"MISSING source: {entry['source']}", file=out)
            missing += 1
            continue
        action = "copy " if apply else "would copy"
        print(f"{action} {entry['source']} -> {dst}", file=out)
        if apply:
            figdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if stubs:
            print(stub(key, entry), file=out)
    return 1 if missing else 0


def _selftest() -> None:
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "results").mkdir()
        (root / "results" / "f.png").write_bytes(b"\x89PNG")
        man = root / "m.json"
        man.write_text(
            json.dumps(
                {
                    "f.png": {
                        "source": "results/f.png",
                        "dest": "f.png",
                        "section": "sections/x.tex",
                        "caption": "Cap",
                    },
                    "review_only.png": {"expectation": "no source -> skipped"},
                }
            )
        )
        ms = root / "ms"
        # dry-run copies nothing
        buf = io.StringIO()
        sync(man, ms, apply=False, stubs=True, repo=root, out=buf)
        assert not (ms / "figures" / "f.png").exists(), "dry-run must not copy"
        assert "review_only" not in buf.getvalue(), "review-only entry must be skipped"
        assert "\\includegraphics" in buf.getvalue()
        # apply copies
        sync(man, ms, apply=True, stubs=False, repo=root, out=io.StringIO())
        assert (ms / "figures" / "f.png").read_bytes() == b"\x89PNG", "apply must copy bytes"
    print("selftest OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    ap.add_argument("--apply", action="store_true", help="copy (default: dry-run)")
    ap.add_argument("--stubs", action="store_true", help="print LaTeX stubs")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    return sync(args.manifest, args.manuscript, args.apply, args.stubs)


if __name__ == "__main__":
    raise SystemExit(main())
