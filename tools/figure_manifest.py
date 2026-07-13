"""Emit a figures.manifest.json so the figure-review gate can require a visual check.

Any figure-producing script should call `write_manifest(out_dir, [(png, expectation), ...])`
after saving its PNGs. The Stop hook (.claude/hooks/figure-review-gate.sh) then blocks
end-of-turn until a figures.review.json (per-figure visual verdicts) exists and is at least
as new as the manifest. See docs/dev/figure-review-protocol.md.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterable, Tuple


def write_manifest(out_dir: str, figures: Iterable[Tuple[str, str]]) -> str:
    """Write <out_dir>/figures.manifest.json.

    figures: iterable of (filename, expectation) — filename relative to out_dir,
    expectation a one-line description of what the figure SHOULD show (used by the
    reviewer to judge match vs anomaly).
    """
    path = os.path.join(out_dir, "figures.manifest.json")
    existing = []
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh).get("figures", [])
    by_path = {str(figure["path"]): figure for figure in existing}
    for filename, expectation in figures:
        by_path[str(filename)] = {"path": str(filename), "expect": str(expectation)}
    manifest = {"created": time.time(), "figures": list(by_path.values())}
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


__all__ = ["write_manifest"]
