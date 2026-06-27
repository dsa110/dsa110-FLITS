"""Authorized network re-run of the foreground-galaxy search for the 3 EXCESS sightlines
that have zero committed foreground galaxies (Wilhelm/Hamilton/Chromatica).

Non-destructive: writes to results/network_rerun/ (NOT over the committed *_galaxies.csv).
Enables the opt-in DESI DR1 engine via in-namespace monkeypatch (no config.py edit);
ENABLE_CLUSTER_ENGINE is already True, ENABLE_ENRICHERS is unused in the package.
"""

from galaxies.v2_0 import search as s

s.ENABLE_EXTRA_ENGINES = True  # adds DesiDr1Engine (covers Whitney/Phineas/Casey; empty elsewhere)
s.TARGETS = [t for t in s.TARGETS if t[0] in {"Wilhelm", "Hamilton", "Chromatica"}]
assert len(s.TARGETS) == 3, s.TARGETS

s.run_search(output_dir="results/network_rerun", build_unified=True)
