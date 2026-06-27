"""Probe each foreground-search engine for a single target with a hard per-query
timeout, to isolate which Vizier/NED query hangs the full search regeneration."""

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FTimeout

import astropy.units as u

from galaxies.v2_0 import config
from galaxies.v2_0.engines import NedEngine, VizierEngine
from galaxies.v2_0.engines_extra import ClusterEngine
from galaxies.v2_0.utils import get_angular_radius, parse_coord

NAME, RA, DEC, Z = config.TARGETS[0]  # Zach (lowest effective z -> largest cone)
coord = parse_coord(RA, DEC)
radius = min(
    get_angular_radius(min(Z, config.MIN_Z_SEARCH), max(100.0, config.DEFAULT_CLUSTER_IMPACT_KPC)),
    config.MAX_SEARCH_RADIUS_DEG * u.deg,
)
print(f"target={NAME} coord=({coord.ra.deg:.4f},{coord.dec.deg:.4f}) radius={radius.to(u.deg):.3f}")

engines = [("NED", NedEngine())]
for cat_name, cat_id in config.VIZIER_CATALOGS.items():
    engines.append((f"VIZIER:{cat_name}={cat_id}", VizierEngine(cat_id)))
for cat_name, cat_id in config.CLUSTER_VIZIER_CATALOGS.items():
    engines.append((f"CLUSTER:{cat_name}={cat_id}", VizierEngine(cat_id)))
engines.append(("ClusterEngine(all 3)", ClusterEngine()))

TIMEOUT = 60.0


def run_one(eng):
    t0 = time.time()
    df = eng.query(coord, radius)
    return len(df), time.time() - t0


with ThreadPoolExecutor(max_workers=len(engines)) as ex:
    futs = {ex.submit(run_one, eng): label for label, eng in engines}
    for fut, label in list(futs.items()):
        pass
    for label, eng in engines:
        pass
    # Submit fresh so each has its own future keyed by label.
    futs = {label: ex.submit(run_one, eng) for label, eng in engines}
    for label, fut in futs.items():
        try:
            n, dt = fut.result(timeout=TIMEOUT)
            print(f"  [{dt:6.1f}s] {label}: {n} rows")
        except FTimeout:
            print(f"  [TIMEOUT >{TIMEOUT:.0f}s] {label}: HANG")
        except Exception as e:
            print(f"  [ERROR] {label}: {type(e).__name__}: {e}")
