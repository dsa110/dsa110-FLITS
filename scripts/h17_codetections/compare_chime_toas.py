#!/usr/bin/env python3
"""Compare re-extracted CHIME 400 MHz TOAs against notebook-derived values.

Pure verification: residual_ms = (reextract toa_unix_400 - fixture toa_unix_400)*1e3.
Because measured_offset = chime_400 - dsa_400 and the DSA timing is held fixed,
the implied new offset is simply golden_offset + residual_ms.

Reads only; never writes crossmatching/toa_crossmatch_results.json.
Runs on the host (stdlib only).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data/research/astrophysics/frbs/chime-dsa-codetections")
REPO = Path("/home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS")
REEXTRACT = ROOT / "results" / "chime_singlebeam_toa_reextract.json"
FIXTURE = REPO / "crossmatching" / "notebook_reproduction_fixture.json"
GOLDEN = REPO / "crossmatching" / "toa_crossmatch_results.json"
OUT_JSON = ROOT / "results" / "chime_singlebeam_toa_comparison.json"
OUT_CSV = ROOT / "results" / "chime_singlebeam_toa_comparison.csv"

# tolerance ladder (ms)
SAMPLE_MS = 2.56e-3  # one CHIME baseband sample
TOL_MATCH = 0.1  # sub-0.1 ms == reproduces notebook value


def classify(resid_ms: float) -> str:
    a = abs(resid_ms)
    if a <= SAMPLE_MS:
        return "exact (<=1 sample)"
    if a <= TOL_MATCH:
        return "match (<=0.1 ms)"
    if a <= 1.0:
        return "close (<=1 ms)"
    return "MISMATCH (>1 ms)"


def main() -> int:
    reextract = json.load(open(REEXTRACT))
    fixture = {b["chime_id"]: b for b in json.load(open(FIXTURE))["bursts"]}
    golden = json.load(open(GOLDEN))
    golden_by_id = {v["chime_id"]: (k, v) for k, v in golden.items()}

    rows = []
    for r in reextract:
        cid = str(r.get("event_id"))
        fx = fixture.get(cid)
        gname, g = golden_by_id.get(cid, (None, {}))
        if r.get("status") != "ok" or fx is None:
            rows.append(
                {
                    "name": r.get("name", gname),
                    "event_id": cid,
                    "status": r.get("status", "?"),
                    "verdict": "not-extracted",
                }
            )
            continue
        fx_unix = fx["chime"]["toa_unix_400"]
        resid_ms = (r["toa_unix_400"] - fx_unix) * 1e3
        golden_offset = g.get("measured_offset_ms")
        implied_offset = (golden_offset + resid_ms) if golden_offset is not None else None
        rows.append(
            {
                "name": gname or r.get("name"),
                "event_id": cid,
                "dm": r["dm"],
                "peak_idx": r["peak_idx"],
                "reextract_toa_unix_400": r["toa_unix_400"],
                "fixture_toa_unix_400": fx_unix,
                "residual_ms": resid_ms,
                "residual_samples": resid_ms / SAMPLE_MS,
                "golden_measured_offset_ms": golden_offset,
                "implied_new_offset_ms": implied_offset,
                "n_rfi_masked": r.get("n_rfi_channels_masked"),
                "snr_like": r.get("snr_like"),
                "verdict": classify(resid_ms),
            }
        )

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    cols = [
        "name",
        "event_id",
        "dm",
        "peak_idx",
        "residual_ms",
        "residual_samples",
        "golden_measured_offset_ms",
        "implied_new_offset_ms",
        "snr_like",
        "n_rfi_masked",
        "verdict",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(
        f"{'name':12s} {'event_id':10s} {'peak':>7s} {'resid_ms':>12s} "
        f"{'resid_samp':>11s} {'verdict':>20s}"
    )
    print("-" * 78)
    ok = []
    for r in rows:
        if r.get("verdict") == "not-extracted":
            print(
                f"{str(r.get('name')):12s} {r['event_id']:10s} {'--':>7s} "
                f"{'--':>12s} {'--':>11s} {r['status']:>20s}"
            )
            continue
        ok.append(r["residual_ms"])
        print(
            f"{r['name']:12s} {r['event_id']:10s} {r['peak_idx']:7d} "
            f"{r['residual_ms']:+12.5f} {r['residual_samples']:+11.2f} {r['verdict']:>20s}"
        )
    if ok:
        import statistics

        print("-" * 78)
        print(
            f"n={len(ok)}  median|resid|={statistics.median([abs(x) for x in ok]):.5f} ms  "
            f"max|resid|={max(abs(x) for x in ok):.5f} ms"
        )
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
