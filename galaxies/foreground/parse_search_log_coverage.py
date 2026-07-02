#!/usr/bin/env python3
"""Backfill survey_coverage.csv from a saved run_search stdout log (pre-instrumentation runs)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from galaxies.foreground.config import TARGETS
from galaxies.foreground.survey_coverage import (
    classify_coverage,
    survey_in_footprint,
    write_survey_coverage_csv,
)
from galaxies.foreground.utils import parse_coord

PROC = re.compile(
    r"^Processing (?P<nick>\w+) \(Target \d+\): (?P<ra>[^,]+), (?P<dec>[^ ]+) \(z=(?P<z>[-0-9.]+)\)"
)
RAW = re.compile(r"^\s+(?P<engine>\S+(?:\([^)]+\))?) returned (?P<n>\d+) raw results\.$")
ZERO = re.compile(r"^\s+(?P<engine>\S+(?:\([^)]+\))?) returned 0 results\.$")
FG = re.compile(
    r"^\s+(?P<engine>\S+(?:\([^)]+\))?): Found (?P<fg>\d+) matches \(from (?P<wz>\d+) with z\)\.$"
)
NOMATCH = re.compile(
    r"^\s+(?P<engine>\S+(?:\([^)]+\))?): 0 matches \(from (?P<wz>\d+) with z\)\.$"
)

ENGINE_TO_SURVEY = {
    "NedTapEngine": "NED",
    "VizierEngine(GLADE+)": "GLADE+",
    "VizierEngine(DESI_DR8_NORTH)": "DESI_DR8_NORTH",
    "VizierEngine(SDSS_DR12)": "SDSS_DR12",
    "ClusterEngine": "CLUSTERS",
}


def parse_log(text: str) -> list[dict]:
    rows: list[dict] = []
    nick = ra = dec = z_frb = None
    coord = None
    pending: dict[str, dict] = {}

    def flush_engine(engine: str) -> None:
        if nick is None:
            return
        survey = ENGINE_TO_SURVEY.get(engine, engine)
        rec = pending.pop(engine, {"raw_count": 0, "with_z_count": 0, "foreground_count": 0})
        in_fp = survey_in_footprint(survey, coord)
        rows.append(
            {
                "nickname": nick,
                "ra": ra,
                "dec": dec,
                "z_frb": z_frb,
                "survey": survey,
                "engine": engine,
                "in_footprint": in_fp,
                "queried": True,
                **rec,
                "status": classify_coverage(
                    in_footprint=in_fp,
                    raw_count=rec["raw_count"],
                    foreground_count=rec["foreground_count"],
                ),
            }
        )

    for line in text.splitlines():
        m = PROC.match(line)
        if m:
            for eng in list(pending):
                flush_engine(eng)
            nick = m.group("nick")
            ra, dec, z_frb = m.group("ra"), m.group("dec"), float(m.group("z"))
            coord = parse_coord(ra, dec)
            pending = {}
            continue
        m = RAW.match(line)
        if m:
            pending[m.group("engine")] = {
                "raw_count": int(m.group("n")),
                "with_z_count": 0,
                "foreground_count": 0,
            }
            continue
        m = ZERO.match(line)
        if m:
            pending[m.group("engine")] = {
                "raw_count": 0,
                "with_z_count": 0,
                "foreground_count": 0,
            }
            continue
        m = FG.match(line)
        if m:
            eng = m.group("engine")
            pending[eng] = {
                "raw_count": pending.get(eng, {}).get("raw_count", int(m.group("wz"))),
                "with_z_count": int(m.group("wz")),
                "foreground_count": int(m.group("fg")),
            }
            continue
        m = NOMATCH.match(line)
        if m:
            eng = m.group("engine")
            wz = int(m.group("wz"))
            pending[eng] = {
                "raw_count": pending.get(eng, {}).get("raw_count", wz),
                "with_z_count": wz,
                "foreground_count": 0,
            }

    for eng in list(pending):
        flush_engine(eng)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True, help="foreground_search_*.out log")
    ap.add_argument("--output-dir", type=Path, default=Path("scratch/repro-foreground-search-hpcc"))
    args = ap.parse_args(argv)
    rows = parse_log(args.log.read_text())
    path = write_survey_coverage_csv(rows, str(args.output_dir))
    print(f"wrote {path} ({len(rows)} rows, {len(TARGETS)} sightlines expected)")


if __name__ == "__main__":
    main()
