#!/usr/bin/env python3
"""D1 read-only map: arc CHIME_bursts basenames ↔ iacobus burst_npys.

Uses vls (arc) and ssh iacobus (burst_npys listing only). No transfers or deletes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required") from exc

REPO = Path(__file__).resolve().parents[2]
BURSTS_YAML = REPO / "configs" / "bursts.yaml"
DEFAULT_CSV = REPO / "reports" / "d1_chime_burst_map.csv"
DEFAULT_JSON = REPO / "reports" / "d1_chime_burst_map.json"

ARC_ROOT = "arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/data/CHIME_bursts"
IACOBUS = "iacobus"
IACOBUS_BURST_NPYS = "/Users/iacobus/Research/CHIME_DSA_Codetections/burst_npys"

VLS = "vls"
DATE_PREFIX = re.compile(r"^\w{3}\s+\d{1,2}(?:\s+\d{4}|\s+\d{2}:\d{2})\s+")
VLS_LINE = re.compile(r"^([drw-]+)\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(.+)$")
ARC_NICKNAME = re.compile(r"^([A-Za-z]+)_chime_I_", re.I)
ARC_UNCORRECTED = re.compile(r"^([A-Za-z]+)_I_\d", re.I)
ANALYSIS_DIR = re.compile(r"^analysis_\d{8}_\d{6}$")
IACOBUS_NICKNAME_ALIASES = {"johndoeii": ("johndoe",)}


def sh(host: str | None, cmd: str) -> str:
    if host:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{host or 'local'}: {cmd}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def parse_vls_line(line: str) -> tuple[int, str, bool] | None:
    m = VLS_LINE.match(line)
    if not m:
        return None
    perms, size_s, rest = m.group(1), m.group(2), m.group(3)
    dm = DATE_PREFIX.match(rest)
    if not dm:
        return None
    name = rest[dm.end() :]
    return int(size_s), name, perms.startswith("d")


def arc_list(uri: str) -> list[tuple[int, str, bool]]:
    proc = subprocess.run([VLS, "-l", uri], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()[0] if proc.stderr else "vls failed"
        raise RuntimeError(f"{uri}: {err}")
    entries: list[tuple[int, str, bool]] = []
    for line in proc.stdout.splitlines():
        parsed = parse_vls_line(line)
        if not parsed:
            continue
        size, name, is_dir = parsed
        if name.startswith("."):
            continue
        entries.append((size, name, is_dir))
    return entries


def arc_walk(root: str, max_depth: int = 2) -> list[tuple[str, str, bool]]:
    """Return (relative_path, basename, is_dir) under root (depth-limited)."""
    out: list[tuple[str, str, bool]] = []

    def _walk(uri: str, rel: str, depth: int) -> None:
        for _size, name, is_dir in arc_list(uri):
            child_rel = f"{rel}/{name}" if rel else name
            out.append((child_rel, name, is_dir))
            if is_dir and depth < max_depth:
                _walk(f"{uri.rstrip('/')}/{name}", child_rel, depth + 1)

    _walk(root, "", 0)
    return out


def iacobus_basenames() -> list[str]:
    cmd = (
        f"find '{IACOBUS_BURST_NPYS}' -maxdepth 1 \\( -type f -o -type d \\) "
        "! -name wrong_npys ! -name .ipynb_checkpoints 2>/dev/null | sort"
    )
    lines = [Path(p).name for p in sh(IACOBUS, cmd).splitlines() if p.strip()]
    return [n for n in lines if n and n != Path(IACOBUS_BURST_NPYS).name]


def load_burst_catalog() -> dict[str, dict]:
    data = yaml.safe_load(BURSTS_YAML.read_text())
    return {k.lower(): v for k, v in data.get("bursts", {}).items()}


def tns_date_code(nickname: str, catalog: dict[str, dict]) -> str | None:
    meta = catalog.get(nickname.lower())
    if not meta:
        return None
    utc = meta.get("utc")
    if not utc:
        return None
    dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%y%m%d")


def guess_nickname(basename: str) -> str | None:
    for pat in (ARC_NICKNAME, ARC_UNCORRECTED):
        m = pat.match(basename)
        if m:
            return m.group(1).lower()
    return None


def iacobus_search_keys(nickname: str) -> set[str]:
    nick = nickname.lower()
    keys = {nick}
    keys.update(IACOBUS_NICKNAME_ALIASES.get(nick, ()))
    return keys


def match_iacobus(
    nickname: str | None,
    arc_basename: str,
    iacobus_names: list[str],
    catalog: dict[str, dict],
) -> tuple[str, str]:
    """Return (iacobus_match, notes)."""
    if nickname is None:
        if ANALYSIS_DIR.match(arc_basename):
            return "", "CANFAR analysis session dir; no iacobus counterpart expected"
        if arc_basename in (".ipynb_checkpoints", "uncorrected"):
            return "", "arc subtree; not a codetection fit product"
        return "", "no nickname pattern"

    nick = nickname.lower()
    search_keys = iacobus_search_keys(nick)
    matches: set[str] = set()
    date_code = tns_date_code(nick, catalog)

    for name in iacobus_names:
        lower = name.lower()
        if any(lower.startswith(f"{key}_") for key in search_keys):
            matches.add(name)
            continue
        if any(lower.startswith("i_") and lower.endswith(f"_{key}.npy") for key in search_keys):
            matches.add(name)
            continue
        if any(lower.startswith("i_") and f"_{key}." in lower for key in search_keys):
            matches.add(name)
            continue
        if date_code and date_code in lower:
            if any(key in lower for key in search_keys) or lower.startswith(f"i_{date_code}"):
                matches.add(name)

    if not matches:
        note = "zero iacobus basename overlap (expected for arc fit-ready _chime_I_ names)"
        if date_code:
            note += f"; TNS date code {date_code} also absent in iacobus top-level names"
        return "", note

    exact = arc_basename in matches
    note_parts = [f"nickname {nick} ↔ {len(matches)} iacobus entry/entries"]
    if date_code:
        note_parts.append(f"TNS date code {date_code}")
    if exact:
        note_parts.append("exact basename match")
    else:
        note_parts.append("namespace differs from arc _chime_I_ fit naming")
    return "; ".join(sorted(matches)), "; ".join(note_parts)


def build_rows(dry_run: bool = False) -> list[dict[str, str]]:
    catalog = load_burst_catalog()
    if dry_run:
        iacobus_names = [
            "casey_240229aaad",
            "casey_14000_16500.npy",
            "I_230325aaag_freya.npy",
            "zach_220207aabh",
        ]
        arc_entries = [
            ("dmphase/casey_chime_I_491_2085_32000b_cntr_bpc.npy", "casey_chime_I_491_2085_32000b_cntr_bpc.npy", False),
            ("dmphase/analysis_20260619_055230", "analysis_20260619_055230", True),
        ]
    else:
        iacobus_names = iacobus_basenames()
        arc_entries = [
            (rel, base, is_dir)
            for rel, base, is_dir in arc_walk(ARC_ROOT, max_depth=2)
            if rel and not rel.endswith("/.ipynb_checkpoints")
        ]

    rows: list[dict[str, str]] = []
    for rel, basename, is_dir in arc_entries:
        if is_dir:
            if not ANALYSIS_DIR.match(basename):
                continue
        elif not basename.endswith(".npy"):
            continue
        nick = guess_nickname(basename)
        iacobus_match, notes = match_iacobus(nick, basename, iacobus_names, catalog)
        rows.append(
            {
                "arc_path": f"{ARC_ROOT}/{rel}",
                "arc_basename": basename,
                "burst_nickname_guess": nick or "",
                "iacobus_match": iacobus_match,
                "notes": notes,
            }
        )
    rows.sort(key=lambda r: (r["burst_nickname_guess"], r["arc_basename"]))
    return rows


def write_reports(rows: list[dict[str, str]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arc_path", "arc_basename", "burst_nickname_guess", "iacobus_match", "notes"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    nicknamed = [r for r in rows if r["burst_nickname_guess"]]
    matched = [r for r in nicknamed if r["iacobus_match"]]
    summary = {
        "generated_by": "scripts/migration/map_chime_bursts_namespaces.py",
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "arc_root": ARC_ROOT,
        "iacobus_burst_npys": IACOBUS_BURST_NPYS,
        "row_count": len(rows),
        "codetection_npy_rows": len(nicknamed),
        "iacobus_linked_rows": len(matched),
        "exact_basename_overlap": sum(1 for r in rows if r["arc_basename"] in r["iacobus_match"]),
        "policy": "read-only map; no bulk rsync/vcp/delete",
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--dry-run", action="store_true", help="Offline smoke (no vls/ssh)")
    parser.add_argument("--stdout", action="store_true", help="Print summary counts")
    args = parser.parse_args()

    rows = build_rows(dry_run=args.dry_run)
    write_reports(rows, args.csv, args.json)

    if args.stdout:
        linked = sum(1 for r in rows if r["iacobus_match"])
        print(
            f"d1 map: {len(rows)} arc rows, {linked} with iacobus match, "
            f"exact basename overlap {sum(1 for r in rows if r['arc_basename'] in r['iacobus_match'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
