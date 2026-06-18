"""Maistro provenance sidecar for FLITS galaxy sightline searches."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from galaxies.v2_0 import config as galaxy_config
from galaxies.v2_0 import search as galaxy_search

DEFAULT_RPC_URL = "http://localhost:8787"
AGENT_ID = "flits:galaxy-search"
MATCH_SUMMARY_KIND = "flits.galaxy.match_summary"


class MaistroRpcError(RuntimeError):
    """Raised when the Maistro RPC writer is unavailable or rejects a request."""


class MaistroClient:
    """Minimal client for the Maistro RPC writer surface."""

    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("ORCH_RPC_URL") or DEFAULT_RPC_URL).rstrip("/")
        self.token = token if token is not None else os.environ.get("ORCH_RPC_TOKEN")
        self.timeout = timeout

    def ensure_ready(self) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/ready",
            headers=self._headers(include_json=False),
            method="GET",
        )
        data = self._open_json(request, "/ready")
        if not data.get("ready"):
            raise MaistroRpcError(f"/ready returned not ready: {data}")
        return data

    def write_batch(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post_json("/write.batch", body)

    def stage(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post_json("/stage", body)

    def state_all(self, run_id: str) -> Mapping[str, Any]:
        return self._post_json("/state.all", {"run_id": run_id})

    def _post_json(self, path: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers=self._headers(include_json=True),
            method="POST",
        )
        return self._open_json(request, path)

    def _headers(self, include_json: bool) -> dict[str, str]:
        headers: dict[str, str] = {}
        if include_json:
            headers["content-type"] = "application/json"
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def _open_json(self, request: urllib.request.Request, path: str) -> Mapping[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise MaistroRpcError(f"{path} {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise MaistroRpcError(f"{path}: {exc.reason}") from exc


def default_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"flits-galaxy-{stamp}"


def git_metadata(repo_root: Path | None = None) -> tuple[str | None, bool | None]:
    root = Path(repo_root or Path.cwd())
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_galaxy_payloads(
    *,
    run_id: str,
    impact_kpc: float,
    output_dir: str | Path,
    z_eps: float,
    command: Sequence[str],
    git_sha: str | None,
    git_dirty: bool | None,
    targets: Sequence[Sequence[Any]],
    catalogs: Mapping[str, str],
    pipeline_status: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    summary_path = output_path / "search_summary.csv"
    summary = _read_summary(summary_path)
    target_items = _target_items(targets)

    state_items = [
        ("pipeline.kind", "flits-galaxy-search"),
        ("pipeline.status", pipeline_status),
        ("pipeline.git_sha", git_sha),
        ("pipeline.git_dirty", git_dirty),
        ("pipeline.command", list(command)),
        ("galaxy.params.impact_kpc", impact_kpc),
        ("galaxy.params.z_eps", z_eps),
        ("galaxy.catalogs", dict(catalogs)),
        ("galaxy.targets", {"hash": _stable_hash(target_items), "items": target_items}),
        ("artifact.search_summary_csv", str(summary_path)),
        ("artifact.output_dir", str(output_path)),
        ("summary.total_targets", len(summary)),
        ("summary.targets_with_matches", sum(1 for row in summary if int(row["num_galaxies"]) > 0)),
        ("summary.total_matches", sum(int(row["num_galaxies"]) for row in summary)),
    ]
    write_batch = {
        "run_id": run_id,
        "items": [{"op": "setState", "key": key, "value": value} for key, value in state_items],
        "agent_id": AGENT_ID,
    }
    stage = [
        {
            "run_id": run_id,
            "kind": MATCH_SUMMARY_KIND,
            "payload": _candidate_payload(row, output_path),
            "derived_from": [],
        }
        for row in summary
    ]
    return {"run_id": run_id, "write_batch": write_batch, "stage": stage}


def run_galaxy_provenance(
    *,
    run_id: str,
    impact_kpc: float,
    output_dir: str | Path,
    z_eps: float,
    command: Sequence[str],
    dry_run: bool,
    client: Any | None = None,
) -> int:
    rpc = client or MaistroClient()
    if not dry_run:
        rpc.ensure_ready()

    output_path = Path(output_dir)
    git_sha, git_dirty = git_metadata()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            galaxy_search.run_search(
                impact_kpc=impact_kpc,
                output_dir=str(output_path),
                z_eps=z_eps,
            )
    except Exception:
        if not dry_run:
            _write_failed_state(rpc, run_id, impact_kpc, output_path, z_eps, command, git_sha, git_dirty)
        raise

    payloads = build_galaxy_payloads(
        run_id=run_id,
        impact_kpc=impact_kpc,
        output_dir=output_path,
        z_eps=z_eps,
        command=command,
        git_sha=git_sha,
        git_dirty=git_dirty,
        targets=galaxy_config.TARGETS,
        catalogs=galaxy_config.VIZIER_CATALOGS,
        pipeline_status="passed",
    )

    if dry_run:
        print(json.dumps(payloads, indent=2, sort_keys=True))
        return 0

    rpc.write_batch(payloads["write_batch"])
    for stage in payloads["stage"]:
        rpc.stage(stage)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or default_run_id()
    command = [sys.executable, "scripts/run_maistro_galaxy_search.py", *list(argv or sys.argv[1:])]
    client = None if args.dry_run else MaistroClient(base_url=args.orch_rpc_url)
    return run_galaxy_provenance(
        run_id=run_id,
        impact_kpc=args.impact_kpc,
        output_dir=args.output_dir,
        z_eps=args.z_eps,
        command=command,
        dry_run=args.dry_run,
        client=client,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLITS galaxy search with Maistro provenance.")
    parser.add_argument("--impact-kpc", type=float, default=galaxy_config.DEFAULT_IMPACT_KPC)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--z-eps", type=float, default=galaxy_config.DEFAULT_Z_EPS)
    parser.add_argument("--run-id")
    parser.add_argument("--orch-rpc-url", default=os.environ.get("ORCH_RPC_URL", DEFAULT_RPC_URL))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _write_failed_state(
    client: Any,
    run_id: str,
    impact_kpc: float,
    output_dir: Path,
    z_eps: float,
    command: Sequence[str],
    git_sha: str | None,
    git_dirty: bool | None,
) -> None:
    items = [
        ("pipeline.kind", "flits-galaxy-search"),
        ("pipeline.status", "failed"),
        ("pipeline.git_sha", git_sha),
        ("pipeline.git_dirty", git_dirty),
        ("pipeline.command", list(command)),
        ("galaxy.params.impact_kpc", impact_kpc),
        ("galaxy.params.z_eps", z_eps),
        ("artifact.output_dir", str(output_dir)),
    ]
    client.write_batch(
        {
            "run_id": run_id,
            "items": [{"op": "setState", "key": key, "value": value} for key, value in items],
            "agent_id": AGENT_ID,
        }
    )


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Galaxy search did not create {path}")
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "name": str(row["name"]),
                "target_id": int(row["target_id"]),
                "ra": str(row["ra"]),
                "dec": str(row["dec"]),
                "z_frb": _json_value(row["z_frb"]),
                "num_galaxies": int(row["num_galaxies"]),
            }
        )
    return rows


def _candidate_payload(summary_row: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    target_name = str(summary_row["name"])
    csv_path = output_dir / f"{target_name.lower()}_galaxies.csv"
    payload: dict[str, Any] = {
        "target_name": target_name,
        "target_id": int(summary_row["target_id"]),
        "ra": summary_row["ra"],
        "dec": summary_row["dec"],
        "z_frb": _json_value(summary_row["z_frb"]),
        "num_galaxies": int(summary_row["num_galaxies"]),
        "csv_path": str(csv_path),
        "review_note": "foreground matches found"
        if int(summary_row["num_galaxies"]) > 0
        else "no foreground matches found",
    }
    best_match = _best_match(csv_path)
    if best_match is not None:
        payload["best_match"] = best_match
    return payload


def _best_match(csv_path: Path) -> dict[str, Any] | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "impact_kpc" not in df.columns:
        return None
    impact = pd.to_numeric(df["impact_kpc"], errors="coerce")
    if impact.notna().sum() == 0:
        return None
    row = df.loc[impact.idxmin()]
    return {str(key): _json_value(value) for key, value in row.to_dict().items()}


def _target_items(targets: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    return [
        {"target_id": index, "name": str(name), "ra": str(ra), "dec": str(dec), "z_frb": _json_value(z_frb)}
        for index, (name, ra, dec, z_frb) in enumerate(targets, start=1)
    ]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
