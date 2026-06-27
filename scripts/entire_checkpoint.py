#!/usr/bin/env python3
"""Append a compact Entire tracing checkpoint to the repo ledger."""

from __future__ import annotations

import argparse
import fnmatch
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def maybe_run(cmd: list[str], cwd: Path) -> str:
    try:
        return run(cmd, cwd)
    except subprocess.CalledProcessError:
        return ""


def format_entire_ref(line: str) -> str:
    if "\t" in line:
        sha, ref = line.split("\t", 1)
        return f"{ref.removeprefix('refs/heads/')} -> {sha[:7]}"
    parts = line.split()
    if len(parts) >= 2:
        return f"{parts[0]} -> {parts[1]}"
    return line


WATCHED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "DATA_LOCATIONS.md",
    "DATA_SOURCES.md",
    "ENTIRE_AUTH_HANDOFF.md",
    "docs/",
    "docs-analysis/",
    "scripts/",
    "configs/",
    "analyses/",
    "analysis/",
)


def should_checkpoint(repo: Path) -> tuple[bool, str]:
    subject = maybe_run(["git", "log", "-1", "--pretty=%s"], repo)
    if subject.startswith("docs: add Entire tracing checkpoint") or (
        "Entire tracing checkpoint" in subject
    ):
        return False, "checkpoint commit"

    changed = maybe_run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "HEAD"],
        repo,
    )
    changed_paths = [line.strip() for line in changed.splitlines() if line.strip()]
    if not changed_paths:
        return False, "no changed paths"

    if changed_paths == ["docs/entire-tracing-checkpoints.md"]:
        return False, "ledger-only commit"

    for path in changed_paths:
        if any(
            path == watched or path.startswith(watched) or fnmatch.fnmatch(path, watched)
            for watched in WATCHED_PATHS
        ):
            return True, f"watched path changed: {path}"

    return False, "no watched paths changed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="stage and commit the checkpoint ledger after appending the snapshot",
    )
    parser.add_argument(
        "--message",
        default="docs: add Entire tracing checkpoint",
        help="commit message to use with --commit",
    )
    parser.add_argument(
        "--note",
        default="",
        help="optional short note to record with the checkpoint",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="skip unless the current commit/merge changed watched paths",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ledger = repo / "docs" / "entire-tracing-checkpoints.md"

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    host = socket.gethostname().split(".")[0]

    branch = run(["git", "branch", "--show-current"], repo)
    head = run(["git", "rev-parse", "--short", "HEAD"], repo)
    origin_main = maybe_run(["git", "rev-parse", "--short", "origin/main"], repo) or "unavailable"
    status = maybe_run(["git", "status", "--short", "--untracked-files=no"], repo)
    status_lines = status.splitlines() if status else []

    local_entire = maybe_run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short) %(objectname:short)",
            "refs/heads/entire",
        ],
        repo,
    )
    local_entire_lines = local_entire.splitlines() if local_entire else []

    origin_entire = maybe_run(["git", "ls-remote", "--heads", "origin", "entire/*"], repo)
    origin_entire_lines = origin_entire.splitlines() if origin_entire else []

    if args.auto:
        ok, reason = should_checkpoint(repo)
        if not ok:
            print(f"Skipped checkpoint: {reason}")
            return 0
        if not args.note:
            args.note = reason

    lines: list[str] = [
        "",
        f"## {timestamp} — {host}",
        f"- repo: `{repo}`",
        f"- branch: `{branch}`",
        f"- head: `{head}`",
        f"- origin/main: `{origin_main}`",
        f"- worktree: {'clean' if not status_lines else 'dirty'}",
    ]
    if args.note:
        lines.append(f"- note: {args.note}")
    if status_lines:
        lines.append("- tracked status:")
        lines.extend(f"  - {line}" for line in status_lines)
    else:
        lines.append("- tracked status: clean")

    lines.append("- local `entire/*` refs:")
    if local_entire_lines:
        lines.extend(f"  - {format_entire_ref(line)}" for line in local_entire_lines)
    else:
        lines.append("  - none")

    lines.append("- `origin` `entire/*` refs:")
    if origin_entire_lines:
        lines.extend(f"  - {format_entire_ref(line)}" for line in origin_entire_lines)
    else:
        lines.append("  - none")

    lines.extend(
        [
            "- note: `.entire/` remains host-local runtime state; only this ledger is tracked.",
            "",
        ]
    )

    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text(
            "# Entire tracing checkpoints\n\n"
            "This ledger captures only the compact, commit-worthy subset of Entire tracing\n"
            "state. Do not mirror `.entire/` runtime artifacts here.\n\n"
            "Use `scripts/entire_checkpoint.py` on whichever host you are working from to\n"
            "append a new checkpoint, then commit this file when the snapshot is worth\n"
            "preserving for future agents.\n",
            encoding="utf-8",
        )

    with ledger.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Appended checkpoint to {ledger}")

    if args.commit:
        subprocess.run(["git", "add", "docs/entire-tracing-checkpoints.md"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", args.message], cwd=repo, check=True)
        print(f"Committed checkpoint with: {args.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
