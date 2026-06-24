#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.10.14"]
# ///
from __future__ import annotations

import nox

nox.needs_version = ">=2025.10.14"
nox.options.default_venv_backend = "uv|virtualenv"

LINT_TARGETS = ("noxfile.py",)


@nox.session(default=True)
def tests(session: nox.Session) -> None:
    """Run the Python test suite."""
    session.install("-e", ".[nested,perf,ne2025]", "pytest>=7.0")
    session.run("pytest", *session.posargs)


@nox.session
def cov(session: nox.Session) -> None:
    """Run the suite under coverage to nominate untested/dead code.

    Skips slow-marked tests by default; pass paths/markers after ``--``.
    """
    session.install("-e", ".[nested,perf,ne2025]", "pytest>=7.0", "pytest-cov>=4.0")
    args = session.posargs or ["-m", "not slow"]
    session.run("pytest", "--cov", "--cov-report=term-missing", *args)


@nox.session
def lint(session: nox.Session) -> None:
    """Run Ruff checks without modifying files.

    Pass paths after ``--`` to lint a broader surface.
    """
    session.install("ruff>=0.15.18")
    targets = session.posargs or LINT_TARGETS
    session.run("ruff", "check", *targets)
    session.run("ruff", "format", "--check", *targets)


if __name__ == "__main__":
    nox.main()
