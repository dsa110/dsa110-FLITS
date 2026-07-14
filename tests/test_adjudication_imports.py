"""P0.2 adjudication-package tests.

Deviation from plan-trust-reset-revalidation.md P0.2: the plan's template test
imported the four modules directly, but these are frozen-era execution scripts
whose top-level code runs the full adjudication pipeline (validate_foreground
issues live TAP queries at import). Byte-equivalence to the frozen era is the
actual Phase 3 requirement, so instead we (a) pin each script's sha256 as
copied from the flits-rerun worktree (scratch/codetection/, 2026-06 frozen
era), (b) assert the modules are locatable as package members, and (c) compile
each without executing it.
"""
import hashlib
import importlib.util
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).parents[1]
PKG = ROOT / "galaxies/foreground/adjudication"

FROZEN_SHA256 = {
    "validate_foreground.py":
        "5ef20abcf7debce8118ce3802b2e56f6a58f813e96bc521ce6bf2dc9876b4588",
    "ps1_strm_adjudicate.py":
        "63997f830bc819ab96ce31fd5f1efbd5195f55a61750606d05315487d6596071",
    "merge_final.py":
        "ae333d9b06fd594311028f9eb8f17c2b94ddfec2c4df6ebc8e2a556a43572b7d",
    "make_catalog_table.py":
        "90dd796de68494464d11e5a463a14a4e04410441075af6ea3e9f38bcfa5d31ae",
}


def test_adjudication_scripts_byte_frozen():
    for name, sha in FROZEN_SHA256.items():
        p = PKG / name
        assert p.exists(), f"missing {p}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == sha, f"{name}: {got} != frozen {sha}"


def test_adjudication_modules_locatable():
    for name in FROZEN_SHA256:
        mod = f"galaxies.foreground.adjudication.{name.removesuffix('.py')}"
        assert importlib.util.find_spec(mod) is not None, mod


def test_adjudication_scripts_compile():
    for name in FROZEN_SHA256:
        py_compile.compile(str(PKG / name), doraise=True)
