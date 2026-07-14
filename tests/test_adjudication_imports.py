"""P0.2 adjudication-package tests.

Deviation from plan-trust-reset-revalidation.md P0.2: the plan's template test
imported the four modules directly, but these are frozen-era execution scripts
whose top-level code runs the full adjudication pipeline (validate_foreground
issues live TAP queries at import). The restored scripts intentionally differ
from the scratch copies only in path plumbing: all stages use one configurable
adjudication directory and the docs emitter resolves the repository root.
The tests therefore assert those path contracts, module locatability, and
compilation without executing live catalog queries.
"""
import importlib.util
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).parents[1]
PKG = ROOT / "galaxies/foreground/adjudication"

SCRIPTS = (
    "validate_foreground.py",
    "ps1_strm_adjudicate.py",
    "merge_final.py",
    "make_catalog_table.py",
)


def test_adjudication_scripts_share_one_data_directory():
    for name in SCRIPTS:
        source = (PKG / name).read_text()
        assert "FLITS_FOREGROUND_ADJUDICATION_DIR" in source, name
        assert '"data", "frozen_census"' in source, name


def test_catalog_emitter_resolves_repository_root():
    source = (PKG / "make_catalog_table.py").read_text()
    assert "os.path.dirname(os.path.dirname(os.path.dirname(HERE)))" in source


def test_adjudication_modules_locatable():
    for name in SCRIPTS:
        mod = f"galaxies.foreground.adjudication.{name.removesuffix('.py')}"
        assert importlib.util.find_spec(mod) is not None, mod


def test_adjudication_scripts_compile():
    for name in SCRIPTS:
        py_compile.compile(str(PKG / name), doraise=True)
