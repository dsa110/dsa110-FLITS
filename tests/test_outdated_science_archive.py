"""Guards for science products retired to the dated archive."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / ".archive" / "outdated-science" / "2026-07-17"


def test_outdated_result_bytes_are_archived():
    moved = {
        ROOT / "analysis/beta_campaign/two_screen_consistency.json":
            ARCHIVE / "analysis/beta_campaign/two_screen_consistency.json",
        ROOT / "analysis/beta_campaign/two_screen_consistency.md":
            ARCHIVE / "analysis/beta_campaign/two_screen_consistency.md",
        ROOT / "galaxies/foreground/data/sightline_attribution_matrix.csv":
            ARCHIVE / "galaxies/foreground/data/sightline_attribution_matrix.csv",
        ROOT / "analysis/chime-scintillation/INVENTORY.yaml":
            ARCHIVE / "analysis/chime-scintillation/INVENTORY.yaml",
    }
    for active, archived in moved.items():
        assert archived.is_file(), archived
        assert not active.exists(), active


def test_joint_summary_is_a_tombstone_and_old_bytes_are_archived():
    tombstone = (ROOT / "results/joint_fit_summary.md").read_text()
    archived = ARCHIVE / "results/joint_fit_summary.md"
    assert archived.is_file()
    assert "ARCHIVED" in tombstone
    assert "remain trustworthy" not in tombstone


def test_legacy_chime_readme_routes_to_the_final_campaign():
    current = (ROOT / "analysis/chime-scintillation/README.md").read_text()
    archived = ARCHIVE / "analysis/chime-scintillation/README.md"
    assert archived.is_file()
    assert "window-tuning-campaign-2026-07-17" in current
    assert "canonical index" not in current


def test_historical_generators_default_to_archive():
    for path in (
        ROOT / "analysis/beta_campaign/two_screen.py",
        ROOT / "analysis/scattering-refit-2026-06/gen_joint_summary.py",
        ROOT / "galaxies/foreground/attribution_matrix.py",
    ):
        source = path.read_text()
        assert '".archive"' in source and '"2026-07-17"' in source, path


def test_archive_has_a_review_index():
    index = (ARCHIVE / "README.md").read_text()
    assert "Do not cite" in index
    assert "Original path" in index
