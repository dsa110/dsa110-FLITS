"""Guards for science products retired to the dated quarantine."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUARANTINE = ROOT / "quarantine" / "2026-07-17-outdated-science"


def test_outdated_result_bytes_are_quarantined():
    moved = {
        ROOT / "analysis/beta_campaign/two_screen_consistency.json":
            QUARANTINE / "analysis/beta_campaign/two_screen_consistency.json",
        ROOT / "analysis/beta_campaign/two_screen_consistency.md":
            QUARANTINE / "analysis/beta_campaign/two_screen_consistency.md",
        ROOT / "galaxies/foreground/data/sightline_attribution_matrix.csv":
            QUARANTINE / "galaxies/foreground/data/sightline_attribution_matrix.csv",
        ROOT / "analysis/chime-scintillation/INVENTORY.yaml":
            QUARANTINE / "analysis/chime-scintillation/INVENTORY.yaml",
    }
    for active, quarantined in moved.items():
        assert quarantined.is_file(), quarantined
        assert not active.exists(), active


def test_joint_summary_is_a_tombstone_and_old_bytes_are_quarantined():
    tombstone = (ROOT / "results/joint_fit_summary.md").read_text()
    archived = QUARANTINE / "results/joint_fit_summary.md"
    assert archived.is_file()
    assert "QUARANTINED" in tombstone
    assert "remain trustworthy" not in tombstone


def test_legacy_chime_readme_routes_to_the_final_campaign():
    current = (ROOT / "analysis/chime-scintillation/README.md").read_text()
    archived = QUARANTINE / "analysis/chime-scintillation/README.md"
    assert archived.is_file()
    assert "window-tuning-campaign-2026-07-17" in current
    assert "canonical index" not in current


def test_historical_generators_default_to_quarantine():
    for path in (
        ROOT / "analysis/beta_campaign/two_screen.py",
        ROOT / "analysis/scattering-refit-2026-06/gen_joint_summary.py",
        ROOT / "galaxies/foreground/attribution_matrix.py",
    ):
        source = path.read_text()
        assert "2026-07-17-outdated-science" in source, path


def test_quarantine_has_a_review_index():
    index = (QUARANTINE / "README.md").read_text()
    assert "Do not cite" in index
    assert "Original path" in index
