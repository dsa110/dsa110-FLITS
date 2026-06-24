import matplotlib

matplotlib.use("Agg")

import pandas as pd

from galaxies.v2_0.plotting import _split_galaxies_clusters, plot_sightline


def test_split_galaxies_clusters_separates_by_classification():
    df = pd.DataFrame(
        {
            "name": ["g1", "clu1", "g2", "clu2"],
            "ra": [10.0, 10.1, 10.2, 10.3],
            "dec": [20.0, 20.0, 20.0, 20.0],
            "z": [0.10, 0.12, 0.10, 0.15],
            # NED 'G'/'' are galaxies; 'GClstr'/'ClG' are clusters (search._CLUSTER_RE).
            "classification": ["G", "GClstr", "", "ClG"],
        }
    )
    gals, clusters = _split_galaxies_clusters(df)
    assert sorted(gals["name"]) == ["g1", "g2"]
    assert sorted(clusters["name"]) == ["clu1", "clu2"]


def test_plot_sightline_renders_galaxies_and_clusters(tmp_path):
    # A near galaxy (~50 kpc) and a far NED cluster (~3 Mpc) — the on-sky angular
    # canvas must show both and label the cluster population distinctly.
    df = pd.DataFrame(
        {
            "name": ["gal", "NSC clu"],
            "ra": [170.43, 170.60],
            "dec": [70.65, 70.70],
            "z": [0.04, 0.15],
            "classification": ["G", "GClstr"],
            "impact_kpc": [50.0, 3000.0],
        }
    )
    info = {"name": "Casey", "ra": "11h19m56.05s", "dec": "+70d40m34.4s", "z_frb": 0.287}
    out = tmp_path / "casey_sky.png"
    fig, ax = plot_sightline(info, df, output_path=str(out))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Cluster" in lbl for lbl in labels)
    assert any("Galax" in lbl for lbl in labels)
    assert out.exists()


def test_plot_sightline_handles_empty(tmp_path):
    info = {"name": "Zach", "ra": "20h40m47.886s", "dec": "+72d52m56.378s", "z_frb": 0.043}
    fig, ax = plot_sightline(info, pd.DataFrame(), output_path=str(tmp_path / "zach_sky.png"))
    assert (tmp_path / "zach_sky.png").exists()
