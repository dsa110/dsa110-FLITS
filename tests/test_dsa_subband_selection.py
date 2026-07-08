from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_driver():
    path = (
        Path(__file__).resolve().parents[1]
        / "analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py"
    )
    spec = importlib.util.spec_from_file_location("dsa_lorentzian_driver", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def _candidate(n_subbands: int, *, n_fit_points: int = 60, fit_range_mhz: float = 12.0):
    return {
        "requested_num_subbands": n_subbands,
        "num_subbands": n_subbands,
        "subbands": [
            {
                "index": i,
                "num_channels": 800,
                "fit_range_mhz": fit_range_mhz,
                "n_fit_points": n_fit_points,
                "selected_components": [{"quality_flags": []}],
            }
            for i in range(n_subbands)
        ],
    }


def _candidate_with_flagged_subband(n_subbands: int, flagged_index: int):
    candidate = _candidate(n_subbands)
    candidate["subbands"][flagged_index]["selected_components"] = [
        {"quality_flags": ["fractional_dnu_err_gt_1"]}
    ]
    return candidate


def test_selects_largest_viable_subband_count():
    selected, report = driver._select_subband_candidate(
        [_candidate(2), _candidate(3), _candidate(4)]
    )

    assert selected["requested_num_subbands"] == 4
    assert report["selected_num_subbands"] == 4
    assert report["policy"] == "explicit_equal_snr_subband_candidate_selection"
    assert report["selected_policy"] == "largest_viable_equal_snr_subband_count"
    assert all(candidate["viable"] for candidate in report["candidates"])


def test_rejects_over_split_candidates_with_too_short_a_fit_window():
    selected, report = driver._select_subband_candidate(
        [
            _candidate(2, fit_range_mhz=25.0),
            _candidate(3, fit_range_mhz=13.0),
            _candidate(4, fit_range_mhz=6.0),
        ]
    )

    assert selected["requested_num_subbands"] == 3
    rejected = [candidate for candidate in report["candidates"] if not candidate["viable"]]
    assert rejected == [
        {
            "num_subbands": 4,
            "viable": False,
            "reasons": ["subband 0 fit_range_mhz 6 < 8"],
        }
    ]


def test_rejects_candidates_with_subbands_that_have_no_unflagged_component():
    selected, report = driver._select_subband_candidate(
        [
            _candidate(2),
            _candidate(3),
            _candidate_with_flagged_subband(4, flagged_index=2),
        ]
    )

    assert selected["requested_num_subbands"] == 3
    rejected = [candidate for candidate in report["candidates"] if not candidate["viable"]]
    assert rejected == [
        {
            "num_subbands": 4,
            "viable": False,
            "reasons": ["subband 2 has no unflagged selected component"],
        }
    ]


def test_falls_back_to_least_pathological_candidate_when_none_are_viable():
    selected, report = driver._select_subband_candidate(
        [
            _candidate_with_flagged_subband(2, flagged_index=0),
            _candidate_with_flagged_subband(3, flagged_index=0),
            _candidate_with_flagged_subband(4, flagged_index=0),
        ]
    )

    assert selected["requested_num_subbands"] == 2
    assert report["selected_policy"] == "least_pathological_equal_snr_subband_count"


def test_write_markdown_accepts_absolute_figure_paths(tmp_path):
    figure = tmp_path / "figures" / "casey_dsa_acf_lorentzian_fits.png"
    figure.parent.mkdir()
    figure.write_bytes(b"not a real png; path handling only")
    out = Path("relative-output.md")
    target = tmp_path / out

    driver._write_markdown(
        [
            {
                "burst": "casey",
                "num_subbands": 2,
                "n_per_subband": [1, 1],
                "burst_preferred_n": 1,
                "component_usable_median_dnu_mhz": {"1": 1.0},
                "subband_selection": {
                    "candidates": [
                        {"num_subbands": 2, "viable": True, "reasons": []},
                    ]
                },
                "figure_png": str(figure.resolve()),
            }
        ],
        [],
        target,
    )

    text = target.read_text()
    assert "figures/casey_dsa_acf_lorentzian_fits.png" in text
