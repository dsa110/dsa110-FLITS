import json
from pathlib import Path

import pytest

from crossmatching.toa_crossmatch import (
    DsaTimingProvenance,
    crossmatch_input_from_dict,
    reproduce_model_result,
    reproduce_notebook_result,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "crossmatching" / "notebook_reproduction_fixture.json"
GOLDEN = ROOT / "crossmatching" / "toa_crossmatch_results.json"


def _load_json(path):
    return json.loads(path.read_text())


def test_reproduces_notebook_crossmatch_results():
    """Legacy peak-based path.

    The golden ``measured_offset_ms`` is now the model (scatter-corrected)
    offset, so the legacy peak offset is asserted against
    ``peak_measured_offset_ms``. ``combined_dm_uncertainty_ms`` remains the pure
    DM-referral term and is unchanged by the model switch.
    """
    fixture = _load_json(FIXTURE)
    golden = _load_json(GOLDEN)

    assert len(fixture["bursts"]) == len(golden) == 12

    for row in fixture["bursts"]:
        expected = golden[row["name"]]
        result = reproduce_notebook_result(crossmatch_input_from_dict(row)).to_legacy_dict()

        for key in ("chime_id", "dm", "dm_mjd", "fwhm_ms"):
            assert result[key] == expected[key]

        assert result["toa_dsa_utc_400"] == expected["toa_dsa_utc_400"]
        assert result["combined_dm_uncertainty_ms"] == pytest.approx(
            expected["combined_dm_uncertainty_ms"], abs=1e-12
        )
        # Legacy peak offset -> golden peak_measured_offset_ms.
        assert result["measured_offset_ms"] == pytest.approx(
            expected["peak_measured_offset_ms"], abs=2e-3
        )
        assert result["geometric_delay_ms"] == pytest.approx(
            expected["geometric_delay_ms"], abs=1e-2
        )


def test_reproduces_model_crossmatch_results():
    """Model (scatter-corrected) path reproduces the golden model offsets.

    Asserts the intrinsic offset, the per-band scatter corrections, the
    differential shift, and the folded error budget. The model ToA subtracts
    each band's joint-fit scattering peak-shift, so ``measured_offset_ms`` =
    peak offset - (Delta_scat_C - Delta_scat_D).
    """
    fixture = _load_json(FIXTURE)
    golden = _load_json(GOLDEN)

    for row in fixture["bursts"]:
        expected = golden[row["name"]]
        result = reproduce_model_result(crossmatch_input_from_dict(row)).to_model_dict()

        # Model offset and its provenance.
        assert result["measured_offset_ms"] == pytest.approx(
            expected["measured_offset_ms"], abs=2e-3
        )
        assert result["peak_measured_offset_ms"] == pytest.approx(
            expected["peak_measured_offset_ms"], abs=2e-3
        )
        for key in (
            "scatter_corr_chime_ms",
            "scatter_corr_dsa_ms",
            "differential_scatter_shift_ms",
            "loc_err_chime_ms",
            "loc_err_dsa_ms",
        ):
            assert result[key] == pytest.approx(expected[key], abs=1e-9)

        # Offset relation: model = peak - differential.
        assert result["measured_offset_ms"] == pytest.approx(
            result["peak_measured_offset_ms"]
            - result["differential_scatter_shift_ms"],
            abs=2e-3,
        )

        # Folded error budget: combined = hypot(chime_model, dsa_model).
        assert result["combined_error_ms"] == pytest.approx(
            (result["error_chime_ms"] ** 2 + result["error_dsa_ms"] ** 2) ** 0.5,
            abs=1e-9,
        )
        # Scattering peak-shift is non-negative and CHIME >= DSA.
        assert result["scatter_corr_chime_ms"] >= 0.0
        assert result["scatter_corr_dsa_ms"] >= 0.0
        assert result["differential_scatter_shift_ms"] >= -1e-9

        # Extended budget: full per-band error = quadrature of the model error,
        # the scatter-correction uncertainty, and the multi-component ambiguity.
        assert result["error_chime_full_ms"] == pytest.approx(
            (
                result["error_chime_ms"] ** 2
                + result["scatter_corr_unc_chime_ms"] ** 2
                + result["comp_ambig_chime_ms"] ** 2
            )
            ** 0.5,
            abs=1e-9,
        )
        assert result["error_dsa_full_ms"] == pytest.approx(
            (
                result["error_dsa_ms"] ** 2
                + result["scatter_corr_unc_dsa_ms"] ** 2
                + result["comp_ambig_dsa_ms"] ** 2
            )
            ** 0.5,
            abs=1e-9,
        )
        # Combined-full folds both full per-band errors and the inter-site
        # geometric-position term.
        assert result["combined_error_full_ms"] == pytest.approx(
            (
                result["error_chime_full_ms"] ** 2
                + result["error_dsa_full_ms"] ** 2
                + result["geo_pos_err_ms"] ** 2
            )
            ** 0.5,
            abs=1e-9,
        )
        # Extended terms are non-negative and never shrink the budget.
        for key in (
            "scatter_corr_unc_chime_ms",
            "scatter_corr_unc_dsa_ms",
            "comp_ambig_chime_ms",
            "comp_ambig_dsa_ms",
            "geo_pos_err_ms",
        ):
            assert result[key] >= 0.0
        assert result["error_chime_full_ms"] >= result["error_chime_ms"] - 1e-9
        assert result["error_dsa_full_ms"] >= result["error_dsa_ms"] - 1e-9
        assert result["combined_error_full_ms"] >= result["combined_error_ms"] - 1e-9


def test_legacy_dict_schema_is_unchanged():
    """to_legacy_dict must remain the original 13-key row (no model leakage)."""
    fixture = _load_json(FIXTURE)
    row = fixture["bursts"][0]
    legacy = reproduce_notebook_result(crossmatch_input_from_dict(row)).to_legacy_dict()
    assert set(legacy.keys()) == {
        "chime_id", "dm", "fwhm_ms", "toa_chime_unix_400", "toa_chime_utc_400",
        "dm_mjd", "toa_dsa_utc_400", "dm_uncertainty", "error_chime_ms",
        "error_dsa_ms", "measured_offset_ms", "combined_dm_uncertainty_ms",
        "geometric_delay_ms",
    }


def test_dsa_filterbank_header_is_provenance_not_curated_time():
    dsa = DsaTimingProvenance(
        dsa_mjd=60369.37095224303,
        filterbank_tstart_mjd=60369.37095,
        tsamp_s=3.2768e-05,
        nchans=6144,
        fch1_mhz=1498.75,
        foff_mhz=-0.03051757812,
    )

    assert dsa.curated_time.mjd == pytest.approx(60369.37095224303)
    assert dsa.filterbank_tstart_mjd == pytest.approx(60369.37095)
    assert abs((dsa.curated_time.mjd - dsa.filterbank_tstart_mjd) * 86400) > dsa.tsamp_s


def test_chime_baseband_paths_are_verified_vospace_locations():
    fixture = _load_json(FIXTURE)
    by_name = {row["name"]: row["chime"] for row in fixture["bursts"]}

    for chime in by_name.values():
        assert chime["baseband_verified_exists"] is True
        assert chime["baseband_path"].startswith("/arc/projects/chime_frb/")
        assert chime["baseband_vospace_uri"] == "arc:" + chime["baseband_path"].removeprefix(
            "/arc/"
        )
        assert chime["baseband_path"].endswith(".h5")
        assert "singlebeam_" in chime["baseband_vls_listing"]

    assert "Run_UpdatedCalSep25" in by_name["oran"]["baseband_path"]
    assert "Run_UpdatedCalSep25" in by_name["wilhelm"]["baseband_path"]
    assert "old_processed_files" in by_name["chromatica"]["baseband_path"]
