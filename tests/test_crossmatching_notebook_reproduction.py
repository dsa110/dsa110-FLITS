import json
from pathlib import Path

import pytest

from crossmatching.toa_crossmatch import (
    DsaTimingProvenance,
    crossmatch_input_from_dict,
    reproduce_notebook_result,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "crossmatching" / "notebook_reproduction_fixture.json"
GOLDEN = ROOT / "crossmatching" / "toa_crossmatch_results.json"


def _load_json(path):
    return json.loads(path.read_text())


def test_reproduces_notebook_crossmatch_results():
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
        assert result["measured_offset_ms"] == pytest.approx(
            expected["measured_offset_ms"], abs=2e-3
        )
        assert result["geometric_delay_ms"] == pytest.approx(
            expected["geometric_delay_ms"], abs=1e-2
        )


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
