"""load_tns_name resolves the canonical nickname->TNS map.

chimedsa_burst_specs.csv is gitignored, so in a clean checkout this exercises the
committed _FALLBACK_TNS map -- including the two designations corrected against a
TNS cone search (mahi, johndoeii). A future edit that reverts either correction
fails here.
"""

from scattering.scat_analysis.burst_metadata import load_tns_name


def test_load_tns_name_corrected_designations():
    assert load_tns_name("mahi") == "FRB 20240122A"  # was 20240119A
    assert load_tns_name("johndoeii") == "FRB 20230814A"  # was 20230814B


def test_load_tns_name_anchors_and_case_insensitive():
    assert load_tns_name("casey") == "FRB 20240229A"
    assert load_tns_name("Wilhelm") == "FRB 20221203A"
    assert load_tns_name("ZACH") == "FRB 20220207C"


def test_load_tns_name_unknown_falls_back_to_upper():
    assert load_tns_name("notaburst") == "NOTABURST"
