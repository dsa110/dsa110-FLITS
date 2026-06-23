"""load_tns_name resolves the canonical nickname->TNS map.

chimedsa_burst_specs.csv is gitignored, so in a clean checkout this exercises the
committed _FALLBACK_TNS map -- including mahi (corrected against a TNS cone search)
and johndoeii (the DSA-110 archive designation FRB 20230814B). A future edit that
reverts either fails here.
"""

from scattering.scat_analysis.burst_metadata import load_tns_name


def test_load_tns_name_corrected_designations():
    assert load_tns_name("mahi") == "FRB 20240122A"  # was 20240119A
    assert load_tns_name("johndoeii") == "FRB 20230814B"  # DSA-110 archive (a.k.a. johndoe)


def test_load_tns_name_anchors_and_case_insensitive():
    assert load_tns_name("casey") == "FRB 20240229A"
    assert load_tns_name("Wilhelm") == "FRB 20221203A"
    assert load_tns_name("ZACH") == "FRB 20220207C"


def test_load_tns_name_unknown_falls_back_to_upper():
    assert load_tns_name("notaburst") == "NOTABURST"
