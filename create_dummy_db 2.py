#
# Copyright 2024, by the California Institute of Technology.
# ALL RIGHTS RESERVED.
# United States Government sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# This software may be subject to U.S. export control laws and regulations.
# By accepting this document, the user agrees to comply with all applicable
# U.S. export laws and regulations. User has the responsibility to obtain
# export licenses, or other export authority as may be required before
# exporting such information to foreign countries or providing access to
# foreign persons.
"""
Create a dummy FLITS results database for testing purposes.
"""

from flits.batch.results_db import ResultsDatabase, ScatteringResult, ScintillationResult

def create_dummy_db(db_path: str = "flits_results.db"):
    """
    Creates and populates a dummy FLITS results database.
    
    Args:
        db_path: The path to the database file to be created.
    """
    db = ResultsDatabase(db_path)
    
    # Add CHIME scattering
    db.add_scattering_result(ScatteringResult(
        burst_name="casey",
        telescope="chime",
        tau_1ghz=0.5,
        tau_1ghz_err=0.05,
        alpha=4.0,
        alpha_err=0.3,
    ))
    
    # Add DSA scattering
    db.add_scattering_result(ScatteringResult(
        burst_name="casey",
        telescope="dsa",
        tau_1ghz=0.48,
        tau_1ghz_err=0.04,
        alpha=4.1,
        alpha_err=0.2,
    ))
    
    # Add another burst for more data
    db.add_scattering_result(ScatteringResult(
        burst_name="freya",
        telescope="chime",
        tau_1ghz=1.2,
        tau_1ghz_err=0.1,
        alpha=3.9,
        alpha_err=0.2,
    ))
    
    # Add CHIME scintillation
    db.add_scintillation_result(ScintillationResult(
        burst_name="casey",
        telescope="chime",
        delta_nu_dc=0.3,
        delta_nu_dc_err=0.05,
    ))
    
    # Add DSA scintillation
    db.add_scintillation_result(ScintillationResult(
        burst_name="casey",
        telescope="dsa",
        delta_nu_dc=2.0,
        delta_nu_dc_err=0.2,
    ))

    # Add another burst for more data
    db.add_scintillation_result(ScintillationResult(
        burst_name="freya",
        telescope="chime",
        delta_nu_dc=0.1,
        delta_nu_dc_err=0.02,
    ))
    
    db.close()
    print(f"Dummy database created at {db_path}")

if __name__ == "__main__":
    create_dummy_db()
