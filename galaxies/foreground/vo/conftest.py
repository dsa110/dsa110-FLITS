import pandas as pd
import pytest
import yaml

from .targets import Target


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast offline unit tests")
    config.addinivalue_line("markers", "integration: slower tests, may hit live services")
    config.addinivalue_line("markers", "network: tests requiring network access")


@pytest.fixture
def sample_targets():
    return [
        Target(name="FRB_Test_A", ra=150.1149, dec=2.2058, redshift=0.225),
        Target(name="FRB_Test_B", ra=210.8023, dec=54.3482, redshift=0.321),
        Target(name="FRB_Test_C", ra=45.6789, dec=-12.3456, redshift=None),
    ]


@pytest.fixture
def sample_targets_yaml(tmp_path, sample_targets):
    # deliberately emits the legacy z_host key so the loader alias stays exercised
    targets_file = tmp_path / "test_targets.yaml"
    data = {
        "targets": [
            {"name": t.name, "ra": t.ra, "dec": t.dec, "z_host": t.redshift} for t in sample_targets
        ]
    }
    with targets_file.open("w") as f:
        yaml.dump(data, f)
    return targets_file


@pytest.fixture
def sample_cone_query_result():
    return pd.DataFrame(
        {
            "source_id": ["src_001", "src_002", "src_003"],
            "ra": [150.115, 150.116, 150.114],
            "dec": [2.205, 2.206, 2.204],
            "z_spec": [0.1234, 0.2345, 0.3456],
            "richness": [15.2, 23.1, 8.7],
            "r200": [850.0, 1200.0, 600.0],
            "m200": [1.2e14, 2.3e14, 0.8e14],
        }
    )


@pytest.fixture
def sample_normalized_result():
    return pd.DataFrame(
        {
            "name": ["src_001", "src_002", "src_003"],
            "id": ["src_001", "src_002", "src_003"],
            "ra": [150.115, 150.116, 150.114],
            "dec": [2.205, 2.206, 2.204],
            "z": [0.1234, 0.2345, 0.3456],
            "z_type": ["spec", "spec", "spec"],
            "richness": [15.2, 23.1, 8.7],
            "m_delta": [1.2e14, 2.3e14, 0.8e14],
            "r_delta": [850.0, 1200.0, 600.0],
            "delta_def": [200, 200, 200],
            "service": ["https://example.com/tap"] * 3,
            "table": ["galaxy.main"] * 3,
            "provenance_json": ['{"service": "https://example.com/tap", "table": "galaxy.main"}']
            * 3,
        }
    )
