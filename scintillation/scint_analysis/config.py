# ==============================================================================
# File: scint_analysis/scint_analysis/config.py
# ==============================================================================
import yaml
import os
import logging

log = logging.getLogger(__name__)

def load_config(burst_config_path):
    """
    Loads and merges telescope and burst-specific configuration files.

    Args:
        burst_config_path (str): The full path to the burst's YAML config file.

    Returns:
        dict: A single dictionary containing the merged configuration.
    """
    log.info(f"Loading burst configuration from: {burst_config_path}")
    try:
        with open(burst_config_path, 'r') as f:
            burst_config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error(f"Burst config file not found: {burst_config_path}")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing burst YAML file: {e}")
        raise

    # Determine the path to the telescope config file
    telescope_name = burst_config.get('telescope')
    if not telescope_name:
        log.error("Burst config must contain a 'telescope' key.")
        raise ValueError("Missing 'telescope' key in burst config.")
    
    # Assume telescope configs are in a subdir relative to the burst config dir
    base_dir = os.path.dirname(burst_config_path)
    telescope_config_path = os.path.join(base_dir, '..', 'telescopes', f"{telescope_name}.yaml")
    
    log.info(f"Loading telescope configuration from: {telescope_config_path}")
    try:
        with open(telescope_config_path, 'r') as f:
            telescope_config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error(f"Telescope config file not found: {telescope_config_path}")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing telescope YAML file: {e}")
        raise

    # Merge the configurations, with burst-specific values overriding telescope defaults
    merged_config = {**telescope_config, **burst_config}
    log.info("Configurations successfully loaded and merged.")
    
    return merged_config