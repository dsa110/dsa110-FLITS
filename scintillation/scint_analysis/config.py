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

def update_yaml_config(config_path, key_path, new_value):
    """
    Updates a specific nested key in a YAML file.

    Args:
        config_path (str): Path to the YAML file.
        key_path (list): A list of keys representing the path to the value.
                         For example: ['analysis', 'acf', 'num_subbands']
        new_value: The new value to set.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Navigate to the target dictionary, creating keys if they don't exist
        d = config_data
        for key in key_path[:-1]:
            d = d.setdefault(key, {})
        
        # Set the new value on the final key
        d[key_path[-1]] = new_value

        # Write the modified dictionary back to the file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Successfully updated '{'.'.join(key_path)}' to {new_value}")

    except Exception as e:
        print(f"Error updating YAML file: {e}")

def update_yaml_guesses(config_path, model_name, new_params_dict):
    """
    Reads a YAML config file, updates the initial guesses for a specific
    model, and writes the changes back to the file.

    Args:
        config_path (str): The full path to the YAML configuration file.
        model_name (str): The key for the model to update (e.g., '2c_lor').
        new_params_dict (dict): A dictionary of the new initial guess parameters.
    """
    try:
        # Read the entire YAML file into a Python dictionary
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Safely navigate and create nested keys if they don't exist
        # This gets config_data['analysis']['fitting']['init_guess']
        init_guess_section = config_data.setdefault('analysis', {})\
                                        .setdefault('fitting', {})\
                                        .setdefault('init_guess', {})

        # Update the parameters for the specified model
        init_guess_section[model_name] = new_params_dict

        # Write the modified dictionary back to the YAML file
        with open(config_path, 'w') as f:
            # default_flow_style=False keeps the block format
            # sort_keys=False preserves the original order as much as possible
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"Successfully updated initial guesses for '{model_name}' in {config_path}")

    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
    except Exception as e:
        print(f"An error occurred while updating the YAML file: {e}")

def update_fitting_parameter(config_path, param_name, new_value):
    """
    Reads a YAML config file, updates a specific parameter in the
    'analysis:fitting' section, and writes the changes back.

    Args:
        config_path (str): The full path to the YAML configuration file.
        param_name (str): The name of the parameter to change (e.g., 'fit_lagrange_mhz').
        new_value: The new value for the parameter.
    """
    try:
        # Read the entire YAML file into a Python dictionary
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Safely navigate to the 'fitting' section
        fitting_section = config_data.setdefault('analysis', {})\
                                     .setdefault('fitting', {})\
                                     .setdefault('pipeline_options', {})

        # Update the specified parameter with the new value
        fitting_section[param_name] = new_value

        # Write the modified dictionary back to the YAML file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"Successfully updated '{param_name}' to '{new_value}' in {config_path}")

    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
    except Exception as e:
        print(f"An error occurred while updating the YAML file: {e}")
        
def update_pipeline_parameter(config_path, param_name, new_value):
    """
    Reads a YAML config file, updates a specific parameter in the
    'analysis:fitting' section, and writes the changes back.

    Args:
        config_path (str): The full path to the YAML configuration file.
        param_name (str): The name of the parameter to change (e.g., 'fit_lagrange_mhz').
        new_value: The new value for the parameter.
    """
    try:
        # Read the entire YAML file into a Python dictionary
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Safely navigate to the 'fitting' section
        fitting_section = config_data.setdefault('pipeline_options', {})

        # Update the specified parameter with the new value
        fitting_section[param_name] = new_value

        # Write the modified dictionary back to the YAML file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"Successfully updated '{param_name}' to '{new_value}' in {config_path}")

    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
    except Exception as e:
        print(f"An error occurred while updating the YAML file: {e}")