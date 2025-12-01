# ==============================================================================
# File: scint_analysis/scint_analysis/config.py
# ==============================================================================
import yaml
import os
import logging
from pathlib import Path
from typing import Union, Optional

log = logging.getLogger(__name__)


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolve a path, supporting relative paths and environment variables.
    
    Parameters
    ----------
    path : str or Path
        The path to resolve. Can be:
        - Absolute path: returned as-is
        - Relative path: resolved relative to base_dir or CWD
        - Contains $VAR or ${VAR}: environment variables are expanded
        - Starts with ~: user home directory is expanded
    base_dir : Path, optional
        Base directory for resolving relative paths.
        
    Returns
    -------
    Path
        The resolved, absolute path.
    """
    path_str = str(path)
    path_str = os.path.expandvars(path_str)
    path_str = os.path.expanduser(path_str)
    resolved = Path(path_str)
    
    if resolved.is_absolute():
        return resolved
    
    if base_dir is not None:
        resolved = base_dir / resolved
    else:
        resolved = Path.cwd() / resolved
    
    return resolved.resolve()

def load_config(burst_config_path, workspace_root: Optional[Union[str, Path]] = None):
    """
    Loads and merges telescope and burst-specific configuration files.

    Args:
        burst_config_path (str): The full path to the burst's YAML config file.
        workspace_root (str or Path, optional): Root directory for resolving 
            relative paths. If None, auto-detected from config location.

    Returns:
        dict: A single dictionary containing the merged configuration.
        
    Notes
    -----
    Relative paths in the config are resolved using `resolve_path`, which
    supports environment variables ($VAR) and home directory (~) expansion.
    """
    log.info(f"Loading burst configuration from: {burst_config_path}")
    
    config_path = Path(burst_config_path).expanduser().resolve()
    config_dir = config_path.parent
    
    # Auto-detect workspace root if not provided
    if workspace_root is None:
        workspace_root = config_dir
        for _ in range(5):
            if (workspace_root / "pyproject.toml").exists() or \
               (workspace_root / "setup.py").exists() or \
               (workspace_root / ".git").exists():
                break
            if workspace_root.parent == workspace_root:
                break
            workspace_root = workspace_root.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    # Ensure configs can use ${FLITS_ROOT} without requiring users to pre-set it.
    os.environ.setdefault("FLITS_ROOT", str(workspace_root))
    
    try:
        with open(config_path, 'r') as f:
            burst_config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error(f"Burst config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing burst YAML file: {e}")
        raise

    # Resolve input_data_path relative to workspace root
    if 'input_data_path' in burst_config:
        raw_data_path = burst_config['input_data_path']
        burst_config['input_data_path'] = str(resolve_path(raw_data_path, base_dir=workspace_root))
        log.info(f"Resolved data path: {burst_config['input_data_path']}")

    # Determine the path to the telescope config file
    telescope_name = burst_config.get('telescope')
    if not telescope_name:
        log.error("Burst config must contain a 'telescope' key.")
        raise ValueError("Missing 'telescope' key in burst config.")
    
    # Try multiple possible locations for telescope config
    telescope_config_path = None
    possible_paths = [
        config_dir / '..' / 'telescopes' / f"{telescope_name}.yaml",
        config_dir / 'telescopes' / f"{telescope_name}.yaml",
        workspace_root / 'scintillation' / 'configs' / 'telescopes' / f"{telescope_name}.yaml",
    ]
    
    for path in possible_paths:
        resolved = path.resolve()
        if resolved.exists():
            telescope_config_path = resolved
            break
    
    if telescope_config_path is None:
        log.error(f"Telescope config file not found. Searched: {[str(p.resolve()) for p in possible_paths]}")
        raise FileNotFoundError(f"Could not find telescope config for '{telescope_name}'")
    
    log.info(f"Loading telescope configuration from: {telescope_config_path}")
    try:
        with open(telescope_config_path, 'r') as f:
            telescope_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error(f"Error parsing telescope YAML file: {e}")
        raise

    # Merge the configurations, with burst-specific values overriding telescope defaults
    merged_config = {**telescope_config, **burst_config}
    
    # Store workspace root for downstream use
    merged_config['_workspace_root'] = str(workspace_root)
    merged_config['_config_dir'] = str(config_dir)
    
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
