"""Generic utilities shared across FLITS modules."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import astropy.units as u
from astropy.time import Time

from .constants import K_DM


def downsample_time(data, t_factor):
    """Block-average by integer factor along the time axis."""
    if t_factor < 1:
        raise ValueError(f"t_factor must be ≥1, got {t_factor}")
    arr = np.asarray(data)
    if arr.ndim == 1:
        nt = arr.shape[0]
        nt_trim = nt - (nt % t_factor)
        return arr[:nt_trim].reshape(nt_trim // t_factor, t_factor).mean(axis=1)
    elif arr.ndim == 2:
        nfreq, nt = arr.shape
        nt_trim = nt - (nt % t_factor)
        blocks = arr[:, :nt_trim].reshape(nfreq, nt_trim // t_factor, t_factor)
        return blocks.mean(axis=2)
    else:
        raise ValueError(f"Unsupported array shape {arr.shape}; expected 1D or 2D.")


def calculate_dm_timing_error(dDM, f_obs, f_ref, K_DM: float = K_DM):
    """Calculate timing error due to DM uncertainty."""
    time_shift = K_DM * dDM * (1 / f_obs.value**2 - 1 / f_ref.value**2) * u.s
    return np.abs(time_shift.to(u.ms))


def clean_and_serialize_dict(burst_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert astropy objects in a dictionary to JSON-serializable types."""
    clean_dict: Dict[str, Any] = {}
    for key, value in burst_dict.items():
        if isinstance(value, u.Quantity):
            clean_dict[key] = value.value
        elif isinstance(value, Time):
            clean_dict[key] = value.iso
        else:
            clean_dict[key] = value
    return clean_dict


def append_to_json(new_data_dict: Dict[str, Any], filename: str) -> None:
    """Append a dictionary to a JSON file, creating the file if needed."""
    clean_new_data = clean_and_serialize_dict(new_data_dict)
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, "r") as f:
                data_list = json.load(f)
            if not isinstance(data_list, list):
                print(f"Error: JSON file '{filename}' does not contain a list.")
                data_list = [clean_new_data]
            else:
                data_list.append(clean_new_data)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{filename}'. Starting a new file.")
            data_list = [clean_new_data]
    else:
        data_list = [clean_new_data]
    with open(filename, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"Successfully appended data to {filename}")
