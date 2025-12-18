#!/usr/bin/env python3
"""
Standardize frequency axis ordering for all FRB dynamic spectra.

This script ensures all .npy data files have ASCENDING frequency order:
    - data[0, :] = lowest frequency channel
    - data[-1, :] = highest frequency channel

The ordering is determined using dispersion physics:
    - Lower frequencies arrive LATER due to dispersion delay
    - If the top of the array (index 0) shows later arrival → already ascending
    - If the bottom of the array shows later arrival → descending, needs flip

For data with minimal dispersion sweep (e.g., DSA at 1.4 GHz), we use
telescope configuration to determine expected dispersion and validate.

Usage:
    python scripts/standardize_freq_axis.py [--dry-run] [--telescope TELESCOPE]

Options:
    --dry-run       Show what would be done without modifying files
    --telescope     Process only files for specified telescope (chime, dsa)
    --backup        Create backups before modifying (default: True)
"""

import argparse
import numpy as np
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TelescopeSpec:
    """Telescope frequency specifications."""

    f_min_ghz: float
    f_max_ghz: float
    df_mhz_raw: float
    dt_ms_raw: float

    @property
    def center_freq_ghz(self) -> float:
        return (self.f_min_ghz + self.f_max_ghz) / 2

    @property
    def bandwidth_ghz(self) -> float:
        return self.f_max_ghz - self.f_min_ghz


# Telescope configurations
TELESCOPES = {
    "chime": TelescopeSpec(
        f_min_ghz=0.400,
        f_max_ghz=0.800,
        df_mhz_raw=0.390625,
        dt_ms_raw=0.00256,
    ),
    "dsa": TelescopeSpec(
        f_min_ghz=1.311,
        f_max_ghz=1.499,
        df_mhz_raw=0.030517578,
        dt_ms_raw=0.032768,
    ),
}


def expected_dispersion_delay_samples(
    dm: float,
    f_low_ghz: float,
    f_high_ghz: float,
    dt_ms: float,
) -> float:
    """
    Calculate expected dispersion delay in samples between two frequencies.

    Dispersion delay: Δt = 4.149 ms × DM × (f_low^-2 - f_high^-2)
    where f is in GHz.
    """
    K_DM = 4.148808  # ms GHz^2 pc^-1 cm^3
    delay_ms = K_DM * dm * (f_low_ghz**-2 - f_high_ghz**-2)
    return delay_ms / dt_ms


def detect_frequency_ordering(
    data: np.ndarray,
    telescope: str,
    dm_estimate: float = 300.0,
) -> Tuple[str, float, str]:
    """
    Detect the frequency ordering of a dynamic spectrum using dispersion.

    Returns:
        ordering: 'ascending' (index 0 = low freq) or 'descending' (index 0 = high freq)
        confidence: 0.0 to 1.0
        method: description of detection method used
    """
    n_freq, _ = data.shape
    spec = TELESCOPES.get(telescope)

    if spec is None:
        raise ValueError(f"Unknown telescope: {telescope}")

    # Expected dispersion delay across the band
    expected_delay_samples = expected_dispersion_delay_samples(
        dm_estimate, spec.f_min_ghz, spec.f_max_ghz, spec.dt_ms_raw
    )

    # Measure actual delay between top and bottom of array
    n_avg = max(10, n_freq // 20)  # Average over ~5% of channels

    # Get time profiles for top and bottom frequency bands
    top_profile = np.nanmean(data[:n_avg, :], axis=0)
    bot_profile = np.nanmean(data[-n_avg:, :], axis=0)

    # Find peak times using centroid for robustness
    def find_peak_centroid(profile, width=50):
        """Find peak using centroid around maximum."""
        peak_idx = np.argmax(profile)
        left = max(0, peak_idx - width)
        right = min(len(profile), peak_idx + width)
        window = profile[left:right]
        indices = np.arange(left, right)
        if np.sum(window) > 0:
            centroid = np.sum(indices * window) / np.sum(window)
            return centroid
        return peak_idx

    top_peak = find_peak_centroid(top_profile)
    bot_peak = find_peak_centroid(bot_profile)

    measured_delay = top_peak - bot_peak  # Positive if top arrives later

    # Interpret the result
    # If top arrives LATER, top = low freq = ASCENDING
    # If top arrives EARLIER, top = high freq = DESCENDING

    abs_delay = abs(measured_delay)
    expected_abs = abs(expected_delay_samples)

    # Confidence based on how much delay we measured relative to expected
    if expected_abs > 0:
        confidence = min(1.0, abs_delay / (expected_abs * 0.3))
    else:
        confidence = 0.0

    # Determine ordering
    if abs_delay < 20:
        # Very small delay - use telescope-specific heuristics
        # CHIME at 400-800 MHz should have large dispersion
        # DSA at 1.3-1.5 GHz has smaller dispersion
        if telescope == "chime":
            # CHIME should have clear dispersion, small delay suggests issue
            method = f"weak_signal (delay={measured_delay:.0f} samples, expected ~{expected_delay_samples:.0f})"
            confidence = 0.3
            # Default to descending for CHIME (historical convention)
            ordering = "descending"
        else:
            # DSA has smaller dispersion, less certain
            method = f"small_delay (delay={measured_delay:.0f} samples)"
            confidence = 0.2
            ordering = "descending"  # DSA native format appears to be descending
    elif measured_delay > 0:
        ordering = "ascending"
        method = f"dispersion (delay={measured_delay:.0f} samples, top arrives later)"
        confidence = min(1.0, abs_delay / 100)
    else:
        ordering = "descending"
        method = (
            f"dispersion (delay={measured_delay:.0f} samples, bottom arrives later)"
        )
        confidence = min(1.0, abs_delay / 100)

    return ordering, confidence, method


def standardize_file(
    filepath: Path,
    telescope: str,
    dry_run: bool = False,
    backup: bool = True,
    dm_estimate: float = 300.0,
) -> dict:
    """
    Standardize a single file to ascending frequency order.

    Returns dict with results.
    """
    result = {
        "file": filepath.name,
        "action": "none",
        "ordering": None,
        "confidence": 0.0,
        "method": None,
    }

    try:
        data = np.load(filepath)
        result["shape"] = data.shape

        ordering, confidence, method = detect_frequency_ordering(
            data, telescope, dm_estimate
        )

        result["ordering"] = ordering
        result["confidence"] = confidence
        result["method"] = method

        if ordering == "ascending":
            result["action"] = "keep"
        else:
            result["action"] = "flip"

            if not dry_run:
                if backup:
                    backup_dir = filepath.parent / "backup_before_standardize"
                    backup_dir.mkdir(exist_ok=True)
                    backup_path = backup_dir / filepath.name
                    if not backup_path.exists():
                        shutil.copy2(filepath, backup_path)

                # Flip the data
                data_flipped = np.flipud(data)
                np.save(filepath, data_flipped)

    except Exception as e:
        result["action"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Standardize frequency axis ordering to ascending",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files",
    )
    parser.add_argument(
        "--telescope",
        choices=["chime", "dsa", "all"],
        default="all",
        help="Process only specified telescope data",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backups",
    )
    parser.add_argument(
        "--dm",
        type=float,
        default=300.0,
        help="Estimated DM for dispersion calculation (default: 300)",
    )

    args = parser.parse_args()

    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    telescopes = ["chime", "dsa"] if args.telescope == "all" else [args.telescope]

    print("=" * 80)
    print("FREQUENCY AXIS STANDARDIZATION")
    print("Target: All data files → ASCENDING order (index 0 = low freq)")
    print("=" * 80)

    if args.dry_run:
        print("*** DRY RUN - No files will be modified ***\n")

    all_results = []

    for telescope in telescopes:
        tel_dir = data_dir / telescope
        if not tel_dir.exists():
            print(f"Warning: {tel_dir} not found, skipping")
            continue

        print(f"\n=== {telescope.upper()} ({tel_dir}) ===\n")

        npy_files = sorted(tel_dir.glob("*.npy"))
        npy_files = [f for f in npy_files if not f.name.startswith("FRB")]

        for filepath in npy_files:
            result = standardize_file(
                filepath,
                telescope,
                dry_run=args.dry_run,
                backup=not args.no_backup,
                dm_estimate=args.dm,
            )
            all_results.append(result)

            action_symbol = {
                "keep": "✓",
                "flip": "↕",
                "error": "✗",
                "none": "?",
            }.get(result["action"], "?")

            conf_str = f"{result['confidence']:.0%}" if result["confidence"] else "N/A"

            print(f"  {action_symbol} {result['file']}")
            print(f"      Ordering: {result['ordering']}, Confidence: {conf_str}")
            print(f"      Method: {result['method']}")
            print(f"      Action: {result['action']}")
            print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    kept = sum(1 for r in all_results if r["action"] == "keep")
    flipped = sum(1 for r in all_results if r["action"] == "flip")
    errors = sum(1 for r in all_results if r["action"] == "error")

    print(f"  Kept (already ascending): {kept}")
    print(f"  Flipped to ascending:     {flipped}")
    print(f"  Errors:                   {errors}")
    print(f"  Total processed:          {len(all_results)}")

    if args.dry_run:
        print("\n*** DRY RUN - No files were modified ***")
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
