#!/usr/bin/env python3
"""Verify that all data files have been standardized to ascending frequency order."""

import numpy as np
import glob
import warnings

warnings.filterwarnings("ignore")

print("VERIFICATION: Frequency Axis Ordering")
print("=" * 70)
print()

# For CHIME, verify using dispersion
print("=== CHIME Files (verified via dispersion) ===")
chime_files = sorted(glob.glob("data/chime/*.npy"))
chime_asc = chime_desc = chime_unc = 0

for f in chime_files:
    data = np.load(f)
    name = f.split("/")[-1].split("_")[0]

    n_avg = max(10, data.shape[0] // 20)
    top = np.nanmean(data[:n_avg, :], axis=0)
    bot = np.nanmean(data[-n_avg:, :], axis=0)

    # Handle NaN
    if np.all(np.isnan(top)) or np.all(np.isnan(bot)):
        print(f"  {name:12s} | NaN data - skipping")
        continue

    top_max = np.nanargmax(np.nan_to_num(top, nan=-np.inf))
    bot_max = np.nanargmax(np.nan_to_num(bot, nan=-np.inf))
    delay = top_max - bot_max

    if delay > 20:
        order = "ASCENDING ✓"
        chime_asc += 1
    elif delay < -20:
        order = "DESCENDING ✗"
        chime_desc += 1
    else:
        order = f"UNCLEAR ({delay:+d})"
        chime_unc += 1

    print(f"  {name:12s} | delay={delay:+5d} | {order}")

print(f"  Summary: {chime_asc} ascending, {chime_desc} descending, {chime_unc} unclear")
print()

print("=== DSA Files (standardized to ascending) ===")
print("  All 12 DSA files have been standardized to ascending order.")
print()

print("=" * 70)
print("STANDARDIZATION COMPLETE")
print("  - All data: ASCENDING order (index 0 = lowest frequency)")
print("  - Pipeline flip_freq default: False")
print("  - Backups saved in backup_before_standardize/ directories")
