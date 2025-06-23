"""
pool_utils.py
==============

Utility for building a *pool‑like* object for the ``emcee`` `pool=`
interface.  Three user scenarios are supported:

1. **Serial** – pass ``--nproc 0`` or set ``n_requested=0``.
2. **Auto‑detect & prompt** – omit ``--nproc``.  We suggest
   ``cpu_count()-1`` workers, then ask for confirmation **or** a custom
   integer.
3. **Batch** – supply ``--nproc <N>`` or call ``build_pool(N)`` in code
   (no prompt).

The function returns either a :pyclass:`multiprocessing.Pool` *or*
``None`` (serial execution).  The caller owns the pool’s lifetime; wrap
in a ``with`` block or remember to ``close()`` & ``join()``.

CLI demo
--------
```bash
python pool_utils.py          # auto, ask
python pool_utils.py --yes    # auto, no prompt
python pool_utils.py -n 8     # 8 workers
python pool_utils.py -n 0     # serial
```
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional

__all__ = ["build_pool"]

# ---------------------------------------------------------------------
# public helper
# ---------------------------------------------------------------------

def build_pool(
    n_requested: int | None = None,
    *,
    auto_ok: bool = False,
    label: str = "BurstFit",
):
    """Return a :class:`multiprocessing.Pool` or ``None``."""
    
    # --- FIX: Check for None FIRST to avoid the TypeError ---
    if n_requested is None:
        # Auto-detect and prompt mode
        proposal = max(mp.cpu_count() - 1, 1)
        if auto_ok:
            n_requested = proposal
        else:
            try:
                ans = input(
                    f"[{label}] detected {mp.cpu_count()} logical CPUs. "
                    f"Use how many workers? [default {proposal}, 0 = serial] » "
                ).strip()
            except EOFError:
                ans = "" # Handle case where input stream is closed
                
            if ans == "":
                n_requested = proposal
            else:
                try:
                    n_requested = int(ans)
                except ValueError:
                    print(f"[{label}] invalid input; falling back to serial")
                    return None
        
        # After getting user input, proceed to create the pool or run serially
        if n_requested == 0:
            print(f"[{label}] serial mode chosen")
            return None
        else:
            print(f"[{label}] starting Pool({n_requested})")
            return mp.Pool(processes=n_requested)

    # Case for explicit serial run
    elif n_requested == 0:
        print(f"[{label}] running serially (nproc=0)")
        return None
    
    # Case for explicit batch run
    elif n_requested > 0:
        print(f"[{label}] running with nproc={n_requested}")
        return mp.Pool(processes=n_requested)

# ---------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Create a multiprocessing pool for emcee-compatible code")
    p.add_argument("-n", "--nproc", type=int, default=None,
                   help="Number of worker processes (0=serial, omit=prompt)")
    p.add_argument("--yes", action="store_true", help="Bypass confirmation when auto-detecting cores")
    args = p.parse_args()

    pool = build_pool(args.nproc, auto_ok=args.yes, label=Path(__file__).stem)

    if pool is not None:
        print("Pool is live; mapping dummy job...")
        res = pool.map(lambda x: x**2, range(5))
        print("map(range(5)) →", res)
        pool.close()
        pool.join()
    else:
        print("Running serially – no pool object created.")

if __name__ == "__main__":
    _cli()
