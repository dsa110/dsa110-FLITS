"""Validate the RFI-mask + raw-input fix on isha (was noise) and whitney (was marginal)
before re-running all 12."""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
from extract_chime_side_inputs import extract_one

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
cases = [("isha", "252069198", 411.568), ("whitney", "215063905", 462.174)]
for name, cid, dm in cases:
    rec, fig = extract_one(
        f"/data/Faber2026/data/chime-frb/{name.lower()}/singlebeam_{cid}.h5", dm
    )
    fig.savefig(f"{ROOT}/diagnostics/validate_{name}.png", dpi=110)
    print(
        f"[{name}] DM_chime={rec['dm_chime']:.3f}±{rec['dm_chime_err']:.2f} "
        f"interior={rec['interior']} flat={rec['flat_ratio']:.2f} snr~{rec['snr']:.1f} "
        f"n_ch_ok={rec['n_chan_ok']}",
        flush=True,
    )
