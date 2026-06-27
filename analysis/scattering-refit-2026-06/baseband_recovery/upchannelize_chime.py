#!/usr/bin/env python3
"""CHIME baseband upchannelization for the resolution-limited co-detection sightlines.

Runs INSIDE the `chimefrb/baseband-analysis:latest` docker image on h17 (lxd110h17), which carries
baseband_analysis 1.9.0 + the CADC `vos` client. h17 reaches the CHIME baseband store on CANFAR/arc
directly (verified: `vls arc:projects/chime_frb/...` works in-container with ~/.ssl/cadcproxy.pem),
so there is NO CANFAR Science-Platform / Harbor dependency. Per target this:
  1. vcp's the ~1 GB singlebeam_<id>.h5 from arc to local scratch (idempotent),
  2. coherently dedisperses the complex per-channel baseband at the burst DM,
  3. upchannelizes each 0.390625 MHz CHIME coarse channel by the verified per-target factor,
  4. forms a Stokes-I dynamic spectrum and writes a small <name>_chime_upchan.npy + _freq.npy.

WHY coherent dedispersion + PFB upchannelization (not a cheap incoherent rechannel):
  The scintillation measurement is a spectral autocorrelation (ACF) of the time-integrated burst
  spectrum; its diffractive bandwidth Dnu_d is the HWHM of the ACF's central Lorentzian. Two
  systematics counterfeit a scintle and bias Dnu_d if not removed at the baseband level:
    1. Intra-channel dispersive smearing. At these DMs (462-960 pc/cc) the sweep across one CHIME
       coarse channel is many microseconds; upchannelizing a smeared channel imprints a dispersive
       chirp that survives as spurious narrow-band ACF structure. Coherent dedispersion on the raw
       voltages removes the sweep EXACTLY (the only way to recover sub-channel spectral resolution
       without re-smearing). Incoherent (post-detection) dedispersion cannot -- the phase needed to
       de-chirp is gone after squaring.
    2. PFB channel-edge response. The CHIME polyphase-filterbank inverse (baseband_analysis) gives
       the synthesized fine channels a flat passband; a naive FFT-rechannel leaves the coarse-channel
       scallop, a deterministic ripple that contaminates the small-lag ACF exactly where Dnu_d lives.
  The limiting scintle at CHIME is NARROWER than one 0.390625 MHz coarse channel (NE2025/NE2001
  predict sub-channel Dnu_d for all of these), so it is unresolved at native CHIME resolution;
  upchannelizing exposes it -- but only if the channel is de-chirped first.

UPCHAN FACTOR (skeptic-corrected, sized to the DOMINANT/narrower scintle at >=4 ch across its HWHM):
  - casey   U=16  host-dominated  (host Dnu_d 0.187 < MW floor 0.207 MHz) -- the one clean host case.
  - whitney U=16  MW-floor-dominated (resolves the 0.140 MHz floor; native x16 over-resolves, fine).
  - phineas U=16  MW-floor-dominated (0.206 MHz floor; min factor is 8 but native x16 is cleaner).
  - mahi    U=512 MW-floor-dominated (0.0036 MHz floor); only non-smeared because FWHM=24 ms. Uses
            the _upchannel(fftsize=2U) generalization (see _waterfall) -- the SLOW python-loop path.
  - isha    OFF by default: NOT cleanly resolvable -- DSA input gamma railed at the 0.06 MHz fit floor
            AND the dominant scale needs U>=256-512 while the 1.8 ms burst smears to <3 time elements.
            Run only as a lower-confidence upper bound with --run-unresolvable.

API NOTE: baseband_analysis.analysis.waterfall_from_beamformed is BROKEN in this image (v1.9.0): it
feeds upchannel()'s 3-tuple straight into incoherent_dedisp, which does matrix_in.copy() ->
AttributeError on the tuple. So this worker drives the package's own primitives directly for ALL
factors: coherent_dedisp + the internal _upchannel(fftsize=2U, downfreq=2) (which returns
(spec, freq, chan_id); upchan factor U = fftsize/downfreq), forming Stokes I from the complex spectrum
without the incoherent step (coherent dedispersion already de-chirps fully). See _waterfall.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

CHIME_COARSE_DF_MHZ = 0.390625  # CHIME coarse channel width (400 MHz / 1024)
CHIME_NATIVE_DT_S = 2.56e-6  # CHIME single-pol baseband sample time

ARC_VOS_ROOT = "arc:projects/chime_frb/data/chime/baseband/processed"  # vcp source (CADC vos URI)
# The 5 singlebeam .h5 are already staged on h17 here -> use in place, no vcp / no arc dependency.
LOCAL_H5_DIR = "/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam"
DEFAULT_SCRATCH = (
    "/data/jfaber/chime_singlebeam"  # vcp fallback landing if a file is NOT pre-staged
)
DEFAULT_OUT_DIR = "/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections"

# id/dm/fwhm_ms from crossmatching/notebook_reproduction_fixture.json (DM also in configs/bursts.yaml).
TARGETS = {
    "casey": {
        "id": "362593221",
        "dm": 491.207,
        "fwhm_ms": 0.1798,
        "upchan": 16,
        "recoverable": True,
        "h5_relpath": "2024/02/29/astro_362593221/singlebeam_362593221.h5",
        "note": "host-dominated (host 0.187 < MW floor 0.207 MHz); cleanest DSA input -- the clean host recovery.",
    },
    "whitney": {
        "id": "215063905",
        "dm": 462.174,
        "fwhm_ms": 0.4865,
        "upchan": 16,
        "recoverable": True,
        "h5_relpath": "2022/03/10/astro_215063905/singlebeam_215063905.h5",
        "note": "MW-floor-dominated (0.140 MHz floor); native x16 (24 kHz) resolves it. Galactic, not host.",
    },
    "phineas": {
        "id": "274819243",
        "dm": 610.274,
        "fwhm_ms": 2.9886,
        "upchan": 16,
        "recoverable": True,
        "h5_relpath": "2023/03/07/astro_274819243/singlebeam_274819243.h5",
        "note": "MW-floor-dominated (0.206 MHz floor); native x16. Long FWHM, time-res ample. Galactic, not host.",
    },
    "mahi": {
        "id": "354049284",
        "dm": 960.128,
        "fwhm_ms": 24.286,
        "upchan": 512,
        "recoverable": True,
        "h5_relpath": "2024/01/22/astro_354049284/singlebeam_354049284.h5",
        "note": "MW-floor-dominated (0.0036 MHz floor) -> U=512 via _upchannel(fftsize=1024); slow. Safe only b/c FWHM=24 ms.",
    },
    "isha": {
        "id": "252069198",
        "dm": 411.568,
        "fwhm_ms": 1.8053,
        "upchan": 256,
        "recoverable": False,
        "h5_relpath": "2022/11/13/astro_252069198/singlebeam_252069198.h5",
        "note": "NOT cleanly resolvable: railed DSA input + dominant scale at/past the time-smearing wall. Upper-bound only.",
    },
}


def _fetch_h5(relpath: str, scratch: str) -> str:
    """Locate the singlebeam .h5: prefer the h17 pre-staged copy; vcp from arc only if absent."""
    name = Path(relpath).name
    local = Path(LOCAL_H5_DIR) / name
    if local.exists():
        return str(local)
    dst = Path(scratch) / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["vcp", f"{ARC_VOS_ROOT}/{relpath}", str(dst)], check=True)
    return str(dst)


def _waterfall(h5_path: str, dm: float, U: int):
    """Coherently-dedispersed, upchannelized Stokes-I waterfall (n_fine_freq, n_time) + freq[MHz].

    We assemble the chain by hand rather than via baseband_analysis.analysis.waterfall_from_beamformed
    because that function is BROKEN in this image (v1.9.0): it passes upchannel()'s 3-tuple return
    straight into incoherent_dedisp, which does `matrix_in.copy()` -> AttributeError on the tuple.
    The pieces we use ARE the package's: coherent_dedisp + the internal _upchannel. After coherent
    dedispersion the baseband is fully de-chirped, so no incoherent_dedisp step is needed -- we form
    Stokes I directly from the upchannelized complex spectrum.
    """
    from baseband_analysis.core.bbdata import BBData  # noqa: PLC0415
    from baseband_analysis.core.dedispersion import coherent_dedisp  # noqa: PLC0415
    from baseband_analysis.core.sampling import _upchannel  # noqa: PLC0415

    data = BBData.from_file(h5_path)
    coherent_dedisp(
        data, dm, time_shift=True
    )  # exact de-chirp on the complex voltages at the burst DM

    # _upchannel returns (spec, freq, chan_id): spec is (npol, nblock, nfine) complex, freq the
    # fine-channel centres (MHz) ordered high->low. upchan factor U = fftsize/downfreq.
    spec, freq, _ = _upchannel(
        data["tiedbeam_baseband"][:],
        freq_id=data.index_map["freq"]["id"][:],
        fftsize=2 * U,
        downfreq=2,
    )
    # Stokes I = |X|^2 + |Y|^2 over the two pols -> (nblock, nfine), transpose to (nfine, n_time).
    stokes_i = (np.abs(spec[0]) ** 2 + np.abs(spec[1]) ** 2).T
    return stokes_i, np.asarray(freq, dtype=np.float64)


def recover_target(name: str, scratch: str, out_dir: str, run_unresolvable: bool = False) -> Path:
    t = TARGETS[name]
    if not t["recoverable"] and not run_unresolvable:
        raise SystemExit(
            f"{name} is flagged NOT cleanly resolvable ({t['note']}). "
            f"Re-run with --run-unresolvable to produce a lower-confidence upper-bound spectrum."
        )

    h5_path = _fetch_h5(t["h5_relpath"], scratch)
    U = t["upchan"]
    stokes_i, freq = _waterfall(h5_path, t["dm"], U)

    # Ascending frequency to match the FLITS BurstDataset convention.
    if freq[0] > freq[-1]:
        freq = freq[::-1]
        stokes_i = stokes_i[::-1, :]

    df_fine = CHIME_COARSE_DF_MHZ / U
    n_fine, n_time = stokes_i.shape

    # --- ponytail self-check: the recovered grid must match the requested upchannelization ---
    # NaN channels are EXPECTED (CHIME masks RFI/missing channels); the downstream ACF uses nansum.
    # So require a healthy finite FRACTION, not all-finite -- only an all-NaN/empty result is a failure.
    assert n_time > 0, f"{name}: empty time axis"
    assert n_fine >= 1024, f"{name}: only {n_fine} channels -- not upchannelized beyond native 1024"
    finite_frac = float(np.isfinite(stokes_i).mean())
    assert finite_frac > 0.3, f"{name}: only {finite_frac:.1%} finite Stokes-I -- effectively empty"
    measured_df = abs(np.nanmedian(np.diff(freq)))
    assert np.isclose(measured_df, df_fine, rtol=0.05), (
        f"{name}: fine channel width {measured_df:.6f} MHz != expected {df_fine:.6f} MHz (U={U})"
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / f"{name}_chime_upchan.npy"
    np.save(spec_path, stokes_i.astype(np.float32))
    np.save(out / f"{name}_chime_freq.npy", freq)
    print(
        f"[{name}] U={U} shape={stokes_i.shape} df={df_fine * 1e3:.3f} kHz "
        f"dt={CHIME_NATIVE_DT_S * 2 * U * 1e3:.4f} ms finite={finite_frac:.1%} -> {spec_path.name}"
    )
    return spec_path


def main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("targets", nargs="*", default=list(TARGETS), help="targets (default: all)")
    p.add_argument("--scratch", default=DEFAULT_SCRATCH, help="local landing dir for the .h5 files")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="output dir for the .npy products")
    p.add_argument(
        "--run-unresolvable",
        action="store_true",
        help="also process targets flagged NOT cleanly resolvable (isha), as an upper bound",
    )
    args = p.parse_args(argv)

    targets = args.targets or list(TARGETS)
    unknown = [n for n in targets if n not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; known: {list(TARGETS)}")
    for name in targets:
        recover_target(name, args.scratch, args.out, run_unresolvable=args.run_unresolvable)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
