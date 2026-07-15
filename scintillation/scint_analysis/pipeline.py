# ==============================================================================
# File: scint_analysis/scint_analysis/pipeline.py
# ==============================================================================
import hashlib
import json
import logging
import os
import pickle

import numpy as np

# Make sure to import the new noise module
from . import analysis, core, noise, plotting

log = logging.getLogger(__name__)


class ScintillationAnalysis:
    """
    An object-oriented controller for running the end-to-end scintillation pipeline.
    """

    def __init__(self, config):
        self.config = config
        self.masked_spectrum = None
        self.noise_descriptor = None
        self.acf_results = None
        # On/off-pulse time windows resolved in run(); exposed so a downstream
        # off-pulse ACF null (chime_artifact_guards) can reuse the identical
        # windows the ACF normalization used. None until run() sets them.
        self.burst_lims = None
        self.off_pulse_lims = None
        self.all_subband_fits = None
        self.final_results = None
        self.all_powerlaw_fits = None
        self.intra_pulse_results = None
        self.modulation_over_time = None
        self.data_prepared = False

        self.cache_dir = self.config.get("pipeline_options", {}).get("cache_directory", "./cache")
        if self.config.get("pipeline_options", {}).get("save_intermediate_steps"):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

    def _config_fingerprint(self):
        """Short hash of every config field that shapes cached pipeline products.

        Cache files are pickles keyed by burst_id; without a fingerprint,
        toggling a preprocessing flag (grid_regularization,
        bandpass_normalization, RFI masking, downsample factors, ...) after a
        cached run would silently reload the stale spectrum/ACF built under
        the old settings (#120 review r2, P1). Fingerprinting the whole
        `analysis` block plus input path and downsample factors errs toward
        recomputation, never toward stale reuse.
        """
        relevant = {
            "input_data_path": self.config.get("input_data_path"),
            "downsample": self.config.get("pipeline_options", {}).get("downsample", {}),
            "analysis": self.config.get("analysis", {}),
        }
        payload = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _get_cache_path(self, stage_name):
        """Generates a standard path for a cache file."""
        burst_id = self.config.get("burst_id", "unknown_burst")
        return os.path.join(
            self.cache_dir, f"{burst_id}_{self._config_fingerprint()}_{stage_name}.pkl"
        )

    def _create_diagnostic_plots(self, burst_lims, off_pulse_lims, baseline_info=None):
        """Internal helper to generate and save diagnostic plots."""
        diag_config = self.config.get("pipeline_options", {}).get("diagnostic_plots", {})
        if not diag_config.get("enable", False):
            return

        log.info("Generating diagnostic plots...")
        plot_dir = diag_config.get("directory", "./plots/diagnostics")
        os.makedirs(plot_dir, exist_ok=True)
        burst_id = self.config.get("burst_id", "unknown_burst")

        # --- On-pulse and Off-pulse Window Plots ---
        try:
            # 1. Prepare and plot the on-pulse window
            on_pulse_power = self.masked_spectrum.power[:, burst_lims[0] : burst_lims[1]]
            on_pulse_times = self.masked_spectrum.times[burst_lims[0] : burst_lims[1]]
            on_pulse_ds_obj = core.DynamicSpectrum(
                on_pulse_power, self.masked_spectrum.frequencies, on_pulse_times
            )
            on_pulse_save_path = os.path.join(plot_dir, f"{burst_id}_on_pulse_diagnostic.png")

            plotting.plot_pulse_window_diagnostic(
                on_pulse_ds_obj, title="On-Pulse Region", save_path=on_pulse_save_path
            )

            # 2. Prepare and plot the off-pulse (noise) window
            off_pulse_power = self.masked_spectrum.power[:, off_pulse_lims[0] : off_pulse_lims[1]]
            off_pulse_times = self.masked_spectrum.times[off_pulse_lims[0] : off_pulse_lims[1]]
            off_pulse_ds_obj = core.DynamicSpectrum(
                off_pulse_power, self.masked_spectrum.frequencies, off_pulse_times
            )
            off_pulse_save_path = os.path.join(plot_dir, f"{burst_id}_off_pulse_diagnostic.png")

            plotting.plot_pulse_window_diagnostic(
                off_pulse_ds_obj, title="Off-Pulse (Noise) Region", save_path=off_pulse_save_path
            )

            log.info(f"On/Off pulse diagnostic plots saved to: {plot_dir}")

        except Exception as e:
            log.error(f"Failed to generate on/off pulse diagnostic plots: {e}")

        if baseline_info:
            log.info("Generating baseline fit diagnostic plot.")
            baseline_save_path = os.path.join(plot_dir, f"{burst_id}_baseline_diagnostic.png")
            plotting.plot_baseline_fit(
                off_pulse_spectrum=baseline_info["original_data"],
                fitted_baseline=baseline_info["model"],
                frequencies=self.masked_spectrum.frequencies,
                poly_order=baseline_info["poly_order"],
                save_path=baseline_save_path,
            )

    def prepare_data(self):
        """
        Loads data from file and performs initial RFI masking.
        Populates self.masked_spectrum.
        """
        if self.data_prepared:
            log.info("Data already prepared. Skipping.")
            return

        log.info("--- Preparing Data ---")

        self.cache_dir = self.config.get("pipeline_options", {}).get("cache_directory", "./cache")
        if self.config.get("pipeline_options", {}).get("save_intermediate_steps"):
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(f"Intermediate results will be cached in: {self.cache_dir}")

        processed_spec_cache = self._get_cache_path("processed_spectrum")

        if os.path.exists(processed_spec_cache) and not self.config.get("pipeline_options", {}).get(
            "force_recalc", False
        ):
            log.info(f"Loading cached processed spectrum from {processed_spec_cache}")
            with open(processed_spec_cache, "rb") as f:
                # The cache now only needs to store the masked spectrum
                self.masked_spectrum = pickle.load(f)

        else:
            log.info("Loading and processing raw data...")
            # spectrum = core.DynamicSpectrum.from_numpy_file(self.config['input_data_path'])
            # --- optional down-sampling factors ---------------------------------
            ds_cfg = self.config.get("pipeline_options", {}).get("downsample", {})
            f_factor = int(ds_cfg.get("f_factor", 1))
            t_factor = int(ds_cfg.get("t_factor", 1))

            # --------------------------------------------------------------------
            spectrum = core.DynamicSpectrum.from_numpy_file(self.config["input_data_path"])
            # Gapped-grid regularization (analysis.grid_regularization) must run
            # before downsampling and shares gating with the freya CLI path so a
            # config enabling it cannot be silently bypassed here (issue #120).
            # Function-level import: freya_scintillation does not import pipeline,
            # so this stays acyclic.
            from .freya_scintillation import apply_grid_regularization

            spectrum = apply_grid_regularization(spectrum, self.config)
            spectrum = spectrum.downsample(f_factor, t_factor)
            # The mask_rfi function now correctly uses the manual window if present
            self.masked_spectrum = spectrum.mask_rfi(self.config)

            if self.config.get("pipeline_options", {}).get("save_intermediate_steps"):
                preprocessing = self.config.get("analysis", {}).get("preprocessing", {})
                if preprocessing.get("mode") == "canfar_reference":
                    mask = np.ma.getmaskarray(self.masked_spectrum.power)
                    # RECIPE.md:171-175 and scint_funcs.py:158-167 make the
                    # zero-derived mask an inspectable input to the later ACF.
                    np.savez_compressed(
                        os.path.join(
                            self.cache_dir,
                            f"{self.config.get('burst_id', 'unknown_burst')}_canfar_clean.npz",
                        ),
                        cleaned_spectrum=self.masked_spectrum.power.data,
                        full_mask=mask,
                        channel_mask=np.all(mask, axis=1),
                        time_mask=np.all(mask, axis=0),
                        frequencies_mhz=self.masked_spectrum.frequencies,
                        times=self.masked_spectrum.times,
                    )
                with open(processed_spec_cache, "wb") as f:
                    pickle.dump(self.masked_spectrum, f)

        self.data_prepared = True
        log.info("--- Data Preparation Finished ---")

    def _apply_bandpass_normalization(self, off_pulse_lims):
        """Apply the selected per-channel normalization.

        The modern mean-only flat-field remains flag-gated.  The opt-in
        ``canfar_reference`` preprocessing mode instead uses the reference
        off-mean/off-RMS normalization before additive baseline subtraction.
        """
        preprocessing = self.config.get("analysis", {}).get("preprocessing", {})
        canfar_reference = preprocessing.get("mode") == "canfar_reference"
        bandpass_cfg = self.config.get("analysis", {}).get("bandpass_normalization", {})
        if not canfar_reference and not bandpass_cfg.get("enable", False):
            return
        from .freya_scintillation import (
            _MIN_BANDPASS_OFF_BINS,
            normalize_bandpass,
            normalize_snr_per_channel,
        )

        # The per-channel gain is static in time, so any off-pulse bins are
        # valid gain samples. Short captures (mahi: 55 bins total) can't reach
        # the floor from the pre-burst window alone; augment with the
        # post-burst region, keeping a small guard band past the configured
        # burst window for any residual scattering tail.
        segments = [(int(off_pulse_lims[0]), int(off_pulse_lims[1]))]
        n_off = segments[0][1] - segments[0][0]
        if n_off < _MIN_BANDPASS_OFF_BINS and self.burst_lims is not None:
            n_time = self.masked_spectrum.power.shape[1]
            burst_start, burst_end = self.burst_lims
            pad = max(3, (burst_end - burst_start) // 10)
            post = (min(burst_end + pad, n_time), n_time)
            if post[1] > post[0]:
                segments.append(post)
                log.warning(
                    "Off-pulse window %s has %d bins (< %d); augmenting with "
                    "post-burst segment %s.",
                    off_pulse_lims, n_off, _MIN_BANDPASS_OFF_BINS, post,
                )
        if canfar_reference:
            # kenzie_funcs.py:94-109 and RECIPE.md:147-155 use (I-mu)/sigma.
            log.info("Applying CANFAR-reference per-channel S/N normalization...")
            self.masked_spectrum = normalize_snr_per_channel(self.masked_spectrum, segments)
        else:
            log.info("Applying per-channel bandpass flat-fielding...")
            kwargs = {}
            if "floor_frac" in bandpass_cfg:
                kwargs["floor_frac"] = float(bandpass_cfg["floor_frac"])
            try:
                self.masked_spectrum = normalize_bandpass(
                    self.masked_spectrum,
                    segments,
                    **kwargs,
                )
            except ValueError as exc:
                # Not enough off-pulse data anywhere in the capture: skip the
                # flat-field rather than kill the run, and clear the enable
                # flag so chime_provenance_status demotes the result to
                # diagnostic_only (missing required mitigation) instead of
                # certifying an un-flat-fielded measurement.
                log.warning("Bandpass flat-fielding skipped: %s", exc)
                bandpass_cfg["enable"] = False
                bandpass_cfg["skipped_reason"] = str(exc)

    @staticmethod
    def _revalidate_dnu(spectrum, channel_width_mhz, **kwargs):
        from .revalidation import revalidate_dnu

        try:
            value = revalidate_dnu(spectrum, channel_width_mhz, **kwargs)
        except Exception as exc:
            log.debug("CHIME artifact re-fit failed: %s", exc)
            return None
        return float(value) if np.isfinite(value) and value > 0 else None

    def _off_pulse_dnu_slices(
        self,
        channel_slice,
        channel_width_mhz,
        *,
        first_lag,
        max_lag_mhz,
        max_slices=6,
    ):
        """Re-fit burst-length off-pulse slices on the reference sub-band.

        max_slices sits above off_pulse_null_verdict's min_off_fits=3 so the
        null still runs if a couple of slice re-fits fail (a cap of exactly 3
        made any single failure a trivial null_pass).
        """
        if self.burst_lims is None or self.off_pulse_lims is None:
            return []
        width = max(self.burst_lims[1] - self.burst_lims[0], 4)
        lo = self.off_pulse_lims[0] + 2
        hi = self.off_pulse_lims[1] - width
        c0, c1 = channel_slice
        values = []
        for start in list(range(lo, hi, width + 4))[:max_slices]:
            spectrum = self.masked_spectrum.get_spectrum((start, start + width))[c0:c1]
            value = self._revalidate_dnu(
                spectrum,
                channel_width_mhz,
                first_lag=first_lag,
                max_lag_mhz=max_lag_mhz,
            )
            if value is not None:
                values.append(value)
        return values

    @staticmethod
    def _fit_acf_width(
        lags,
        acf,
        err,
        *,
        fit_range_mhz,
        harmonic_cfg,
        channel_width_mhz,
        excision_bins=0,
    ):
        """Re-fit an existing ACF after fit-window/mask/low-lag selection."""
        from . import chime_artifact_guards as guards
        from .revalidation import compare_lorentzian_components

        lags = np.asarray(lags, float)
        acf = np.asarray(acf, float)
        err = None if err is None else np.asarray(err, float)
        keep = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range_mhz)
        if excision_bins:
            keep &= np.abs(lags) > (int(excision_bins) + 0.5) * channel_width_mhz
        if err is not None:
            keep &= np.isfinite(err) & (err > 0)
        selected_err = None if err is None else err[keep]
        lags, acf, selected_err, _record = guards.apply_harmonic_mask_to_fit(
            lags[keep], acf[keep], selected_err, harmonic_cfg
        )
        if lags.size < 4:
            return None
        try:
            verdict = compare_lorentzian_components(
                lags, acf, max_components=1, acf_err=selected_err
            )
        except Exception as exc:
            log.debug("CHIME artifact ACF re-fit failed: %s", exc)
            return None
        fit = verdict.get("fits", [{}])[0]
        components = fit.get("components", []) if fit.get("success") else []
        return float(components[0]["dnu_mhz"]) if components else None

    def _finalize_chime_status(self):
        """Run the CHIME-only physical artifact gates and persist their evidence."""
        if str(self.config.get("telescope", "")).lower() != "chime":
            return

        from . import chime_artifact_guards as guards

        acf_cfg = self.config.get("analysis", {}).get("acf", {})
        fit_cfg = self.config.get("analysis", {}).get("fitting", {})
        first_lag = int(acf_cfg.get("first_fit_lag", 1))
        fit_range = float(fit_cfg.get("fit_lagrange_mhz", 45.0))
        centers = np.asarray(self.acf_results.get("subband_center_freqs_mhz", []), float)
        slices = self.acf_results.get("subband_channel_slices", [])
        widths = self.acf_results.get("subband_channel_widths_mhz", [])
        if not centers.size or not slices or not widths:
            on_dnu = None
            off_dnu = []
            excision_widths = {}
            scan_records = []
        else:
            ref_freq = float(fit_cfg.get("reference_frequency_mhz", np.nanmedian(centers)))
            index = int(np.nanargmin(np.abs(centers - ref_freq)))
            channel_slice = tuple(slices[index])
            channel_width = float(widths[index])
            lags = self.acf_results["subband_lags_mhz"][index]
            acf = self.acf_results["subband_acfs"][index]
            err_values = self.acf_results.get("subband_acfs_err", [])
            err = err_values[index] if len(err_values) > index else None
            harmonic_cfg = fit_cfg.get("harmonic_mask", {})
            on_dnu = self._fit_acf_width(
                lags,
                acf,
                err,
                fit_range_mhz=fit_range,
                harmonic_cfg=harmonic_cfg,
                channel_width_mhz=channel_width,
            )
            off_dnu = self._off_pulse_dnu_slices(
                channel_slice,
                channel_width,
                first_lag=first_lag,
                max_lag_mhz=fit_range,
            )
            excisions = acf_cfg.get("low_lag_excision_bins", (1, 2, 3, 6))
            excision_widths = {
                int(k): self._fit_acf_width(
                    lags,
                    acf,
                    err,
                    fit_range_mhz=fit_range,
                    harmonic_cfg=harmonic_cfg,
                    channel_width_mhz=channel_width,
                    excision_bins=int(k),
                )
                for k in excisions
            }
            scan_records = []
            for window in fit_cfg.get("fit_lag_scan_mhz", []):
                window = float(window)
                value = (
                    on_dnu
                    if window == fit_range
                    else self._fit_acf_width(
                        lags,
                        acf,
                        err,
                        fit_range_mhz=window,
                        harmonic_cfg=harmonic_cfg,
                        channel_width_mhz=channel_width,
                    )
                )
                scan_records.append(
                    {"fit_lag_mhz": window, "dnu_mhz": value, "success": value is not None}
                )

        provenance = guards.chime_provenance_status(self.config)
        null = guards.off_pulse_null_verdict(on_dnu, off_dnu)
        stability = guards.low_lag_stability_verdict(on_dnu, excision_widths)
        # P4e blind spots: the null/stability gates only probe the reference
        # sub-band, so an unphysical fitted amplitude (whitney m = 3.53) or a
        # zero-dof two-sub-band width (casey_hi) sailed through. Judge the
        # fitted m values and the valid-sub-band count from the components.
        all_mods: list = []
        n_valid_bw = 0
        for comp in (self.final_results.get("components") or {}).values():
            for meas in comp.get("subband_measurements", []) or []:
                bw = meas.get("bw")
                if bw is not None and np.isfinite(bw) and bw > 0:
                    n_valid_bw += 1
                all_mods.append(meas.get("mod"))
        modulation = guards.modulation_index_verdict(all_mods)
        support = guards.subband_support_verdict(n_valid_bw)
        status = guards.finalize_measurement_status(
            provenance,
            off_pulse_null=null,
            low_lag_stability=stability,
            modulation_index=modulation,
            subband_support=support,
            provenance_caveat=self.config.get("provenance_caveat"),
        )
        # Fail closed: finalize_measurement_status only downgrades on an
        # explicit False, but a swallowed re-fit exception or missing ACF
        # metadata leaves on_dnu None and the verdicts inconclusive (None).
        # A gate that could not run must not certify a measurement.
        inconclusive = [
            name
            for name, ran in (
                ("on_pulse_refit", on_dnu is not None),
                ("off_pulse_null", null.get("null_pass") is not None),
                ("low_lag_stability", stability.get("stable") is not None),
                ("modulation_index", modulation.get("physical") is not None),
            )
            if not ran
        ]
        if inconclusive:
            status = {
                **{k: v for k, v in status.items() if k == "provenance_caveat"},
                "status": guards.DIAGNOSTIC_ONLY,
                "downgraded": True,
                "failed_checks": status["failed_checks"]
                + ["inconclusive:" + ",".join(inconclusive)],
            }
        scan_widths = [record["dnu_mhz"] for record in scan_records if record["success"]]
        systematic = max(scan_widths) - min(scan_widths) if len(scan_widths) >= 2 else None
        self.final_results.update(
            {
                "measurement_status": status,
                "chime_provenance": provenance,
                "off_pulse_null": null,
                "low_lag_stability": stability,
                "modulation_index_verdict": modulation,
                "subband_support_verdict": support,
                "systematic_scan": {
                    "fit_windows": scan_records,
                    "fit_window_systematic_mhz": systematic,
                },
                "analysis_windows": {
                    "on_pulse_bins": list(self.burst_lims),
                    "off_pulse_bins": list(self.off_pulse_lims),
                },
                "fit_lag_policy": {
                    "first_fit_lag": first_lag,
                    "harmonic_mask": fit_cfg.get("harmonic_mask", {}),
                    "reported_dnu_definition": "HWHM",
                },
            }
        )
        self.final_results.setdefault("reported_dnu_definition", "HWHM")

    def run(self):
        """
        Executes the full scintillation analysis pipeline from start to finish.
        """
        self.prepare_data()  # Ensures data is loaded

        log.info(f"--- Starting Scintillation Pipeline for {self.config['burst_id']} ---")

        rfi_config = self.config.get("analysis", {}).get("rfi_masking", {})

        # --- CENTRALIZED WINDOW DETERMINATION ---
        manual_on_pulse = rfi_config.get("manual_burst_window")
        if manual_on_pulse and len(manual_on_pulse) == 2:
            burst_lims = manual_on_pulse
            log.warning(f"RUN: Using manually specified on-pulse window: {burst_lims}")
        else:
            log.info("RUN: Using automated burst detection for on-pulse window.")
            burst_lims = self.masked_spectrum.find_burst_envelope(
                thres=rfi_config.get("find_burst_thres", 5.0),
                padding_factor=rfi_config.get("padding_factor", 0.2),
            )

        manual_off_pulse = rfi_config.get("manual_noise_window")
        if manual_off_pulse and len(manual_off_pulse) == 2:
            off_pulse_lims = manual_off_pulse
            log.warning(f"RUN: Using manually specified off-pulse (noise) window: {off_pulse_lims}")
        else:
            noise_end_bin = burst_lims[0] - 200  # Default buffer
            off_pulse_lims = (max(0, noise_end_bin - 500), noise_end_bin)  # Default off-pulse
            log.info(f"RUN: Using automated off-pulse window: {off_pulse_lims}")
        # --- END CENTRALIZED WINDOW DETERMINATION ---
        # Expose the resolved windows for downstream off-pulse diagnostics.
        self.burst_lims = tuple(int(v) for v in burst_lims)
        self.off_pulse_lims = tuple(int(v) for v in off_pulse_lims)

        # --- BANDPASS FLAT-FIELDING (before any additive baseline step) ---
        self._apply_bandpass_normalization(off_pulse_lims)

        # --- BASELINE SUBTRACTION (MOVED HERE) ---
        baseline_info_for_plotting = None
        baseline_config = self.config.get("analysis", {}).get("baseline_subtraction", {})
        if baseline_config.get("enable", False):
            log.info("Applying polynomial baseline subtraction...")
            if off_pulse_lims[1] > off_pulse_lims[0] + 50:  # Check for a valid off-pulse region
                poly_order = baseline_config.get("poly_order", 1)
                # Use the finalized off_pulse_lims to get the spectrum for baseline fitting
                off_pulse_spectrum_1d = self.masked_spectrum.get_spectrum(off_pulse_lims)

                # Create a temporary variable to hold the spectrum before subtraction for the plot
                spec_before_baseline = self.masked_spectrum

                self.masked_spectrum, baseline_model = self.masked_spectrum.subtract_poly_baseline(
                    off_pulse_spectrum_1d, poly_order=poly_order
                )
                if baseline_model is not None:
                    baseline_info_for_plotting = {
                        "original_data": spec_before_baseline.get_spectrum(off_pulse_lims),
                        "model": baseline_model,
                        "poly_order": poly_order,
                    }
            else:
                log.warning("Not enough off-pulse data to model baseline. Skipping subtraction.")

        # --- DIAGNOSTIC PLOTS ---
        # This function is now called AFTER the final windows are determined.
        self._create_diagnostic_plots(
            burst_lims, off_pulse_lims, baseline_info=baseline_info_for_plotting
        )

        # --- NOISE CHARACTERIZATION ---
        if self.config.get("analysis", {}).get("noise", {}).get("disable", False):
            log.info("Noise modelling disabled by config.")
            self.noise_descriptor = None
        elif off_pulse_lims[1] > off_pulse_lims[0] + 100:
            log.info("Characterizing off-pulse noise...")
            off_pulse_data = self.masked_spectrum.power.data[
                :, off_pulse_lims[0] : off_pulse_lims[1]
            ].T
            self.noise_descriptor = noise.estimate_noise_descriptor(off_pulse_data)
            log.info(
                f"Noise characterization complete. Detected kind: '{self.noise_descriptor.kind}'"
            )
        else:
            log.warning("Not enough pre-burst data for robust noise characterization. Skipping.")
            self.noise_descriptor = None

        # --- ACF CALCULATION ---
        acf_results_cache = self._get_cache_path("acf_results")
        if os.path.exists(acf_results_cache) and not self.config.get("pipeline_options", {}).get(
            "force_recalc", False
        ):
            log.info(f"Loading cached ACF results from {acf_results_cache}")
            with open(acf_results_cache, "rb") as f:
                self.acf_results = pickle.load(f)
        else:
            log.info("Calculating ACFs for all sub-bands...")
            self.acf_results = analysis.calculate_acfs_for_subbands(
                self.masked_spectrum,
                self.config,
                burst_lims=burst_lims,
                noise_desc=self.noise_descriptor,
            )
            if self.config.get("pipeline_options", {}).get("save_intermediate_steps"):
                with open(acf_results_cache, "wb") as f:
                    pickle.dump(self.acf_results, f)
                log.info(f"Saved ACF results to cache: {acf_results_cache}")

        # --- HALT CHECK ---
        if self.config.get("pipeline_options", {}).get("halt_after_acf", False):
            log.info("'halt_after_acf' is set to True. Halting pipeline as requested.")
            return

        # --- Run the intra-pulse analysis ---
        acf_config = self.config.get("analysis", {}).get("acf", {})
        if acf_config.get("enable_intra_pulse_analysis", False):
            log.info("Running intra-pulse analysis...")
            self.intra_pulse_results = analysis.analyze_intra_pulse_scintillation(
                self.masked_spectrum, burst_lims, self.config, self.noise_descriptor
            )
            self.modulation_over_time = analysis.modulation_index_over_time(
                self.masked_spectrum.power,
                burst_lims,
                chunk_bins=acf_config.get("time_chunk_size_bins", 3),
                overlap_bins=acf_config.get("time_overlap_bins", 2),
                times=self.masked_spectrum.times,
            )

        # --- Stage 4: Fit Models and Derive Parameters ---
        if not self.acf_results or not self.acf_results["subband_acfs"]:
            log.error("ACF results are empty, cannot proceed to fitting. Exiting.")
            return

        log.info("Fitting models and deriving final scintillation parameters...")
        self.final_results, self.all_subband_fits, self.all_powerlaw_fits = (
            analysis.analyze_scintillation_from_acfs(self.acf_results, self.config)
        )
        analysis.attach_modulation_index_frequency(self.final_results)
        if acf_config.get("enable_intra_pulse_analysis", False):
            acf_measurements = [
                {
                    "time_s": result.get("time_s"),
                    "m": result.get("mod"),
                    "m_err": result.get("mod_err"),
                    **analysis._bandwidth_fields(result.get("bw"), result.get("bw_err")),
                }
                for result in (self.intra_pulse_results or [])
            ]
            self.final_results["modulation_index_time"] = {
                "acf_fitted": {
                    "definition": analysis.INTRA_PULSE_ACF_MODULATION_DEFINITION,
                    "measurements": acf_measurements,
                },
                "direct_std_mean": self.modulation_over_time,
            }

        self._finalize_chime_status()

        # Attach two-screen / emission-size / consistency interpretation per component
        # (bridge fills config['source'] from tau_consistency + optional multi-scale Δν).
        from galaxies.foreground.scintillation_bridge import attach_interpretation_with_bridge

        nick = self.config.get("burst_id") or (self.config.get("source") or {}).get("nickname")
        self.config = attach_interpretation_with_bridge(
            self.final_results,
            self.config,
            nickname=nick,
            acf_results=self.acf_results,
            masked_spectrum=self.masked_spectrum,
            burst_lims=burst_lims,
        )

        # Attach the NE2025 MW scattering floor + extragalactic-excess flag when the
        # burst sky position is in config['source'] (no-op without it or the optional
        # mwprop/pygedm dep).
        src = self.config.get("source", {})
        if src.get("ra_deg") is not None and src.get("dec_deg") is not None:
            from .floor_wiring import attach_galactic_floor_all

            attach_galactic_floor_all(self.final_results, src["ra_deg"], src["dec_deg"])

        # --- 2D GLOBAL SCINTILLATION FIT ---
        self.fit_2d_result = None
        fit_2d_config = self.config.get("analysis", {}).get("fit_2d", {})
        if fit_2d_config.get("enable", True):  # Enabled by default
            log.info("Running 2D global scintillation fit across all sub-bands...")
            self.fit_2d_result = self._run_2d_scintillation_fit(fit_2d_config)

        log.info("--- Pipeline execution finished. ---")

    def _run_2d_scintillation_fit(self, fit_2d_config):
        """
        Run 2D global scintillation fit across all sub-bands.

        This enforces physical frequency scaling: γ(ν) = γ₀ × (ν/ν_ref)^α
        and provides direct measurement of the scaling index α.
        """
        try:
            from .fitting_2d import fit_2d_scintillation
        except ImportError as e:
            log.warning(f"Could not import fitting_2d module: {e}. Skipping 2D fit.")
            return None

        if self.acf_results is None or not self.acf_results.get("subband_acfs"):
            log.warning("No ACF results available for 2D fitting.")
            return None

        try:
            result = fit_2d_scintillation(
                self.acf_results,
                model_type=fit_2d_config.get("model_type", "lorentzian"),
                fit_range_mhz=fit_2d_config.get("fit_range_mhz", 25.0),
                nu_ref=fit_2d_config.get("nu_ref", None),
                gamma_0_init=fit_2d_config.get("gamma_0_init", 1.0),
                alpha_init=fit_2d_config.get("alpha_init", 4.0),
                m_0_init=fit_2d_config.get("m_0_init", 0.5),
                vary_alpha=fit_2d_config.get("vary_alpha", True),
                include_self_noise=fit_2d_config.get("include_self_noise", False),
            )

            log.info(
                f"2D fit complete: γ₀ = {result.gamma_0:.3f} ± {result.gamma_0_err:.3f} MHz, "
                f"α = {result.alpha:.2f} ± {result.alpha_err:.2f}, "
                f"χ²_red = {result.redchi:.2f}"
            )

            # Store in final_results for convenience
            if self.final_results is not None:
                self.final_results["fit_2d"] = {
                    "gamma_0": result.gamma_0,
                    "gamma_0_err": result.gamma_0_err,
                    "alpha": result.alpha,
                    "alpha_err": result.alpha_err,
                    "m_0": result.m_0,
                    "m_0_err": result.m_0_err,
                    "nu_ref": result.nu_ref,
                    "redchi": result.redchi,
                    "success": result.success,
                    **analysis._bandwidth_fields(result.gamma_0, result.gamma_0_err),
                }
                ref_freq = (
                    self.config.get("analysis", {})
                    .get("fitting", {})
                    .get("reference_frequency_mhz", 600.0)
                )
                joint_estimator = analysis.joint_2d_gamma_scaling(result, ref_freq)
                for component in self.final_results.get("components", {}).values():
                    scaling = component.get("gamma_scaling")
                    if scaling is not None:
                        scaling["joint_2d"] = joint_estimator.copy()

            return result

        except Exception as e:
            log.error(f"2D scintillation fit failed: {e}")
            return None
