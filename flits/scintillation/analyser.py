"""High-level scintillation analyser for FRB dynamic spectra.

The :class:`ScintillationAnalyser` ties together the preprocessing, ACF,
fitting, secondary-spectrum and physics modules into a single stateful
workflow, migrated from the legacy `scint_pipeline` script. The missing
``Tuple`` and ``matplotlib`` imports of the legacy module are fixed here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .acf import calculate_acf_2d
from .fitting import fit_lorentzian_acf, fit_scint_bandwidth_freq_relation
from .physics import (
    scintillation_bandwidth_to_timescale,
    screen_distance_from_curvature,
)
from .preprocessing import scrunch, upchannelize
from .secondary import calculate_secondary_spectrum

__all__ = ["ScintillationAnalyser"]


class ScintillationAnalyser:
    """Perform scintillation analysis on an FRB dynamic spectrum.

    Parameters
    ----------
    dyn_spec : ndarray
        Dynamic spectrum with shape ``(ntime, nfreq)``.
    freqs_mhz : ndarray
        Channel centre frequencies in MHz.
    time_res_s : float
        Time resolution in seconds.
    source_name : str, optional
        Identifier for the source/observation. Default "FRB".

    Attributes
    ----------
    freq_res_mhz : float
        Channel width in MHz, inferred from ``freqs_mhz``.
    params : dict
        Parameters used by each analysis stage.
    results : dict
        Analysis products (processed spectrum, ACF, fits, secondary spectrum).

    Raises
    ------
    ValueError
        If ``dyn_spec`` is not 2D or its channel count does not match
        ``freqs_mhz``.
    """

    def __init__(
        self,
        dyn_spec: NDArray[np.floating],
        freqs_mhz: NDArray[np.floating],
        time_res_s: float,
        source_name: str = "FRB",
    ) -> None:
        if dyn_spec.ndim != 2:
            raise ValueError("dyn_spec must be 2D (time, freq).")
        if dyn_spec.shape[1] != len(freqs_mhz):
            raise ValueError(
                "Number of frequency channels in dyn_spec must match "
                "length of freqs_mhz."
            )

        self.dyn_spec = dyn_spec
        self.freqs_mhz = freqs_mhz
        self.time_res_s = time_res_s
        if len(freqs_mhz) > 1:
            self.freq_res_mhz = float(np.abs(np.median(np.diff(freqs_mhz))))
        else:
            self.freq_res_mhz = 0.0
            print("Warning: Cannot determine frequency resolution from single channel.")
        self.source_name = source_name

        self.params: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

        print(f"Initialized Analyser for {self.source_name}")
        print(f"Data shape (time, freq): {self.dyn_spec.shape}")
        print(f"Time resolution: {self.time_res_s:.6f} s")
        print(
            f"Frequency range: {self.freqs_mhz.min():.2f} - "
            f"{self.freqs_mhz.max():.2f} MHz"
        )
        print(f"Frequency resolution: {self.freq_res_mhz:.6f} MHz")

    def preprocess(
        self,
        t_scrunch: int = 1,
        f_scrunch: int = 1,
        time_range: Optional[Tuple[int, int]] = None,
        freq_range_mhz: Optional[Tuple[float, float]] = None,
        apply_upchannel: bool = False,
        upchannel_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Preprocess the dynamic spectrum.

        Applies, in order: frequency selection, time selection, scrunching, and
        (optionally) upchannelization, updating ``dyn_spec``, ``freqs_mhz``,
        ``time_res_s`` and ``freq_res_mhz`` and storing the processed spectrum
        in ``results['processed_dyn_spec']``.

        Parameters
        ----------
        t_scrunch, f_scrunch : int, optional
            Block-averaging factors in time and frequency. Default 1.
        time_range : tuple of int, optional
            ``(start, end)`` sample indices to keep.
        freq_range_mhz : tuple of float, optional
            ``(f_min, f_max)`` frequency window in MHz.
        apply_upchannel : bool, optional
            Apply FFT upchannelization. Default False.
        upchannel_params : dict, optional
            Keyword arguments for :func:`~flits.scintillation.preprocessing.upchannelize`.
            Default ``{'fftsize': 32, 'downfreq': 2, 'downtime': 1}``.
        """
        if upchannel_params is None:
            upchannel_params = {"fftsize": 32, "downfreq": 2, "downtime": 1}
        self.params["preprocess"] = {
            "t_scrunch": t_scrunch,
            "f_scrunch": f_scrunch,
            "time_range": time_range,
            "freq_range_mhz": freq_range_mhz,
            "apply_upchannel": apply_upchannel,
            "upchannel_params": upchannel_params,
        }

        current_dyn_spec = self.dyn_spec
        current_freqs = self.freqs_mhz
        current_time_res = self.time_res_s
        current_freq_res = self.freq_res_mhz

        if freq_range_mhz is not None:
            f_min, f_max = freq_range_mhz
            freq_mask = (current_freqs >= f_min) & (current_freqs <= f_max)
            if np.sum(freq_mask) == 0:
                raise ValueError(
                    f"No channels found in frequency range {freq_range_mhz} MHz."
                )
            current_dyn_spec = current_dyn_spec[:, freq_mask]
            current_freqs = current_freqs[freq_mask]
            print(
                f"Selected frequency range: {current_freqs.min():.2f} - "
                f"{current_freqs.max():.2f} MHz"
            )

        if time_range is not None:
            t_start, t_end = time_range
            if not 0 <= t_start < t_end <= current_dyn_spec.shape[0]:
                raise ValueError(
                    f"Invalid time range indices: {time_range} for data with "
                    f"{current_dyn_spec.shape[0]} time samples."
                )
            current_dyn_spec = current_dyn_spec[t_start:t_end, :]
            print(f"Selected time range (indices): {t_start} to {t_end}")

        if t_scrunch > 1 or f_scrunch > 1:
            print(f"Scrunching by T={t_scrunch}, F={f_scrunch}")
            current_dyn_spec = scrunch(current_dyn_spec, t_scrunch, f_scrunch)
            current_time_res *= t_scrunch
            current_freq_res *= f_scrunch
            if f_scrunch > 1:
                current_freqs = current_freqs.reshape(-1, f_scrunch).mean(axis=1)
            print(
                f"New shape: {current_dyn_spec.shape}, "
                f"New T_res: {current_time_res:.6f} s, "
                f"New F_res: {current_freq_res:.6f} MHz"
            )

        if apply_upchannel:
            print(f"Applying upchannelization with params: {upchannel_params}")
            spec_freq_time = current_dyn_spec.T  # (freq, time) for upchannelize
            try:
                upchann_spec, _ = upchannelize(spec_freq_time, **upchannel_params)
                current_dyn_spec = upchann_spec.T  # back to (time, freq)
                fftsize = upchannel_params.get("fftsize", 32)
                downtime = upchannel_params.get("downtime", 1)
                current_time_res = self.time_res_s * fftsize * downtime
                original_bw = len(self.freqs_mhz) * self.freq_res_mhz
                new_n_freq = current_dyn_spec.shape[1]
                current_freq_res = original_bw / new_n_freq if new_n_freq > 0 else 0.0
                current_freqs = np.linspace(
                    self.freqs_mhz.min(), self.freqs_mhz.max(), new_n_freq
                )
                print(
                    f"Upchannelized shape: {current_dyn_spec.shape}, "
                    f"New T_res: {current_time_res:.6f} s, "
                    f"New F_res: {current_freq_res:.6f} MHz"
                )
            except ValueError as exc:
                print(f"Upchannelization failed: {exc}. Skipping.")

        self.dyn_spec = current_dyn_spec
        self.freqs_mhz = current_freqs
        self.time_res_s = current_time_res
        self.freq_res_mhz = current_freq_res
        self.results["processed_dyn_spec"] = self.dyn_spec

        print("Preprocessing complete.")
        print(f"Final shape (time, freq): {self.dyn_spec.shape}")
        print(f"Final Time resolution: {self.time_res_s:.6f} s")
        print(
            f"Final Frequency range: {self.freqs_mhz.min():.2f} - "
            f"{self.freqs_mhz.max():.2f} MHz"
        )
        print(f"Final Frequency resolution: {self.freq_res_mhz:.6f} MHz")

    def _data_for_analysis(self) -> NDArray[np.floating]:
        """Return the processed spectrum if available, else the raw one."""
        if "processed_dyn_spec" in self.results:
            return self.results["processed_dyn_spec"]
        print("Run preprocess() first or use original data.")
        return self.dyn_spec

    def calculate_acf(self, axis: int = 1, norm: bool = True) -> None:
        """Compute the averaged ACF along an axis.

        Parameters
        ----------
        axis : int, optional
            0 for time, 1 for frequency. Default 1.
        norm : bool, optional
            Normalize per-slice ACFs before averaging. Default True.
        """
        data_to_use = self._data_for_analysis()
        axis_name = "frequency" if axis == 1 else "time"
        print(f"Calculating ACF along axis {axis} ({axis_name})...")
        lags, avg_acf = calculate_acf_2d(data_to_use, axis=axis, norm=norm)
        self.results["acf"] = {"axis": axis, "lags": lags, "acf": avg_acf}
        print("ACF calculation complete.")

    def fit_acf_lorentzian(self, const_offset: bool = True) -> None:
        """Fit a Lorentzian to the frequency-axis ACF.

        Parameters
        ----------
        const_offset : bool, optional
            Include a constant offset in the fit. Default True.

        Raises
        ------
        RuntimeError
            If a frequency-axis ACF has not been computed yet.
        """
        if "acf" not in self.results or self.results["acf"]["axis"] != 1:
            raise RuntimeError("Calculate ACF along frequency axis (axis=1) first.")

        print("Fitting Lorentzian to ACF...")
        acf_data = self.results["acf"]
        lags = acf_data["lags"]
        acf = acf_data["acf"]
        center_guess = lags[len(lags) // 2]

        params, model, fit_result = fit_lorentzian_acf(
            lags, acf, errs=None, center_guess=center_guess, const_offset=const_offset
        )

        if params is not None:
            print("Lorentzian fit successful.")
            hwhm = params["wid"].value
            hwhm_err = (
                params["wid"].stderr if params["wid"].stderr is not None else np.nan
            )
            self.results["acf_fit"] = {
                "params": params,
                "model": model,
                "fit_result": fit_result,
                "scint_bandwidth_hwhm": hwhm,
                "scint_bandwidth_hwhm_err": hwhm_err,
                "fit_report": fit_result.fit_report(),
            }
            print(f"  Scintillation Bandwidth (HWHM): {hwhm:.4f} +/- {hwhm_err:.4f} MHz")
        else:
            print("Lorentzian fit failed.")
            self.results["acf_fit"] = None

    def run_subband_analysis(
        self, n_subbands: int, fit_model: str = "lorentzian"
    ) -> None:
        """Fit the scintillation bandwidth in sub-bands and its frequency scaling.

        Splits the band into ``n_subbands`` sub-bands, fits the ACF in each, and
        fits \u0394\u03bd_d \u221d \u03bd^\u03b1 across sub-bands.

        Parameters
        ----------
        n_subbands : int
            Number of sub-bands.
        fit_model : str, optional
            Per-sub-band ACF model. Only "lorentzian" is supported. Default
            "lorentzian".

        Raises
        ------
        ValueError
            If there are fewer channels than requested sub-bands.
        """
        data_to_use = self._data_for_analysis()
        freqs_to_use = self.freqs_mhz

        _, nf = data_to_use.shape
        if nf < n_subbands:
            raise ValueError(
                f"Number of channels ({nf}) is less than requested subbands "
                f"({n_subbands})."
            )

        print(f"Performing analysis across {n_subbands} subbands...")
        sub_indices = np.array_split(np.arange(nf), n_subbands)
        subband_results: Dict[str, Any] = {
            "center_freq_mhz": [],
            "scint_bw_hwhm": [],
            "scint_bw_hwhm_err": [],
            "fit_params": [],
            "fit_success": [],
        }

        for i, indices in enumerate(sub_indices):
            if len(indices) == 0:
                continue
            sub_spec = data_to_use[:, indices]
            sub_freqs = freqs_to_use[indices]
            center_freq = float(np.mean(sub_freqs))
            print(
                f"  Subband {i + 1}/{n_subbands} "
                f"(Freq ~ {center_freq:.2f} MHz, {len(indices)} chans)"
            )

            sub_freq_res = (
                float(np.abs(np.median(np.diff(sub_freqs))))
                if len(indices) > 1
                else self.freq_res_mhz
            )
            lags, acf = calculate_acf_2d(sub_spec, axis=1, norm=True)
            lags_mhz = lags * sub_freq_res

            if fit_model == "lorentzian":
                params, _, fit_result = fit_lorentzian_acf(
                    lags_mhz, acf, const_offset=True
                )
            else:
                print(f"Warning: Unsupported fit model '{fit_model}'. Skipping fit.")
                params, fit_result = None, None

            subband_results["center_freq_mhz"].append(center_freq)
            if params is not None and fit_result is not None and fit_result.success:
                hwhm = params["wid"].value
                hwhm_err = (
                    params["wid"].stderr
                    if params["wid"].stderr is not None
                    else np.nan
                )
                subband_results["scint_bw_hwhm"].append(hwhm)
                subband_results["scint_bw_hwhm_err"].append(hwhm_err)
                subband_results["fit_params"].append(params)
                subband_results["fit_success"].append(True)
                print(f"    Fit OK. HWHM: {hwhm:.4f} +/- {hwhm_err:.4f} MHz")
            else:
                subband_results["scint_bw_hwhm"].append(np.nan)
                subband_results["scint_bw_hwhm_err"].append(np.nan)
                subband_results["fit_params"].append(None)
                subband_results["fit_success"].append(False)
                print("    Fit Failed.")

        for key in subband_results:
            if key != "fit_params":
                subband_results[key] = np.array(subband_results[key])

        self.results["subband_analysis"] = subband_results

        print("\nFitting scintillation bandwidth vs. frequency (\u0394\u03bd_d \u221d \u03bd^\u03b1)...")
        valid_mask = subband_results["fit_success"] & np.isfinite(
            subband_results["scint_bw_hwhm"]
        )
        if np.sum(valid_mask) >= 2:
            freqs_fit = subband_results["center_freq_mhz"][valid_mask]
            bw_fit = subband_results["scint_bw_hwhm"][valid_mask]
            bw_err_fit = subband_results["scint_bw_hwhm_err"][valid_mask]
            use_errors = np.all(np.isfinite(bw_err_fit)) and np.all(bw_err_fit > 0)

            params_pl, model_pl, fit_result_pl = fit_scint_bandwidth_freq_relation(
                freqs_fit, bw_fit, errs=bw_err_fit if use_errors else None
            )

            if params_pl is not None:
                print("Power law fit successful.")
                alpha = params_pl["index"].value
                alpha_err = (
                    params_pl["index"].stderr
                    if params_pl["index"].stderr is not None
                    else np.nan
                )
                amp = params_pl["amp"].value
                amp_err = (
                    params_pl["amp"].stderr
                    if params_pl["amp"].stderr is not None
                    else np.nan
                )
                subband_results["power_law_fit"] = {
                    "params": params_pl,
                    "model": model_pl,
                    "fit_result": fit_result_pl,
                    "alpha": alpha,
                    "alpha_err": alpha_err,
                    "amplitude": amp,
                    "amplitude_err": amp_err,
                    "fit_report": fit_result_pl.fit_report(),
                }
                print(f"  alpha = {alpha:.2f} +/- {alpha_err:.2f}")
                print(f"  Fit: d_nu_d = ({amp:.2e}) * nu_MHz^({alpha:.2f})")
            else:
                print("Power law fit failed.")
                subband_results["power_law_fit"] = None
        else:
            print("Not enough valid points to fit power law.")
            subband_results["power_law_fit"] = None

    def calculate_secondary(self) -> None:
        """Compute the secondary spectrum and store it in ``results``."""
        data_to_use = self._data_for_analysis()
        print("Calculating secondary spectrum...")
        freq_res_hz = self.freq_res_mhz * 1e6
        sec_spec, fd_axis, tau_axis = calculate_secondary_spectrum(
            data_to_use, self.time_res_s, freq_res_hz
        )
        self.results["secondary_spectrum"] = {
            "spec": sec_spec,
            "fd_hz": fd_axis,
            "tau_us": tau_axis * 1e6,
        }
        print("Secondary spectrum calculation complete.")

    def fit_secondary_arc(self, *args: Any, **kwargs: Any) -> None:
        """Fit parabolic arc(s) to the secondary spectrum (not yet implemented).

        Raises
        ------
        RuntimeError
            If the secondary spectrum has not been computed.
        """
        if "secondary_spectrum" not in self.results:
            raise RuntimeError("Calculate secondary spectrum first.")
        print("Placeholder: Secondary spectrum arc fitting not implemented yet.")
        self.results["secondary_fit"] = None

    def derive_timescale_from_bandwidth(
        self,
        freq_mhz: Optional[float] = None,
        coefficient: float = 1.0,
    ) -> None:
        """Estimate \u03c4_d from the ACF bandwidth at a reference frequency.

        Parameters
        ----------
        freq_mhz : float, optional
            Reference frequency in MHz. Defaults to the current mean frequency.
        coefficient : float, optional
            Uncertainty relation scaling constant C. Default 1.0. Typically C ≈ 1.16 for a
            thin Kolmogorov screen, or C ≈ 0.72 for a thick Kolmogorov screen.
        """
        if "acf_fit" not in self.results or self.results["acf_fit"] is None:
            print("Warning: Cannot estimate timescale. Run fit_acf_lorentzian first.")
            return
        if freq_mhz is None:
            freq_mhz = float(np.mean(self.freqs_mhz))

        delta_nu_d_hz = self.results["acf_fit"]["scint_bandwidth_hwhm"] * 1e6
        alpha = 4.0
        subband = self.results.get("subband_analysis")
        if subband and subband.get("power_law_fit"):
            alpha = subband["power_law_fit"]["alpha"]

        tau_d_s = scintillation_bandwidth_to_timescale(
            delta_nu_d_hz, freq_mhz, alpha, coefficient=coefficient
        )
        tau_d_ms = tau_d_s * 1000.0
        self.results["derived_timescale_ms"] = tau_d_ms
        print(
            f"Derived scintillation timescale tau_d ~ {tau_d_ms:.4f} ms "
            f"(at {freq_mhz:.1f} MHz, assuming alpha={alpha:.2f}, coefficient={coefficient:.2f})"
        )

    def derive_screen_distance(
        self,
        source_dist_mpc: Optional[float] = None,
        v_eff_kms: Optional[float] = None,
        select_host_root: bool = False,
    ) -> None:
        """Estimate the screen distance from the secondary-arc curvature.

        Parameters
        ----------
        source_dist_mpc : float, optional
            Source distance in Mpc. If given, derive the lens distance D_L;
            otherwise the effective distance D_eff.
        v_eff_kms : float, optional
            Effective transverse velocity in km/s. Default 100 km/s.
        select_host_root : bool, optional
            If True, return the larger root for D_L (host-galaxy screen) rather than
            the smaller root (Milky Way screen). Default False.
        """
        secondary_fit = self.results.get("secondary_fit")
        if not secondary_fit:
            print("Warning: Cannot estimate screen distance. Run fit_secondary_arc first.")
            return
        if "curvature" not in secondary_fit:
            print("Warning: Arc curvature measurement not found in secondary_fit results.")
            return

        curvature = secondary_fit["curvature"]
        freq_ghz = float(np.mean(self.freqs_mhz)) / 1000.0
        dist_pc = screen_distance_from_curvature(
            curvature,
            freq_ghz,
            source_dist_mpc,
            v_eff_kms=v_eff_kms,
            select_host_root=select_host_root,
        )

        v_eff = 100.0 if v_eff_kms is None else v_eff_kms
        if source_dist_mpc is not None:
            self.results["derived_lens_distance_pc"] = dist_pc
            print(
                f"Derived Lens distance D_L ~ {dist_pc:.2f} pc "
                f"(assuming D_S = {source_dist_mpc} Mpc, V_eff = {v_eff:.1f} km/s, select_host_root={select_host_root})"
            )
        else:
            self.results["derived_effective_distance_pc"] = dist_pc
            print(
                f"Derived Effective distance D_eff ~ {dist_pc:.2f} pc "
                f"(assuming V_eff = {v_eff:.1f} km/s)"
            )

    def plot_dynamic_spectrum(self, processed: bool = True, **kwargs: Any) -> None:
        """Plot the (processed or original) dynamic spectrum."""
        if processed and "processed_dyn_spec" in self.results:
            spec = self.results["processed_dyn_spec"]
            title = f"{self.source_name} Processed Dynamic Spectrum"
        else:
            spec = self.dyn_spec
            title = f"{self.source_name} Original Dynamic Spectrum"

        t_vec = np.arange(spec.shape[0]) * self.time_res_s
        f_vec = self.freqs_mhz
        plt.figure(figsize=kwargs.pop("figsize", (10, 5)))
        plt.imshow(
            spec.T,
            aspect="auto",
            origin="lower",
            extent=[t_vec[0], t_vec[-1], f_vec[0], f_vec[-1]],
            **kwargs,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.colorbar(label="Intensity (Arb. Units)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_acf(self, **kwargs: Any) -> None:
        """Plot the averaged ACF and overlay the Lorentzian fit if present."""
        if "acf" not in self.results:
            print("No ACF calculated yet.")
            return

        acf_data = self.results["acf"]
        lags = acf_data["lags"]
        acf = acf_data["acf"]
        axis = acf_data["axis"]
        axis_name = "Frequency Lag (MHz)" if axis == 1 else "Time Lag (samples)"

        plt.figure(figsize=kwargs.pop("figsize", (8, 5)))
        plt.plot(lags, acf, **kwargs)
        plt.xlabel(axis_name)
        plt.ylabel("Autocorrelation")
        plt.title(
            f"{self.source_name} Averaged ACF "
            f"({'Freq Axis' if axis == 1 else 'Time Axis'})"
        )
        plt.grid(True)

        acf_fit = self.results.get("acf_fit")
        if acf_fit is not None and axis == 1:
            fit_result = acf_fit["fit_result"]
            hwhm = fit_result.params["wid"].value
            plt.plot(
                lags,
                fit_result.best_fit,
                "r--",
                label=f"Lorentzian Fit (HWHM={hwhm:.3f} MHz)",
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_subband_analysis(self, **kwargs: Any) -> None:
        """Plot scintillation bandwidth vs frequency with the power-law fit."""
        if "subband_analysis" not in self.results:
            print("No subband analysis performed yet.")
            return

        res = self.results["subband_analysis"]
        freqs = res["center_freq_mhz"]
        bw = res["scint_bw_hwhm"]
        bw_err = res["scint_bw_hwhm_err"]
        success = res["fit_success"]

        plt.figure(figsize=kwargs.pop("figsize", (8, 5)))
        plt.errorbar(
            freqs[success],
            bw[success],
            yerr=bw_err[success],
            fmt="o",
            label="Successful Fits",
            capsize=3,
        )
        if np.any(~success):
            plt.plot(freqs[~success], bw[~success], "x", color="red", label="Failed Fits")

        if res.get("power_law_fit") is not None:
            pl_fit = res["power_law_fit"]
            alpha = pl_fit["alpha"]
            amp = pl_fit["amplitude"]
            fit_freqs = np.linspace(freqs.min(), freqs.max(), 100)
            plt.plot(
                fit_freqs,
                amp * (fit_freqs ** alpha),
                "r--",
                label=f"Fit: alpha = {alpha:.2f} +/- {pl_fit['alpha_err']:.2f}",
            )

        plt.xlabel("Center Frequency (MHz)")
        plt.ylabel("Scintillation Bandwidth HWHM (MHz)")
        plt.title(f"{self.source_name} Scintillation Bandwidth vs. Frequency")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_secondary_spectrum(self, **kwargs: Any) -> None:
        """Plot the secondary spectrum on a log intensity scale."""
        if "secondary_spectrum" not in self.results:
            print("No secondary spectrum calculated yet.")
            return

        sec = self.results["secondary_spectrum"]
        spec = sec["spec"]
        fd = sec["fd_hz"]
        tau = sec["tau_us"]

        vmin = kwargs.pop("vmin", np.percentile(spec, 5))
        vmax = kwargs.pop("vmax", np.percentile(spec, 99.5))
        norm = kwargs.pop("norm", matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

        plt.figure(figsize=kwargs.pop("figsize", (8, 6)))
        plt.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[fd[0], fd[-1], tau[0], tau[-1]],
            norm=norm,
            cmap=kwargs.pop("cmap", "viridis"),
            **kwargs,
        )
        plt.xlabel("Doppler Frequency f_D (Hz)")
        plt.ylabel("Delay tau (us)")
        plt.colorbar(label="Secondary Spectrum Power (Arb. Units)")
        plt.title(f"{self.source_name} Secondary Spectrum")
        plt.tight_layout()
        plt.show()
