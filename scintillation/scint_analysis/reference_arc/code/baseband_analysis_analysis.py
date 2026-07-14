# In baseband_analysis/analysis/snr.py

"""Summary missing

******* PREVIOUS DOCSTRING *******
<FRESHLY_INSERTED>
^^^^^^^ PREVIOUS DOCSTRING ^^^^^^^
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.signal import (
    get_onsource_beam,
    get_profile,
    normalize_and_refine,
)
from baseband_analysis.utilities.plotting import plot_waterfall

log = logging.getLogger(__name__)


def get_snr_matrix(snr, prof_lim=None):
    """
    Simple function to retrieve optimum SNR from `get_profile`.

    Parameters
    ----------
    snr : `np.ndarray`
        Description

    Returns
    -------
    snr_t : TYPE
        Description
    """
    prof = get_profile(snr)
    if prof_lim is None:
        argmax = prof.argmax()
    else:
        argmax = prof[prof_lim[0] : prof_lim[1]].argmax()
    # It would be better to downsample to W_50
    snr_t = snr[..., argmax]
    return snr_t


def beam_snr(snr, prof_lim=None):
    """
    Simple function to retrieve optimum beam SNR from `get_profile`.

    Parameters
    ----------
    snr : TYPE
        Description

    Returns
    -------
    snr_f : TYPE
        Description
    """
    if len(snr.shape) == 3:
        snr = get_snr_matrix(snr, prof_lim=prof_lim)
    snr_f = np.nansum(snr, axis=0) / np.sqrt(
        np.count_nonzero(~np.isnan(np.sum(snr, axis=1)))
    )
    return snr_f


def get_snr(
    data,
    DM=None,
    diagnostic_plots=False,
    w=None,
    valid_channels=None,
    time_range=None,
    return_full=False,
    spectrum_lim=True,
    spectrum_thresh=2,
    DM_range=10,
    downsample=False,
    refine_RFI=False,
    fill_missing_time=None,
    thres_mean=5,
    thres_std=3,
    doublecheck_RFI=True,
    DM_step=0.01,
    raise_missing_signal=True,
    check_old_process=True,
    check_channel_number=True,
    fill_nan=True,
):
    """Normalize power, refine DM, remove RFI, and produce a waterfall plot of baseband data.
    Parameters
    ----------
    data : BBD data object
        BBD data object using BBData.from_file(path_to_h5_data_file)
    DM : float, optional
        Value of DM to which de-disperse the waterfall.
        If not specified, use DM from the BBData object attribute (data["tiedbeam_power"].attrs["DM_coherent"]).
    diagnostic_plots: bool, optional
        Whether to show diagnostic plots or not. There is at least 1 and at most 5 diagnostic plots.
        - Waterfall plot of data.
          Should contain the burst in the middle. [Last plot]
        - Mask where noise was added (yellow) and untouched signal (purple).
          Should have a yellow triangular region.
        - Followed by mask that shows where extremely bright pixels (in yellow) were removed.
          Should be all purple or have a few yellow pixels.
        - DM vs time. Should have a sloped line with a bright pixel half way through (at the S/N maximising DM).
          If you can't see the line, try decreasing the DM_range.
        - If fill_missing_time is True, there will be a second copy of the last 3 plots described.
          This is because once the noise is filled in, we have to redo the processing to make sure the burst is included.
    w : array of floats, optional
        Weight and offset to normalise the data by. If not specified, the function calls get_weight on the power.
    valid_channels : array of bools, optional
        You can specify RFI channels you wish to discard by setting them to False.
        Note that the first element of the array corresponds to 800 MHz, and the last element of the array to 400 MHz.
        The array size should be the same as the first dimension of power i.e. power.shape[0].
        If not specified, we call get_RFI_channels on the power.
    time_range : tuple or list, optional
        Start and end time bins indicating where the data is valid
        i.e. the time range that does not contain any NANs or artefacts such as missing data.
        If not specified, we call get_valid_time_range **will be removed soon.
    return_full : bool, optional
        If set to False, only returns freq_id, freq, power.
        If set to True, returns freq_id, freq, power, offset, weight, valid_channels, time_range, DM, downsampling_factor.
    spectrum_lim : bool, optional
        If True, cuts out the frequencies which do not contain any burst data using get_spectrum_lim
    DM_range : float, optional
        If you want to refine the DM, provide a positive float to define the range for DM refinement.
        DM refinement will be over DM +/- DM_range / 2  .
    DM_step : float, optional
        Size of DM step to consider when refining DM i.e. the DM trials will be np.arange(DM - DM_range / 2, DM + DM_range / 2, DM_step).
        If not specified, default is 0.1 pc/cm^3.
    downsample : bool or int, optional
        If True, downsamples baseband data to intensity resolution. If an integer is input, downsamples by the given factor.
    refine_RFI : bool, optional
        Whether to refine RFI or not.
    fill_missing_time : bool, optional
        If not specified or set to None, get_snr decides whether we need to fill missing data with noise or not.
        If set to False, never add noise. If set to True, always add noise.
    thres : float, optional
        Threshold that goes into get_RFI_channels.
        High threshold means almost no RFI will be removed i.e. very bright signals allowed.
        Low threshold means most signal will be removed.
        The threshold should be at least 1.5. The default is 3.
    fill_nan : bool, optional
        If NaN values are present, replace them with noise.

    Returns
    -------
    freq_id : array_like
        Array of frequency numbers i.e. 1 ~ 800 MHz, and 1024 ~ 400 MHz.
    freq : array_like
        Array of frequencies present in the power, same shape as first dimension of power.
        These are the frequencies in MHz corresponding to the freq_id.
    power : array_like
        Normalized power.
    Examples
    --------
    >>> from baseband_analysis.core import BBData
    >>> from baseband_analysis.utilities import get_snr, plot_waterfall
    >>> file_location = '/data/frb-baseband/baseband_products/Daniela/singlebeam_81110473.h5' #Crab
    >>> data = BBData.from_file(file_location)
    >>> freq_id, freq, power = get_snr(data, DM_range = 5, time_range = [5000,10000])
    >>> plot_waterfall(power, data, freq)
    """
    if isinstance(data, str):
        data = BBData.from_file(data)

    if check_channel_number:
        if data["tiedbeam_power"].shape[0] < 100:
            raise RuntimeError(
                "Less than 100 channels are present in the beamformed file. Use check_channel_number=False to ignore this"
            )

    if fill_missing_time is True:
        if time_range is None:
            time_range = [0, data["tiedbeam_power"].shape[-1]]

    (
        power,
        valid_channels_out,
        offset,
        weight,
        freq,
        freq_id,
        downsampling_factor,
        time_range_out,
        DM_out,
    ) = normalize_and_refine(
        data,
        DM=DM,
        downsample=downsample,
        refine_RFI=refine_RFI,
        valid_channels=valid_channels,
        time_range=time_range,
        thres_mean=thres_mean,
        thres_std=thres_std,
        spectrum_lim=spectrum_lim,
        spectrum_thresh=spectrum_thresh,
        w=w,
        diagnostic_plots=diagnostic_plots,
        DM_range=DM_range,
        doublecheck_RFI=doublecheck_RFI,
        DM_step=DM_step,
        check_old_process=check_old_process,
        fill_nan=fill_nan,
    )

    if fill_missing_time is None:
        # decide if pulse will be cut off
        profile = get_profile(power)
        peak = np.nanargmax(profile)
        if (
            (np.nanmax(profile) < 6)
            or (peak > power.shape[-1] * 0.9)
            or (peak < power.shape[-1] * 0.1)
        ):
            log.warning(
                "Pulse not in the valid range, noise will be added to search the full dump."
            )
            if time_range is None:
                time_range = [0, data["tiedbeam_power"].shape[-1]]

            (
                power,
                valid_channels_out,
                offset,
                weight,
                freq,
                freq_id,
                downsampling_factor,
                time_range_out,
                DM_out,
            ) = normalize_and_refine(
                data,
                DM=DM,
                downsample=downsample,
                refine_RFI=refine_RFI,
                valid_channels=valid_channels,
                time_range=time_range,
                thres_mean=thres_mean,
                thres_std=thres_std,
                spectrum_lim=spectrum_lim,
                spectrum_thresh=spectrum_thresh,
                w=w,
                diagnostic_plots=diagnostic_plots,
                DM_range=DM_range,
                doublecheck_RFI=doublecheck_RFI,
                DM_step=DM_step,
                check_old_process=check_old_process,
                fill_nan=fill_nan,
            )
            profile = get_profile(power)
            peak = np.nanargmax(profile)
            end_bin = int(
                peak + 100e-3 / data.attrs["delta_time"] / downsampling_factor  # s
            )
            power = power[..., :end_bin]
            time_range_out[1] = end_bin * downsampling_factor
    # Calculate the profile where more than 10 valid channels are present
    profile = get_profile(power, min_channels=10)
    # Raise an exception if no peak larger than 6 is found
    if np.nanmax(profile) < 5:
        error_msg = "Signal not found in the data"
        if raise_missing_signal:
            raise RuntimeError(error_msg)
        else:
            log.warning(error_msg)

    if diagnostic_plots:
        # Choose brightest beam to plot
        if len(power.shape) > 2:
            wfall = power[:, get_onsource_beam(power)].copy()
        else:
            wfall = power.copy()
        fig = plot_waterfall(wfall, data, f=freq, t_downsample=downsampling_factor)
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "snr.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")

    if return_full:
        time_range_out = np.array(time_range_out)
        if downsample:
            time_range_out *= data["tiedbeam_power"].attrs["time_downsample_factor"]
        return (
            freq_id,
            freq,
            power,
            offset,
            weight,
            valid_channels_out,
            time_range_out,
            DM_out,
            downsampling_factor,
        )
    else:
        return freq_id, freq, power


def get_snr_beam(
    data,
    DM=None,
    diagnostic_plots=False,
    w=None,
    time_range=None,
    spectrum_lim=True,
    DM_range=10,
    fill_missing_time=None,
):

    """
    Wrapper for `get_snr`.

    Parameters
    ----------
    data : TYPE
        Description
    DM : `float`
        Dispersion Measure (DM) in pc/cm3 units.
    diagnostic_plots : `bool`
        If `True` it will plot diagnostic plot.
    w : `np.ndarray`
        Weight and offset to normalise the data by, an array of float numbers.
    time_range : `list` or `np.ndarray`
        start and end time bins where the burst is located.
    spectrum_lim : `bool`
        If `True` will cut out the frequencies which do not contain any burst data.
    DM_range : `float`
        DM deviation to consider when refining DM (see `DM_SNR`).
    fill_missing_time : `bool`
        If `True` noise will be added.
    data : `BBData`

    Returns
    -------
    snr_beam : `float`
    """

    if isinstance(data, str):
        data = BBData.from_file(data)

    # Fit S/N with model
    freq_id, freq, power = get_snr(
        data,
        DM=DM,
        diagnostic_plots=diagnostic_plots,
        w=w,
        time_range=time_range,
        spectrum_lim=spectrum_lim,
        DM_range=DM_range,
        refine_RFI=True,
        fill_missing_time=fill_missing_time,
    )
    snr_beam = beam_snr(power)

    if diagnostic_plots:
        from baseband_analysis.utilities import plotting

        cbeam = get_onsource_beam(snr_beam)
        wfall = power[:, cbeam]

        if isinstance(diagnostic_plots, bool):
            save_dir = None
        else:
            save_dir = diagnostic_plots

        plotting.snr_beam_diagnostic(wfall, data, save_dir=save_dir)

    return snr_beam