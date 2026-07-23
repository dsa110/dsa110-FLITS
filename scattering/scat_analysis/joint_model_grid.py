"""Median-posterior model-grid production shared by controlled and legacy fits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from .burstfit import FRBParams
from .burstfit_joint import gain_marginal_multi_band_solution


def recover(model, params_list):
    kernels = np.stack([model(replace(params, c0=1.0, gamma=0.0), "M3") for params in params_list])
    data = np.asarray(model.data, float)
    noise_support = np.asarray(model.noise_std).reshape(-1)
    noise = np.clip(np.asarray(noise_support, dtype=float), 1e-9, None)
    matrix = np.einsum("nft,mft->fnm", kernels, kernels)
    right = np.einsum("nft,ft->fn", kernels, data)
    count = len(params_list)
    jitter = 1e-9 * max(float(np.einsum("fnn->f", matrix).mean()), 1e-30)
    gain = np.linalg.solve(matrix + jitter * np.eye(count), right[..., None])[..., 0]
    prediction = np.einsum("fn,nft->ft", gain, kernels)
    valid = model.valid
    valid = (
        np.ones(data.shape[0], bool)
        if valid is None
        else np.asarray(valid).reshape(-1).astype(bool)
    )
    residual = ((data - prediction) / noise[:, None])[valid]
    residual = residual[np.isfinite(residual)]
    residual_mean_square = float(np.mean(residual**2))
    valid_gain = np.where(valid[:, None], gain, 0.0)
    fluence = np.einsum("fn,nft->n", np.clip(valid_gain, 0.0, None), kernels)
    return {
        "data": data,
        "model": prediction,
        "freq": np.asarray(model.freq, float),
        "time": np.asarray(model.time, float),
        "noise": noise_support,
        "valid": valid,
        "fluence": np.asarray(fluence, float),
    }, residual_mean_square


def recover_proper_gain(model, params_list, s2: float | None):
    """Recover the same finite-prior gains used by the multi-component likelihood."""
    valid = np.asarray(model.valid, dtype=bool)
    gains, diagnostics = gain_marginal_multi_band_solution(
        model,
        params_list,
        ["M3"] * len(params_list),
        s2=s2,
    )
    kernels = np.stack(
        [
            model(replace(params, c0=1.0, gamma=0.0), "M3", freq_subset=valid)
            for params in params_list
        ]
    )
    prediction_valid = np.einsum("fn,nft->ft", gains, kernels)
    data = np.asarray(model.data, dtype=float)
    prediction = np.zeros_like(data)
    prediction[valid] = prediction_valid
    noise_support = np.asarray(model.noise_std).reshape(-1)
    noise = np.clip(np.asarray(noise_support, dtype=float), 1e-9, None)
    residual = (data[valid] - prediction_valid) / noise[valid, None]
    residual = residual[np.isfinite(residual)]
    residual_mean_square = float(np.mean(residual**2))
    fluence = np.einsum("fn,nft->n", np.clip(gains, 0.0, None), kernels)
    return (
        {
            "data": data,
            "model": prediction,
            "freq": np.asarray(model.freq, float),
            "time": np.asarray(model.time, float),
            "noise": noise_support,
            "valid": valid,
            "fluence": np.asarray(fluence, float),
        },
        residual_mean_square,
        diagnostics,
    )


def band_params(
    percentiles: Mapping[str, float], band: str, count: int, tau: float, beta: float
) -> list[FRBParams]:
    delta_dm = float(percentiles.get(f"delta_dm_{band}", 0.0))
    output = []
    for index in range(1, count + 1):
        t0 = percentiles.get(f"t0_{band}{index}", percentiles.get(f"t0_{band}"))
        zeta = percentiles.get(f"zeta_{band}{index}", percentiles.get(f"zeta_{band}"))
        output.append(
            FRBParams(
                c0=1.0,
                t0=float(t0),
                gamma=0.0,
                zeta=float(zeta),
                tau_1ghz=tau,
                beta=beta,
                delta_dm=delta_dm,
            )
        )
    return output


def build_model_grid_arrays(model_C, model_D, fit_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact median-posterior data/model packet for both bands."""
    values = {key: value["median"] for key, value in fit_summary["percentiles"].items()}
    tau, beta = values["tau_1ghz"], values["beta"]
    count_c = int(fit_summary.get("components_C", 1))
    count_d = int(fit_summary.get("components_D", 1))
    if fit_summary.get("shared_zeta"):
        zeta_c = values["zeta_1ghz"] * np.asarray(model_C.freq, float) ** values["x_zeta"]
        zeta_d = values["zeta_1ghz"] * np.asarray(model_D.freq, float) ** values["x_zeta"]
        params_c = [
            FRBParams(
                c0=1.0,
                t0=values["t0_C"],
                gamma=0.0,
                zeta=zeta_c,
                tau_1ghz=tau,
                beta=beta,
                delta_dm=values["delta_dm_C"],
            )
        ]
        params_d = [
            FRBParams(
                c0=1.0,
                t0=values["t0_D"],
                gamma=0.0,
                zeta=zeta_d,
                tau_1ghz=tau,
                beta=beta,
                delta_dm=values["delta_dm_D"],
            )
        ]
    else:
        params_c = band_params(values, "C", count_c, tau, beta)
        params_d = band_params(values, "D", count_d, tau, beta)
    gain_model = fit_summary.get("gain_model", "ordinary_least_squares")
    if gain_model == "proper_gaussian":
        recovered_c, residual_mean_square_c, gain_diagnostics_c = recover_proper_gain(
            model_C, params_c, fit_summary.get("gain_s2")
        )
        recovered_d, residual_mean_square_d, gain_diagnostics_d = recover_proper_gain(
            model_D, params_d, fit_summary.get("gain_s2")
        )
    elif gain_model == "ordinary_least_squares":
        recovered_c, residual_mean_square_c = recover(model_C, params_c)
        recovered_d, residual_mean_square_d = recover(model_D, params_d)
        gain_diagnostics_c = {"s2": np.nan}
        gain_diagnostics_d = {"s2": np.nan}
    else:
        raise ValueError(f"unsupported gain model: {gain_model}")
    return {
        "dataC": recovered_c["data"],
        "modelC": recovered_c["model"],
        "freqC": recovered_c["freq"],
        "timeC": recovered_c["time"],
        "noiseC": recovered_c["noise"],
        "validC": recovered_c["valid"],
        "dataD": recovered_d["data"],
        "modelD": recovered_d["model"],
        "freqD": recovered_d["freq"],
        "timeD": recovered_d["time"],
        "noiseD": recovered_d["noise"],
        "validD": recovered_d["valid"],
        "alpha": fit_summary["alpha"]["median"],
        "beta": beta,
        "tau_1ghz": tau,
        "residual_mean_squareC": residual_mean_square_c,
        "residual_mean_squareD": residual_mean_square_d,
        "nC": count_c,
        "nD": count_d,
        "fluenceC": recovered_c["fluence"],
        "fluenceD": recovered_d["fluence"],
        "burst": fit_summary["burst"],
        "gain_model": gain_model,
        "gain_s2_C": gain_diagnostics_c["s2"],
        "gain_s2_D": gain_diagnostics_d["s2"],
        "dm_initC": float(model_C.dm_init),
        "dm_initD": float(model_D.dm_init),
        "df_MHzC": float(model_C.df_MHz),
        "df_MHzD": float(model_D.df_MHz),
        "dispersion_betaC": float(model_C.beta),
        "dispersion_betaD": float(model_D.beta),
    }
