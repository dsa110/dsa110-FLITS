"""Residual analysis and fit quality validation for FLITS."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class ResidualDiagnostics:
    """Results from residual analysis."""

    normality_pass: bool
    normality_pvalue: float

    bias_pass: bool
    bias_mean: float
    bias_num_sigma: float

    autocorr_pass: bool
    durbin_watson: float

    chi_sq_red: float
    r_squared: float

    quality_flag: str  # "PASS", "MARGINAL", "FAIL"
    validation_notes: list

    def __str__(self):
        lines = [
            "=" * 80,
            f"RESIDUAL DIAGNOSTICS: {self.quality_flag}",
            "=" * 80,
            f"χ²_red = {self.chi_sq_red:.2f}",
            f"R² = {self.r_squared:.3f}",
            f"",
            f"Normality: {'PASS' if self.normality_pass else 'FAIL'} (p={self.normality_pvalue:.4f})",
            f"Bias: {'PASS' if self.bias_pass else 'FAIL'} (mean={self.bias_mean:.4f}, σ={self.bias_num_sigma:.2f})",
            f"Autocorr: {'PASS' if self.autocorr_pass else 'FAIL'} (DW={self.durbin_watson:.2f})",
        ]
        if self.validation_notes:
            lines.append("\nNotes:")
            for note in self.validation_notes:
                lines.append(f"  - {note}")
        lines.append("=" * 80)
        return "\n".join(lines)


def analyze_residuals(
    data: NDArray,
    model_pred: NDArray,
    noise_std: float = 1.0,
    normality_threshold: float = 0.05,
    output_path: str = None,
) -> ResidualDiagnostics:
    """Analyze residuals for fit quality.

    Parameters
    ----------
    data : ndarray
        Observed data
    model_pred : ndarray
        Model prediction
    noise_std : float
        Noise standard deviation
    normality_threshold : float
        p-value threshold for Shapiro-Wilk test
    output_path : str, optional
        If provided, save diagnostic plots

    Returns
    -------
    ResidualDiagnostics
    """

    data_flat = np.asarray(data).flatten()
    model_flat = np.asarray(model_pred).flatten()
    noise_flat = np.atleast_1d(noise_std)
    if noise_flat.size == 1:
        noise_flat = np.full_like(data_flat, noise_flat[0])
    else:
        noise_flat = np.asarray(noise_flat).flatten()

    residuals = data_flat - model_flat

    # Chi-squared and R-squared
    chi_sq = np.sum((residuals / noise_flat) ** 2)
    dof = len(data_flat) - 1
    chi_sq_red = chi_sq / dof

    ss_res = np.sum((residuals / noise_flat) ** 2)
    ss_tot = np.sum(((data_flat - np.mean(data_flat)) / noise_flat) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Normality test
    test_residuals = residuals[::max(1, len(residuals)//5000)]
    try:
        shapiro_stat, normality_pvalue = stats.shapiro(test_residuals)
        normality_pass = normality_pvalue > normality_threshold
    except Exception:
        normality_pvalue = np.nan
        normality_pass = True

    # Bias test
    bias_mean = np.mean(residuals)
    bias_std = np.std(residuals)
    bias_sem = bias_std / np.sqrt(len(residuals))

    if bias_sem > 0:
        bias_num_sigma = abs(bias_mean) / bias_sem
        bias_pass = bias_num_sigma < 3.0
    else:
        bias_num_sigma = 0.0
        bias_pass = True

    # Autocorrelation test
    diffs = np.diff(residuals)
    durbin_watson = np.sum(diffs ** 2) / np.sum(residuals ** 2)
    autocorr_pass = 1.0 <= durbin_watson <= 3.0

    # Assign quality flag
    validation_notes = []
    quality_flag = "PASS"

    if chi_sq_red > 10:
        quality_flag = "FAIL"
        validation_notes.append(f"χ²_red = {chi_sq_red:.1f} >> threshold")
    elif chi_sq_red > 3:
        quality_flag = "FAIL"
        validation_notes.append(f"χ²_red = {chi_sq_red:.2f} > 3.0")
    elif chi_sq_red > 1.5:
        quality_flag = "MARGINAL"
        validation_notes.append(f"χ²_red = {chi_sq_red:.2f} slightly high")
    elif chi_sq_red < 0.3:
        quality_flag = "MARGINAL"
        validation_notes.append(f"χ²_red = {chi_sq_red:.2f} suspiciously low")
    else:
        validation_notes.append(f"χ²_red = {chi_sq_red:.2f} ✓")

    if r_squared < 0.5:
        quality_flag = "FAIL"
        validation_notes.append(f"R² = {r_squared:.3f} << 0.5")
    elif r_squared < 0.7:
        quality_flag = "FAIL"
        validation_notes.append(f"R² = {r_squared:.3f} < 0.7")
    elif r_squared < 0.85:
        if quality_flag == "PASS":
            quality_flag = "MARGINAL"
        validation_notes.append(f"R² = {r_squared:.3f} marginal")
    else:
        validation_notes.append(f"R² = {r_squared:.3f} ✓")

    if not normality_pass:
        if quality_flag == "PASS":
            quality_flag = "MARGINAL"
        validation_notes.append(f"Residuals non-normal (p={normality_pvalue:.4f})")

    if not bias_pass:
        quality_flag = "FAIL"
        validation_notes.append(f"Systematic bias ({bias_num_sigma:.1f}σ)")

    if not autocorr_pass:
        quality_flag = "FAIL"
        validation_notes.append(f"Autocorrelation (DW={durbin_watson:.2f})")

    # Generate plots if requested
    if output_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(data_flat, 'k.', alpha=0.5, label='Data', markersize=2)
        axes[0, 0].plot(model_flat, 'r-', linewidth=1.5, label='Model')
        axes[0, 0].set_title('Data vs. Model')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(residuals, 'b.', alpha=0.5, markersize=2)
        axes[0, 1].axhline(0, color='k', linestyle='--', linewidth=1)
        axes[0, 1].axhline(3*bias_sem, color='r', linestyle=':', linewidth=1)
        axes[0, 1].axhline(-3*bias_sem, color='r', linestyle=':', linewidth=1)
        axes[0, 1].set_title('Residuals')
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(alpha=0.3, axis='y')

        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    diagnostics = ResidualDiagnostics(
        normality_pass=bool(normality_pass),
        normality_pvalue=float(normality_pvalue),
        bias_pass=bool(bias_pass),
        bias_mean=float(bias_mean),
        bias_num_sigma=float(bias_num_sigma),
        autocorr_pass=bool(autocorr_pass),
        durbin_watson=float(durbin_watson),
        chi_sq_red=float(chi_sq_red),
        r_squared=float(r_squared),
        quality_flag=quality_flag,
        validation_notes=validation_notes,
    )

    return diagnostics


__all__ = ["ResidualDiagnostics", "analyze_residuals"]
