import numpy as np
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit

__all__ = ["DMPhaseEstimator", "quadratic"]

# ------------------------------------------------------------------
# Constants & helpers
# ------------------------------------------------------------------
K_DM_S = 4.148808e3  # MHz^2 pc^-1 cm^3 ms
EPS = 1e-30


def quadratic(x, a, b, c):
    """Quadratic a x² + b x + c."""
    return a * x ** 2 + b * x + c


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class DMPhaseEstimator:
    """Vectorised phase‑coherence DM estimator with bootstrap‑robust σ_DM."""

    def __init__(
        self,
        waterfall: np.ndarray,
        freqs: np.ndarray,
        dt: float,
        dm_grid: np.ndarray,
        ref: str | float = "top",
        weights=None,
        f_cut=None,
        n_boot: int = 200,
        random_state: int | None = None,
    ):
        self.wf = np.asarray(waterfall, complex)
        self.freqs = np.asarray(freqs, float)
        self.dt = float(dt)
        self.dm_grid = np.asarray(dm_grid, float)
        self.n_boot = max(20, int(n_boot))  # need some resamples
        self.rng = np.random.default_rng(random_state)
        self.f_cut = f_cut

        self.nu_ref = (
            self.freqs.max() if ref == "top" else self.freqs.min() if ref == "bottom" else
            0.5 * (self.freqs.max() + self.freqs.min()) if ref == "centre" else float(ref)
        )

        self.weights = self._init_weights(weights)
        self.n_t, self.n_ch = self.wf.shape
        self.fft_wf = fft(self.wf, axis=0)
        self.freq_axis = fftfreq(self.n_t, self.dt)
        self.delay_sec = 1e-3 * K_DM_S * (1 / self.freqs**2 - 1 / self.nu_ref**2)

        self.dm_curve, self.dm_err, self._bs_curves = self._make_dm_curve()
        self.dm_best, self.dm_sigma = self._fit_peak_bootstrap()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, weights):
        if weights is None:
            mad = np.median(np.abs(self.wf - np.median(self.wf, axis=0)), axis=0)
            sig = 1.4826 * mad + 1e-12
            w = 1 / sig**2
        else:
            w = np.asarray(weights, float)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Core vectorised ops
    # ------------------------------------------------------------------
    def _phase_cube(self):
        phase = np.exp(
            -2j * np.pi * self.freq_axis[:, None, None]
            * (self.dm_grid[:, None] * self.delay_sec)[None, :, :]
        )
        return (self.fft_wf[:, None, :] * phase).transpose(1, 0, 2)  # (dm, t, ch)

    def _coherent_power(self, spec):
        ph = spec / np.maximum(np.abs(spec), EPS)
        ph *= self.weights[None, None, :]
        s = ph.sum(axis=2)
        return np.abs(s) ** 2 * (self.freq_axis ** 2)[None, :]

    def _window_mask(self, power):
        if self.f_cut is not None:
            lo, hi = self.f_cut
            return (np.abs(self.freq_axis) >= lo) & (np.abs(self.freq_axis) <= hi)
        tot = power.sum(0)
        pk = np.argmax(tot)
        thresh = 0.1 * tot[pk]
        try:
            cut = np.where(tot[pk:] < thresh)[0][0] + pk
        except IndexError:
            cut = len(tot) - 1
        return np.abs(self.freq_axis) <= np.abs(self.freq_axis[cut])

    def _make_dm_curve(self):
        spec = self._phase_cube()
        power = self._coherent_power(spec)
        mask = self._window_mask(power)
        curve = power[:, mask].sum(1)

        # channel bootstrap resamples
        bs_curves = []
        for _ in range(self.n_boot):
            idx = self.rng.choice(self.n_ch, self.n_ch, True)
            ph = spec[:, :, idx] / np.maximum(np.abs(spec[:, :, idx]), EPS)
            ph *= self.weights[idx][None, None, :]
            bs_p = np.abs(ph.sum(2)) ** 2 * (self.freq_axis ** 2)[None, :]
            bs_curves.append(bs_p[:, mask].sum(1))
        bs_curves = np.stack(bs_curves)
        err = np.std(bs_curves, 0, ddof=1)
        return curve, err, bs_curves

    # ------------------------------------------------------------------
    # Peak fit using bootstrap distribution
    # ------------------------------------------------------------------
    def _fit_peak_bootstrap(self):
        # main‑curve quadratic fit over ±2 grid points
        i_pk = int(np.argmax(self.dm_curve))
        window = slice(max(i_pk - 2, 0), min(i_pk + 3, len(self.dm_grid)))
        x = self.dm_grid[window]
        y = self.dm_curve[window]
        coef, _ = curve_fit(quadratic, x, y)
        a, b, _ = coef
        dm_peak = -b / (2 * a)
        dm_peak = float(np.clip(dm_peak, self.dm_grid.min(), self.dm_grid.max()))

        # bootstrap peaks
        peaks = []
        for c in self._bs_curves:
            y_bs = c[window]
            try:
                p_bs, _ = curve_fit(quadratic, x, y_bs)
                a_bs, b_bs, _ = p_bs
                pk = -b_bs / (2 * a_bs)
            except Exception:
                pk = x[np.argmax(y_bs)]
            peaks.append(np.clip(pk, self.dm_grid.min(), self.dm_grid.max()))
        dm_sigma = np.std(peaks, ddof=1)
        # guard against zero
        if dm_sigma == 0:
            dm_sigma = self.dm_grid[1] - self.dm_grid[0]
        return dm_peak, dm_sigma

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def result(self):
        return {
            "dm_best": self.dm_best,
            "dm_sigma": self.dm_sigma,
            "dm_curve": self.dm_curve,
            "dm_curve_sigma": self.dm_err,
            "dm_grid": self.dm_grid,
        }

    def get_dm(self):
        return self.dm_best, self.dm_sigma

