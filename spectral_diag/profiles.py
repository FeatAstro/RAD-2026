"""Theoretical line profiles.

Planned contents:
- optical_depth_gaussian: Doppler-only (Gaussian) tau(wave) -- a
  pedagogical stepping stone, implemented first
- voigt / optical_depth: full Voigt profile via scipy.special.wofz,
  adding natural/pressure (damping) broadening on top of the Gaussian
- InstrumentGrid: turn a model spectrum into what the instrument
  records, by convolving it with the line spread function (LSF) and
  averaging it over each pixel
"""

from dataclasses import dataclass

import numpy as np
from scipy import constants as _const
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz

from .core import ANGSTROM_TO_CM, C_KMS, FWHM_PER_SIGMA, KMS_TO_CMS

# tau0 = sqrt(pi) * r_e * c * f * lambda_rest * N / b, with r_e the
# classical electron radius (see NOTES_perso.tex Sec. 4 -- this is the
# same combination as e^2/(m_e*c) in cgs, via r_e = e^2/(m_e*c^2)).
# Built from CODATA constants rather than a truncated literature value,
# then converted so N is in cm^-2, lambda_rest in Angstrom, b in km/s.
_R_E_CM = _const.physical_constants["classical electron radius"][0] * 1e2  # m -> cm
_C_CMS = _const.c * 1e2  # m/s -> cm/s

TAU0_CONST = np.sqrt(np.pi) * _R_E_CM * _C_CMS * ANGSTROM_TO_CM / KMS_TO_CMS
# tau0 = TAU0_CONST * N[cm^-2] * f * lambda_rest[Angstrom] / b[km/s]

# a = Gamma * lambda_rest / (4 pi b), the Voigt damping parameter (see
# NOTES_perso.tex Sec. 5 and 6) -- a pure unit-conversion factor
# (Angstrom -> cm, km/s -> cm/s), no fundamental constants beyond pi.
A_CONST = ANGSTROM_TO_CM / (4 * np.pi * KMS_TO_CMS)
# a = A_CONST * gamma[s^-1] * lambda_rest[Angstrom] / b[km/s]

# Largest allowed offset of a pixel from the log-uniform grid, in pixels.
# SDSS stores log10(wave) in float32, which shifts pixels by ~0.001
PIXEL_GRID_TOLERANCE = 0.01


def optical_depth_gaussian(wave, line, logN, b, z):
    """Gaussian (Doppler-only) optical depth tau(wave) for one transition.

    Ignores natural/pressure (damping) broadening -- i.e. this is the
    narrow-line limit of the full Voigt profile. It breaks down once
    a line starts to saturate, where the Lorentzian wings matter.

    Parameters
    ----------
    wave : array
        Observed wavelength grid, Angstrom.
    line : Line
        Atomic transition (wave_rest in Angstrom, f).
    logN : float
        log10(column density / cm^-2).
    b : float
        Doppler (velocity dispersion) parameter, km/s.
    z : float
        Redshift of the absorber.

    Returns
    -------
    tau : array
        Optical depth at each wavelength in `wave` (dimensionless).
    """
    N = 10**logN
    lam0_obs = line.wave_rest * (1 + z)
    dv = C_KMS * (wave - lam0_obs) / lam0_obs  # km/s
    tau0 = TAU0_CONST * N * line.f * line.wave_rest / b
    return tau0 * np.exp(-((dv / b) ** 2))


def voigt(x, a):
    """Voigt function H(x, a) = Re[wofz(x + i a)].

    Reduces to exp(-x**2) (pure Gaussian) as a -> 0 -- this is the
    sanity check that ties this back to optical_depth_gaussian.
    """
    return wofz(x + 1j * a).real


def optical_depth(wave, line, logN, b, z):
    """Full Voigt optical depth tau(wave) for one transition.

    Adds natural/pressure (damping) broadening on top of the Gaussian
    (Doppler) term in `optical_depth_gaussian`, via the transition's
    damping constant `line.gamma`. Reduces to `optical_depth_gaussian`
    in the limit of negligible damping (a -> 0): dominates the line
    core the same way, but keeps growing in the wings with N even
    after the core saturates (the a != 0 behaviour `optical_depth_gaussian`
    cannot capture).

    Parameters
    ----------
    wave : array
        Observed wavelength grid, Angstrom.
    line : Line
        Atomic transition (wave_rest in Angstrom, f, gamma).
    logN : float
        log10(column density / cm^-2).
    b : float
        Doppler (velocity dispersion) parameter, km/s.
    z : float
        Redshift of the absorber.

    Returns
    -------
    tau : array
        Optical depth at each wavelength in `wave` (dimensionless).
    """
    N = 10**logN
    lam0_obs = line.wave_rest * (1 + z)
    dv = C_KMS * (wave - lam0_obs) / lam0_obs  # km/s
    x = dv / b
    a = A_CONST * line.gamma * line.wave_rest / b
    tau0 = TAU0_CONST * N * line.f * line.wave_rest / b
    return tau0 * voigt(x, a)


@dataclass(frozen=True)
class InstrumentGrid:
    """Model grid matched to the pixels and the LSF of a spectrum.

    The instrument convolves the true spectrum with its LSF, then each
    pixel integrates the result over its width. A line narrower than a
    pixel is lost if the model is only evaluated at the pixel centres,
    so the model is computed on `oversample` subpixels per pixel.
    Requires pixels uniform in log10(wave) (gaps allowed), where a
    Gaussian LSF of constant FWHM in km/s has a constant width in
    pixels.

    Attributes
    ----------
    wave : array
        Subpixel centres, Angstrom. Covers the gaps between observed
        pixels, so that the convolution sees a contiguous grid.
    subpixels : array of int, shape (n_pixels, oversample)
        Indices into `wave` of the subpixels of each observed pixel.
    lsf_sigma : float
        Gaussian sigma of the LSF, in subpixels.
    """

    wave: np.ndarray
    subpixels: np.ndarray
    lsf_sigma: float

    @classmethod
    def for_spectrum(cls, spectrum, oversample):
        """Build the grid for the pixels and LSF of `spectrum`.

        Parameters
        ----------
        spectrum : Spectrum
            `meta` must hold pixel_dex and fwhm_kms.
        oversample : int
            Subpixels per pixel. The subpixel width,
            ln(10) * c * pixel_dex / oversample in km/s, must be smaller
            than the narrowest line (b).

        Returns
        -------
        grid : InstrumentGrid
        """
        pixel_dex = spectrum.meta["pixel_dex"]
        subpixel_dex = pixel_dex / oversample
        pixel_numbers = _pixel_numbers(spectrum.wave, pixel_dex)
        n_subpixels = (pixel_numbers[-1] + 1) * oversample
        # Pixel centres are at wave, so pixel 0 starts half a pixel below
        start = np.log10(spectrum.wave[0]) - pixel_dex / 2
        log_wave = start + subpixel_dex * (np.arange(n_subpixels) + 0.5)
        lsf_sigma_kms = spectrum.meta["fwhm_kms"] / FWHM_PER_SIGMA
        return cls(
            wave=10**log_wave,
            subpixels=pixel_numbers[:, None] * oversample + np.arange(oversample),
            lsf_sigma=_kms_to_dex(lsf_sigma_kms) / subpixel_dex,
        )

    def observe(self, flux):
        """Convolve `flux` with the LSF and average it over each pixel.

        Parameters
        ----------
        flux : array
            Model flux on `self.wave`, e.g. exp(-tau). Multiply by the
            continuum afterwards: it barely changes across the LSF.

        Returns
        -------
        observed : array
            One value per pixel of the spectrum the grid was built for.
            Within ~3 lsf_sigma of the grid edges it assumes the flux
            continues constant beyond them.
        """
        smoothed = gaussian_filter1d(flux, self.lsf_sigma, mode="nearest")
        return smoothed[self.subpixels].mean(axis=1)


def _pixel_numbers(wave, pixel_dex):
    """Position of each pixel on a grid uniform in log10(wave).

    Parameters
    ----------
    wave : array
        Increasing pixel centres, Angstrom.
    pixel_dex : float
        Grid step, log10(Angstrom).

    Returns
    -------
    numbers : array of int
        0 for the first pixel; jumps by more than 1 across gaps.

    Raises
    ------
    ValueError
        If a pixel is more than PIXEL_GRID_TOLERANCE off the grid.
    """
    position = (np.log10(wave) - np.log10(wave[0])) / pixel_dex
    numbers = np.round(position).astype(int)
    if np.any(np.abs(position - numbers) > PIXEL_GRID_TOLERANCE):
        raise ValueError("wave is not uniform in log10(wave) with step pixel_dex")
    return numbers


def _kms_to_dex(velocity):
    """Width in log10(wave) of a velocity interval.

    d(ln wave) = dv / c, and log10 = ln / ln(10).

    Parameters
    ----------
    velocity : float
        km/s.

    Returns
    -------
    dex : float
    """
    return velocity / (C_KMS * np.log(10))
