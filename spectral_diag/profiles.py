"""Theoretical line profiles.

Planned contents:
- optical_depth_gaussian: Doppler-only (Gaussian) tau(wave) -- a
  pedagogical stepping stone, implemented first
- voigt / optical_depth: full Voigt profile via scipy.special.wofz,
  adding natural/pressure (damping) broadening on top of the Gaussian
- apply_lsf: convolve a model spectrum with the instrument's line
  spread function
"""

import numpy as np
from scipy.special import wofz

C_KMS = 299792.458  # speed of light, km/s


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
    # tau0 = sqrt(pi) * e^2 * f * lambda_rest * N / (m_e * c * b), with the
    # cgs constant sqrt(pi)*e^2/(m_e*c) folded in and converted for
    # lambda_rest in Angstrom and b in km/s (N in cm^-2).
    tau0 = 1.4974e-15 * N * line.f * line.wave_rest / b
    return tau0 * np.exp(-(dv / b) ** 2)


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
    # a = Gamma * lambda_rest / (4 pi b), the ratio of the Lorentzian
    # (damping) width to the Doppler width; see NOTES_perso.tex. The
    # 7.9577e-15 folds in the unit conversion for lambda_rest in
    # Angstrom and b in km/s (Gamma in s^-1).
    a = 7.9577e-15 * line.gamma * line.wave_rest / b
    tau0 = 1.4974e-15 * N * line.f * line.wave_rest / b
    return tau0 * voigt(x, a)
