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
from scipy import constants as _const
from scipy.special import wofz

from .core import ANGSTROM_TO_CM, C_KMS, KMS_TO_CMS

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
    a = A_CONST * line.gamma * line.wave_rest / b
    tau0 = TAU0_CONST * N * line.f * line.wave_rest / b
    return tau0 * voigt(x, a)
