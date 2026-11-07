"""Continuum fitting and normalisation.

The continuum is a cubic spline fitted by least squares to the pixels
that are not in a line. Lines are found by sigma clipping: pixels more
than `clip_low` errors below the fit are rejected, the spline is
refitted, and so on until the set of used pixels stops changing. Only
pixels below the fit are rejected, because absorption only removes
flux.
"""

# TODO: for Doppler-broadened lines (quasar absorbers, 21 cm), give
# knot_spacing in km/s instead of Angstrom, since a line of given b
# gets wider in Angstrom as lambda grows. Not needed for the A0 star.

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

# Cubic spline: smooth up to the second derivative
SPLINE_DEGREE = 3
# Fewest used pixels allowed between two knots: the 4 coefficients of
# the cubic in that interval must be well constrained
MIN_PIXELS_PER_INTERVAL = 10


@dataclass(frozen=True)
class ContinuumSettings:
    """Parameters of the continuum fit.

    Attributes
    ----------
    knot_spacing : float
        Distance between spline knots, Angstrom. Wider than the broadest
        line, or the spline follows it; narrow enough to follow the
        shape of the continuum.
    clip_low : float
        Pixels more than clip_low errors below the fit are rejected.
    max_iter : int
        Maximum number of fit-and-clip iterations.
    """

    knot_spacing: float = 250.0
    clip_low: float = 2.0
    max_iter: int = 20


@dataclass(frozen=True)
class ContinuumFit:
    """Result of `fit_continuum`.

    Attributes
    ----------
    continuum : array
        Continuum at every pixel, units of the flux. NaN beyond the first
        and last used pixel, where the spline is not constrained.
    error : array
        1-sigma statistical error of `continuum`. NaN where it is.
    used : array of bool
        Pixels used in the final fit.
    n_iter : int
        Fit-and-clip iterations done.
    status : str
        "converged" (the used pixels stopped changing) or "max_iter".
    """

    continuum: np.ndarray
    error: np.ndarray
    used: np.ndarray
    n_iter: int
    status: str


def fit_continuum(spectrum, settings=ContinuumSettings()):
    """Fit the continuum of `spectrum` by iterative sigma clipping.

    Parameters
    ----------
    spectrum : Spectrum
        Wavelengths in Angstrom, strictly increasing; `error` > 0.
    settings : ContinuumSettings

    Returns
    -------
    fit : ContinuumFit
    """
    spectrum.check_valid()
    used, n_iter, status = _clip_until_stable(spectrum, settings)
    spline = _weighted_spline(spectrum, used, settings.knot_spacing)

    used_wave = spectrum.wave[used]
    inside = (spectrum.wave >= used_wave[0]) & (spectrum.wave <= used_wave[-1])
    continuum = np.full(len(spectrum.wave), np.nan)
    error = np.full(len(spectrum.wave), np.nan)
    continuum[inside] = spline(spectrum.wave[inside])
    error[inside] = _continuum_error(spline, spectrum, used, spectrum.wave[inside])
    return ContinuumFit(continuum, error, used, n_iter, status)


def normalise(spectrum, continuum):
    """Return a copy of `spectrum` divided by `continuum`.

    Parameters
    ----------
    spectrum : Spectrum
    continuum : array
        e.g. `ContinuumFit.continuum`.

    Returns
    -------
    normalised : Spectrum
        Flux and error divided by the continuum, which becomes 1. The
        error does not include the error of the continuum.
    """
    return replace(
        spectrum,
        flux=spectrum.flux / continuum,
        error=spectrum.error / continuum,
        continuum=np.ones_like(continuum),
    )


def _clip_until_stable(spectrum, settings):
    """Fit, reject the pixels far below the fit, and repeat until stable.

    Every pixel is tested against each new fit, so pixels rejected
    earlier can come back: the first fits run through the lines, below
    the true continuum, and rise step by step to it.

    Parameters
    ----------
    spectrum : Spectrum
    settings : ContinuumSettings

    Returns
    -------
    used : array of bool
        Pixels used in the last fit.
    n_iter : int
    status : str
        "converged" or "max_iter".
    """
    used = np.ones(len(spectrum.wave), dtype=bool)
    for n_iter in range(1, settings.max_iter + 1):
        spline = _weighted_spline(spectrum, used, settings.knot_spacing)
        # Evaluated at every pixel, also beyond the used ones, so that
        # pixels rejected at the ends can come back
        deviation = (spectrum.flux - spline(spectrum.wave)) / spectrum.error
        new_used = deviation >= -settings.clip_low
        if np.array_equal(new_used, used):
            return used, n_iter, "converged"
        used = new_used
    warnings.warn(f"continuum fit not converged after {settings.max_iter} iterations")
    return used, settings.max_iter, "max_iter"


def _weighted_spline(spectrum, used, knot_spacing):
    """Least-squares cubic spline through the used pixels, weighted by 1/error.

    Parameters
    ----------
    spectrum : Spectrum
    used : array of bool
    knot_spacing : float
        Angstrom.

    Returns
    -------
    spline : scipy.interpolate.BSpline
    """
    wave = spectrum.wave[used]
    knots = _knots(wave, knot_spacing)
    weights = 1 / spectrum.error[used]
    return make_lsq_spline(wave, spectrum.flux[used], knots, SPLINE_DEGREE, weights)


def _knots(wave, knot_spacing):
    """Knots about `knot_spacing` apart, from wave[0] to wave[-1].

    Starts from evenly spaced knots. While two knots enclose fewer than
    MIN_PIXELS_PER_INTERVAL used pixels (e.g. inside a broad line whose
    pixels were all rejected), one of them is removed, so the spline
    bridges the gap. A B-spline needs its two end knots repeated
    SPLINE_DEGREE + 1 times.

    Parameters
    ----------
    wave : array
        Sorted wavelengths of the used pixels, Angstrom.
    knot_spacing : float
        Angstrom.

    Returns
    -------
    knots : array
    """
    first, last = wave[0], wave[-1]
    n_intervals = max(1, round((last - first) / knot_spacing))
    interior_knots = list(np.linspace(first, last, n_intervals + 1)[1:-1])
    sparse_interval = _first_sparse_interval(wave, interior_knots)
    while sparse_interval is not None:
        # Interval i ends at interior knot i, except the last interval,
        # which ends at wave[-1]
        if sparse_interval == len(interior_knots):
            # No knot on its right: remove its left one
            del interior_knots[sparse_interval - 1]
        else:
            del interior_knots[sparse_interval]
        sparse_interval = _first_sparse_interval(wave, interior_knots)
    end_knots = SPLINE_DEGREE + 1
    return np.concatenate([[first] * end_knots, interior_knots, [last] * end_knots])


def _first_sparse_interval(wave, interior_knots):
    """Index of the first interval between knots with too few used pixels.

    Parameters
    ----------
    wave : array
        Sorted wavelengths of the used pixels, Angstrom.
    interior_knots : list of float

    Returns
    -------
    index : int or None
        0 for the interval from wave[0] to the first knot. None if every
        interval holds at least MIN_PIXELS_PER_INTERVAL pixels, or if
        there is no interior knot left.
    """
    if not interior_knots:
        return None
    edges = [wave[0], *interior_knots, wave[-1]]
    pixels_per_interval, _ = np.histogram(wave, bins=edges)
    for index, pixel_count in enumerate(pixels_per_interval):
        if pixel_count < MIN_PIXELS_PER_INTERVAL:
            return index
    return None


def _continuum_error(spline, spectrum, used, wave):
    """1-sigma statistical error of the fitted spline at each wavelength of `wave`.

    The spline is C(lambda) = sum_j c_j B_j(lambda). Least squares gives
    the covariance of the coefficients c,
        Cov = (sum over used pixels i of B_i B_i^T / error_i^2)^-1,
    where B_i holds the values of the basis functions B_j at pixel i, and
    the variance of the continuum at lambda is B(lambda)^T Cov B(lambda).
    Cov is multiplied by the reduced chi^2 when it exceeds 1, so that
    pixel errors that are too small inflate the continuum error.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
        Fit to the `used` pixels.
    spectrum : Spectrum
    used : array of bool
    wave : array
        Wavelengths where the error is wanted, Angstrom.

    Returns
    -------
    error : array
        Units of the flux.
    """
    n_coefficients = len(spline.c)
    used_wave = spectrum.wave[used]
    used_flux = spectrum.flux[used]
    used_error = spectrum.error[used]

    normal_matrix = np.zeros((n_coefficients, n_coefficients))
    used_basis = BSpline.design_matrix(used_wave, spline.t, SPLINE_DEGREE)
    for basis_row, pixel_error in zip(used_basis.toarray(), used_error):
        normal_matrix += np.outer(basis_row, basis_row) / pixel_error**2
    covariance = np.linalg.pinv(normal_matrix)

    residuals = (used_flux - spline(used_wave)) / used_error
    reduced_chi2 = np.sum(residuals**2) / (len(residuals) - n_coefficients)
    covariance = covariance * max(1.0, reduced_chi2)

    basis = BSpline.design_matrix(wave, spline.t, SPLINE_DEGREE)
    variances = []
    for basis_row in basis.toarray():
        variances.append(basis_row @ covariance @ basis_row)
    return np.sqrt(variances)
