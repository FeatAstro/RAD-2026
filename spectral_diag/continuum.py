"""Continuum fitting and normalisation.

- fit_continuum: cubic spline fitted to the unabsorbed flux level, by
  iterative sigma clipping below the fit, with its statistical error.
  Needs no line list: the pixels clipped below the fit mark candidate
  absorption (plus a few noise pixels)
- normalise: divide a Spectrum's flux/error by the fitted continuum

Written for absorption spectra: absorption only removes flux, so only
pixels far below the fit are rejected.
"""

# TODO: for Doppler-broadened lines (quasar absorbers, ISM, 21 cm), give
# knot_spacing in km/s and place knots uniformly in ln(lambda), since
# dlambda / lambda = dv / c. A fixed spacing in Angstrom is relatively
# stiffer in the blue and looser in the red, because the same b gives
# lines that widen as lambda grows. Not needed for the A0 star: 250 A is
# far wider than any line over the whole SDSS range.

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

# Cubic spline: smooth up to the second derivative
SPLINE_DEGREE = 3
# Fewest fitted pixels allowed between two knots; below this the cubic
# in that interval is poorly constrained, so the interval is merged.
# Heuristic margin (~2-3x the 4 local cubic coefficients); the exact
# value matters little at the default knot spacing (~125-280 pixels
# per interval on SDSS)
MIN_PIXELS_PER_INTERVAL = 10


@dataclass(frozen=True)
class ContinuumSettings:
    """Parameters of the sigma-clipped spline continuum fit.

    Attributes
    ----------
    knot_spacing : float
        Approximate distance between spline knots, Angstrom. Must be
        wider than the broadest absorption line, or the spline follows
        it; too wide and it cannot follow the continuum shape. Choose it
        by comparing fits with several spacings.
    clip_low : float
        Pixels more than clip_low sigma below the fit are rejected.
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
        Continuum level at every pixel, units of the flux. NaN beyond
        the first and last used pixel, where it is unconstrained.
    error : array
        1-sigma statistical error of `continuum`, correlated between
        neighbouring pixels. NaN where `continuum` is.
    used : array of bool
        Pixels used in the final fit (not clipped).
    n_iter : int
        Fit-and-clip iterations done.
    status : str
        "converged" (the set of used pixels stopped changing) or
        "max_iter" (still changing after max_iter iterations).
    """

    continuum: np.ndarray
    error: np.ndarray
    used: np.ndarray
    n_iter: int
    status: str


def fit_continuum(spectrum, settings=ContinuumSettings()):
    """Fit the continuum of `spectrum` by iterative sigma clipping.

    Fits a weighted cubic spline to the used pixels, rejects those more
    than clip_low sigma below it, and refits until the set of used
    pixels stops changing. Every pixel is tested against each new fit,
    so pixels rejected earlier can come back: the first fits run through
    the lines, below the true continuum, and rise step by step to it.

    Parameters
    ----------
    spectrum : Spectrum
        Wavelengths in Angstrom, strictly increasing; `error` > 0.
    settings : ContinuumSettings

    Returns
    -------
    fit : ContinuumFit
    """
    _check_input(spectrum)
    used, n_iter, status = _clip_until_stable(spectrum, settings)
    # The clipping loop can stop right after updating the mask: refit
    # so that continuum and mask always match
    spline = _weighted_spline(spectrum, used, settings.knot_spacing)
    inside = _inside_used_range(spectrum.wave, used)
    continuum = np.full(spectrum.wave.shape, np.nan)
    continuum[inside] = spline(spectrum.wave[inside])
    error = np.full(spectrum.wave.shape, np.nan)
    error[inside] = _spline_error(spline, spectrum, used, inside)
    return ContinuumFit(continuum, error, used, n_iter, status)


def normalise(spectrum, continuum):
    """Return a copy of `spectrum` divided by `continuum`.

    Parameters
    ----------
    spectrum : Spectrum
    continuum : array
        Continuum level at every pixel, e.g. `ContinuumFit.continuum`.

    Returns
    -------
    normalised : Spectrum
        Flux and error relative to the continuum, which becomes 1. The
        error does not include the uncertainty of the continuum.
    """
    return replace(
        spectrum,
        flux=spectrum.flux / continuum,
        error=spectrum.error / continuum,
        continuum=np.ones_like(continuum),
    )


def _check_input(spectrum):
    """Raise ValueError if the spectrum cannot be fitted.

    Parameters
    ----------
    spectrum : Spectrum
    """
    if np.any(np.diff(spectrum.wave) <= 0):
        raise ValueError("wave must be strictly increasing")
    # Written as "not > 0" so that NaN errors are caught too
    if np.any(~(spectrum.error > 0)):
        raise ValueError("error must be positive at every pixel")


def _clip_until_stable(spectrum, settings):
    """Refit and reclip, starting from all pixels, until the mask stops changing.

    Parameters
    ----------
    spectrum : Spectrum
    settings : ContinuumSettings

    Returns
    -------
    used : array of bool
    n_iter : int
    status : str
        "converged" or "max_iter", see `ContinuumFit`.
    """
    used = np.ones(spectrum.wave.shape, dtype=bool)
    for n_iter in range(1, settings.max_iter + 1):
        new_used = _pixels_not_below_fit(spectrum, settings, used)
        if np.array_equal(new_used, used):
            return used, n_iter, "converged"
        used = new_used
    warnings.warn(
        f"continuum fit not converged after {settings.max_iter} iterations",
        stacklevel=3,
    )
    return used, settings.max_iter, "max_iter"


def _pixels_not_below_fit(spectrum, settings, used):
    """Fit the `used` pixels; True for every pixel not clip_low sigma below the fit.

    The spline is evaluated (extrapolated) at every pixel here, not
    only inside the used range: otherwise pixels clipped at the ends
    would get no continuum to be tested against and never come back.

    Parameters
    ----------
    spectrum : Spectrum
    settings : ContinuumSettings
    used : array of bool

    Returns
    -------
    new_used : array of bool
    """
    spline = _weighted_spline(spectrum, used, settings.knot_spacing)
    deviation = (spectrum.flux - spline(spectrum.wave)) / spectrum.error
    return deviation >= -settings.clip_low


def _weighted_spline(spectrum, used, knot_spacing):
    """Least-squares cubic spline through the `used` pixels, 1/sigma weights.

    Parameters
    ----------
    spectrum : Spectrum
    used : array of bool
        Pixels entering the fit.
    knot_spacing : float
        Angstrom.

    Returns
    -------
    spline : scipy.interpolate.BSpline
        Constrained only inside the range of the used pixels.
    """
    wave = spectrum.wave[used]
    return make_lsq_spline(
        wave,
        spectrum.flux[used],
        _knot_vector(wave, knot_spacing),
        k=SPLINE_DEGREE,
        w=1 / spectrum.error[used],
    )


def _knot_vector(wave, knot_spacing):
    """Full knot vector: evenly spaced, with sparse intervals merged.

    A least-squares spline needs enough fitted pixels between its knots
    (Schoenberg-Whitney condition, and a well-constrained cubic). While
    some interval holds fewer than MIN_PIXELS_PER_INTERVAL pixels (most
    clipped in a broad line), one of its knots is dropped, so that
    interval merges with its neighbour and the spline bridges the gap.

    Parameters
    ----------
    wave : array
        Sorted wavelengths of the fitted pixels, Angstrom.
    knot_spacing : float
        Angstrom.

    Returns
    -------
    knots : array
        Interior knots plus SPLINE_DEGREE + 1 repeated end knots at
        wave[0] and wave[-1].
    """
    first, last = wave[0], wave[-1]
    n_intervals = max(1, round((last - first) / knot_spacing))
    evenly_spaced = np.linspace(first, last, n_intervals + 1)
    interior_knots = list(evenly_spaced[1:-1])
    sparse_interval = _first_sparse_interval(wave, interior_knots)
    while sparse_interval is not None:
        interior_knots.pop(_knot_to_remove(sparse_interval, interior_knots))
        sparse_interval = _first_sparse_interval(wave, interior_knots)
    end_knots = SPLINE_DEGREE + 1
    return np.concatenate([[first] * end_knots, interior_knots, [last] * end_knots])


def _first_sparse_interval(wave, interior_knots):
    """Index of the first interval between knots with too few pixels.

    Parameters
    ----------
    wave : array
        Sorted wavelengths of the fitted pixels, Angstrom.
    interior_knots : list of float

    Returns
    -------
    index : int or None
        0 for the interval from wave[0] to the first interior knot. None
        if every interval holds at least MIN_PIXELS_PER_INTERVAL pixels,
        or if there is no interior knot left.
    """
    if not interior_knots:
        return None
    edges = [wave[0], *interior_knots, wave[-1]]
    pixel_counts, _ = np.histogram(wave, bins=edges)
    for index, pixel_count in enumerate(pixel_counts):
        if pixel_count < MIN_PIXELS_PER_INTERVAL:
            return index
    return None


def _knot_to_remove(sparse_interval, interior_knots):
    """Index, in `interior_knots`, of a knot bounding interval `sparse_interval`.

    Interval i lies between interior knots i-1 and i, so its right edge is
    knot i; the last interval has no right interior knot, so its left edge
    is removed instead.

    Parameters
    ----------
    sparse_interval : int
    interior_knots : list of float

    Returns
    -------
    index : int
    """
    last_knot = len(interior_knots) - 1
    return min(sparse_interval, last_knot)


def _inside_used_range(wave, used):
    """True between the first and last used pixel, inclusive.

    Parameters
    ----------
    wave : array
    used : array of bool

    Returns
    -------
    inside : array of bool
    """
    used_wave = wave[used]
    return (wave >= used_wave[0]) & (wave <= used_wave[-1])


def _spline_error(spline, spectrum, used, inside):
    """1-sigma statistical error of the fitted spline at `inside` pixels.

    The spline is C(lambda) = sum_j c_j B_j(lambda), so its variance is
    B(lambda)^T Cov(c) B(lambda), with B(lambda) the values of the basis
    functions at lambda. Still a lower bound: it ignores the choice of
    knots and which pixels were clipped (compare fits with several knot
    spacings for that systematic).

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
        Fit to the `used` pixels.
    spectrum : Spectrum
    used, inside : array of bool

    Returns
    -------
    error : array
        One value per `inside` pixel, units of the flux.
    """
    covariance = _coefficient_covariance(spline, spectrum, used)
    basis = _basis_values(spline, spectrum.wave[inside])
    variances = []
    for basis_row in basis:
        variances.append(basis_row @ covariance @ basis_row)
    return np.sqrt(variances)


def _coefficient_covariance(spline, spectrum, used):
    """Covariance of the spline coefficients, scaled by the reduced chi^2.

    Least squares gives Cov(c) = (sum_i B_i B_i^T / sigma_i^2)^-1, summed
    over the used pixels i. It is multiplied by the reduced chi^2 when
    that exceeds 1, so that pixel errors that are too small, or a spline
    that cannot follow the continuum, inflate the error.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
    spectrum : Spectrum
    used : array of bool

    Returns
    -------
    covariance : array, shape (n_coefficients, n_coefficients)
    """
    basis = _basis_values(spline, spectrum.wave[used])
    n_coefficients = spline.c.size
    normal_matrix = np.zeros((n_coefficients, n_coefficients))
    for basis_row, pixel_error in zip(basis, spectrum.error[used]):
        normal_matrix += np.outer(basis_row, basis_row) / pixel_error**2
    reduced_chi2 = _reduced_chi2(spline, spectrum, used)
    return max(1.0, reduced_chi2) * np.linalg.pinv(normal_matrix)


def _basis_values(spline, wave):
    """Values of every B-spline basis function at every wavelength.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
    wave : array
        Angstrom.

    Returns
    -------
    basis : array, shape (len(wave), n_coefficients)
        Row i holds B_j(wave[i]) for every coefficient j.
    """
    return BSpline.design_matrix(wave, spline.t, SPLINE_DEGREE).toarray()


def _reduced_chi2(spline, spectrum, used):
    """Chi^2 of the spline fit per degree of freedom, over the used pixels.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
    spectrum : Spectrum
    used : array of bool

    Returns
    -------
    reduced_chi2 : float
    """
    wave = spectrum.wave[used]
    normalised_residuals = (spectrum.flux[used] - spline(wave)) / spectrum.error[used]
    degrees_of_freedom = wave.size - spline.c.size
    return np.sum(normalised_residuals**2) / degrees_of_freedom
