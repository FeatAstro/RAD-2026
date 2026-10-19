"""Continuum fitting and normalisation.

- fit_continuum: iterative, asymmetrically sigma-clipped cubic spline
  fit to the unabsorbed flux level, with its statistical error. Needs
  no line list: the pixels clipped below the fit mark candidate
  absorption (plus a few noise pixels)
- normalise: divide a Spectrum's flux/error by the fitted continuum

Written for absorption spectra: pixels far below the fit are treated
as lines, pixels far above it as glitches (cosmic rays, sky residuals).
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

# Cubic spline: smooth up to the second derivative
SPLINE_DEGREE = 3
# Fewest fitted pixels allowed between two knots; below this the cubic
# in that interval is poorly constrained, so the interval is merged
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
    clip_low, clip_high : float
        Pixels more than clip_low sigma below (or clip_high sigma above)
        the fit are rejected. The upper cut is only applied once the
        lower one alone has converged.
    grow_width : float
        Pixels within this distance of a rejected pixel are rejected
        too, Angstrom. Removes line wings too shallow to be clipped on
        their own. In Angstrom rather than pixels because SDSS pixels
        are uniform in log(lambda), not in lambda.
    max_iter : int
        Maximum number of fit-and-clip iterations per clipping stage.
    """

    knot_spacing: float = 150.0
    clip_low: float = 2.0
    clip_high: float = 4.0
    grow_width: float = 1.0
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
        Fit-and-clip iterations done, both stages together.
    status : str
        "converged" (mask stable), "cycling" (mask alternating between
        states that differ by a few pixels at a threshold; harmless) or
        "max_iter" (still changing).
    """

    continuum: np.ndarray
    error: np.ndarray
    used: np.ndarray
    n_iter: int
    status: str


def fit_continuum(spectrum, settings=ContinuumSettings()):
    """Fit the continuum of `spectrum` by iterative sigma clipping.

    Fits a weighted cubic spline to the used pixels, rejects those
    outside [-clip_low, +clip_high] sigma of it (and their neighbours
    within grow_width), and refits until the set of used pixels stops
    changing. Every pixel is tested against each new fit, so pixels
    rejected earlier can come back.

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
    used, n_iter, status = _clip_in_two_stages(spectrum, settings)
    # The clipping loop can stop right after updating the mask: refit
    # so that continuum and mask always match
    spline = _weighted_spline(spectrum, used, settings.knot_spacing)
    inside = _inside_used_range(spectrum.wave, used)
    continuum = np.where(inside, spline(spectrum.wave), np.nan)
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
    if np.any(~(spectrum.error > 0)):
        raise ValueError("error must be positive at every pixel")


def _clip_in_two_stages(spectrum, settings):
    """Clip below the fit only, then on both sides, until stable.

    On the first iterations the lines drag the fit down, so real
    continuum pixels can sit more than clip_high above it and would be
    thrown away as glitches; the upper cut waits until the fit has
    risen to the continuum.

    Parameters
    ----------
    spectrum : Spectrum
    settings : ContinuumSettings

    Returns
    -------
    used : array of bool
    n_iter : int
        Iterations of both stages together.
    status : str
        Status of the second stage, see `ContinuumFit`.
    """
    lower_only = replace(settings, clip_high=np.inf)
    used = np.ones(spectrum.wave.shape, dtype=bool)
    used, first_iter, _ = _clip_until_stable(spectrum, lower_only, used)
    used, second_iter, status = _clip_until_stable(spectrum, settings, used)
    return used, first_iter + second_iter, status


def _clip_until_stable(spectrum, settings, used):
    """Refit and reclip from mask `used` until the mask stops changing.

    Unlike a mask that only shrinks, re-testing every pixel does not
    guarantee convergence: a few pixels at a threshold can flip back
    and forth, so a repeated mask also ends the loop.

    Parameters
    ----------
    spectrum : Spectrum
    settings : ContinuumSettings
    used : array of bool
        Starting mask; not modified.

    Returns
    -------
    used : array of bool
    n_iter : int
    status : str
        "converged", "cycling" or "max_iter", see `ContinuumFit`.
    """
    previous_masks = []
    for n_iter in range(1, settings.max_iter + 1):
        new_used = _clip_against_fit(spectrum, settings, used)
        if np.array_equal(new_used, used):
            return used, n_iter, "converged"
        if any(np.array_equal(new_used, mask) for mask in previous_masks):
            return new_used, n_iter, "cycling"
        previous_masks.append(used)
        used = new_used
    warnings.warn(
        f"continuum fit not converged after {settings.max_iter} iterations",
        stacklevel=3,
    )
    return used, settings.max_iter, "max_iter"


def _clip_against_fit(spectrum, settings, used):
    """Fit the `used` pixels and return the new mask of accepted pixels.

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
    rejected = (deviation < -settings.clip_low) | (deviation > settings.clip_high)
    return ~_grow_mask(spectrum.wave, rejected, settings.grow_width)


def _grow_mask(wave, mask, width):
    """True for pixels within `width` of a True pixel of `mask`.

    Parameters
    ----------
    wave : array
        Sorted wavelengths, Angstrom.
    mask : array of bool
    width : float
        Angstrom.

    Returns
    -------
    grown : array of bool
    """
    marked_wave = wave[mask]
    if marked_wave.size == 0:
        return mask
    next_marked = np.searchsorted(marked_wave, wave).clip(max=marked_wave.size - 1)
    previous_marked = (next_marked - 1).clip(min=0)
    distance = np.minimum(
        np.abs(marked_wave[next_marked] - wave),
        np.abs(wave - marked_wave[previous_marked]),
    )
    return distance <= width


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
    clipped in a broad line), its bounding knot is dropped, so that
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
    interior = list(np.linspace(first, last, n_intervals + 1)[1:-1])
    while interior:
        counts = np.histogram(wave, bins=[first, *interior, last])[0]
        sparse = np.flatnonzero(counts < MIN_PIXELS_PER_INTERVAL)
        if sparse.size == 0:
            break
        interior.pop(min(sparse[0], len(interior) - 1))
    end_knots = SPLINE_DEGREE + 1
    return np.r_[[first] * end_knots, interior, [last] * end_knots]


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

    Propagates the coefficient covariance: var C(lambda) =
    B(lambda)^T Cov(c) B(lambda), with B the B-spline basis. Cov(c) is
    scaled by the reduced chi^2 when it exceeds 1, so that pixel errors
    that are too small or a spline that cannot follow the continuum
    inflate the error. Still a lower bound: it ignores the choice of
    knots and which pixels were clipped (compare fits with several
    knot spacings for that systematic).

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
    basis = BSpline.design_matrix(spectrum.wave[inside], spline.t, SPLINE_DEGREE)
    return np.sqrt(np.sum((basis @ covariance) * basis.toarray(), axis=1))


def _coefficient_covariance(spline, spectrum, used):
    """Covariance of the spline coefficients, (B^T W B)^-1 times max(1, chi2_nu).

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
    spectrum : Spectrum
    used : array of bool

    Returns
    -------
    covariance : array, shape (n_coefficients, n_coefficients)
    """
    wave, error = spectrum.wave[used], spectrum.error[used]
    basis = BSpline.design_matrix(wave, spline.t, SPLINE_DEGREE).toarray()
    weighted_basis = basis / error[:, None]
    normalised_residuals = (spectrum.flux[used] - spline(wave)) / error
    degrees_of_freedom = wave.size - spline.c.size
    reduced_chi2 = np.sum(normalised_residuals**2) / degrees_of_freedom
    return max(1.0, reduced_chi2) * np.linalg.pinv(weighted_basis.T @ weighted_basis)
