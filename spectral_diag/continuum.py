"""Continuum fitting and normalisation.

- fit_continuum: iterative, asymmetrically sigma-clipped cubic spline
  fit to the unabsorbed flux level. Needs no line list: the pixels
  clipped below the fit mark candidate absorption (plus a few noise
  pixels)
- normalise: divide a Spectrum's flux/error by the fitted continuum
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

# Cubic spline: smooth up to the second derivative
SPLINE_DEGREE = 3


@dataclass(frozen=True)
class ContinuumSettings:
    """Parameters of the sigma-clipped spline continuum fit.

    Attributes
    ----------
    knot_spacing : float
        Distance between spline knots, Angstrom. Must be wider than the
        broadest absorption line, or the spline follows it; too wide
        and it cannot follow the continuum shape. Choose it by comparing
        fits with several spacings.
    clip_low, clip_high : float
        Pixels more than clip_low sigma below (and clip_high sigma above)
        the fit are rejected. Absorption only removes flux, so by
        default only the cut below is applied. A finite clip_high
        catches cosmic rays and sky residuals.
    max_iter : int
        Maximum number of fit-and-clip iterations; the loop stops
        earlier once the set of rejected pixels no longer changes.
    """

    knot_spacing: float = 150.0
    clip_low: float = 2.0
    clip_high: float = np.inf
    max_iter: int = 20


def fit_continuum(spectrum, settings=ContinuumSettings()):
    """Fit the continuum of `spectrum` by iterative sigma clipping.

    Fits a weighted cubic spline to all pixels, rejects those outside
    [-clip_low, +clip_high] sigma of it, and refits until the set of
    rejected pixels converges. At each iteration every pixel is tested
    against the new fit, so pixels rejected earlier can come back.
    Warns if it has not converged after `settings.max_iter` iterations.

    Parameters
    ----------
    spectrum : Spectrum
        Wavelengths in Angstrom; `error` > 0.
    settings : ContinuumSettings

    Returns
    -------
    continuum : array
        Continuum level at every pixel, units of `spectrum.flux`. NaN
        beyond the first and last used pixel, where it is unconstrained.
    used : array of bool
        Pixels used in the last fit (not clipped).
    """
    used = np.ones(spectrum.wave.shape, dtype=bool)
    for _ in range(settings.max_iter):
        continuum = _weighted_spline(spectrum, used, settings.knot_spacing)
        # Every pixel is re-tested against the new fit, not only those
        # still used: the fit changes between iterations (it rises out of
        # the lines but can also drop locally), so a pixel rejected by an
        # earlier fit can be valid for the new one. The price: unlike a
        # mask that only shrinks, convergence is not guaranteed, hence
        # max_iter and the warning below
        new_used = _within_clip(spectrum, continuum, settings)
        changed = np.count_nonzero(new_used != used)
        if changed == 0:
            break
        used = new_used
    else:
        warnings.warn(
            f"continuum fit not converged after {settings.max_iter} iterations: "
            f"{changed} pixels still change",
            stacklevel=2,
        )
    return continuum, used


def normalise(spectrum, continuum):
    """Return a copy of `spectrum` divided by `continuum`.

    Parameters
    ----------
    spectrum : Spectrum
    continuum : array
        Continuum level at every pixel, e.g. from `fit_continuum`.

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
    continuum : array
        Spline evaluated at every pixel of `spectrum`; NaN outside the
        range of the used pixels.
    """
    wave = spectrum.wave[used]
    spline = LSQUnivariateSpline(
        wave,
        spectrum.flux[used],
        _interior_knots(wave, knot_spacing),
        w=1 / spectrum.error[used],
        k=SPLINE_DEGREE,
    )
    # A spline extrapolated past its data diverges quickly: leave the
    # continuum undefined there instead
    inside = (spectrum.wave >= wave[0]) & (spectrum.wave <= wave[-1])
    return np.where(inside, spline(spectrum.wave), np.nan)


def _interior_knots(wave, knot_spacing):
    """Evenly spaced knots, dropping those that end an empty interval.

    A least-squares spline needs fitted pixels between its knots
    (Schoenberg-Whitney condition); otherwise the fit is singular. A
    knot whose preceding interval has no fitted pixel (all clipped in a
    broad line) is dropped, so that interval merges with the next one
    and the spline bridges the gap.

    Parameters
    ----------
    wave : array
        Sorted wavelengths of the fitted pixels, Angstrom.
    knot_spacing : float
        Angstrom.

    Returns
    -------
    knots : array
        Strictly inside (wave[0], wave[-1]).
    """
    edges = np.arange(wave[0], wave[-1], knot_spacing)
    pixels_per_interval = np.diff(np.searchsorted(wave, edges))
    knots = edges[1:]
    return knots[pixels_per_interval > 0]


def _within_clip(spectrum, continuum, settings):
    """True for pixels within [-clip_low, +clip_high] sigma of `continuum`.

    Parameters
    ----------
    spectrum : Spectrum
    continuum : array
    settings : ContinuumSettings

    Returns
    -------
    mask : array of bool
    """
    deviation = (spectrum.flux - continuum) / spectrum.error
    return (deviation >= -settings.clip_low) & (deviation <= settings.clip_high)
