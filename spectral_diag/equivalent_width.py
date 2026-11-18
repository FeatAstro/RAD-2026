"""Equivalent width measurement with error propagation.

The EW error has two parts, added in quadrature:
- the flux noise, independent from pixel to pixel;
- the error of the fitted continuum (`ContinuumFit.error`). The spline
  continuum varies over the knot spacing (~250 Angstrom), much wider than
  a line window, so its error is taken as fully correlated over the
  window: the whole continuum under the line moves up or down together.
"""

import numpy as np


def measure_ew(
    wave,
    flux,
    continuum,
    error=None,
    continuum_error=None,
    wave_min=None,
    wave_max=None,
):
    """Equivalent width of a line, integrated over a wavelength window.

    W = integral of (1 - flux/continuum) dwave, via the trapezoidal
    rule. A line that fully blocks all light over a width Δlambda
    gives W = Δlambda (hence "equivalent" width).

    Parameters
    ----------
    wave, flux, continuum : array
        Observed wavelength (Angstrom), flux and fitted continuum,
        already restricted to the line region unless wave_min/wave_max
        are given.
    error : array, optional
        Per-pixel flux uncertainty (same units as flux), independent
        from pixel to pixel.
    continuum_error : array, optional
        1-sigma uncertainty of the continuum (same units as flux), e.g.
        `ContinuumFit.error`. Taken as fully correlated over the window.
    wave_min, wave_max : float, optional
        If given, restrict the integral to this wavelength range
        before integrating (inclusive).

    Returns
    -------
    W : float
        Equivalent width, Angstrom. Positive for an absorption line.
    W_err : float or None
        1-sigma uncertainty on W from the flux noise and the continuum
        error added in quadrature, Angstrom, or None if neither `error`
        nor `continuum_error` was given.
    """
    inside = _inside_window(wave, wave_min, wave_max)
    wave = _select(wave, inside)
    flux = _select(flux, inside)
    continuum = _select(continuum, inside)
    error = _select(error, inside)
    continuum_error = _select(continuum_error, inside)

    equivalent_width = np.trapezoid(1 - flux / continuum, wave)
    if error is None and continuum_error is None:
        return equivalent_width, None

    weights = _trapezoid_weights(wave)
    variance = 0.0
    if error is not None:
        variance += noise_ew_error(weights, continuum, error) ** 2
    if continuum_error is not None:
        variance += continuum_ew_error(weights, flux, continuum, continuum_error) ** 2
    return equivalent_width, np.sqrt(variance)


def noise_ew_error(weights, continuum, error):
    """EW error due to the flux noise, independent from pixel to pixel.

    dW/dflux_i = -weight_i / continuum_i, summed in quadrature.

    Parameters
    ----------
    weights : array
        Trapezoid weights of the pixels, Angstrom.
    continuum, error : array
        Continuum and flux error, units of the flux.

    Returns
    -------
    W_err : float
        Angstrom.
    """
    return np.sqrt(np.sum((weights * error / continuum) ** 2))


def continuum_ew_error(weights, flux, continuum, continuum_error):
    """EW error due to the continuum error, fully correlated over the window.

    dW/dcontinuum_i = weight_i * flux_i / continuum_i^2. A fully correlated
    error shifts every pixel by continuum_error_i at once, so the
    contributions add linearly, not in quadrature. For a flat continuum
    this gives (Δlambda - W) * continuum_error / continuum.

    Parameters
    ----------
    weights : array
        Trapezoid weights of the pixels, Angstrom.
    flux, continuum, continuum_error : array
        Units of the flux.

    Returns
    -------
    W_err : float
        Angstrom.
    """
    return abs(np.sum(weights * flux * continuum_error / continuum**2))


def _inside_window(wave, wave_min, wave_max):
    """Pixels with wave_min <= wave <= wave_max (a missing bound is no limit).

    Parameters
    ----------
    wave : array
        Angstrom.
    wave_min, wave_max : float or None
        Angstrom.

    Returns
    -------
    inside : array of bool
    """
    wave = np.asarray(wave, dtype=float)
    inside = np.ones(len(wave), dtype=bool)
    if wave_min is not None:
        inside &= wave >= wave_min
    if wave_max is not None:
        inside &= wave <= wave_max
    return inside


def _select(values, inside):
    """The pixels `inside` of `values` as a float array, or None if None.

    Parameters
    ----------
    values : array or None
    inside : array of bool

    Returns
    -------
    selected : array or None
    """
    if values is None:
        return None
    return np.asarray(values, dtype=float)[inside]


def _trapezoid_weights(wave):
    """Per-pixel weights such that sum(weights * y) == np.trapezoid(y, wave).

    Each pixel gets half the distance between its two neighbours; the
    first and last pixels, half the distance to their only neighbour.

    Parameters
    ----------
    wave : array
        Increasing wavelengths, Angstrom.

    Returns
    -------
    weights : array
        Angstrom.
    """
    weights = np.zeros(len(wave))
    last = len(wave) - 1
    weights[0] = (wave[1] - wave[0]) / 2
    weights[last] = (wave[last] - wave[last - 1]) / 2
    for pixel in range(1, last):
        weights[pixel] = (wave[pixel + 1] - wave[pixel - 1]) / 2
    return weights
