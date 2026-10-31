"""Equivalent width measurement with error propagation.

Planned contents:
- measure_ew: integrate (1 - flux/continuum) over a line window to
  get the equivalent width and its uncertainty
"""

import numpy as np


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


def measure_ew(wave, flux, continuum, error=None, wave_min=None, wave_max=None):
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
        Per-pixel flux uncertainty (same units as flux). If given,
        the EW uncertainty is propagated assuming independent,
        continuum-free-of-error pixel noise.
    wave_min, wave_max : float, optional
        If given, restrict the integral to this wavelength range
        before integrating (inclusive).

    Returns
    -------
    W : float
        Equivalent width, Angstrom. Positive for an absorption line.
    W_err : float or None
        1-sigma uncertainty on W, Angstrom, or None if `error` was
        not given.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    continuum = np.asarray(continuum, dtype=float)

    if wave_min is not None or wave_max is not None:
        mask = np.ones_like(wave, dtype=bool)
        if wave_min is not None:
            mask &= wave >= wave_min
        if wave_max is not None:
            mask &= wave <= wave_max
        wave, flux, continuum = wave[mask], flux[mask], continuum[mask]
        if error is not None:
            error = np.asarray(error, dtype=float)[mask]

    equivalent_width = np.trapezoid(1 - flux / continuum, wave)

    if error is None:
        return equivalent_width, None

    error = np.asarray(error, dtype=float)
    weights = _trapezoid_weights(wave)
    variance = np.sum((weights * error / continuum) ** 2)
    return equivalent_width, np.sqrt(variance)
