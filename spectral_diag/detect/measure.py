"""Centre, depth and equivalent width of one detected feature.

The centre of a feature comes from a generic Gaussian fit, which needs no
atomic data. The physical Voigt fit (optical_depth + InstrumentGrid, with
N, b, z free) belongs to a later stage, on the identified lines.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import curve_fit

from ..constants import C_KMS, FWHM_PER_SIGMA
from ..equivalent_width import measure_ew
from .significance import relative_signal


def measure_feature(spectrum, window, sign):
    """Centre, window, depth and equivalent width of one feature.

    Parameters
    ----------
    spectrum : Spectrum
        The whole spectrum, with `continuum` and `meta["fwhm_kms"]`.
    window : slice
        Pixels of the feature.
    sign : {-1, +1}
        -1 for absorption, +1 for emission.

    Returns
    -------
    row : dict
        One value per name in FEATURE_COLUMNS, see `spectral_diag.detect`.
        The centre comes from a Gaussian fit (see `_fit_centre`), the depth
        is that of the deepest pixel, the EW is over the whole window.
    """
    feature = _cut(spectrum, window)
    signal, _ = relative_signal(feature, sign)
    centre, centre_err = _fit_centre(feature, sign)
    raw_ew, ew_err = measure_ew(
        feature.wave, feature.flux, feature.continuum, feature.error
    )
    # measure_ew is positive for absorption; flip it so emission is positive too
    ew = -sign * raw_ew
    return {
        "wave": centre,
        "wave_err": centre_err,
        "wave_min": feature.wave[0],
        "wave_max": feature.wave[-1],
        "depth": signal.max(),
        "ew": ew,
        "ew_err": ew_err,
        "snr": ew / ew_err,
    }


def _cut(spectrum, window):
    """The pixels `window` of `spectrum`, as a new Spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        With `continuum` set.
    window : slice
        Pixels to keep.

    Returns
    -------
    cut : Spectrum
        `wave`, `flux`, `error` and `continuum` restricted to `window`;
        `meta` is shared with `spectrum`.
    """
    return replace(
        spectrum,
        wave=spectrum.wave[window],
        flux=spectrum.flux[window],
        error=spectrum.error[window],
        continuum=spectrum.continuum[window],
    )


def _gaussian(wave, amplitude, centre, sigma):
    """Gaussian profile.

    Parameters
    ----------
    wave : array
        Angstrom.
    amplitude : float
        Value at the centre.
    centre, sigma : float
        Position and standard deviation, Angstrom.

    Returns
    -------
    profile : array
    """
    return amplitude * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)


def _fit_centre(feature, sign):
    """Centre of a feature and its 1-sigma error, from a Gaussian fit.

    Fits amplitude, centre and width to the signal of the whole window by
    weighted least squares. The width cannot be smaller than the LSF, the
    narrowest a line can look. The error is that of the centre in the fit
    covariance, scaled by the reduced chi^2 when it exceeds 1 (a Gaussian
    is a poor shape for a broad line with damping wings).

    Parameters
    ----------
    feature : Spectrum
        Only the pixels of the feature (its window), with `continuum` and
        `meta["fwhm_kms"]`.
    sign : {-1, +1}
        -1 for absorption, +1 for emission.

    Returns
    -------
    centre, centre_err : float
        Angstrom. If the fit fails: the wavelength of the deepest pixel
        and NaN.
    """
    signal, inverse_variance = relative_signal(feature, sign)
    peak = feature.wave[np.argmax(signal)]
    sigma_min = peak * feature.meta["fwhm_kms"] / FWHM_PER_SIGMA / C_KMS
    width = feature.wave[-1] - feature.wave[0]
    lower = [0.0, feature.wave[0], sigma_min]
    upper = [np.inf, feature.wave[-1], max(width, 2 * sigma_min)]
    try:
        params, covariance = curve_fit(
            _gaussian,
            feature.wave,
            signal,
            p0=[signal.max(), peak, sigma_min],
            sigma=1 / np.sqrt(inverse_variance),
            absolute_sigma=True,
            bounds=(lower, upper),
        )
    except RuntimeError:
        return peak, np.nan
    residuals = (signal - _gaussian(feature.wave, *params)) * np.sqrt(inverse_variance)
    degrees_of_freedom = signal.size - len(params)
    reduced_chi2 = (
        np.sum(residuals**2) / degrees_of_freedom if degrees_of_freedom > 0 else 1.0
    )
    return params[1], np.sqrt(covariance[1, 1] * max(1.0, reduced_chi2))
