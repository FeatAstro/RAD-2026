"""Centre, depth and equivalent width of one detected feature.

The centre comes from a generic Gaussian fit, which needs no atomic data.
The Voigt fit (N, b, z free) belongs to a later stage, on identified lines.
"""

import numpy as np
from scipy.optimize import curve_fit

from .. import constants
from ..equivalent_width import measure_ew
from ..line_profiles import lsf_sigma_kms
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
        One value per name in FEATURE_COLUMNS. The depth is that of the
        deepest pixel; the EW is over the whole window.
    """
    feature = spectrum.cut(window)
    signal, inverse_variance = relative_signal(feature, sign)
    centre, centre_err = _fit_centre(feature, signal, inverse_variance)
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


# TODO: Gaussian plus linear baseline for windows cut by split_blends (4317 A
# on the wing of H-gamma); split into helpers (over ~20 lines)
def _fit_centre(feature, signal, inverse_variance):
    """Centre of a feature and its 1-sigma error, from a Gaussian fit.

    The width is bounded below by the LSF. The error is scaled by the
    reduced chi^2 when above 1: a Gaussian is a poor shape for damping wings.

    Parameters
    ----------
    feature : Spectrum
        Pixels of the window, with `meta["fwhm_kms"]`.
    signal, inverse_variance : array
        From `relative_signal` on `feature`.

    Returns
    -------
    centre, centre_err : float
        Angstrom. If the fit fails: the deepest pixel and NaN.
    """
    peak = feature.wave[np.argmax(signal)]
    sigma_min = peak * lsf_sigma_kms(feature) / constants.C_KMS
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
