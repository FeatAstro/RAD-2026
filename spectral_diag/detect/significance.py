"""Significance of a feature at each pixel, and the windows of the features.

The signal relative to the continuum is filtered with a Gaussian as wide as
the LSF (matched filter). A window holding several resolved lines is split
at the dips between them.
"""

import numpy as np

from .. import constants
from ..line_profiles import lsf_sigma_kms

# Significance (sigma) where a window ends: low enough to keep the wings of a
# broad line in one window, high enough not to join features through noise
# TODO: calibrate the edge significance (EW kept vs false extension)
WINDOW_EDGE_SIGNIFICANCE = 1.0
# Half-width of the kernel, where a Gaussian is negligible (3e-4 of its peak)
KERNEL_HALF_WIDTH_IN_SIGMAS = 4.0
# Smallest drop of the significance (sigma) between two peaks for separate
# lines. Calibrated on 300 synthetic spectra at SNR 50 (tests/test_detect.py):
# 1% of broad lines wrongly split, 85% of weak pairs 2.5 FWHM apart separated
MIN_SPLIT_DIP = 3.5


def relative_signal(spectrum, sign):
    """Fractional deviation of the flux from the continuum, and its weight.

    Parameters
    ----------
    spectrum : Spectrum
        Uses `flux`, `error` and `continuum` (NaN allowed).
    sign : {-1, +1}
        -1 for absorption, +1 for emission.

    Returns
    -------
    signal : array
        sign * (flux / continuum - 1); 0 where the continuum is NaN.
    inverse_variance : array
        1 / sigma^2 of `signal`; 0 where the continuum is NaN.
    """
    valid = np.isfinite(spectrum.continuum)
    signal = np.where(valid, sign * (spectrum.flux / spectrum.continuum - 1), 0.0)
    # TODO: add the continuum error (needs the covariance), as in `equivalent_width`
    relative_error = spectrum.error / spectrum.continuum
    inverse_variance = np.where(valid, 1 / relative_error**2, 0.0)
    return signal, inverse_variance


def unresolved_line_sigma_in_pixels(spectrum):
    """Sigma of the LSF of the instrument, in pixels.

    Parameters
    ----------
    spectrum : Spectrum
        Uses `wave` and `meta["fwhm_kms"]` (km/s).

    Returns
    -------
    sigma : float
    """
    # Doppler: d(ln wave) = dv / c, so a step in ln(wave) is a fixed velocity
    # step on a log-uniform grid (as SDSS). The median ignores gaps in the data.
    pixel_kms = constants.C_KMS * np.median(np.diff(np.log(spectrum.wave)))
    return lsf_sigma_kms(spectrum) / pixel_kms


def gaussian_kernel(sigma):
    """Gaussian shape of an unresolved line, sampled at whole pixels.

    Parameters
    ----------
    sigma : float
        In pixels.

    Returns
    -------
    kernel : array
        Odd length, peak 1 at the middle element.
    """
    half_width_in_pixels = int(np.ceil(KERNEL_HALF_WIDTH_IN_SIGMAS * sigma))
    pixel_offsets_from_centre = np.arange(
        -half_width_in_pixels, half_width_in_pixels + 1
    )
    return np.exp(-0.5 * (pixel_offsets_from_centre / sigma) ** 2)


def matched_filter_significance(signal, inverse_variance, kernel):
    """Significance, in sigma, of a line with shape `kernel` at each pixel.

    Weighted least-squares amplitude over its error,
    sum(k s w) / sqrt(sum(k^2 w)), with k the kernel, s the signal and w
    the inverse variance.

    Parameters
    ----------
    signal, inverse_variance : array
        From `relative_signal`.
    kernel : array
        From `gaussian_kernel`.

    Returns
    -------
    significance : array
        0 at pixels without weight.
    """
    weighted_sum = _kernel_weighted_sum(signal * inverse_variance, kernel)
    norm = _kernel_weighted_sum(inverse_variance, kernel**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        significance = weighted_sum / np.sqrt(norm)
    return np.where(inverse_variance > 0, significance, 0.0)


def _kernel_weighted_sum(values, kernel):
    """Sum of `values` times `kernel` in a window centred on each pixel.

    Parameters
    ----------
    values : array
        One value per pixel.
    kernel : array
        Window weights, odd length, centred on the pixel.

    Returns
    -------
    sums : array
        Same length as `values`; pixels outside the array count as 0.
    """
    half_width = len(kernel) // 2
    sums = np.zeros(len(values))
    for center in range(len(values)):
        for offset in range(-half_width, half_width + 1):
            pixel = center + offset
            if 0 <= pixel < len(values):
                sums[center] += kernel[offset + half_width] * values[pixel]
    return sums


# TODO: cut windows at gaps in the data (8640-8794 A on the A0 star)
def feature_windows(significance, threshold):
    """Pixel ranges of the features, by two thresholds (hysteresis).

    A window is a run above WINDOW_EDGE_SIGNIFICANCE with a pixel above
    `threshold`.

    Parameters
    ----------
    significance : array
        From `matched_filter_significance`.
    threshold : float
        Detection threshold, in sigma.

    Returns
    -------
    windows : list of slice
        In increasing wavelength.
    """
    runs = _runs_above(significance, WINDOW_EDGE_SIGNIFICANCE)
    return [run for run in runs if significance[run].max() > threshold]


def _runs_above(values, level):
    """Slices of the maximal runs of consecutive values above `level`.

    Parameters
    ----------
    values : array
    level : float

    Returns
    -------
    runs : list of slice
        In increasing index.
    """
    runs = []
    start = None
    for index, value in enumerate(values):
        if value > level and start is None:
            start = index
        elif value <= level and start is not None:
            runs.append(slice(start, index))
            start = None
    if start is not None:
        runs.append(slice(start, len(values)))
    return runs


def split_blends(significance, window, threshold):
    """Split a feature window into one window per resolved line.

    Cuts at the lowest point between two peaks above `threshold` when the
    significance drops by at least MIN_SPLIT_DIP (e.g. NaI D). Lines closer
    than the resolution have no such dip and stay together.

    Parameters
    ----------
    significance : array
        From `matched_filter_significance`.
    window : slice
        One window from `feature_windows`.
    threshold : float
        Detection threshold, in sigma: only peaks above it are split.

    Returns
    -------
    windows : list of slice
        In increasing wavelength, covering `window` without overlap.
    """
    peaks = _peaks_above(significance, window, threshold)
    cuts = _cut_positions(significance, peaks)
    starts = [window.start] + cuts
    stops = cuts + [window.stop]
    return [slice(start, stop) for start, stop in zip(starts, stops)]


def _peaks_above(significance, window, level):
    """Local maxima of the significance inside a window, above a level.

    Parameters
    ----------
    significance : array
    window : slice
    level : float

    Returns
    -------
    peaks : list of int
        Pixel indices in `significance`, increasing. A maximum is higher
        than its left neighbour and at least as high as its right one;
        the first and last pixels of the window are not considered.
    """
    peaks = []
    for index in range(window.start + 1, window.stop - 1):
        value = significance[index]
        is_maximum = (
            value > significance[index - 1] and value >= significance[index + 1]
        )
        if is_maximum and value > level:
            peaks.append(index)
    return peaks


def _cut_positions(significance, peaks):
    """Pixels where to cut a window so that each part holds one line.

    Walks the peaks left to right. If the significance drops by at least
    MIN_SPLIT_DIP between the current peak and the next, cuts at the lowest
    point between them; otherwise keeps the higher one as the current peak.

    Parameters
    ----------
    significance : array
    peaks : list of int
        From `_peaks_above`.

    Returns
    -------
    cuts : list of int
        Pixel indices, increasing.
    """
    cuts = []
    current = peaks[0] if peaks else None
    for following in peaks[1:]:
        valley = current + int(np.argmin(significance[current : following + 1]))
        lower_peak = min(significance[current], significance[following])
        if lower_peak - significance[valley] >= MIN_SPLIT_DIP:
            cuts.append(valley)
            current = following
        elif significance[following] > significance[current]:
            current = following
    return cuts
