"""Significance of a feature at each pixel, and the windows of the features.

The spectrum is converted to a signal relative to the continuum, and a
Gaussian as wide as the LSF is slid along it (matched filter). The
features are the groups of pixels above a detection threshold. A group
holding several resolved lines is split at the dips between them.
"""

import numpy as np

from ..constants import C_KMS, FWHM_PER_SIGMA

# Significance (in sigma) below which the window of a feature ends. The
# window grows from the detected pixels while the significance stays above
# it: low enough to keep the wings of a broad line in one window, high
# enough that noise or a slightly low continuum does not join features
# TODO: calibrate the edge significance (EW kept vs false extension, A0 blends)
WINDOW_EDGE_SIGNIFICANCE = 1.0
# The matched-filter kernel is cut at this many sigma, where a Gaussian
# is already negligible (3e-4 of its peak)
KERNEL_HALF_WIDTH_SIGMA = 4.0
# Smallest drop of the significance (in sigma) between two peaks for them to
# be separate lines. Calibrated on 300 synthetic spectra at SNR 50 (as in
# tests/test_detect.py): a single broad line is wrongly split in 21% of them
# at 2, 11% at 2.5, 3.7% at 3, 1% at 3.5 and 0% at 4, while two weak narrow
# lines 6 A (2.5 FWHM) apart are separated in 94%, 93%, 90%, 85% and 78%.
# 3.5 keeps most pairs while rarely cutting a broad line into pieces
MIN_SPLIT_DIP = 3.5


def check_input(spectrum):
    """Raise ValueError if the features cannot be searched for.

    A pixel with zero error would get infinite weight, so noise-free
    spectra (e.g. make_synthetic with snr=None) are rejected. Without
    the LSF width the filter has no shape.

    Parameters
    ----------
    spectrum : Spectrum
        Checks `error` and `meta["fwhm_kms"]`.
    """
    if np.any(~(spectrum.error > 0)):
        raise ValueError("error must be positive at every pixel")
    if "fwhm_kms" not in spectrum.meta:
        raise ValueError('meta["fwhm_kms"] (LSF FWHM in km/s) is required')


def relative_signal(spectrum, sign):
    """Fractional deviation of the flux from the continuum, and its weight.

    For absorption, signal = 1 - flux / continuum: 0 on the continuum,
    0.7 where a line absorbs 70% of the light. For emission the sign is
    flipped, so features are always positive.

    Parameters
    ----------
    spectrum : Spectrum
        Uses `flux`, `error` and `continuum` (NaN allowed).
    sign : {-1, +1}
        -1 for absorption, +1 for emission.

    Returns
    -------
    signal : array
        sign * (flux / continuum - 1), dimensionless. Where the continuum
        is NaN, the signal is 0.
    inverse_variance : array
        1 / sigma^2, with sigma = error / continuum the noise of
        `signal`. Weight of each pixel in the filter; 0 where the
        continuum is NaN, so those pixels are ignored.
    """
    valid = np.isfinite(spectrum.continuum)
    # Placeholder so that the division is defined; NaN pixels are zeroed below
    continuum = np.where(valid, spectrum.continuum, 1.0)
    signal = np.where(valid, sign * (spectrum.flux / continuum - 1), 0.0)
    # TODO: add the continuum error (needs the covariance), as in `equivalent_width`
    relative_error = spectrum.error / continuum
    inverse_variance = np.where(valid, 1 / relative_error**2, 0.0)
    return signal, inverse_variance


def lsf_sigma_pixels(spectrum):
    """Width, in pixels, of an unresolved line (sigma of the LSF).

    Converts the LSF from km/s to pixels. The pixel width in km/s is
    the median step of ln(wave) times c, since d(ln wave) = dv / c.

    Parameters
    ----------
    spectrum : Spectrum
        Uses `wave` and `meta["fwhm_kms"]`, the FWHM of the LSF in km/s.

    Returns
    -------
    sigma : float
        Gaussian sigma of the LSF, in pixels.
    """
    pixel_kms = C_KMS * np.median(np.diff(np.log(spectrum.wave)))
    return spectrum.meta["fwhm_kms"] / FWHM_PER_SIGMA / pixel_kms


def gaussian_kernel(sigma):
    """Gaussian shape of an unresolved line, sampled at whole pixels.

    Parameters
    ----------
    sigma : float
        Gaussian sigma, in pixels.

    Returns
    -------
    kernel : array
        Values at pixel offsets -n ... +n from the centre, with
        n = ceil(KERNEL_HALF_WIDTH_SIGMA * sigma) (at least 1), so the
        length is odd and the peak is the middle element. Peak value 1;
        the normalisation cancels in the significance.
    """
    half_width = max(1, int(np.ceil(KERNEL_HALF_WIDTH_SIGMA * sigma)))
    offsets = np.arange(-half_width, half_width + 1)
    return np.exp(-0.5 * (offsets / sigma) ** 2)


def matched_filter_significance(signal, inverse_variance, kernel):
    """Significance, in sigma, of a line with shape `kernel` at each pixel.

    At each pixel, the kernel is fitted to the nearby signal by weighted
    least squares. The best-fit amplitude is sum(k s w) / sum(k^2 w) and
    its error 1 / sqrt(sum(k^2 w)), with k the kernel, s the signal and
    w the inverse variance. The significance is amplitude / error:
    sum(k s w) / sqrt(sum(k^2 w)). Near the edges and NaN pixels the
    sums run over the pixels available.

    Parameters
    ----------
    signal, inverse_variance : array
        From `relative_signal`.
    kernel : array
        From `gaussian_kernel`. Symmetric, so convolution and
        correlation coincide.

    Returns
    -------
    significance : array
        Same shape as `signal`; 0 at pixels without weight, so that no
        feature window includes them.
    """
    weighted_sum = np.convolve(signal * inverse_variance, kernel, mode="same")
    norm = np.convolve(inverse_variance, kernel**2, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        significance = weighted_sum / np.sqrt(norm)
    return np.where(inverse_variance > 0, significance, 0.0)


def feature_windows(significance, threshold):
    """Pixel ranges of the features, by two thresholds (hysteresis).

    A feature is detected where the significance exceeds `threshold`,
    and extends in both directions while it stays above
    WINDOW_EDGE_SIGNIFICANCE. This keeps a broad line with its wings as
    one window instead of splitting it where the noise dips.

    Parameters
    ----------
    significance : array
        From `matched_filter_significance`.
    threshold : float
        Detection threshold, in sigma.

    Returns
    -------
    windows : list of slice
        One per feature, in increasing wavelength: a run of pixels above
        WINDOW_EDGE_SIGNIFICANCE with at least one pixel above `threshold`.
    """
    runs = _runs_above(significance, WINDOW_EDGE_SIGNIFICANCE)
    return [run for run in runs if significance[run].max() > threshold]


def _runs_above(values, level):
    """Runs of consecutive elements above a level.

    Parameters
    ----------
    values : array
    level : float

    Returns
    -------
    runs : list of slice
        One per maximal run of consecutive elements with value > `level`,
        in increasing index.
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

    The hysteresis of `feature_windows` keeps two lines in one window
    when the significance between them stays above
    WINDOW_EDGE_SIGNIFICANCE (e.g. the NaI D doublet, or CaII K on the
    wing of H-epsilon). Here the window is cut at the lowest point
    between two peaks above `threshold` when the significance drops by
    at least MIN_SPLIT_DIP. Lines closer than the resolution have no
    such dip and stay together.

    Parameters
    ----------
    significance : array
        From `matched_filter_significance`.
    window : slice
        One window from `feature_windows`.
    threshold : float
        Detection threshold, in sigma: only peaks above it can be split.

    Returns
    -------
    windows : list of slice
        In increasing wavelength, covering `window` without overlap; the
        lowest pixel between two lines starts the right-hand window.
        `[window]` if there is nothing to split.
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

    Walks the peaks from left to right. If the significance drops by at
    least MIN_SPLIT_DIP between the current peak and the next, the window
    is cut at the lowest point between them and the next peak becomes
    the current one. Otherwise both are the same line, and the higher
    peak is kept as the current one.

    Parameters
    ----------
    significance : array
    peaks : list of int
        From `_peaks_above`.

    Returns
    -------
    cuts : list of int
        Pixel indices, increasing; empty if there is nothing to cut.
    """
    cuts = []
    current = peaks[0] if peaks else None
    for following in peaks[1:]:
        valley = _lowest_between(significance, current, following)
        lower_peak = min(significance[current], significance[following])
        if lower_peak - significance[valley] >= MIN_SPLIT_DIP:
            cuts.append(valley)
            current = following
        elif significance[following] > significance[current]:
            current = following
    return cuts


def _lowest_between(significance, left, right):
    """Pixel with the lowest significance between two pixels (included).

    Parameters
    ----------
    significance : array
    left, right : int
        Pixel indices, left < right.

    Returns
    -------
    index : int
    """
    return left + int(np.argmin(significance[left : right + 1]))
