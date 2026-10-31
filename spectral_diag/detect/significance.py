"""Significance of a feature at each pixel, and the windows of the features.

The spectrum is converted to a signal relative to the continuum, and a
Gaussian as wide as the LSF is slid along it (matched filter). The
features are the groups of pixels above a detection threshold.
"""

import numpy as np
from scipy import ndimage

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
    runs, _ = ndimage.label(significance > WINDOW_EDGE_SIGNIFICANCE)
    windows = ndimage.find_objects(runs)
    return [window[0] for window in windows if significance[window].max() > threshold]
