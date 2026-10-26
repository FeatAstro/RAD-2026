"""Detection of the significant features of a spectrum."""

from astropy.table import Table

from .measure import measure_feature
from .significance import (
    check_input,
    feature_windows,
    gaussian_kernel,
    lsf_sigma_pixels,
    matched_filter_significance,
    relative_signal,
)

FEATURE_COLUMNS = (
    "wave",
    "wave_err",
    "wave_min",
    "wave_max",
    "depth",
    "ew",
    "ew_err",
    "snr",
)


def find_features(spectrum, threshold=4.0, sign=-1):
    """Find the significant absorption (or emission) features of a spectrum.

    Steps:
    1. Convert the flux to a signal relative to the continuum.
    2. Slide along the spectrum a Gaussian as wide as the instrument's
       LSF (the shape of an unresolved line). At each pixel this gives
       the significance, in sigma, of a line centred there.
    3. Each group of neighbouring pixels above WINDOW_EDGE_SIGNIFICANCE
       that contains a pixel above `threshold` is one feature, so a
       broad line with its wings comes out as one feature.
    4. Measure the centre, depth and equivalent width of each feature.

    The window is meant for detecting and locating the feature. For a
    broad line it misses the far damping wings, whose depth is below
    the noise of one resolution element (about 6% of the EW of a
    Balmer-like line at SNR 50): measure the EW for the curve of growth
    over a wider window or with a profile fit.

    Parameters
    ----------
    spectrum : Spectrum
        `continuum` fitted (NaN pixels are ignored), `error` positive,
        and `meta["fwhm_kms"]`, the FWHM of the LSF in km/s. Pixels
        roughly uniform in velocity, as in SDSS; small gaps only shift
        the filter slightly.
    threshold : float
        Detection threshold, in sigma.
    sign : {-1, +1}
        -1 for absorption, +1 for emission.

    Returns
    -------
    features : astropy.table.Table
        One row per feature, columns FEATURE_COLUMNS (see
        `spectral_diag.detect`), sorted by wavelength.

    Raises
    ------
    ValueError
        If `spectrum.error` is not positive at every pixel, or
        `meta["fwhm_kms"]` is missing.
    """
    check_input(spectrum)
    signal, inverse_variance = relative_signal(spectrum, sign)
    kernel = gaussian_kernel(lsf_sigma_pixels(spectrum))
    significance = matched_filter_significance(signal, inverse_variance, kernel)
    windows = feature_windows(significance, threshold)
    rows = [measure_feature(spectrum, window, sign) for window in windows]
    return Table(rows=rows, names=FEATURE_COLUMNS) if rows else _empty_table()


def _empty_table():
    """Features table with no rows and the columns of FEATURE_COLUMNS.

    Returns
    -------
    features : astropy.table.Table
    """
    return Table(names=FEATURE_COLUMNS, dtype=[float] * len(FEATURE_COLUMNS))
