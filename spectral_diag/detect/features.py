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
    split_blends,
)

# TODO: flag telluric features (O2, H2O at fixed observed wavelength, ground-based
# only), e.g. a `telluric` column from an ESO SkyCalc transmission model at the
# LSF. Not needed for the A0 star: only O2 A (~7598 A) is clearly telluric
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
       that contains a pixel above `threshold` is one window, so a
       broad line with its wings comes out as one feature.
    4. Split each window at the dips between resolved lines
       (`split_blends`), e.g. the NaI D doublet. Lines closer than the
       resolution stay in one feature.
    5. Measure the centre, depth and equivalent width of each feature.

    The window is meant for detecting and locating the feature. For a
    broad line it misses the far damping wings, whose depth is below
    the noise of one resolution element (about 6% of the EW of a
    Balmer-like line at SNR 50): measure the EW for the curve of growth
    over a wider window or with a profile fit. A line split off the
    wing of a broad one (e.g. CaII K on H-epsilon) has a good centre,
    but its EW includes the wing absorption under it.

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
    windows = _line_windows(significance, threshold)
    rows = [measure_feature(spectrum, window, sign) for window in windows]
    return Table(rows=rows, names=FEATURE_COLUMNS) if rows else _empty_table()


# TODO: flag the features that come from a split (`blended` column, agree with
# Gael): cutting at the valley shares their absorption only roughly (NaI D,
# CaII K on H-epsilon), so their EW needs a joint profile fit after identify
def _line_windows(significance, threshold):
    """Feature windows, split so that each holds one resolved line.

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
    windows = []
    for window in feature_windows(significance, threshold):
        windows.extend(split_blends(significance, window, threshold))
    return windows


def _empty_table():
    """Features table with no rows and the columns of FEATURE_COLUMNS.

    Returns
    -------
    features : astropy.table.Table
    """
    return Table(names=FEATURE_COLUMNS, dtype=[float] * len(FEATURE_COLUMNS))
