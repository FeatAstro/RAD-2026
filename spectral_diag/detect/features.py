"""Detection of the significant features of a spectrum."""

from astropy.table import Table

from .measure import measure_feature
from .significance import (
    feature_windows,
    gaussian_kernel,
    matched_filter_significance,
    relative_signal,
    split_blends,
    unresolved_line_sigma_in_pixels,
)

# TODO: flag telluric features (O2, H2O) with a `telluric` column
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

    The EW is approximate: over the detection window it misses the far
    wings of a broad line (about 6% at SNR 50) and includes the wing under
    a split line (e.g. CaII K on H-epsilon). Remeasure it for the curve of
    growth.

    Parameters
    ----------
    spectrum : Spectrum
        With `continuum` fitted, `error` > 0 and `meta["fwhm_kms"]` (LSF
        FWHM, km/s). Pixels roughly uniform in velocity, as in SDSS.
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
        If `wave` is not strictly increasing, `error` is not positive at
        every pixel or `meta["fwhm_kms"]` is missing.
    """
    spectrum.check_valid()
    signal, inverse_variance = relative_signal(spectrum, sign)
    kernel = gaussian_kernel(unresolved_line_sigma_in_pixels(spectrum))
    significance = matched_filter_significance(signal, inverse_variance, kernel)
    windows = _line_windows(significance, threshold)
    rows = [measure_feature(spectrum, window, sign) for window in windows]
    return Table({name: [row[name] for row in rows] for name in FEATURE_COLUMNS})


# TODO: add a `blended` column for split features (agree with Gael): their EW
# needs a joint profile fit after identify
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
