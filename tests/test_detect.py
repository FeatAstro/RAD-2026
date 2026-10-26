"""Tests of feature detection on synthetic SDSS-like spectra.

Lines are injected at known positions, convolved with the LSF and
binned into pixels by InstrumentGrid. Wavelengths in Angstrom.
"""

import numpy as np
import pytest

from spectral_diag.core import Absorber, Line, Spectrum
from spectral_diag.detect import FEATURE_COLUMNS, find_features
from spectral_diag.profiles import InstrumentGrid, optical_depth

PIXEL_DEX = 1e-4
FWHM_KMS = 150.0
OVERSAMPLE = 10
# log10 of 4500 to ~5500 Angstrom
LOG_WAVE_START = 3.653
N_PIXELS = 870
SNR = 50.0
# Narrow lines: depth ~0.12 at the SDSS resolution, EW signal-to-noise ~7
NARROW_WAVES = (4600.0, 4800.0, 5200.0)
NARROW_LOGN = 13.5
NARROW_B = 10.0
# Broad line standing in for a Balmer line, wings to about +-45 Angstrom
# (same as in test_continuum)
BROAD_LINE = Line("Y", 4860.0, 0.1, 1.5e12)
BROAD_LOGN = 15.5
BROAD_B = 20.0
# Largest difference from the noise-free EW in the same window, in ew_err
MAX_EW_PULL = 4.0
# The window misses the far, sub-noise damping wings (~6% at SNR 50);
# fail if it starts missing much more
MIN_EW_FRACTION_IN_WINDOW = 0.9
# Well above the ~8-10 sigma that the filter gives to the narrow lines
UNREACHABLE_THRESHOLD = 20.0
# A single deep pixel: its core above half depth is that pixel alone
ONE_PIXEL_LINE_INDEX = 400
ONE_PIXEL_LINE_FLUX = 0.4
# Largest distance from the true centre, in units of wave_err
MAX_CENTRE_PULL = 4.0


def narrow_absorbers(waves):
    """One weak-damping test line per wavelength, at z = 0.

    Parameters
    ----------
    waves : sequence of float
        Rest (= observed) wavelengths, Angstrom.

    Returns
    -------
    absorbers : list of Absorber
    """
    return [
        Absorber("T", 0.0, NARROW_LOGN, NARROW_B, [Line("T", wave, 0.1, 2.65e8)])
        for wave in waves
    ]


def sdss_like_spectrum(absorbers, rng, sign=-1):
    """Unit-continuum spectrum observed at the SDSS resolution, with noise.

    Parameters
    ----------
    absorbers : list of Absorber
    rng : numpy.random.Generator
    sign : {-1, +1}
        +1 turns every absorption line into an emission line of the
        same shape.

    Returns
    -------
    spectrum : Spectrum
        With continuum = 1 and the noise-free flux in meta["model"].
    """
    wave = 10 ** (LOG_WAVE_START + PIXEL_DEX * np.arange(N_PIXELS))
    meta = {"pixel_dex": PIXEL_DEX, "fwhm_kms": FWHM_KMS}
    template = Spectrum(wave, np.ones_like(wave), np.ones_like(wave), meta=meta)
    grid = InstrumentGrid.for_spectrum(template, OVERSAMPLE)
    absorption = 1 - grid.observe(np.exp(-total_optical_depth(grid.wave, absorbers)))
    model = 1 + sign * absorption
    error = np.full_like(wave, 1 / SNR)
    flux = model + rng.normal(0, error)
    meta = {**meta, "model": model}
    return Spectrum(wave, flux, error, continuum=np.ones_like(wave), meta=meta)


def total_optical_depth(wave, absorbers):
    """Sum of the optical depths of every line of every absorber.

    Parameters
    ----------
    wave : array
        Angstrom.
    absorbers : list of Absorber

    Returns
    -------
    tau : array
    """
    tau = np.zeros_like(wave)
    for absorber in absorbers:
        for line in absorber.lines:
            tau += optical_depth(wave, line, absorber.logN, absorber.b, absorber.z)
    return tau


def test_noise_only_gives_no_features():
    spectrum = sdss_like_spectrum([], np.random.default_rng(1))
    features = find_features(spectrum)
    assert len(features) == 0
    assert tuple(features.colnames) == FEATURE_COLUMNS


def test_rejects_noise_free_spectrum():
    spectrum = sdss_like_spectrum([], np.random.default_rng(6))
    spectrum.error[:] = 0.0
    with pytest.raises(ValueError):
        find_features(spectrum)


def test_rejects_missing_resolution():
    spectrum = sdss_like_spectrum([], np.random.default_rng(7))
    del spectrum.meta["fwhm_kms"]
    with pytest.raises(ValueError):
        find_features(spectrum)


def test_finds_narrow_lines_at_their_positions():
    spectrum = sdss_like_spectrum(
        narrow_absorbers(NARROW_WAVES), np.random.default_rng(2)
    )
    features = find_features(spectrum)
    assert len(features) == len(NARROW_WAVES)
    assert np.all(features["depth"] > 0)
    pull = (features["wave"] - np.array(NARROW_WAVES)) / features["wave_err"]
    assert np.all(np.abs(pull) < MAX_CENTRE_PULL)
    assert np.all(features["wave_min"] < NARROW_WAVES)
    assert np.all(features["wave_max"] > NARROW_WAVES)


def test_threshold_controls_detections():
    spectrum = sdss_like_spectrum(
        narrow_absorbers(NARROW_WAVES), np.random.default_rng(2)
    )
    assert len(find_features(spectrum, threshold=4.0)) == len(NARROW_WAVES)
    assert len(find_features(spectrum, threshold=UNREACHABLE_THRESHOLD)) == 0


def test_centre_error_of_one_pixel_line_is_not_zero():
    spectrum = sdss_like_spectrum([], np.random.default_rng(8))
    spectrum.flux[:] = 1.0
    spectrum.flux[ONE_PIXEL_LINE_INDEX] = ONE_PIXEL_LINE_FLUX
    features = find_features(spectrum)
    assert len(features) == 1
    assert features["wave_err"][0] > 0
    np.testing.assert_allclose(features["wave"][0], spectrum.wave[ONE_PIXEL_LINE_INDEX])


def test_broad_line_is_one_feature_with_its_ew():
    absorber = Absorber("Y", 0.0, BROAD_LOGN, BROAD_B, [BROAD_LINE])
    spectrum = sdss_like_spectrum([absorber], np.random.default_rng(3))
    features = find_features(spectrum)
    assert len(features) == 1
    feature = features[0]
    absorption = 1 - spectrum.meta["model"]
    window = (spectrum.wave >= feature["wave_min"]) & (
        spectrum.wave <= feature["wave_max"]
    )
    window_ew = np.trapezoid(absorption[window], spectrum.wave[window])
    total_ew = np.trapezoid(absorption, spectrum.wave)
    assert abs(feature["ew"] - window_ew) < MAX_EW_PULL * feature["ew_err"]
    assert window_ew > MIN_EW_FRACTION_IN_WINDOW * total_ew


def test_emission_lines_with_positive_sign():
    absorbers = narrow_absorbers(NARROW_WAVES)
    spectrum = sdss_like_spectrum(absorbers, np.random.default_rng(4), sign=+1)
    assert len(find_features(spectrum, sign=-1)) == 0
    features = find_features(spectrum, sign=+1)
    assert len(features) == len(NARROW_WAVES)
    assert np.all(features["depth"] > 0)
    assert np.all(features["ew"] > 0)
    assert np.all(features["snr"] > 0)


def test_ignores_pixels_without_continuum():
    spectrum = sdss_like_spectrum(
        narrow_absorbers(NARROW_WAVES), np.random.default_rng(5)
    )
    spectrum.continuum[:50] = np.nan
    features = find_features(spectrum)
    assert len(features) == len(NARROW_WAVES)
    assert np.all(np.isfinite(features["ew"]))
