"""Tests of the sigma-clipped spline continuum fit.

Adapted from Gael's tests on the `gael` branch to the merged API.
Wavelengths in Angstrom.
"""

import numpy as np
import pytest
from scipy import constants

from spectral_diag.continuum import (
    ContinuumSettings,
    _weighted_spline,
    fit_continuum,
    normalise,
)
from spectral_diag.constants import C_KMS
from spectral_diag.data_classes import Absorber, Line, Spectrum
from spectral_diag.equivalent_width import measure_ew
from spectral_diag.synthetic_spectrum import make_synthetic

WAVE = np.linspace(4000, 6000, 4000)
# Smooth, sloped continuum
CONTINUUM = 1000 * (WAVE / 5000) ** -2
SNR = 60
# Largest deviation from the true continuum allowed, in units of the
# fitted continuum error
MAX_PULL = 4.0

# SDSS-like spectrum of an A star: constant 69 km/s pixels and a
# blackbody continuum at 8500 K
SDSS_PIXEL_KMS = 69.0
SDSS_WAVE = 3800.0 * np.exp(np.arange(0.0, np.log(9200 / 3800), SDSS_PIXEL_KMS / C_KMS))
A_STAR_TEMPERATURE = 8500.0
END_MARGIN = 50.0

# One broad line standing in for a Balmer line; made-up damping that
# gives wings to about +-45 Angstrom
BROAD_LINE = Line("Y", 4860.0, 0.1, 1.5e12)
BROAD_ABSORBER = Absorber("Y", 0.0, 15.5, 20.0, [BROAD_LINE])
BROAD_EW_HALF_WINDOW = 120.0
N_REALISATIONS = 5
MAX_RELATIVE_EW_BIAS = 0.08


def _blackbody_shape(wave, temperature):
    """Planck B_lambda normalised to a peak of 1000 (arbitrary flux units).

    Parameters
    ----------
    wave : array
        Angstrom.
    temperature : float
        K.

    Returns
    -------
    flux : array
    """
    lam = wave * 1e-10
    planck = 1 / (
        lam**5 * np.expm1(constants.h * constants.c / (lam * constants.k * temperature))
    )
    return 1000 * planck / planck.max()


def _noisy(wave, continuum, absorbers=(), seed=0):
    """Synthetic spectrum at SNR with a fixed random seed.

    Parameters
    ----------
    wave, continuum : array
    absorbers : sequence of Absorber
    seed : int

    Returns
    -------
    spectrum : Spectrum
    """
    rng = np.random.default_rng(seed)
    return make_synthetic(wave, list(absorbers), continuum=continuum, snr=SNR, rng=rng)


def _pull(fit, true_continuum):
    """Deviation of the fit from the true continuum, in continuum errors.

    Parameters
    ----------
    fit : ContinuumFit
    true_continuum : array

    Returns
    -------
    pull : array
        NaN where the fit is undefined.
    """
    return (fit.continuum - true_continuum) / fit.error


def test_no_lines_recovers_continuum_within_errors():
    fit = fit_continuum(_noisy(WAVE, CONTINUUM))
    pull = _pull(fit, CONTINUUM)
    # Clipping at -2 sigma biases the fit slightly high, so allow a
    # mean offset of up to 1 continuum error
    assert abs(np.nanmean(pull)) < 1.0
    assert np.nanmax(np.abs(pull)) < MAX_PULL


def test_narrow_lines_are_rejected():
    lines = [Line("X", wave, 0.3, 1e8) for wave in (4300.0, 4800.0, 5500.0)]
    fit = fit_continuum(
        _noisy(WAVE, CONTINUUM, [Absorber("X", 0.0, 14.0, 60.0, lines)])
    )
    for line in lines:
        assert not fit.used[np.argmin(np.abs(WAVE - line.wave_rest))]
    assert np.nanmax(np.abs(_pull(fit, CONTINUUM))) < MAX_PULL


def test_continuum_matches_returned_mask_when_not_converged():
    line = Line("X", 5000.0, 0.3, 1e8)
    spectrum = _noisy(WAVE, CONTINUUM, [Absorber("X", 0.0, 14.0, 60.0, [line])])
    settings = ContinuumSettings(max_iter=1)
    with pytest.warns(UserWarning):
        fit = fit_continuum(spectrum, settings)
    assert fit.status == "max_iter"
    spline = _weighted_spline(spectrum, fit.used, settings.knot_spacing)
    inside = np.isfinite(fit.continuum)
    np.testing.assert_allclose(fit.continuum[inside], spline(WAVE)[inside])


def test_normalise():
    spectrum = _noisy(WAVE, CONTINUUM)
    fit = fit_continuum(spectrum)
    normalised = normalise(spectrum, fit.continuum)
    np.testing.assert_allclose(normalised.flux, spectrum.flux / fit.continuum)
    np.testing.assert_allclose(normalised.continuum, 1.0)


def test_zero_error_rejected():
    spectrum = Spectrum(WAVE, CONTINUUM, np.zeros_like(WAVE))
    with pytest.raises(ValueError):
        fit_continuum(spectrum)


def test_blackbody_continuum_recovered_away_from_ends():
    true_continuum = _blackbody_shape(SDSS_WAVE, A_STAR_TEMPERATURE)
    fit = fit_continuum(_noisy(SDSS_WAVE, true_continuum, seed=2))
    inner = (SDSS_WAVE > SDSS_WAVE[0] + END_MARGIN) & (
        SDSS_WAVE < SDSS_WAVE[-1] - END_MARGIN
    )
    assert np.nanmax(np.abs(_pull(fit, true_continuum))[inner]) < MAX_PULL
    # At the ends the spline has data on one side only: larger error
    relative_error = fit.error / true_continuum
    last_finite = relative_error[np.isfinite(relative_error)][-1]
    assert last_finite > 2 * np.nanmedian(relative_error)


def _broad_line_ew(spectrum, continuum):
    """Equivalent width of BROAD_LINE over its fixed window, Angstrom.

    Parameters
    ----------
    spectrum : Spectrum
    continuum : array

    Returns
    -------
    W : float
    """
    centre = BROAD_LINE.wave_rest
    W, _ = measure_ew(
        spectrum.wave,
        spectrum.flux,
        continuum,
        wave_min=centre - BROAD_EW_HALF_WINDOW,
        wave_max=centre + BROAD_EW_HALF_WINDOW,
    )
    return W


def _mean_broad_line_ew_bias(knot_spacing):
    """Mean EW error of BROAD_LINE due to the fitted continuum, Angstrom.

    Parameters
    ----------
    knot_spacing : float
        Angstrom.

    Returns
    -------
    bias : float
        Mean over N_REALISATIONS of W(fitted continuum) - W(true continuum).
    """
    true_continuum = _blackbody_shape(SDSS_WAVE, A_STAR_TEMPERATURE)
    settings = ContinuumSettings(knot_spacing=knot_spacing)
    biases = []
    for seed in range(N_REALISATIONS):
        spectrum = _noisy(SDSS_WAVE, true_continuum, [BROAD_ABSORBER], seed)
        fit = fit_continuum(spectrum, settings)
        biases.append(
            _broad_line_ew(spectrum, fit.continuum)
            - _broad_line_ew(spectrum, true_continuum)
        )
    return np.mean(biases)


def test_broad_line_ew_biased_low_more_with_close_knots():
    # The shallow outer wings are kept in the fit and pull it down, so
    # the EW comes out too small; closer knots follow the line more
    true_continuum = _blackbody_shape(SDSS_WAVE, A_STAR_TEMPERATURE)
    noiseless = make_synthetic(SDSS_WAVE, [BROAD_ABSORBER], continuum=true_continuum)
    true_W = _broad_line_ew(noiseless, true_continuum)
    close_knots_bias = _mean_broad_line_ew_bias(100.0)
    wide_knots_bias = _mean_broad_line_ew_bias(250.0)
    assert close_knots_bias < wide_knots_bias < 0
    assert abs(wide_knots_bias) < MAX_RELATIVE_EW_BIAS * true_W
