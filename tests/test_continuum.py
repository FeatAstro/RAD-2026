import numpy as np
import pytest
from scipy import constants

from spectral_diag.continuum import fit_continuum, normalise
from spectral_diag.core import C_KMS, Absorber, Line, Spectrum
from spectral_diag.ew import measure_ew
from spectral_diag.io import make_synthetic

WAVE = np.linspace(4000, 6000, 4000)
CONTINUUM = 1000 * (WAVE / 5000) ** -2  # smooth, sloped


def _spectrum(absorbers, seed=0):
    rng = np.random.default_rng(seed)
    return make_synthetic(WAVE, absorbers, continuum=CONTINUUM, snr=60, rng=rng)


def test_no_lines_recovers_continuum_within_errors():
    fit = fit_continuum(_spectrum([]))
    pull = (fit.continuum - CONTINUUM) / fit.meta["continuum_err"]
    # Clipping at -2 sigma biases the fit slightly high, so allow a
    # mean offset of up to 1 continuum error.
    assert abs(pull.mean()) < 1.0
    assert np.abs(fit.continuum / CONTINUUM - 1).max() < 0.005


def test_narrow_lines_are_rejected():
    lines = [Line("X", w, 0.3, 1e8) for w in (4300.0, 4800.0, 5500.0)]
    fit = fit_continuum(_spectrum([Absorber("X", 0.0, 14.0, 60.0, lines)]))
    for line in lines:
        assert not fit.meta["continuum_mask"][np.argmin(np.abs(WAVE - line.wave_rest))]
    assert np.abs(fit.continuum / CONTINUUM - 1).max() < 0.005


def test_exclude_is_respected():
    fit = fit_continuum(_spectrum([]), exclude=[(4900.0, 5100.0)])
    inside = (WAVE >= 4900) & (WAVE <= 5100)
    assert not fit.meta["continuum_mask"][inside].any()


def test_normalise():
    fit = fit_continuum(_spectrum([]))
    norm = normalise(fit)
    np.testing.assert_allclose(norm.flux, fit.flux / fit.continuum)
    np.testing.assert_allclose(norm.continuum, 1.0)


def test_zero_error_rejected():
    spec = Spectrum(WAVE, CONTINUUM, np.zeros_like(WAVE))
    with pytest.raises(ValueError):
        fit_continuum(spec)


# SDSS-like spectrum of an A star: constant 69 km/s pixels, blackbody
# continuum at 8500 K, one broad line standing in for a Balmer line.
SDSS_WAVE = 3800.0 * np.exp(np.arange(0.0, np.log(9200 / 3800), 69.0 / C_KMS))
_LAM = SDSS_WAVE * 1e-10  # m
_PLANCK = 1 / (_LAM**5 * np.expm1(constants.h * constants.c / (_LAM * constants.k * 8500.0)))
SDSS_CONT = 1000 * _PLANCK / _PLANCK.max()
BROAD = Line("Y", 4860.0, 0.1, 1.5e12)  # made-up damping: wings to about +-45 A
BROAD_ABS = [Absorber("Y", 0.0, 15.5, 20.0, [BROAD])]


def test_blackbody_continuum_recovered_away_from_ends():
    spec = make_synthetic(SDSS_WAVE, [], continuum=SDSS_CONT, snr=60, rng=np.random.default_rng(2))
    fit = fit_continuum(spec)
    inner = (SDSS_WAVE > SDSS_WAVE[0] + 50) & (SDSS_WAVE < SDSS_WAVE[-1] - 50)
    assert np.abs(fit.continuum / SDSS_CONT - 1)[inner].max() < 0.005
    # At the ends the spline has data on one side only: larger error.
    err = fit.meta["continuum_err"] / SDSS_CONT
    assert err[-1] > 2 * np.median(err)


def _broad_bias(knot_spacing, n_real=5):
    lo, hi = BROAD.wave_rest - 120, BROAD.wave_rest + 120
    dW = []
    for seed in range(n_real):
        spec = make_synthetic(SDSS_WAVE, BROAD_ABS, continuum=SDSS_CONT, snr=60,
                              rng=np.random.default_rng(seed))
        C = fit_continuum(spec, knot_spacing=knot_spacing).continuum
        dW.append(measure_ew(SDSS_WAVE, spec.flux, C, wave_min=lo, wave_max=hi)[0]
                  - measure_ew(SDSS_WAVE, spec.flux, SDSS_CONT, wave_min=lo, wave_max=hi)[0])
    return np.mean(dW)


def test_broad_line_ew_biased_low_more_with_close_knots():
    # The shallow outer wings are kept in the fit and pull it down, so
    # the EW comes out too small; closer knots follow the line more.
    W_true = measure_ew(SDSS_WAVE, make_synthetic(SDSS_WAVE, BROAD_ABS, continuum=SDSS_CONT).flux,
                        SDSS_CONT, wave_min=BROAD.wave_rest - 120, wave_max=BROAD.wave_rest + 120)[0]
    bias_100, bias_250 = _broad_bias(100.0), _broad_bias(250.0)
    assert bias_100 < bias_250 < 0
    assert abs(bias_250) < 0.08 * W_true
