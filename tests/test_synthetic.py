import numpy as np

from spectral_diag.cog import ew_linear
from spectral_diag.core import Absorber
from spectral_diag.ew import measure_ew
from spectral_diag.io import make_synthetic


def test_synthetic_noise_free_ew(line):
    wave = np.linspace(1500, 1600, 200_000)
    absorber = Absorber(species="Test", z=0.0, logN=11.0, b=10.0, lines=[line])
    spec = make_synthetic(wave, [absorber])
    assert np.all(spec.error == 0)
    W, _ = measure_ew(spec.wave, spec.flux, spec.continuum)
    np.testing.assert_allclose(W, ew_linear(line, 11.0), rtol=0.01)


def test_ew_error_calibrated_against_noise(line):
    wave = np.linspace(1545, 1555, 2000)
    absorber = Absorber(species="Test", z=0.0, logN=13.5, b=10.0, lines=[line])
    rng = np.random.default_rng(0)
    Ws, W_errs = [], []
    for _ in range(400):
        spec = make_synthetic(wave, [absorber], snr=30.0, rng=rng)
        W, W_err = measure_ew(spec.wave, spec.flux, spec.continuum, error=spec.error)
        Ws.append(W)
        W_errs.append(W_err)
    ratio = np.std(Ws) / np.mean(W_errs)
    assert 0.7 < ratio < 1.3
