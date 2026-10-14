import numpy as np
import pytest

from spectral_diag.cog import ew_linear
from spectral_diag.ew import measure_ew
from spectral_diag.profiles import optical_depth, optical_depth_gaussian

WAVE = np.linspace(1200, 1800, 200_000)
CONTINUUM = np.ones_like(WAVE)


def test_ew_linear_regime_matches_analytic(line):
    tau = optical_depth_gaussian(WAVE, line, logN=11.0, b=10.0, z=0.0)
    W, _ = measure_ew(WAVE, np.exp(-tau), CONTINUUM)
    np.testing.assert_allclose(W, ew_linear(line, 11.0), rtol=0.01)


@pytest.mark.parametrize("b", [3.0, 10.0, 30.0, 60.0])
def test_ew_linear_regime_independent_of_b(line, b):
    tau = optical_depth_gaussian(WAVE, line, logN=11.0, b=b, z=0.0)
    W, _ = measure_ew(WAVE, np.exp(-tau), CONTINUUM)
    np.testing.assert_allclose(W, ew_linear(line, 11.0), rtol=0.01)


def test_cog_regime_slopes(line):
    logNs = np.arange(11.0, 20.01, 0.25)
    W = np.array([
        measure_ew(WAVE, np.exp(-optical_depth(WAVE, line, logN=lN, b=10.0, z=0.0)), CONTINUUM)[0]
        for lN in logNs
    ])
    slope_low = np.polyfit(logNs[:5], np.log10(W[:5]), 1)[0]
    slope_high = np.polyfit(logNs[-5:], np.log10(W[-5:]), 1)[0]
    assert abs(W[0] / ew_linear(line, logNs[0]) - 1) < 0.01
    assert abs(slope_low - 1) < 0.02
    assert abs(slope_high - 0.5) < 0.05
