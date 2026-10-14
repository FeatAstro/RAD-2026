import numpy as np

from spectral_diag.core import Line
from spectral_diag.profiles import optical_depth, optical_depth_gaussian

WAVE = np.linspace(1540, 1560, 4000)


def test_voigt_reduces_to_gaussian():
    line_nodamp = Line(species="Test", wave_rest=1550.0, f=0.1, gamma=1e-6)
    tau_g = optical_depth_gaussian(WAVE, line_nodamp, logN=14, b=10, z=0.0)
    tau_v = optical_depth(WAVE, line_nodamp, logN=14, b=10, z=0.0)
    np.testing.assert_allclose(tau_v, tau_g, rtol=0, atol=1e-10)


def test_voigt_damping_wings(line):
    tau_g = optical_depth_gaussian(WAVE, line, logN=17.0, b=10, z=0.0)
    tau_v = optical_depth(WAVE, line, logN=17.0, b=10, z=0.0)
    far = np.abs(WAVE - 1550) > 5.0
    assert tau_v[far].max() > 100 * tau_g[far].max()
