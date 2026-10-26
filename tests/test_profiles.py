import numpy as np
import pytest

from spectral_diag.core import C_KMS, FWHM_PER_SIGMA, Line, Spectrum
from spectral_diag.profiles import (
    InstrumentGrid,
    optical_depth,
    optical_depth_gaussian,
)

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


PIXEL_DEX = 1e-4
FWHM_KMS = 150.0
OVERSAMPLE = 10
# Pixel width in km/s for PIXEL_DEX
PIXEL_KMS = np.log(10) * C_KMS * PIXEL_DEX


def log_uniform_spectrum(n_pixels=400, start=3.6):
    """Flat Spectrum on a grid uniform in log10(wave)."""
    wave = 10 ** (start + PIXEL_DEX * np.arange(n_pixels))
    meta = {"pixel_dex": PIXEL_DEX, "fwhm_kms": FWHM_KMS}
    return Spectrum(
        wave=wave, flux=np.ones_like(wave), error=np.ones_like(wave), meta=meta
    )


def weak_line_absorption(grid, b):
    """Intrinsic 1 - flux on `grid.wave` of a weak Gaussian line at its centre."""
    centre = np.sqrt(grid.wave[0] * grid.wave[-1])
    velocity = C_KMS * np.log(grid.wave / centre)
    weak_tau0 = 0.01
    return 1 - np.exp(-weak_tau0 * np.exp(-((velocity / b) ** 2)))


def test_observe_keeps_flat_spectrum():
    spectrum = log_uniform_spectrum()
    grid = InstrumentGrid.for_spectrum(spectrum, OVERSAMPLE)
    np.testing.assert_allclose(grid.observe(np.ones_like(grid.wave)), 1.0)


def test_observe_conserves_equivalent_width():
    grid = InstrumentGrid.for_spectrum(log_uniform_spectrum(), OVERSAMPLE)
    intrinsic = weak_line_absorption(grid, b=5.0)
    observed = 1 - grid.observe(1 - intrinsic)
    # Both areas in pixel units: each pixel holds OVERSAMPLE subpixels
    np.testing.assert_allclose(observed.sum(), intrinsic.sum() / OVERSAMPLE, rtol=1e-6)


def test_observed_width_of_unresolved_line():
    # Variances add: LSF, line (sigma = b / sqrt 2) and pixel box (w^2 / 12)
    b = 5.0
    spectrum = log_uniform_spectrum()
    grid = InstrumentGrid.for_spectrum(spectrum, OVERSAMPLE)
    absorption = 1 - grid.observe(1 - weak_line_absorption(grid, b))
    velocity = PIXEL_KMS * (np.arange(absorption.size) - (absorption.size - 1) / 2)
    variance = np.sum(absorption * velocity**2) / np.sum(absorption)
    expected = (FWHM_KMS / FWHM_PER_SIGMA) ** 2 + b**2 / 2 + PIXEL_KMS**2 / 12
    np.testing.assert_allclose(variance, expected, rtol=0.01)


def test_grid_fills_gaps():
    spectrum = log_uniform_spectrum()
    keep = np.ones(spectrum.wave.size, dtype=bool)
    keep[100:160] = False
    gapped = Spectrum(
        spectrum.wave[keep],
        spectrum.flux[keep],
        spectrum.error[keep],
        meta=spectrum.meta,
    )
    grid = InstrumentGrid.for_spectrum(gapped, OVERSAMPLE)
    full = InstrumentGrid.for_spectrum(spectrum, OVERSAMPLE)
    np.testing.assert_allclose(grid.wave, full.wave)
    np.testing.assert_array_equal(grid.subpixels, full.subpixels[keep])


def test_rejects_grid_not_uniform_in_log():
    spectrum = log_uniform_spectrum()
    linear = Spectrum(
        np.linspace(spectrum.wave[0], spectrum.wave[-1], spectrum.wave.size),
        spectrum.flux,
        spectrum.error,
        meta=spectrum.meta,
    )
    with pytest.raises(ValueError):
        InstrumentGrid.for_spectrum(linear, OVERSAMPLE)
