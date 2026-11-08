import numpy as np
import pytest

from spectral_diag.data_classes import Spectrum

N_PIXELS = 10
WINDOW = slice(2, 5)


def _spectrum(continuum):
    """Spectrum with distinct values per pixel, so a wrong cut shows.

    Parameters
    ----------
    continuum : array or None
        Passed through to Spectrum.

    Returns
    -------
    spectrum : Spectrum
    """
    pixels = np.arange(N_PIXELS, dtype=float)
    return Spectrum(
        wave=1000.0 + pixels,
        flux=2.0 * pixels,
        error=3.0 * pixels,
        continuum=continuum,
        meta={"source": "test"},
    )


def test_cut_keeps_window_of_every_array():
    spectrum = _spectrum(continuum=np.linspace(1.0, 2.0, N_PIXELS))
    cut = spectrum.cut(WINDOW)
    for name in ("wave", "flux", "error", "continuum"):
        assert np.array_equal(getattr(cut, name), getattr(spectrum, name)[WINDOW])


def test_cut_without_continuum_keeps_none():
    cut = _spectrum(continuum=None).cut(WINDOW)
    assert cut.continuum is None
    assert len(cut.wave) == WINDOW.stop - WINDOW.start


def test_cut_shares_meta():
    spectrum = _spectrum(continuum=None)
    assert spectrum.cut(WINDOW).meta is spectrum.meta


def test_check_valid_accepts_valid_spectrum():
    spectrum = _spectrum(continuum=None)
    spectrum.error[0] = 1.0
    spectrum.check_valid()


def test_check_valid_rejects_nan_error():
    spectrum = _spectrum(continuum=None)
    spectrum.error[:] = np.nan
    with pytest.raises(ValueError, match="error"):
        spectrum.check_valid()
