import numpy as np
import pytest

from spectral_diag.data_classes import Spectrum

N_PIXELS = 10


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


def test_check_valid_accepts_valid_spectrum():
    spectrum = _spectrum(continuum=None)
    spectrum.error[0] = 1.0
    spectrum.check_valid()


def test_check_valid_rejects_nan_error():
    spectrum = _spectrum(continuum=None)
    spectrum.error[:] = np.nan
    with pytest.raises(ValueError, match="error"):
        spectrum.check_valid()
