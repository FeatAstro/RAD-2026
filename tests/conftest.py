import pytest

from spectral_diag.data_classes import Line


@pytest.fixture
def line():
    """Test transition close to CIV 1550 (wavelength in Å)."""
    return Line(species="Test", wave_rest=1550.0, f=0.1, gamma=2.65e8)
