import numpy as np
import pytest
from scipy import constants as scipy_constants

from spectral_diag import constants
from spectral_diag.line_list import LineSelection, _nist_name, get_lines

# Rydberg constant of hydrogen: R_infinity corrected for the reduced mass, 1/Angstrom
HYDROGEN_RYDBERG = (
    scipy_constants.Rydberg * 1e-10 / (1 + scipy_constants.m_e / scipy_constants.m_p)
)
# Fine structure and Lamb shift move the line centroids from the Bohr
# levels by a few km/s, far below the SDSS and COS resolutions
MAX_RYDBERG_OFFSET_KMS = 5.0
# Highest upper level of the Rydberg comb compared with NIST
MAX_UPPER_LEVEL = 60
# Balmer (n = 2) and Paschen (n = 3) lines of the SDSS range, Angstrom
SDSS_HYDROGEN = LineSelection(
    wave_min=3600, wave_max=9300, e_low_max=np.inf, multiplet_totals=True
)


@pytest.mark.parametrize(
    "species, expected",
    [("CIV", "C IV"), ("SiII", "Si II"), ("HI", "H I"), ("FeII", "Fe II")],
)
def test_nist_name(species, expected):
    """Species names are split into element and ionisation stage."""
    assert _nist_name(species) == expected


def test_nist_name_rejects_bad_species():
    """Names that are not element + roman numeral raise ValueError."""
    with pytest.raises(ValueError):
        _nist_name("civ")


def test_civ_doublet_matches_textbook():
    """CIV 1548, 1550 have the wavelengths and f of Morton 2003."""
    lines = get_lines("CIV", LineSelection(wave_min=1500, wave_max=1600))
    expected = [(1548.204, 0.190), (1550.781, 0.0952)]
    assert len(lines) == len(expected)
    for line, (wave_rest, f) in zip(lines, expected):
        np.testing.assert_allclose(line.wave_rest, wave_rest, atol=0.01)
        np.testing.assert_allclose(line.f, f, rtol=0.01)


def test_unknown_species_has_no_saved_table():
    """A species never downloaded raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_lines("XeIX")


def test_empty_selection_raises():
    """A selection that keeps no line raises ValueError."""
    with pytest.raises(ValueError):
        get_lines("CIV", LineSelection(wave_min=5000, wave_max=5001))


def test_hydrogen_lyman_alpha():
    """Lyman alpha has the textbook total f and the 2p decay rate.

    Fine-structure components only: the multiplet total would double
    the summed f.
    """
    lyman_alpha = get_lines("HI", LineSelection(wave_min=1215, wave_max=1216))
    np.testing.assert_allclose(sum(line.f for line in lyman_alpha), 0.4164, rtol=1e-3)
    for line in lyman_alpha:
        np.testing.assert_allclose(line.gamma, 6.265e8, rtol=1e-3)


def _rydberg_wavelength(n_lower, n_upper):
    """Vacuum wavelength of the hydrogen line n_upper -> n_lower (Bohr model).

    Parameters
    ----------
    n_lower, n_upper : int
        Principal quantum numbers.

    Returns
    -------
    wave : float
        Angstrom.
    """
    wavenumber = HYDROGEN_RYDBERG * (1 / n_lower**2 - 1 / n_upper**2)
    return 1 / wavenumber


def _closest_rydberg_offset_kms(wave):
    """Velocity offset of `wave` from the closest Balmer or Paschen line.

    Parameters
    ----------
    wave : float
        Angstrom.

    Returns
    -------
    offset : float
        km/s, absolute value.
    """
    smallest_offset = np.inf
    for n_lower in (2, 3):
        for n_upper in range(n_lower + 1, MAX_UPPER_LEVEL + 1):
            predicted = _rydberg_wavelength(n_lower, n_upper)
            offset = abs(wave / predicted - 1) * constants.C_KMS
            smallest_offset = min(smallest_offset, offset)
    return smallest_offset


def test_hydrogen_multiplet_totals_follow_rydberg():
    """Each NIST total is one Balmer or Paschen line of the Rydberg formula."""
    lines = get_lines("HI", SDSS_HYDROGEN)
    for line in lines:
        assert _closest_rydberg_offset_kms(line.wave_rest) < MAX_RYDBERG_OFFSET_KMS


def test_hydrogen_multiplet_totals_one_line_per_transition():
    """Halpha and Hbeta come as single lines with their textbook total f."""
    lines = get_lines("HI", SDSS_HYDROGEN)
    for wave_rest, f_total in [(6564.6, 0.6411), (4862.7, 0.1194)]:
        matches = [line for line in lines if abs(line.wave_rest - wave_rest) < 1]
        assert len(matches) == 1
        np.testing.assert_allclose(matches[0].f, f_total, rtol=1e-3)
