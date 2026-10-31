import numpy as np
import pytest
from requests.exceptions import RequestException

from spectral_diag.line_list import LineSelection, _nist_name, get_lines, query_nist


def _query_nist_or_skip(species, selection):
    """Run `query_nist`, skipping the test if NIST is unreachable.

    Only network errors skip, to keep the offline suite green; parsing
    errors still fail the test.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV".
    selection : LineSelection

    Returns
    -------
    lines : list of Line
    """
    try:
        return query_nist(species, selection)
    except RequestException as err:
        pytest.skip(f"NIST unreachable: {err}")


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


def test_nist_matches_table():
    """NIST and Verner+96 agree on the CIV doublet."""
    nist_lines = _query_nist_or_skip("CIV", LineSelection(wave_min=1500, wave_max=1600))
    table_lines = get_lines("CIV")
    assert len(nist_lines) == len(table_lines)
    for nist_line, table_line in zip(nist_lines, table_lines):
        np.testing.assert_allclose(nist_line.wave_rest, table_line.wave_rest, atol=0.05)
        np.testing.assert_allclose(nist_line.f, table_line.f, rtol=0.05)
        np.testing.assert_allclose(nist_line.gamma, table_line.gamma, rtol=0.1)


def test_nist_hydrogen_lyman_alpha():
    """Lyman alpha has the textbook total f and the 2p decay rate.

    Fine-structure components only: the multiplet total would double
    the summed f.
    """
    lyman_alpha = _query_nist_or_skip("HI", LineSelection(wave_min=1215, wave_max=1216))
    np.testing.assert_allclose(sum(line.f for line in lyman_alpha), 0.4164, rtol=1e-3)
    for line in lyman_alpha:
        np.testing.assert_allclose(line.gamma, 6.265e8, rtol=1e-3)
