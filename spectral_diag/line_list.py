"""Atomic data of absorption lines, from the NIST Atomic Spectra Database.

NIST ASD (Kramida et al., https://physics.nist.gov/asd), queried with
astroquery.nist. The lines of each species are downloaded once by
`download_line_table` and saved in data/line_lists/ (committed), so that
`get_lines` works offline. gamma sums A_ki over every decay of the upper
and lower levels that NIST lists.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from astroquery.nist import Nist

from .data_classes import Line

LINE_LIST_DIR = Path(__file__).resolve().parent.parent / "data" / "line_lists"
# Wavelength range of the NIST query, Angstrom. Wider than any selection
# so that gamma sums every decay of each level, not only those in range
_NIST_GAMMA_RANGE = (100, 100_000)
# Wavelength range saved to the line tables, Angstrom: from the Lyman
# limit to the red end of SDSS
_SAVED_WAVE_RANGE = (900, 10_000)
# NIST column holding "E_i - E_k" (eV); the spaces are part of its name
_NIST_ENERGY_COLUMN = "Ei           Ek"
# Separator in NIST level labels "config | term | J"
_NIST_LEVEL_SEP = "|"


@dataclass(frozen=True)
class LineSelection:
    """Criteria to select absorption lines from an atomic database.

    Attributes
    ----------
    wave_min, wave_max : float
        Vacuum rest-wavelength range, Angstrom.
    f_min : float
        Minimum oscillator strength.
    e_low_max : float
        Maximum lower-level energy, eV. 0: ground level only (cold
        ISM/CGM gas); ~0.1: also ground-term fine structure (e.g. CII*
        1335, SiII* 1264); np.inf: all levels (stellar spectra).
    multiplet_totals : bool
        False: the fine-structure components of each line. True: one
        line per n -> n' transition, the total that NIST lists for
        hydrogenic ions (HI, HeII) only. Use it for hydrogen: the
        components of a line lie within ~10 km/s (Halpha: 7 within
        0.2 Angstrom), far below the resolution of SDSS or COS, and
        their f cannot simply be added when the lower level is split
        (n >= 2). gamma is then the mean over the sublevels, e.g.
        4.7e8 s^-1 for Lyman alpha instead of 6.3e8 for the 2p level.
    """

    wave_min: float = 900.0
    wave_max: float = 3000.0
    f_min: float = 0.0
    e_low_max: float = 0.0
    multiplet_totals: bool = False


def get_lines(species, selection=LineSelection()):
    """Absorption lines of `species` that pass `selection`.

    Reads the table saved in LINE_LIST_DIR; run `download_line_table`
    first for a species that has none.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV", "SiII", "HI".
    selection : LineSelection

    Returns
    -------
    lines : list of Line
        Sorted by wavelength.

    Raises
    ------
    FileNotFoundError
        If no table is saved for `species`.
    ValueError
        If no line passes `selection`.
    """
    table = Table.read(_line_table_path(species))
    selected = table[_selection_mask(table, selection)]
    lines = [
        Line(species, float(row["wave"]), float(row["f"]), float(row["gamma"]))
        for row in selected
    ]
    if not lines:
        raise ValueError(f"no lines for {species!r} with {selection}")
    return sorted(lines, key=lambda line: line.wave_rest)


def download_line_table(species):
    """Query NIST for the lines of `species` and save them in LINE_LIST_DIR.

    Saves every line with an oscillator strength in _SAVED_WAVE_RANGE:
    the fine-structure components and, for hydrogenic ions, the n -> n'
    totals, each with its gamma. `get_lines` selects among them.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV" (sent to NIST as "C IV").

    Returns
    -------
    path : Path
        The saved table (ECSV).
    """
    transitions = _nist_transitions(species)
    fine_structure = _is_fine_structure_resolved(transitions)
    table = vstack(
        [
            _line_table(transitions[fine_structure], multiplet_total=False),
            _line_table(transitions[~fine_structure], multiplet_total=True),
        ]
    )
    table.meta["source"] = "NIST Atomic Spectra Database, via astroquery.nist"
    path = _line_table_path(species)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, overwrite=True)
    return path


def _line_table_path(species):
    """Path of the saved line table of `species`.

    Parameters
    ----------
    species : str

    Returns
    -------
    path : Path
    """
    return LINE_LIST_DIR / f"{species}.ecsv"


def _nist_transitions(species):
    """Fetch the NIST transitions of `species` that have an A_ki.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV".

    Returns
    -------
    table : astropy.table.Table
        With the extra columns of `_add_derived_columns`.
    """
    table = Nist.query(
        _NIST_GAMMA_RANGE[0] * u.AA,
        _NIST_GAMMA_RANGE[1] * u.AA,
        linename=_nist_name(species),
        wavelength_type="vacuum",
    )
    _add_derived_columns(table)
    has_einstein_a = ~np.ma.getmaskarray(table["Aki"])
    return table[has_einstein_a]


def _line_table(transitions, multiplet_total):
    """Lines of `transitions` with an f in _SAVED_WAVE_RANGE, as a compact table.

    gamma is computed within `transitions`, so that fine-structure
    components and multiplet totals each use their own decay rates.

    Parameters
    ----------
    transitions : astropy.table.Table
        Output of `_nist_transitions`, either only fine-structure
        components or only multiplet totals.
    multiplet_total : bool
        Value of the "multiplet_total" column.

    Returns
    -------
    table : astropy.table.Table
        Columns wave (Angstrom), f, gamma (s^-1), e_low (eV),
        electric_dipole and multiplet_total.
    """
    f = np.ma.filled(transitions["fik"], np.nan).astype(float)
    wave = np.asarray(transitions["wave"], dtype=float)
    keep = _rows_to_save(f, wave)
    # NIST leaves "Type" empty for allowed electric-dipole (E1) lines
    electric_dipole = np.ma.getmaskarray(transitions["Type"])
    table = Table()
    table["wave"] = wave[keep]
    table["f"] = f[keep]
    table["gamma"] = _damping_constants(transitions)[keep]
    table["e_low"] = np.asarray(transitions["e_low"], dtype=float)[keep]
    table["electric_dipole"] = electric_dipole[keep]
    table["multiplet_total"] = np.full(keep.sum(), multiplet_total)
    return table


def _rows_to_save(f, wave):
    """True for lines with an oscillator strength inside _SAVED_WAVE_RANGE.

    Parameters
    ----------
    f : array
        NaN where NIST gives none.
    wave : array
        Angstrom.

    Returns
    -------
    keep : array of bool
    """
    # NaN fails every comparison, so lines without f are dropped
    return (f > 0) & (wave >= _SAVED_WAVE_RANGE[0]) & (wave <= _SAVED_WAVE_RANGE[1])


def _add_derived_columns(table):
    """Add "wave" (Angstrom), "e_low" (eV), "lower_level" and "upper_level" to `table`.

    Parameters
    ----------
    table : astropy.table.Table
        Raw NIST table, modified in place.
    """
    table["wave"] = _wavelengths(table)
    table["e_low"] = [_lower_energy(entry) for entry in table[_NIST_ENERGY_COLUMN]]
    # Levels are identified by label, not energy: in hydrogenic ions
    # different levels can be degenerate (e.g. HI 3p3/2 and 3d3/2)
    table["lower_level"] = [_level_id(label) for label in table["Lower level"]]
    table["upper_level"] = [_level_id(label) for label in table["Upper level"]]


def _is_fine_structure_resolved(table):
    """True for transitions whose two levels have a J value.

    Hydrogenic ions also list the n -> n' total of each multiplet
    (levels without J) next to its fine-structure components; keeping
    only one of the two avoids counting the same absorption twice.

    Parameters
    ----------
    table : astropy.table.Table
        NIST table with "lower_level" and "upper_level" columns.

    Returns
    -------
    mask : array of bool
    """
    return np.array(
        [
            not (
                lower_level.endswith(_NIST_LEVEL_SEP)
                or upper_level.endswith(_NIST_LEVEL_SEP)
            )
            for lower_level, upper_level in zip(
                table["lower_level"], table["upper_level"]
            )
        ]
    )


def _damping_constants(table):
    """Damping constant of each transition: decay rates of both levels.

    Parameters
    ----------
    table : astropy.table.Table
        Output of `_nist_transitions`, or a subset of it.

    Returns
    -------
    gamma : array
        s^-1.
    """
    decay_rates = _decay_rates(table)
    # The ground level does not decay, so it is missing from decay_rates
    return np.array(
        [
            decay_rates[upper_level] + decay_rates.get(lower_level, 0.0)
            for lower_level, upper_level in zip(
                table["lower_level"], table["upper_level"]
            )
        ]
    )


def _decay_rates(table):
    """Total decay rate of each level: sum of A_ki over all its decays.

    Parameters
    ----------
    table : astropy.table.Table
        Output of `_nist_transitions`.

    Returns
    -------
    decay_rates : dict
        Level label -> decay rate, s^-1.
    """
    decay_rates = {}
    for level, einstein_a in zip(table["upper_level"], table["Aki"]):
        decay_rates[level] = decay_rates.get(level, 0.0) + float(einstein_a)
    return decay_rates


def _selection_mask(table, selection):
    """True for the electric-dipole lines of `table` that pass `selection`.

    Forbidden lines (M1, E2) are too weak to absorb in ISM/CGM gas, and
    f_lu is defined for electric dipole.

    Parameters
    ----------
    table : astropy.table.Table
        Saved line table (see `_line_table`).
    selection : LineSelection

    Returns
    -------
    mask : array of bool
    """
    return (
        np.asarray(table["electric_dipole"])
        & (np.asarray(table["multiplet_total"]) == selection.multiplet_totals)
        & (np.asarray(table["e_low"]) <= selection.e_low_max)
        & (np.asarray(table["f"]) >= selection.f_min)
        & (np.asarray(table["wave"]) >= selection.wave_min)
        & (np.asarray(table["wave"]) <= selection.wave_max)
    )


def _wavelengths(table):
    """Vacuum wavelengths, Ritz if available, else observed.

    Ritz wavelengths come from the level energies and are usually more
    accurate than the measured ones.

    Parameters
    ----------
    table : astropy.table.Table
        Raw NIST table.

    Returns
    -------
    wave : array
        Angstrom, NaN if neither is given.
    """
    waves = []
    for ritz, observed in zip(table["Ritz"], table["Observed"]):
        wave = _first_number(ritz)
        if np.isnan(wave):
            wave = _first_number(observed)
        waves.append(wave)
    return np.array(waves)


def _level_id(label):
    """Normalise a NIST level label by removing all whitespace.

    Parameters
    ----------
    label : str
        e.g. "2p     | 2P*  | 3/2".

    Returns
    -------
    level_id : str
        e.g. "2p|2P*|3/2".
    """
    return "".join(str(label).split())


def _nist_name(species):
    """Convert a species name to NIST notation.

    Parameters
    ----------
    species : str
        e.g. "CIV", "SiII".

    Returns
    -------
    nist_name : str
        e.g. "C IV", "Si II".
    """
    match = re.fullmatch(r"([A-Z][a-z]?)([IVX]+)", species)
    if match is None:
        raise ValueError(
            f"cannot parse species {species!r}; expected e.g. 'CIV', 'SiII'"
        )
    return f"{match[1]} {match[2]}"


def _lower_energy(entry):
    """Lower-level energy from a NIST "E_i - E_k" string.

    Parameters
    ----------
    entry : str
        e.g. "0.000000 - 8.008258"; NIST markers [ ] ? + are skipped.

    Returns
    -------
    e_low : float
        eV, NaN if missing.
    """
    return _first_number(entry)


def _first_number(entry):
    """First number in a NIST table entry.

    NIST entries can be numbers, masked, or strings with markers
    (e.g. Ritz wavelength "6107.955+", energy "[0.000000] - 8.008258").

    Parameters
    ----------
    entry : str, float or masked

    Returns
    -------
    number : float
        NaN if the entry holds no number.
    """
    if entry is np.ma.masked:
        return np.nan
    match = re.search(r"\d+\.?\d*", str(entry))
    if match is None:
        return np.nan
    return float(match[0])
