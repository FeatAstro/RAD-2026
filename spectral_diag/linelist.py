"""Atomic data for common ISM/CGM lines.

Two sources:

- `get_lines` (offline): rest wavelengths (vacuum) and oscillator
  strengths from Verner, Verner & Ferland 1996, Atomic Data and Nuclear
  Data Tables 64, 1 (arXiv:atom-ph/9604003), Table 1. gamma is the sum
  of the tabulated A_ki over every decay of the upper level listed in
  that table (the decays to the ground term). Decays to other terms are
  not in the table, so gamma is a slight underestimate for FeII.
- `query_nist` (online): NIST Atomic Spectra Database (Kramida et al.,
  https://physics.nist.gov/asd), queried with astroquery.nist. Ground
  level only by default; excited lower levels (fine structure, stellar
  lines) via `LineSelection.e_low_max`. gamma sums A_ki over every decay
  of the upper and lower levels that NIST lists.
"""

import re
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astroquery.nist import Nist

from .core import Line

# species -> (wave_rest [Angstrom], f, gamma [s^-1])
_VERNER96_LINES = {
    "CIV": [
        (1548.1950, 1.90e-1, 2.64e8),
        (1550.7700, 9.52e-2, 2.64e8),
    ],
    "SiII": [
        (1190.416, 2.93e-1, 4.100e9),
        (1193.290, 5.84e-1, 4.100e9),
        (1260.422, 1.18e0, 2.972e9),
        (1304.370, 8.71e-2, 1.016e9),
        (1526.707, 1.32e-1, 1.121e9),
        (1808.013, 2.56e-3, 3.125e6),
    ],
    "FeII": [
        (1608.4511, 8.92e-2, 3.812e8),
        (2344.2139, 1.26e-1, 2.542e8),
        (2367.5905, 4.25e-3, 3.293e8),
        (2374.4612, 5.27e-2, 3.304e8),
        (2382.7652, 3.43e-1, 3.350e8),
        (2586.6500, 5.48e-2, 2.864e8),
        (2600.1729, 2.40e-1, 2.892e8),
    ],
}

# Wavelength range of the NIST query, Angstrom. Wider than any selection
# so that gamma sums every decay of each level, not only those in range
_NIST_GAMMA_RANGE = (100, 100_000)
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
    """

    wave_min: float = 900.0
    wave_max: float = 3000.0
    f_min: float = 0.0
    e_low_max: float = 0.0


def get_lines(species):
    """Return the Verner+96 transitions of `species` (offline).

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV", "SiII", "FeII".

    Returns
    -------
    lines : list of Line
    """
    if species not in _VERNER96_LINES:
        raise KeyError(
            f"no lines tabulated for {species!r}; available: {sorted(_VERNER96_LINES)}"
        )
    return [
        Line(species, wave_rest, f, gamma)
        for wave_rest, f, gamma in _VERNER96_LINES[species]
    ]


def query_nist(species, selection=LineSelection()):
    """Query NIST ASD for the absorption lines of `species`.

    Keeps electric-dipole transitions that pass `selection`: forbidden
    lines are too weak to absorb in ISM/CGM gas, and f_lu is defined
    for electric dipole.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV" (sent to NIST as "C IV").
    selection : LineSelection
        Wavelength range, minimum f and maximum lower-level energy.

    Returns
    -------
    lines : list of Line
        Sorted by wavelength.
    """
    transitions = _nist_transitions(species)
    _add_damping_constants(transitions)
    selected = transitions[_absorption_mask(transitions, selection)]
    lines = [
        Line(species, float(row["wave"]), float(row["fik"]), float(row["gamma"]))
        for row in selected
    ]
    if not lines:
        raise ValueError(f"no NIST lines for {species!r} with {selection}")
    return sorted(lines, key=lambda line: line.wave_rest)


def _nist_transitions(species):
    """Fetch the NIST transitions of `species` usable for absorption lines.

    Parameters
    ----------
    species : str
        Species name, e.g. "CIV".

    Returns
    -------
    table : astropy.table.Table
        Transitions with A_ki between fine-structure levels, with the
        extra columns of `_add_derived_columns`.
    """
    table = Nist.query(
        _NIST_GAMMA_RANGE[0] * u.AA,
        _NIST_GAMMA_RANGE[1] * u.AA,
        linename=_nist_name(species),
        wavelength_type="vacuum",
    )
    _add_derived_columns(table)
    has_einstein_a = ~np.ma.getmaskarray(table["Aki"])
    return table[_is_fine_structure_resolved(table) & has_einstein_a]


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
    (levels without J) next to its fine-structure components; dropping
    the totals avoids counting the same absorption twice.

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


def _add_damping_constants(table):
    """Add "gamma" (s^-1), the decay rates of both levels, to `table`.

    Parameters
    ----------
    table : astropy.table.Table
        Output of `_nist_transitions`, modified in place.
    """
    decay_rates = _decay_rates(table)
    # The ground level does not decay, so it is missing from decay_rates
    table["gamma"] = [
        decay_rates[upper_level] + decay_rates.get(lower_level, 0.0)
        for lower_level, upper_level in zip(table["lower_level"], table["upper_level"])
    ]


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


def _absorption_mask(table, selection):
    """True for electric-dipole transitions that pass `selection`.

    Parameters
    ----------
    table : astropy.table.Table
        Output of `_nist_transitions`.
    selection : LineSelection

    Returns
    -------
    mask : array of bool
    """
    # NaN (missing f) fails every comparison, so those lines are dropped
    fik = np.ma.filled(table["fik"], np.nan)
    wave = np.asarray(table["wave"])
    # Only allowed electric-dipole (E1) lines: forbidden ones (M1, E2) are
    # too weak to absorb in ISM/CGM gas, and f_lu is defined for E1.
    # NIST leaves "Type" empty for E1 transitions
    is_electric_dipole = np.ma.getmaskarray(table["Type"])
    return (
        is_electric_dipole
        & (np.asarray(table["e_low"]) <= selection.e_low_max)
        & (fik >= selection.f_min)
        & (wave >= selection.wave_min)
        & (wave <= selection.wave_max)
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
    ritz = np.ma.filled(table["Ritz"], np.nan)
    observed = np.ma.filled(table["Observed"], np.nan)
    return np.where(np.isnan(ritz), observed, ritz)


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
    match = re.search(r"\d+\.?\d*", str(entry))
    return float(match[0]) if match else np.nan
