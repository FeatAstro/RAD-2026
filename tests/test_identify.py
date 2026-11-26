"""Tests of the comb identification on hand-built features tables.

The true features are the lines of a Milky-Way-like absorber in HST/COS
G130M (1142-1444 Angstrom, gap at 1289-1298 Angstrom), with the equivalent
widths of the curve of growth; spurious features are placed at random.
Wavelengths in Angstrom, velocities in km/s.
"""

import numpy as np
from astropy.table import Table

from spectral_diag import constants
from spectral_diag.curve_of_growth import theoretical_ew
from spectral_diag.data_classes import Spectrum
from spectral_diag.detect import CombSettings, identify
from spectral_diag.line_list import LineSelection, get_lines

COS_SEGMENTS = ((1142.0, 1289.0), (1298.0, 1444.0))
SPECIES = ["HI", "CII", "NI", "NV", "OI", "SiII", "SiIII", "SiIV", "SII", "FeII"]
# log10 N (cm^-2) of the test absorber, of the order of the Milky Way
# towards 3C 273
LOG_COLUMNS = {
    "CII": 15.5,
    "NI": 14.8,
    "OI": 15.8,
    "SiII": 15.0,
    "SiIII": 13.5,
    "SiIV": 13.3,
    "SII": 15.2,
    "FeII": 14.7,
}
DOPPLER_B = 15.0
SYSTEM_VELOCITY = -12.0
# Errors of a feature centre and of its EW, of the order of COS at S/N 20
WAVE_ERR = 0.005
EW_ERR = 0.004
# Lines weaker than this many EW errors are not detected
DETECTION_IN_EW_ERRORS = 4.0
SPURIOUS_EW_RANGE = (0.02, 0.3)
N_SPURIOUS = 40
N_RANDOM_SEEDS = 200
# Measured 0 false systems in 200 sets of 40 random features; fail if
# clearly worse
MAX_FALSE_SYSTEMS = 2
MAX_VELOCITY_ERROR = 2.0


def cos_like_spectrum(segments=COS_SEGMENTS):
    """Flat spectrum used only for its wavelength coverage.

    Parameters
    ----------
    segments : sequence of (float, float)
        Covered wavelength ranges, Angstrom.

    Returns
    -------
    spectrum : Spectrum
    """
    wave = np.concatenate([np.arange(start, end, 0.01) for start, end in segments])
    return Spectrum(wave, np.ones_like(wave), np.ones_like(wave))


def is_inside(wave, segments):
    """True if `wave` falls inside one of `segments`.

    Parameters
    ----------
    wave : float
    segments : sequence of (float, float)

    Returns
    -------
    inside : bool
    """
    for start, end in segments:
        if start < wave < end:
            return True
    return False


def absorber_rows(rng, log_columns=LOG_COLUMNS, segments=COS_SEGMENTS):
    """Detectable lines of the test absorber, as features table rows.

    Parameters
    ----------
    rng : numpy.random.Generator
    log_columns : dict
        Species -> log10 N.
    segments : sequence of (float, float)

    Returns
    -------
    rows : list of tuple
        (wave, ew, ew_err, truth), truth = Line.name.
    """
    rows = []
    z = SYSTEM_VELOCITY / constants.C_KMS
    for species, log_column in log_columns.items():
        for line in get_lines(species, LineSelection(1100, 1500, f_min=1e-3)):
            wave = line.wave_rest * (1 + z) + rng.normal(0, WAVE_ERR)
            ew = theoretical_ew(line, log_column, DOPPLER_B, z) + rng.normal(0, EW_ERR)
            if is_inside(wave, segments) and ew > DETECTION_IN_EW_ERRORS * EW_ERR:
                rows.append((wave, ew, EW_ERR, line.name))
    return rows


def random_rows(rng, n_rows):
    """Features at random wavelengths, with random EWs.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_rows : int

    Returns
    -------
    rows : list of tuple
        Same layout as `absorber_rows`, truth = "".
    """
    rows = []
    for _ in range(n_rows):
        wave = rng.uniform(COS_SEGMENTS[0][0], COS_SEGMENTS[-1][1])
        rows.append((wave, rng.uniform(*SPURIOUS_EW_RANGE), EW_ERR, ""))
    return rows


def features_table(rows):
    """Features table sorted by wavelength, with the true line names.

    Parameters
    ----------
    rows : list of tuple

    Returns
    -------
    features : astropy.table.Table
        Columns wave, ew, ew_err and truth.
    """
    return Table(rows=sorted(rows), names=["wave", "ew", "ew_err", "truth"])


def test_finds_the_system_and_labels_every_line():
    """The system velocity is recovered; spurious features stay blank."""
    rng = np.random.default_rng(1)
    rows = absorber_rows(rng) + random_rows(rng, N_SPURIOUS)
    identified = identify(features_table(rows), cos_like_spectrum(), SPECIES)
    assert abs(identified.meta["systems"][0] - SYSTEM_VELOCITY) < MAX_VELOCITY_ERROR
    for row in identified:
        assert row["line"] == row["truth"]


def test_tooth_in_a_gap_does_not_count_as_missing():
    """SiII is found although its strongest line, 1260, falls in a gap."""
    segments = ((1142.0, 1255.0), (1265.0, 1444.0))
    rows = absorber_rows(np.random.default_rng(2), {"SiII": 15.0}, segments)
    spectrum = cos_like_spectrum(segments)
    identified = identify(features_table(rows), spectrum, ["SiII"])
    assert all(identified["species"] == "SiII")


def test_line_too_strong_for_its_tooth_is_rejected():
    """An unrelated strong line on the position of SII 1250 is not SII."""
    rows = absorber_rows(np.random.default_rng(3))
    for index, (wave, ew, ew_err, truth) in enumerate(rows):
        if truth == "SII 1250.584":
            rows[index] = (wave, 10 * ew, ew_err, "")
    identified = identify(features_table(rows), cos_like_spectrum(), SPECIES)
    for row in identified:
        assert row["line"] == row["truth"]


def test_chance_systems_rare_in_random_features():
    """Random features alone rarely give a system."""
    false_systems = 0
    for seed in range(N_RANDOM_SEEDS):
        rng = np.random.default_rng(100 + seed)
        features = features_table(random_rows(rng, N_SPURIOUS))
        identified = identify(features, cos_like_spectrum(), SPECIES)
        false_systems += len(identified.meta["systems"])
    assert false_systems <= MAX_FALSE_SYSTEMS


def test_balmer_series_found_with_excited_levels():
    """The Balmer lines (lower level n = 2) form a comb once e_low_max allows it."""
    balmer = get_lines(
        "HI",
        LineSelection(
            wave_min=3800, wave_max=7000, e_low_max=11.0, multiplet_totals=True
        ),
    )
    rows = []
    for line in balmer:
        wave = line.wave_rest * (1 + SYSTEM_VELOCITY / constants.C_KMS)
        rows.append((wave, 10.0, 0.3, line.name))
    wave = np.arange(3800.0, 9200.0, 1.0)
    spectrum = Spectrum(wave=wave, flux=np.ones(wave.size), error=np.ones(wave.size))
    features = features_table(rows)

    ground_only = identify(features, spectrum, ["HI"])
    assert ground_only.meta["systems"] == []

    settings = CombSettings(e_low_max=11.0, strength_tolerance_in_errors=np.inf)
    with_excited = identify(features, spectrum, ["HI"], settings)
    for row in with_excited:
        assert row["line"] == row["truth"]
