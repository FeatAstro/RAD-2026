"""Tests of measured_ews and of the curve-of-growth plot. Wavelengths in Angstrom."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from astropy.table import Table

from spectral_diag.curve_of_growth import fit_cog, measured_ews, theoretical_ew
from spectral_diag.line_list import get_lines
from spectral_diag.plots import plot_curve_of_growth

TRUE_LOGN = 15.3
TRUE_B = 20.0
REDSHIFT = -4e-5
RELATIVE_EW_ERROR = 0.02


def identified_table():
    """Identified features: the S II triplet at its true EWs, one Si II line, one blank.

    Returns
    -------
    identified : astropy.table.Table
    """
    sulphur_lines = get_lines("SII")
    silicon_line = get_lines("SiII")[0]
    species, names, ews = [], [], []
    for line in sulphur_lines:
        species.append("SII")
        names.append(line.name)
        ews.append(theoretical_ew(line, TRUE_LOGN, TRUE_B, REDSHIFT))
    species += ["SiII", ""]
    names += [silicon_line.name, ""]
    ews += [0.5, 0.1]
    ew_errors = [RELATIVE_EW_ERROR * ew for ew in ews]
    return Table({"species": species, "line": names, "ew": ews, "ew_err": ew_errors})


def test_measured_ews_keeps_only_the_species():
    ews = measured_ews(identified_table(), "SII")
    expected_names = [line.name for line in get_lines("SII")]
    assert sorted(ews) == sorted(expected_names)


def test_fit_on_measured_ews_recovers_true_values():
    ews = measured_ews(identified_table(), "SII")
    cog_fit = fit_cog(get_lines("SII"), ews, z=REDSHIFT)
    assert abs(cog_fit["logN"] - TRUE_LOGN) < 0.01
    assert abs(cog_fit["b"] - TRUE_B) < 0.1
    assert cog_fit["z"] == REDSHIFT


def test_plot_has_one_point_per_measured_line():
    ews = measured_ews(identified_table(), "SII")
    cog_fit = fit_cog(get_lines("SII"), ews, z=REDSHIFT)
    fig, ax = plot_curve_of_growth(get_lines("SII"), ews, cog_fit)
    # Each measured line is drawn as one error bar
    assert len(ax.containers) == len(ews)
    assert np.isfinite(ax.get_xlim()).all()
