"""Tests of the chi2 map of curve-of-growth fits. Wavelengths in Angstrom."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from spectral_diag.curve_of_growth import fit_cog, theoretical_ew
from spectral_diag.line_list import get_lines
from spectral_diag.plots import plot_chi2_map, plot_profiles
from spectral_diag.uncertainty import (
    ONE_PARAMETER_95,
    Chi2Map,
    _chi2_scale,
    _refined_minimum,
    chi2_map,
    profile_interval,
)

TRUE_LOGN = 14.5
TRUE_B = 20.0
# Large enough for the 68 % interval to span several grid steps (0.025 dex)
RELATIVE_EW_ERROR = 0.2
# Hand-built maps: chi2 = ((logN - CENTRE) / WIDTH)**2, the same at every b
CENTRE = 15.0
WIDTH = 0.1


def sulphur_ews():
    """Exact EWs of the S II triplet, {line name: (W, W_err)}.

    Returns
    -------
    lines : list of Line
    ews : dict
    """
    lines = get_lines("SII")
    ews = {}
    for line in lines:
        width = theoretical_ew(line, TRUE_LOGN, TRUE_B)
        ews[line.name] = (width, RELATIVE_EW_ERROR * width)
    return lines, ews


def parabola_map(chi2_offset=0.0, scale=1.0):
    """Chi2Map whose chi2 is a parabola in logN, flat in b.

    Parameters
    ----------
    chi2_offset : float
        Added everywhere, as the chi2 of a poor fit.
    scale : float

    Returns
    -------
    chi2_map : Chi2Map
    """
    logN_values = np.linspace(14.0, 16.0, 201)
    b_values = np.geomspace(1.0, 100.0, 5)
    chi2 = np.zeros((len(b_values), len(logN_values)))
    for row in range(len(b_values)):
        chi2[row] = ((logN_values - CENTRE) / WIDTH) ** 2 + chi2_offset
    return Chi2Map(logN_values, b_values, chi2, chi2_offset, scale, CENTRE, 10.0)


def test_map_agrees_with_the_fit():
    """The map is lowest at the fit, and its interval matches the covariance error."""
    lines, ews = sulphur_ews()
    cog_fit = fit_cog(lines, ews)
    result = chi2_map(lines, ews, cog_fit)
    interval = profile_interval(result, "logN")
    assert result.scale == 1.0
    assert abs(interval.best - TRUE_LOGN) < 0.025
    half_width = (interval.high - interval.low) / 2
    assert half_width == pytest.approx(cog_fit["logN_err"], rel=0.1)


def test_interval_of_a_parabola():
    """Delta chi2 = 1 at one width, 3.84 at 1.96 widths."""
    result = parabola_map()
    interval = profile_interval(result, "logN")
    assert interval.low == pytest.approx(CENTRE - WIDTH, abs=1e-3)
    assert interval.high == pytest.approx(CENTRE + WIDTH, abs=1e-3)
    wide = profile_interval(result, "logN", ONE_PARAMETER_95)
    assert wide.high == pytest.approx(CENTRE + 1.96 * WIDTH, abs=1e-3)


def test_scale_widens_the_interval():
    """Dividing chi2 by 4 doubles the interval."""
    interval = profile_interval(parabola_map(scale=4.0), "logN")
    assert interval.high == pytest.approx(CENTRE + 2 * WIDTH, abs=1e-3)


def test_unscaled_map_with_replace():
    """replace(scale=1) gives back the raw interval."""
    raw = replace(parabola_map(scale=4.0), scale=1.0)
    interval = profile_interval(raw, "logN")
    assert interval.high == pytest.approx(CENTRE + WIDTH, abs=1e-3)


def test_chi2_scale():
    """Only scaled up, and not at all with no degree of freedom."""
    assert _chi2_scale(10.0, 2) == 5.0
    assert _chi2_scale(0.5, 2) == 1.0
    assert _chi2_scale(3.0, 0) == 1.0


def test_flat_parameter_reaches_the_grid_edges():
    """b does not change chi2 here: its interval is the whole grid, as limits."""
    interval = profile_interval(parabola_map(), "b")
    assert interval.low_at_grid_edge
    assert interval.high_at_grid_edge


def test_unknown_parameter():
    """Only logN and b exist."""
    with pytest.raises(ValueError):
        profile_interval(parabola_map(), "z")


def test_plots_return_figures():
    """The map and profile plots run on a map."""
    fig, ax = plot_chi2_map(parabola_map(), "test")
    assert ax.get_yscale() == "log"
    fig, axes = plot_profiles(parabola_map(), "test")
    assert len(axes) == 2


def test_refined_minimum_of_a_parabola():
    """The vertex of a sampled parabola is found between the samples."""
    positions = np.arange(-3.0, 4.0)
    samples = (positions - 0.3) ** 2 + 2.0
    assert _refined_minimum(samples) == pytest.approx(2.0)
