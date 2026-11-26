"""Uncertainties of a curve-of-growth fit from a chi2 map over (logN, b).

The errors of `fit_cog` come from the curvature of chi2 at its minimum, which
assumes chi2 is a parabola there. For saturated lines N and b trade against
each other and chi2 is a long curved valley instead. The map evaluates chi2
on a grid, so the confidence regions follow its real shape.

When the best fit is poor (chi2_min / dof > 1), chi2 is divided by
chi2_min / dof before the confidence levels are applied: the errors are
widened as if the EW errors had been underestimated by that factor. For
the unscaled map, use `dataclasses.replace(chi2_map, scale=1.0)`.
"""

from dataclasses import dataclass

import numpy as np

from .curve_of_growth import theoretical_ew

# Delta chi2 of the 68, 95 and 99.7 % joint regions of two parameters
JOINT_LEVELS = (2.30, 6.17, 11.8)
# Delta chi2 of the 68 and 95 % intervals of one parameter, the other one minimised
ONE_PARAMETER_68 = 1.0
ONE_PARAMETER_95 = 3.84
# Grid of the map: logN (cm^-2) around the fit, b (km/s) log-spaced
LOGN_HALF_RANGE = 3.0
LOGN_POINTS = 241
B_RANGE = (1.0, 100.0)
B_POINTS = 150


@dataclass(frozen=True)
class Chi2Map:
    """chi2 of a curve-of-growth fit on a (logN, b) grid.

    Attributes
    ----------
    logN_values : array
        log10 N (cm^-2) of the grid columns.
    b_values : array
        b (km/s) of the grid rows.
    chi2 : array, shape (len(b_values), len(logN_values))
    chi2_min : float
        Lowest chi2, of the fit or of the grid.
    scale : float
        chi2_min / dof if above 1, else 1 (also 1 with no degree of
        freedom).
    best_logN, best_b : float
        Best fit of `fit_cog`.
    """

    logN_values: np.ndarray
    b_values: np.ndarray
    chi2: np.ndarray
    chi2_min: float
    scale: float
    best_logN: float
    best_b: float

    def delta_chi2(self):
        """(chi2 - chi2_min) / scale at every grid point.

        Returns
        -------
        delta : array, same shape as `chi2`
        """
        return (self.chi2 - self.chi2_min) / self.scale


@dataclass(frozen=True)
class Interval:
    """Confidence interval of one parameter, from the profile of the chi2 map.

    Attributes
    ----------
    best : float
        Value at the lowest chi2 of the profile.
    low, high : float
        Ends of the interval. At an edge of the grid, the edge itself.
    low_at_grid_edge, high_at_grid_edge : bool
        The interval reaches that edge of the grid: that end is only a
        limit, the true end lies beyond it.
    """

    best: float
    low: float
    high: float
    low_at_grid_edge: bool
    high_at_grid_edge: bool


def chi2_map(lines, ews, cog_fit):
    """chi2 of a curve-of-growth fit on a (logN, b) grid around it.

    Parameters
    ----------
    lines : list of Line
        Lines of the species; only those with an entry in `ews` are used.
    ews : dict
        {line name: (W, W_err)}, Angstrom, as `fit_cog` takes them.
    cog_fit : dict
        Output of `fit_cog` (uses logN, b, chi2, dof and z).

    Returns
    -------
    chi2_map : Chi2Map
    """
    measured_lines = [line for line in lines if line.name in ews]
    logN_values = np.linspace(
        cog_fit["logN"] - LOGN_HALF_RANGE,
        cog_fit["logN"] + LOGN_HALF_RANGE,
        LOGN_POINTS,
    )
    b_values = np.geomspace(B_RANGE[0], B_RANGE[1], B_POINTS)
    chi2 = np.zeros((len(b_values), len(logN_values)))
    for line in measured_lines:
        width, width_error = ews[line.name]
        model_widths = _model_ew_grid(line, logN_values, b_values, cog_fit["z"])
        chi2 += ((model_widths - width) / width_error) ** 2
    chi2_min = min(cog_fit["chi2"], np.min(chi2))
    scale = _chi2_scale(chi2_min, cog_fit["dof"])
    return Chi2Map(
        logN_values, b_values, chi2, chi2_min, scale, cog_fit["logN"], cog_fit["b"]
    )


def profile(chi2_map, parameter):
    """Lowest delta chi2 over the other parameter, at each value of this one.

    The valley of saturated lines can be narrower than a grid step of the
    other parameter, so the lowest grid value is refined with a parabola
    through it and its two neighbours (`_refined_minimum`).

    Parameters
    ----------
    chi2_map : Chi2Map
    parameter : str
        "logN" or "b".

    Returns
    -------
    values : array
        Grid of the parameter.
    profile : array
        Delta chi2 (rescaled, see `Chi2Map.delta_chi2`) at each value.
    """
    delta = chi2_map.delta_chi2()
    if parameter == "logN":
        values = chi2_map.logN_values
        lines_of_other_parameter = delta.T
    elif parameter == "b":
        values = chi2_map.b_values
        lines_of_other_parameter = delta
    else:
        raise ValueError(f"parameter must be 'logN' or 'b', not {parameter!r}")
    profile_values = []
    for samples in lines_of_other_parameter:
        profile_values.append(_refined_minimum(samples))
    return values, np.array(profile_values)


def profile_interval(chi2_map, parameter, level=ONE_PARAMETER_68):
    """Interval of logN or b where the profile of delta chi2 is below `level`.

    The ends are interpolated linearly between grid points, so an interval
    is only resolved if it spans a few of them: logN steps are
    2 * LOGN_HALF_RANGE / (LOGN_POINTS - 1) = 0.025 dex, b steps 3 %.

    Parameters
    ----------
    chi2_map : Chi2Map
    parameter : str
        "logN" or "b".
    level : float
        ONE_PARAMETER_68 or ONE_PARAMETER_95.

    Returns
    -------
    interval : Interval
    """
    values, profile_values = profile(chi2_map, parameter)
    return _interval_below(values, profile_values, level)


def _model_ew_grid(line, logN_values, b_values, z):
    """Theoretical EW of one line at every (b, logN) of the grid.

    Parameters
    ----------
    line : Line
    logN_values, b_values : array
    z : float

    Returns
    -------
    widths : array, shape (len(b_values), len(logN_values))
        Angstrom.
    """
    widths = np.zeros((len(b_values), len(logN_values)))
    for row, b in enumerate(b_values):
        for column, logN in enumerate(logN_values):
            widths[row, column] = theoretical_ew(line, logN, b, z)
    return widths


def _refined_minimum(samples):
    """Lowest value of a parabola through the lowest sample and its neighbours.

    On a narrow valley the lowest grid sample can miss the bottom by up to
    ~1 in delta chi2; the parabola brings it within a few hundredths
    (checked against a 2000-point grid in notebook 08).

    Parameters
    ----------
    samples : array
        Values on an evenly spaced grid (logN, or log b).

    Returns
    -------
    minimum : float
        The lowest sample itself at an edge of the grid, or if the three
        points do not curve upwards.
    """
    index = int(np.argmin(samples))
    if index == 0 or index == len(samples) - 1:
        return samples[index]
    before, lowest, after = samples[index - 1], samples[index], samples[index + 1]
    curvature = after - 2 * lowest + before
    if curvature <= 0:
        return lowest
    return lowest - (after - before) ** 2 / (8 * curvature)


def _chi2_scale(chi2_min, dof):
    """Factor that brings chi2_min / dof down to 1, never below 1.

    Parameters
    ----------
    chi2_min : float
    dof : int

    Returns
    -------
    scale : float
        1 with no degree of freedom: the fit then passes through every
        point and says nothing about the errors.
    """
    if dof < 1:
        return 1.0
    return max(1.0, chi2_min / dof)


def _interval_below(values, profile, level):
    """Range around the minimum of `profile` where it stays below `level`.

    A second region below `level`, separated from the minimum by a bump
    above it, is not included.

    Parameters
    ----------
    values : array
        Grid of the parameter, increasing.
    profile : array
        Delta chi2 at each value.
    level : float

    Returns
    -------
    interval : Interval
    """
    best_index = int(np.argmin(profile))
    first = best_index
    while first > 0 and profile[first - 1] <= level:
        first -= 1
    last = best_index
    while last < len(values) - 1 and profile[last + 1] <= level:
        last += 1
    low_at_grid_edge = first == 0
    high_at_grid_edge = last == len(values) - 1
    low = values[first]
    if not low_at_grid_edge:
        low = _crossing(values, profile, first - 1, level)
    high = values[last]
    if not high_at_grid_edge:
        high = _crossing(values, profile, last, level)
    return Interval(values[best_index], low, high, low_at_grid_edge, high_at_grid_edge)


def _crossing(values, profile, index, level):
    """Value where `profile` crosses `level` between two neighbouring points.

    Parameters
    ----------
    values, profile : array
    index : int
        The crossing lies between `index` and `index + 1`.
    level : float

    Returns
    -------
    value : float
        Linear interpolation.
    """
    step = values[index + 1] - values[index]
    rise = profile[index + 1] - profile[index]
    return values[index] + (level - profile[index]) * step / rise
