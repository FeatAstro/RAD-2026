"""Theoretical curves of growth and multi-line fits for N and b.

- theoretical_ew: W(logN, b) for a given Line, valid in the linear,
  saturated and damped regimes
- ew_linear: analytic W in the optically thin limit (independent of b)
- fit_cog: given measured EWs for several lines of the same species,
  solve for (logN, b)

Assumes a single absorbing cloud (one velocity component) and no
instrumental broadening.
"""

import numpy as np
from scipy.optimize import least_squares

from .constants import C_KMS
from .line_profiles import A_CONST, TAU0_CONST, voigt

# Beyond this many Doppler widths from the centre the Gaussian core is
# negligible (exp(-64)) and only the Lorentzian wing a / (sqrt(pi) x^2)
# remains
CORE_HALF_WIDTH = 8.0
# Number of integration points across the core, and in the far wing
CORE_POINTS = 801
WING_POINTS = 600
# The numerical integral stops where the wing optical depth falls below
# this; the rest of the wing is added analytically
WING_END_OPTICAL_DEPTH = 1e-3

# Coarse (logN, b) grid that gives the starting points of the fit
GRID_STEP_LOGN = 0.5
GRID_POINTS_B = 7
# Saturated lines can leave several local minima along the N-b valley,
# so the fit is refined from this many of the best grid points
NUMBER_OF_STARTS = 5
# Typical change of (logN, b) in one fit step: 1 dex and 10 km/s
PARAMETER_SCALE = [1.0, 10.0]


def theoretical_ew(line, logN, b, z=0.0):
    """Equivalent width of one transition from a single cloud.

    W = 2 * (lambda_rest * b / c) * (1 + z) * F(tau0, a), where F is
    the integral of 1 - exp(-tau0 * H(x, a)) over x = dv / b >= 0.

    Parameters
    ----------
    line : Line
        Atomic transition (wave_rest in Angstrom, f, gamma).
    logN : float or array
        log10(column density / cm^-2).
    b : float or array
        Doppler parameter, km/s. Arrays of logN and b are paired
        element by element (numpy broadcasting rules).
    z : float
        Redshift of the absorber. The result is the observed-frame
        EW, i.e. the rest-frame EW times (1 + z).

    Returns
    -------
    equivalent_width : float or array
        Angstrom. A float if logN and b are single numbers, else an
        array of their common shape.
    """
    if np.ndim(logN) == 0 and np.ndim(b) == 0:
        return _single_cloud_ew(line, logN, b, z)
    logN_values, b_values = np.broadcast_arrays(logN, b)
    equivalent_widths = np.zeros(logN_values.shape)
    for index in np.ndindex(logN_values.shape):
        equivalent_widths[index] = _single_cloud_ew(
            line, logN_values[index], b_values[index], z
        )
    return equivalent_widths


def _single_cloud_ew(line, logN, b, z):
    """Equivalent width of one transition for one value of (logN, b).

    Parameters
    ----------
    line : Line
    logN : float
        log10(column density / cm^-2).
    b : float
        Doppler parameter, km/s.
    z : float
        Redshift of the absorber.

    Returns
    -------
    equivalent_width : float
        Observed frame, Angstrom.
    """
    central_optical_depth = TAU0_CONST * 10**logN * line.f * line.wave_rest / b
    damping_parameter = A_CONST * line.gamma * line.wave_rest / b
    doppler_width = line.wave_rest * b / C_KMS
    reduced_width = _reduced_ew(central_optical_depth, damping_parameter)
    return float(2 * doppler_width * (1 + z) * reduced_width)


def _reduced_ew(central_optical_depth, damping_parameter):
    """Integral of 1 - exp(-tau0 * H(x, a)) over x from 0 to infinity.

    Parameters
    ----------
    central_optical_depth : float
        tau0.
    damping_parameter : float
        a.

    Returns
    -------
    reduced_width : float
        Equivalent width in units of the Doppler width lambda_rest b / c,
        for half of the line (x >= 0).
    """
    # Far from the core the optical depth is wing_strength / x^2
    wing_strength = central_optical_depth * damping_parameter / np.sqrt(np.pi)
    offsets = _integration_offsets(wing_strength)
    optical_depth = central_optical_depth * voigt(offsets, damping_parameter)
    # -expm1(-tau) is 1 - exp(-tau), but stays accurate when tau is tiny
    absorbed_fraction = -np.expm1(-optical_depth)
    integral = np.trapezoid(absorbed_fraction, offsets)
    # Beyond the last offset tau << 1, so 1 - exp(-tau) is tau itself, and
    # the integral of wing_strength / x^2 from there to infinity is:
    remaining_wing = wing_strength / offsets[-1]
    return integral + remaining_wing


def _integration_offsets(wing_strength):
    """Offsets x = dv / b at which the line profile is integrated.

    Evenly spaced over the core; for damped lines, extended with
    logarithmically spaced points out to where the wing optical depth
    falls to WING_END_OPTICAL_DEPTH.

    Parameters
    ----------
    wing_strength : float
        tau0 * a / sqrt(pi), so that tau = wing_strength / x^2 in the wing.

    Returns
    -------
    offsets : array
        Increasing, from 0.
    """
    core_offsets = np.linspace(0.0, CORE_HALF_WIDTH, CORE_POINTS)
    wing_end = np.sqrt(wing_strength / WING_END_OPTICAL_DEPTH)
    if wing_end <= CORE_HALF_WIDTH:
        return core_offsets
    wing_offsets = np.geomspace(CORE_HALF_WIDTH, wing_end, WING_POINTS)
    # The first wing offset equals the last core offset: skip it
    return np.concatenate([core_offsets, wing_offsets[1:]])


def fit_cog(lines, ews, z=0.0, logN_bounds=(10.0, 22.0), b_bounds=(1.0, 200.0)):
    """Fit (logN, b) of one species to the measured EWs of several lines.

    Minimises chi2 = sum(((W_measured - W_model) / W_err)**2), starting
    from the best points of a coarse (logN, b) grid.

    Parameters
    ----------
    lines : list of Line
        Candidate transitions of one species. Only those whose `name`
        has an entry in `ews` are used.
    ews : dict
        Measured observed-frame EWs, {line.name: (W, W_err)} in
        Angstrom, as returned by `equivalent_width.measure_ew`.
    z : float
        Redshift of the absorber.
    logN_bounds, b_bounds : (float, float)
        Allowed range of each parameter.

    Returns
    -------
    result : dict
        logN, b : best-fit values.
        logN_err, b_err : 1-sigma uncertainties from the covariance
            matrix (nan if the fit does not constrain them).
        cov : 2x2 covariance matrix of (logN, b).
        chi2, dof : chi-square at the best fit and degrees of freedom.
    """
    measured_lines = _lines_with_measurements(lines, ews)

    def normalised_residuals(parameters):
        logN, b = parameters
        residuals = []
        for line in measured_lines:
            measured_width, measured_error = ews[line.name]
            model_width = theoretical_ew(line, logN, b, z)
            residuals.append((model_width - measured_width) / measured_error)
        return np.array(residuals)

    starts = _best_grid_points(normalised_residuals, logN_bounds, b_bounds)
    fit = _best_fit_from_starts(normalised_residuals, starts, logN_bounds, b_bounds)
    covariance = _parameter_covariance(fit)
    return {
        "logN": fit.x[0],
        "b": fit.x[1],
        "logN_err": np.sqrt(covariance[0, 0]),
        "b_err": np.sqrt(covariance[1, 1]),
        "cov": covariance,
        "chi2": float(np.sum(fit.fun**2)),
        "dof": len(measured_lines) - 2,
    }


def _lines_with_measurements(lines, ews):
    """The lines that have a measured EW, after checking the inputs.

    Parameters
    ----------
    lines : list of Line
    ews : dict
        {line.name: (W, W_err)}.

    Returns
    -------
    measured_lines : list of Line

    Raises
    ------
    ValueError
        If two lines share a name, an EW has no matching line, or fewer
        than two lines are measured.
    """
    names = [line.name for line in lines]
    if len(set(names)) != len(names):
        raise ValueError("line names must be unique")
    unknown_names = set(ews) - set(names)
    if unknown_names:
        raise ValueError(f"no line named {sorted(unknown_names)} in `lines`")
    measured_lines = [line for line in lines if line.name in ews]
    if len(measured_lines) < 2:
        raise ValueError("need measured EWs for at least two lines")
    return measured_lines


def _best_grid_points(normalised_residuals, logN_bounds, b_bounds):
    """The NUMBER_OF_STARTS points of a coarse (logN, b) grid with lowest chi2.

    Parameters
    ----------
    normalised_residuals : function
        (logN, b) -> array of (W_model - W_measured) / W_err.
    logN_bounds, b_bounds : (float, float)

    Returns
    -------
    starts : list of (float, float)
        Best first.
    """
    logN_grid = np.arange(
        logN_bounds[0] + GRID_STEP_LOGN, logN_bounds[1], GRID_STEP_LOGN
    )
    b_grid = np.geomspace(b_bounds[0] * 2, b_bounds[1] / 2, GRID_POINTS_B)
    grid_points = []
    for logN in logN_grid:
        for b in b_grid:
            chi2 = np.sum(normalised_residuals((logN, b)) ** 2)
            grid_points.append((chi2, logN, b))
    grid_points.sort()
    best_points = grid_points[:NUMBER_OF_STARTS]
    return [(logN, b) for chi2, logN, b in best_points]


def _best_fit_from_starts(normalised_residuals, starts, logN_bounds, b_bounds):
    """Least-squares fit from each starting point; the one with the lowest chi2.

    Parameters
    ----------
    normalised_residuals : function
        (logN, b) -> array of (W_model - W_measured) / W_err.
    starts : list of (float, float)
    logN_bounds, b_bounds : (float, float)

    Returns
    -------
    best_fit : scipy.optimize.OptimizeResult
        `x` holds (logN, b), `fun` the residuals, `jac` their Jacobian.
    """
    lower_bounds = [logN_bounds[0], b_bounds[0]]
    upper_bounds = [logN_bounds[1], b_bounds[1]]
    best_fit = None
    for start in starts:
        fit = least_squares(
            normalised_residuals,
            start,
            bounds=(lower_bounds, upper_bounds),
            x_scale=PARAMETER_SCALE,
        )
        if best_fit is None or fit.cost < best_fit.cost:
            best_fit = fit
    return best_fit


def _parameter_covariance(fit):
    """Covariance of (logN, b), (J^T J)^-1 with J the Jacobian of the residuals.

    Parameters
    ----------
    fit : scipy.optimize.OptimizeResult

    Returns
    -------
    covariance : array, shape (2, 2)
        NaN everywhere if the fit does not constrain both parameters.
    """
    jacobian = fit.jac
    try:
        return np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan)


def ew_linear(line, logN):
    """Rest-frame equivalent width in the linear (optically thin) regime.

    W = sqrt(pi) * TAU0_CONST * N * f * lambda_rest**2 / c, independent of b.

    Parameters
    ----------
    line : Line
        Atomic transition (wave_rest in Angstrom, f).
    logN : float or array
        log10(column density / cm^-2).

    Returns
    -------
    equivalent_width : float or array
        Angstrom.
    """
    column_density = 10**logN
    return (
        TAU0_CONST
        * np.sqrt(np.pi)
        * column_density
        * line.f
        * line.wave_rest**2
        / C_KMS
    )
