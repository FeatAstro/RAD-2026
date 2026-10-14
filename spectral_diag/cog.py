"""Theoretical curves of growth and multi-line fits for N and b.

- theoretical_ew: W(logN, b) for a given Line, valid in the linear,
  saturated and damped regimes
- fit_cog: given measured EWs for several lines of the same species,
  solve for (logN, b)

Assumes a single absorbing cloud (one velocity component) and no
instrumental broadening.
"""

import numpy as np
from scipy.optimize import least_squares

from .core import C_KMS
from .profiles import A_CONST, TAU0_CONST, voigt

# Beyond x = _X_CORE Doppler widths the Gaussian core is negligible
# (exp(-64)) and only the Lorentzian wing a / (sqrt(pi) x^2) remains.
_X_CORE = 8.0
_TAU_STOP = 1e-3


def _reduced_ew(tau0, a):
    """Integral of 1 - exp(-tau0 * H(x, a)) over x from 0 to infinity."""
    x = np.linspace(0.0, _X_CORE, 801)
    wing = tau0 * a / np.sqrt(np.pi)  # tau ~ wing / x^2 far from the core
    x_max = np.sqrt(wing / _TAU_STOP)
    if x_max > _X_CORE:
        x = np.concatenate([x, np.geomspace(_X_CORE, x_max, 600)[1:]])
    integral = np.trapezoid(-np.expm1(-tau0 * voigt(x, a)), x)
    # Beyond the grid tau << 1, so the integrand is tau itself and the
    # remaining area is wing / x_end.
    return integral + wing / x[-1]


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
        Doppler parameter, km/s.
    z : float
        Redshift of the absorber. The result is the observed-frame
        EW, i.e. the rest-frame EW times (1 + z).

    Returns
    -------
    W : float or array
        Equivalent width, Angstrom.
    """
    logN, b = np.broadcast_arrays(np.asarray(logN, dtype=float), np.asarray(b, dtype=float))
    tau0 = TAU0_CONST * 10**logN * line.f * line.wave_rest / b
    a = A_CONST * line.gamma * line.wave_rest / b
    reduced = np.array([_reduced_ew(t, ai) for t, ai in zip(tau0.ravel(), a.ravel())])
    W = 2 * line.wave_rest * b / C_KMS * (1 + z) * reduced.reshape(b.shape)
    return W if W.ndim else float(W)


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
        Angstrom, as returned by `ew.measure_ew`.
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
    names = [line.name for line in lines]
    if len(set(names)) != len(names):
        raise ValueError("line names must be unique")
    unknown = set(ews) - set(names)
    if unknown:
        raise ValueError(f"no line named {sorted(unknown)} in `lines`")
    used = [line for line in lines if line.name in ews]
    if len(used) < 2:
        raise ValueError("need measured EWs for at least two lines")

    W = np.array([ews[line.name][0] for line in used])
    W_err = np.array([ews[line.name][1] for line in used])

    def residuals(p):
        model = np.array([theoretical_ew(line, p[0], p[1], z) for line in used])
        return (model - W) / W_err

    grid_logN = np.arange(logN_bounds[0] + 0.5, logN_bounds[1], 0.5)
    grid_b = np.geomspace(b_bounds[0] * 2, b_bounds[1] / 2, 7)
    starts = sorted(
        ((logN, b) for logN in grid_logN for b in grid_b),
        key=lambda p: np.sum(residuals(p) ** 2),
    )
    # Saturated lines can leave several local minima along the N-b
    # valley, so refine from the few best grid points and keep the best.
    fit = min(
        (
            least_squares(
                residuals,
                start,
                bounds=([logN_bounds[0], b_bounds[0]], [logN_bounds[1], b_bounds[1]]),
                x_scale=[1.0, 10.0],
            )
            for start in starts[:5]
        ),
        key=lambda f: f.cost,
    )

    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        logN_err, b_err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
        logN_err = b_err = np.nan

    return {
        "logN": fit.x[0],
        "b": fit.x[1],
        "logN_err": logN_err,
        "b_err": b_err,
        "cov": cov,
        "chi2": float(np.sum(fit.fun**2)),
        "dof": len(used) - 2,
    }
