"""Interactive spectrum and fit visualisation.

Planned contents:
- plot_spectrum: flux/error/continuum with identified line windows
  marked
- plot_fit: data vs. model profile for a fitted Absorber
- plot_curve_of_growth: measured EWs over a theoretical curve of growth
- plot_chi2_map: confidence regions of a curve-of-growth fit in (logN, b)
- plot_profiles: delta chi2 of logN and of b, each with the other minimised
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from .curve_of_growth import theoretical_ew
from .uncertainty import JOINT_LEVELS, ONE_PARAMETER_68, ONE_PARAMETER_95, profile


# --- Font -----------------------------------------------------------------------------

FONT_SIZE = 14
LATEX_FONT_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "font.size": FONT_SIZE,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}

# All figures of the project share the LaTeX font, so importing this module sets it
plt.rcParams.update(LATEX_FONT_STYLE)


# --- Shared style ---------------------------------------------------------------------

RAW_LINE_WIDTH = 0.5
WAVELENGTH_LABEL = r"$\lambda$ [$\mathrm{\AA}$] (vacuum)"


# --- Shared helpers -------------------------------------------------------------------


def latex_unit(unit_text):
    """Convert a unit string to an inline LaTeX label.

    Parameters
    ----------
    unit_text : str
        Unit readable by astropy, e.g. ``"1e-17 erg s^-1 cm^-2 Angstrom^-1"``.

    Returns
    -------
    str
        The unit in LaTeX, between ``$`` signs.
    """
    latex_text = u.Unit(unit_text).to_string("latex_inline")
    # astropy writes a scale of 1e-17 as "1 x 10^-17"; the "1 x" adds nothing
    return latex_text.replace(r"1 \times ", "")


# --- Raw spectrum ---------------------------------------------------------------------

RAW_SPECTRUM_FIGSIZE = (12, 5)
# The flux panel is taller: the error only needs to show its trend
FLUX_TO_ERROR_HEIGHT_RATIO = (3, 1)


def plot_raw_spectrum(spectrum):
    """Plot the flux and, in a panel below, its 1-sigma error against wavelength.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to plot; ``meta["flux_unit"]`` is used for the title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Flux and error axes, so the caller can add markers or call ``plt.show()``.
    """
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=RAW_SPECTRUM_FIGSIZE,
        gridspec_kw={"height_ratios": FLUX_TO_ERROR_HEIGHT_RATIO},
    )
    flux_ax, error_ax = axes
    unit_label = latex_unit(spectrum.meta["flux_unit"])
    flux_ax.plot(spectrum.wave, spectrum.flux, color="k", linewidth=RAW_LINE_WIDTH)
    # Flux and error share the unit, so it goes once in the title
    flux_ax.set_title(f"Flux and error in {unit_label}")
    flux_ax.set_ylabel("Flux")
    error_ax.plot(
        spectrum.wave, spectrum.error, color="tab:blue", linewidth=RAW_LINE_WIDTH
    )
    error_ax.set_ylabel("Error")
    error_ax.set_xlabel(WAVELENGTH_LABEL)
    return fig, axes


# --- Continuum ------------------------------------------------------------------------

CONTINUUM_FIGSIZE = (12, 4)
REJECTED_MARKER_SIZE = 2


def plot_continuum(spectrum, fit):
    """Plot the flux, the fitted continuum and the pixels rejected by the fit.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum that was fitted; ``meta["flux_unit"]`` is used for the title.
    fit : ContinuumFit
        Result of ``continuum.fit_continuum`` on ``spectrum``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot, so the caller can zoom or call ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=CONTINUUM_FIGSIZE)
    rejected = ~fit.used
    ax.plot(
        spectrum.wave, spectrum.flux, color="k", linewidth=RAW_LINE_WIDTH, label="Flux"
    )
    ax.plot(
        spectrum.wave[rejected],
        spectrum.flux[rejected],
        ".",
        markersize=REJECTED_MARKER_SIZE,
        color="tab:red",
        label="Rejected",
    )
    ax.plot(spectrum.wave, fit.continuum, color="tab:blue", label="Continuum")
    ax.set_title(f"Flux in {latex_unit(spectrum.meta['flux_unit'])}")
    ax.set_ylabel("Flux")
    ax.set_xlabel(WAVELENGTH_LABEL)
    ax.legend(loc="upper right")
    return fig, ax


def plot_normalised(spectrum):
    """Plot a normalised spectrum, with a line at the continuum level 1.

    Parameters
    ----------
    spectrum : Spectrum
        Flux already divided by the continuum, e.g. the output of
        ``continuum.normalise``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot, so the caller can zoom or call ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=CONTINUUM_FIGSIZE)
    ax.plot(
        spectrum.wave, spectrum.flux, color="k", linewidth=RAW_LINE_WIDTH, label="Flux"
    )
    ax.axhline(1, color="tab:blue", label="Continuum")
    ax.set_ylabel("Normalised flux")
    ax.set_xlabel(WAVELENGTH_LABEL)
    ax.legend(loc="lower right")
    return fig, ax


# --- Features -------------------------------------------------------------------------

WINDOW_COLOR = "tab:orange"
# Light enough that the flux stays visible through the shading
WINDOW_ALPHA = 0.3
CENTRE_LINE_WIDTH = 1
# Just right of the axes, vertically centred (axes fractions)
LEGEND_OUTSIDE_RIGHT = (1.01, 0.5)


def plot_features(spectrum, features):
    """Plot a normalised spectrum with the window of each feature shaded.

    Parameters
    ----------
    spectrum : Spectrum
        Normalised spectrum, e.g. the output of ``continuum.normalise``.
    features : astropy.table.Table
        Output of ``detect.find_features``; uses ``wave``, ``wave_min`` and
        ``wave_max``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot, so the caller can zoom or call ``plt.show()``.
    """
    fig, ax = plot_normalised(spectrum)
    window_label, centre_label = "Window", "Centre"
    for feature in features:
        ax.axvspan(
            feature["wave_min"],
            feature["wave_max"],
            color=WINDOW_COLOR,
            alpha=WINDOW_ALPHA,
            linewidth=0,
            label=window_label,
        )
        ax.axvline(
            feature["wave"],
            color="tab:red",
            linewidth=CENTRE_LINE_WIDTH,
            label=centre_label,
        )
        # Only the first window and centre go in the legend
        window_label, centre_label = None, None
    # Outside the axes: the features fill the plot, so no corner is free
    ax.legend(loc="center left", bbox_to_anchor=LEGEND_OUTSIDE_RIGHT)
    return fig, ax


# --- Curve of growth ------------------------------------------------------------------

COG_FIGSIZE = (7, 5)
# Range of log10 N (cm^-2) around the fit drawn by the theoretical curve
COG_LOGN_HALF_RANGE = 3.0
COG_CURVE_POINTS = 61


def plot_curve_of_growth(lines, ews, cog_fit):
    """Plot measured EWs of one species on its fitted curve of growth.

    Each line sits at log10(N f lambda) for the fitted N, so all lines fall
    on the single curve of the fitted b when the fit is good.

    Parameters
    ----------
    lines : list of Line
        Lines of the species; only those with an entry in `ews` are drawn.
    ews : dict
        {line name: (W, W_err)}, Angstrom, e.g. from
        ``curve_of_growth.measured_ews``.
    cog_fit : dict
        Output of ``curve_of_growth.fit_cog`` (uses logN, b and z).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot.
    """
    measured_lines = [line for line in lines if line.name in ews]
    fig, ax = plt.subplots(figsize=COG_FIGSIZE)
    _plot_theoretical_curve(ax, measured_lines[0], cog_fit)
    for line in measured_lines:
        _plot_measured_point(ax, line, ews[line.name], cog_fit)
    ax.set_xlabel(r"$\log_{10}(N f \lambda)$ [cm$^{-2}$ $\mathrm{\AA}$]")
    ax.set_ylabel(r"$\log_{10}(W / \lambda)$")
    ax.set_title(
        f"{measured_lines[0].species}: "
        f"log N = {cog_fit['logN']:.2f}, b = {cog_fit['b']:.1f} km/s"
    )
    ax.legend()
    return fig, ax


def _plot_theoretical_curve(ax, line, cog_fit):
    """Draw the curve of growth of `line` at the fitted b.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    line : Line
    cog_fit : dict
        Output of ``curve_of_growth.fit_cog``.
    """
    logN_values = np.linspace(
        cog_fit["logN"] - COG_LOGN_HALF_RANGE,
        cog_fit["logN"] + COG_LOGN_HALF_RANGE,
        COG_CURVE_POINTS,
    )
    widths = theoretical_ew(line, logN_values, cog_fit["b"], cog_fit["z"])
    ax.plot(
        logN_values + np.log10(line.f * line.wave_rest),
        np.log10(widths / line.wave_rest),
        color="k",
        label=f"Theory, b = {cog_fit['b']:.1f} km/s",
    )


def _plot_measured_point(ax, line, measurement, cog_fit):
    """Draw one measured EW at log10(N f lambda) for the fitted N.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    line : Line
    measurement : (float, float)
        W and W_err, Angstrom.
    cog_fit : dict
        Output of ``curve_of_growth.fit_cog``.
    """
    width, width_error = measurement
    # Error of log10(W): dW / (W ln 10)
    log_error = width_error / (width * np.log(10))
    ax.errorbar(
        cog_fit["logN"] + np.log10(line.f * line.wave_rest),
        np.log10(width / line.wave_rest),
        yerr=log_error,
        fmt="o",
        label=line.name,
    )


# --- chi2 map -------------------------------------------------------------------------

CHI2_MAP_FIGSIZE = (7, 5)
JOINT_LEVEL_LABELS = ("68 %", "95 %", "99.7 %")
JOINT_LEVEL_COLORS = ("tab:blue", "tab:orange", "tab:red")


def plot_chi2_map(chi2_map, title=""):
    """Plot the 68, 95 and 99.7 % joint regions of (logN, b) and the best fit.

    Parameters
    ----------
    chi2_map : uncertainty.Chi2Map
        Its rescaled delta chi2 is drawn.
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot.
    """
    delta = chi2_map.delta_chi2()
    fig, ax = plt.subplots(figsize=CHI2_MAP_FIGSIZE)
    ax.contour(
        chi2_map.logN_values,
        chi2_map.b_values,
        delta,
        levels=JOINT_LEVELS,
        colors=JOINT_LEVEL_COLORS,
    )
    _plot_contour_legend(ax)
    ax.plot(
        chi2_map.best_logN,
        chi2_map.best_b,
        "k+",
        markersize=12,
        label="best fit",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10} N$ [cm$^{-2}$]")
    ax.set_ylabel(r"$b$ [km s$^{-1}$]")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def _plot_contour_legend(ax):
    """Add one legend entry per confidence level (contours have none).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    for label, color in zip(JOINT_LEVEL_LABELS, JOINT_LEVEL_COLORS):
        ax.plot([], [], color=color, label=label)


PROFILES_FIGSIZE = (11, 4)
PROFILE_AXIS_LABELS = {"logN": r"$\log_{10} N$ [cm$^{-2}$]", "b": r"$b$ [km s$^{-1}$]"}
# Highest delta chi2 shown: a little above the 95 % level
PROFILE_DELTA_MAX = 6.0


def plot_profiles(chi2_map, title=""):
    """Plot the profiles of delta chi2 for logN and b, with the 68 and 95 % levels.

    Parameters
    ----------
    chi2_map : uncertainty.Chi2Map
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure holding the plot.
    axes : array of matplotlib.axes.Axes
        logN, then b.
    """
    fig, axes = plt.subplots(1, 2, figsize=PROFILES_FIGSIZE)
    for ax, parameter in zip(axes, ("logN", "b")):
        values, profile_values = profile(chi2_map, parameter)
        ax.plot(values, profile_values, color="k")
        ax.axhline(ONE_PARAMETER_68, color="tab:blue", linestyle="--", label="68 %")
        ax.axhline(ONE_PARAMETER_95, color="tab:orange", linestyle="--", label="95 %")
        ax.set_ylim(0, PROFILE_DELTA_MAX)
        ax.set_xlabel(PROFILE_AXIS_LABELS[parameter])
        ax.set_ylabel(r"$\Delta\chi^2$")
    axes[1].set_xscale("log")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
