"""Interactive spectrum and fit visualisation.

Planned contents:
- plot_spectrum: flux/error/continuum with identified line windows
  marked
- plot_fit: data vs. model profile for a fitted Absorber
- plot_cog: measured EWs over a theoretical curve of growth
"""

import astropy.units as u
import matplotlib.pyplot as plt


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
