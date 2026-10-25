"""Shared data structures used across the pipeline.

Agreed early so the two sides of the project (data -> lines, lines ->
physics) can work in parallel against a fixed interface.

Units: wavelengths in Angstrom, velocities in km/s, column densities
in cm^-2 (log10 N in outputs).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import constants as _const

C_KMS = _const.c * 1e-3  # speed of light, km/s
ANGSTROM_TO_CM = 1e-8
KMS_TO_CMS = 1e5


@dataclass
class Spectrum:
    """An observed or synthetic spectrum.

    Attributes
    ----------
    wave : array
        Observed wavelength grid, Angstrom (vacuum).
    flux : array
        Flux at each wavelength, same length as `wave`.
    error : array
        1-sigma flux uncertainty per pixel, same units as `flux`.
    continuum : array, optional
        Continuum level at each wavelength, filled in by `continuum`.
    meta : dict
        Free-form metadata, filled in by the readers in `io`. Keys
        used so far (all optional):
        instrument, source, obs_id -- provenance;
        ra, dec (deg), mjd, exptime (s) -- observation;
        wave_frame, flux_unit -- conventions;
        fwhm_kms, R -- instrumental resolution;
        snr -- median signal-to-noise;
        absorbers -- injected truth (synthetic spectra).
    """

    wave: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    continuum: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Line:
    """A single atomic transition from the line list (immutable).

    Attributes
    ----------
    species : str
        Species name, e.g. "HI", "CIV".
    wave_rest : float
        Rest wavelength in vacuum, Angstrom. Air-calibrated spectra must
        be converted to vacuum on loading.
    f : float
        Absorption oscillator strength f_lu.
    gamma : float
        Total damping constant: decay rate of the upper level plus that
        of the lower level (0 for the ground level), s^-1.
    name : str, optional
        Unique label used to match EWs to lines and in plots. Defaults
        to species + wave_rest to 3 decimals, e.g. "CIV 1548.195".
    """

    species: str
    wave_rest: float
    f: float
    gamma: float
    name: str | None = None

    # TODO: g_lower, g_upper and a derived A_ul property (radio demo).
    # TODO: atomic mass, to split b into thermal and turbulent parts.

    def __post_init__(self):
        if self.name is None:
            object.__setattr__(self, "name", f"{self.species} {self.wave_rest:.3f}")


@dataclass
class Absorber:
    """A fitted or injected absorption system for one species.

    Attributes
    ----------
    species : str
        Species name, e.g. "HI", "CIV".
    z : float
        Redshift of the absorber.
    logN : float
        log10(column density / cm^-2).
    b : float
        Doppler (velocity dispersion) parameter, km/s.
    lines : list of Line
        Transitions of this species used to model or fit the absorber.
    """

    species: str
    z: float
    logN: float
    b: float
    lines: list[Line] = field(default_factory=list)
