"""Shared data structures used across the pipeline.

Agreed early so the two sides of the project (data -> lines, lines ->
physics) can work in parallel against a fixed interface.

Units: wavelengths in Angstrom, velocities in km/s, column densities
in cm^-2 (log10 N in outputs).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    """An observed or synthetic spectrum.

    wave, flux, error are 1D arrays of equal length. continuum is the
    fitted/normalised continuum level, filled in by `continuum`
    once available.
    """

    wave: np.ndarray  # Angstrom
    flux: np.ndarray
    error: np.ndarray
    continuum: np.ndarray | None = None


@dataclass
class Line:
    """A single atomic transition from the line list."""

    species: str  # e.g. "HI", "CIV"
    wave_rest: float  # Angstrom
    f: float  # oscillator strength
    gamma: float  # damping constant, s^-1


@dataclass
class Absorber:
    """A fitted or injected absorption system for one species."""

    species: str
    z: float
    logN: float  # log10(N / cm^-2)
    b: float  # km/s
    lines: list[Line] = field(default_factory=list)
