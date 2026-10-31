"""Synthetic spectra with injected absorbers at known (z, logN, b), to
validate the rest of the pipeline."""

import numpy as np

from .data_classes import Spectrum
from .line_profiles import optical_depth


def make_synthetic(wave, absorbers, continuum=1.0, snr=None, rng=None):
    """Build a synthetic Spectrum with injected absorbers at known (z, N, b).

    No instrumental LSF is applied: this is the intrinsic model spectrum.

    Parameters
    ----------
    wave : array
        Wavelength grid, Angstrom.
    absorbers : list of Absorber
        Every line of every absorber is added to the optical depth.
    continuum : float or array
        Continuum level, broadcastable to `wave`.
    snr : float, optional
        Signal-to-noise ratio relative to the continuum. None (default):
        noise-free spectrum with error = 0.
    rng : numpy.random.Generator, optional
        For reproducible noise. Defaults to a fresh unseeded Generator.

    Returns
    -------
    spectrum : Spectrum
        `meta["absorbers"]` holds the injected truth.
    """
    wave = np.asarray(wave, dtype=float)
    continuum = np.broadcast_to(continuum, wave.shape).astype(float)
    flux = continuum * np.exp(-_total_optical_depth(wave, absorbers))

    if snr is None:
        error = np.zeros_like(wave)
    else:
        error = continuum / snr
        rng = rng if rng is not None else np.random.default_rng()
        flux = flux + rng.normal(0, error)

    meta = {"instrument": "synthetic", "absorbers": list(absorbers), "snr": snr}
    return Spectrum(wave=wave, flux=flux, error=error, continuum=continuum, meta=meta)


def _total_optical_depth(wave, absorbers):
    """Optical depth of all the lines of all the absorbers.

    Optical depths add because each line attenuates the same beam:
    exp(-tau1) * exp(-tau2) = exp(-(tau1 + tau2)).

    Parameters
    ----------
    wave : array
        Wavelength grid, Angstrom.
    absorbers : list of Absorber

    Returns
    -------
    tau : array
        Same shape as `wave`.
    """
    tau = np.zeros_like(wave)
    for absorber in absorbers:
        for line in absorber.lines:
            tau += optical_depth(
                wave, line, logN=absorber.logN, b=absorber.b, z=absorber.z
            )
    return tau
