"""Load spectra from files or archives, and generate synthetic spectra.

Planned contents:
- read_fits / read_ascii: load an observed Spectrum from disk
- query_archive: fetch a spectrum from an online archive (astroquery)
- make_synthetic: build a Spectrum with injected absorbers at known
  (z, logN, b), for validating the rest of the pipeline
"""

import numpy as np

from .core import Spectrum
from .profiles import optical_depth


def make_synthetic(wave, absorbers, continuum=1.0, snr=None, rng=None):
    """Build a synthetic Spectrum with injected absorbers at known (z, N, b).

    Combines every line of every absorber into a single optical depth
    (summing tau is exact, not an approximation -- each species/line
    independently attenuates the same beam, so transmissions multiply
    and optical depths add: exp(-tau1)*exp(-tau2) = exp(-(tau1+tau2))).

    No instrumental LSF is applied yet (`profiles.apply_lsf` doesn't
    exist yet) -- this is the intrinsic model spectrum, not a
    realistic stand-in for a specific instrument's resolution.

    Parameters
    ----------
    wave : array
        Wavelength grid, Angstrom.
    absorbers : list of Absorber
        Each absorber's `lines` are summed into the total optical
        depth, using that absorber's logN, b, z.
    continuum : float or array
        Continuum level, broadcastable to `wave`. No continuum shape
        model beyond this (see `continuum.py`).
    snr : float, optional
        Target signal-to-noise ratio relative to the continuum. If
        None (default), returns a noise-free spectrum with
        error = 0 -- useful for checking against closed-form truth
        before testing against noise.
    rng : numpy.random.Generator, optional
        For reproducible noise. Defaults to a fresh unseeded
        Generator.

    Returns
    -------
    spectrum : Spectrum
    """
    wave = np.asarray(wave, dtype=float)
    continuum_arr = np.broadcast_to(continuum, wave.shape).astype(float)

    tau_total = np.zeros_like(wave)
    for absorber in absorbers:
        for line in absorber.lines:
            tau_total += optical_depth(
                wave, line, logN=absorber.logN, b=absorber.b, z=absorber.z
            )

    flux_clean = continuum_arr * np.exp(-tau_total)

    if snr is None:
        return Spectrum(
            wave=wave,
            flux=flux_clean,
            error=np.zeros_like(wave),
            continuum=continuum_arr,
        )

    rng = rng if rng is not None else np.random.default_rng()
    sigma = continuum_arr / snr
    flux = flux_clean + rng.normal(0, sigma)
    return Spectrum(wave=wave, flux=flux, error=sigma, continuum=continuum_arr)
