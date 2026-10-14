"""Load spectra from files or archives, and generate synthetic spectra.

Planned contents:
- read_fits / read_ascii: load an observed Spectrum from disk
- query_mast / download_mast_spectra: search and fetch spectra from
  the MAST archive (astroquery.mast)
- read_sdss_spec: load an SDSS spec / spec-lite FITS file
- make_synthetic: build a Spectrum with injected absorbers at known
  (z, logN, b), for validating the rest of the pipeline
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations

from .core import C_KMS, Spectrum
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
    # Injected truth, to compare against recovered parameters
    meta = {"instrument": "synthetic", "absorbers": list(absorbers), "snr": snr}

    if snr is None:
        return Spectrum(
            wave=wave,
            flux=flux_clean,
            error=np.zeros_like(wave),
            continuum=continuum_arr,
            meta=meta,
        )

    rng = rng if rng is not None else np.random.default_rng()
    sigma = continuum_arr / snr
    flux = flux_clean + rng.normal(0, sigma)
    return Spectrum(
        wave=wave, flux=flux, error=sigma, continuum=continuum_arr, meta=meta
    )


def query_mast(target, radius=5 * u.arcsec, **criteria):
    """Search MAST for spectroscopic observations around a target.

    Parameters
    ----------
    target : str or SkyCoord
        Object name (resolved by MAST) or sky position.
    radius : Quantity
        Search radius around the target.
    **criteria
        Extra `Observations.query_criteria` filters, e.g.
        obs_collection="HST", instrument_name="COS/FUV". Valid
        fields: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html
        (or `Observations.get_metadata("observations")`). Usage:
        https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html

    Returns
    -------
    obs : astropy.table.Table
        One row per observation with dataproduct_type == "spectrum".
    """
    position = {"coordinates": target} if isinstance(target, SkyCoord) else {"objectname": target}
    return Observations.query_criteria(
        **position, radius=radius, dataproduct_type="spectrum", **criteria
    )


def download_mast_spectra(obs, download_dir="data/mast", max_size_mb=50, **filters):
    """Download the science FITS spectra of MAST observations.

    Parameters
    ----------
    obs : astropy.table.Table
        Rows returned by `query_mast`.
    download_dir : str or Path
        Files are written directly in this directory (flat layout).
    max_size_mb : float
        Skip products larger than this. Guards against catalogue-wide
        files attached to some collections (e.g. SDSS specObj, ~7 GB).
    **filters
        Extra `Observations.filter_products` filters, e.g.
        productSubGroupDescription="X1DSUM" (HST) or "SPECTRA" (SDSS).

    Returns
    -------
    paths : list of Path
        Local paths of the downloaded files.
    """
    products = Observations.get_product_list(obs)
    products = Observations.filter_products(
        products, productType="SCIENCE", dataproduct_type="spectrum",
        extension="fits", **filters
    )
    products = products[products["size"] <= max_size_mb * 1e6]
    if len(products) == 0:
        return []

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    manifest = Observations.download_products(
        products, download_dir=str(download_dir), flat=True
    )
    return [Path(p) for p in manifest["Local Path"]]


def read_sdss_spec(path):
    """Load an SDSS spec / spec-lite file into a Spectrum.

    SDSS wavelengths are already vacuum, heliocentric. Pixels with
    ivar == 0 (no data, masked by the pipeline) are dropped.

    Parameters
    ----------
    path : str or Path
        spec-PLATE-MJD-FIBER.fits or spec-lite-*.fits.

    Returns
    -------
    spectrum : Spectrum
        Flux and error in 1e-17 erg s^-1 cm^-2 Angstrom^-1. `meta`
        holds the standard keys (see `Spectrum`) plus the pipeline
        redshift and classification: z, z_err, class, subclass.
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
        coadd = Table(hdul["COADD"].data)  # co-added spectrum, one row per pixel
        specobj = hdul["SPECOBJ"].data[0]  # pipeline results, single row

    ivar = np.asarray(coadd["ivar"], dtype=float)  # inverse variance, 1/sigma^2
    good = ivar > 0  # ivar == 0 marks pixels with no valid data

    # wdisp: LSF Gaussian sigma in pixels of 1e-4 dex in log10(lambda)
    sigma_kms = coadd["wdisp"][good] * 1e-4 * np.log(10) * C_KMS
    fwhm_kms = float(np.median(2 * np.sqrt(2 * np.log(2)) * sigma_kms))

    meta = {
        "instrument": "SDSS",
        "source": Path(path).name,
        "obs_id": f"{specobj['PLATE']}-{specobj['MJD']}-{specobj['FIBERID']:04d}",
        "ra": float(specobj["PLUG_RA"]),  # fibre position, deg
        "dec": float(specobj["PLUG_DEC"]),
        "mjd": int(specobj["MJD"]),
        "exptime": float(header["EXPTIME"]),  # s, summed over exposures
        "wave_frame": "vacuum, heliocentric",
        "flux_unit": "1e-17 erg s^-1 cm^-2 Angstrom^-1",
        "fwhm_kms": fwhm_kms,  # median over the spectrum
        "R": C_KMS / fwhm_kms,
        "snr": float(specobj["SN_MEDIAN_ALL"]),
        "z": float(specobj["Z"]),
        "z_err": float(specobj["Z_ERR"]),
        "class": specobj["CLASS"].strip(),  # STAR / GALAXY / QSO
        "subclass": specobj["SUBCLASS"].strip(),  # e.g. spectral type
    }
    return Spectrum(
        wave=10 ** np.asarray(coadd["loglam"], dtype=float)[good],  # log10(lambda/A) -> A
        flux=np.asarray(coadd["flux"], dtype=float)[good],
        error=1 / np.sqrt(ivar[good]),  # sigma = 1/sqrt(ivar)
        meta=meta,
    )
