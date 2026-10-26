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

from .core import C_KMS, FWHM_PER_SIGMA, Spectrum
from .profiles import optical_depth

BYTES_PER_MB = 1e6
# SDSS pixels are uniform in log10(lambda), 1e-4 dex wide
SDSS_PIXEL_DEX = 1e-4
SDSS_FLUX_UNIT = "1e-17 erg s^-1 cm^-2 Angstrom^-1"


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
    return Observations.query_criteria(
        **_mast_position(target), radius=radius, dataproduct_type="spectrum", **criteria
    )


def _mast_position(target):
    """Express a target as the position keyword of `query_criteria`.

    Parameters
    ----------
    target : str or SkyCoord
        Object name or sky position.

    Returns
    -------
    position : dict
        {"coordinates": target} or {"objectname": target}.
    """
    if isinstance(target, SkyCoord):
        return {"coordinates": target}
    return {"objectname": target}


def download_mast_spectra(obs, download_dir="data/mast", max_size_mb=50, **filters):
    """Download the science FITS spectra of MAST observations.

    Parameters
    ----------
    obs : astropy.table.Table
        Rows returned by `query_mast`.
    download_dir : str or Path
        Created if needed; files are written directly in it (flat layout).
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
    products = _science_spectra(obs, max_size_mb, **filters)
    if len(products) == 0:
        return []

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    manifest = Observations.download_products(
        products, download_dir=str(download_dir), flat=True
    )
    return [Path(local_path) for local_path in manifest["Local Path"]]


def _science_spectra(obs, max_size_mb, **filters):
    """List the science FITS spectra of `obs` no larger than `max_size_mb`.

    Parameters
    ----------
    obs : astropy.table.Table
        Rows returned by `query_mast`.
    max_size_mb : float
        Size limit per product, MB.
    **filters
        Extra `Observations.filter_products` filters.

    Returns
    -------
    products : astropy.table.Table
    """
    products = Observations.get_product_list(obs)
    products = Observations.filter_products(
        products,
        productType="SCIENCE",
        dataproduct_type="spectrum",
        extension="fits",
        **filters,
    )
    return products[products["size"] <= max_size_mb * BYTES_PER_MB]


def read_sdss_spec(path):
    """Load an SDSS spec / spec-lite file into a Spectrum.

    Reads the co-added spectrum (HDU COADD, one row per pixel) and the
    pipeline results (HDU SPECOBJ). SDSS wavelengths are vacuum,
    heliocentric. Pixels with ivar == 0 (no valid data) are dropped.

    Parameters
    ----------
    path : str or Path
        spec-PLATE-MJD-FIBER.fits or spec-lite-*.fits.

    Returns
    -------
    spectrum : Spectrum
        Flux and error in `SDSS_FLUX_UNIT`. `meta` holds the standard
        keys (see `Spectrum`) plus the pipeline redshift and
        classification: z, z_err, class, subclass.
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
        coadd = Table(hdul["COADD"].data)
        specobj = hdul["SPECOBJ"].data[0]

    inverse_variance = np.asarray(coadd["ivar"], dtype=float)
    good = inverse_variance > 0
    fwhm_kms = _sdss_fwhm_kms(coadd["wdisp"][good])
    meta = {**_sdss_meta(path, header, specobj), **_resolution_meta(fwhm_kms)}
    return Spectrum(
        wave=10 ** np.asarray(coadd["loglam"], dtype=float)[good],
        flux=np.asarray(coadd["flux"], dtype=float)[good],
        error=1 / np.sqrt(inverse_variance[good]),
        meta=meta,
    )


def _sdss_fwhm_kms(wdisp):
    """Median instrumental FWHM from the SDSS `wdisp` column.

    Parameters
    ----------
    wdisp : array
        Gaussian sigma of the LSF, in pixels of `SDSS_PIXEL_DEX`.

    Returns
    -------
    fwhm_kms : float
        Median over the spectrum, km/s.
    """
    # A step of d(log10 lambda) is a velocity of ln(10) * d(log10 lambda) * c
    sigma_kms = np.asarray(wdisp, dtype=float) * SDSS_PIXEL_DEX * np.log(10) * C_KMS
    return float(np.median(FWHM_PER_SIGMA * sigma_kms))


def _resolution_meta(fwhm_kms):
    """Resolution keys of `Spectrum.meta`.

    Parameters
    ----------
    fwhm_kms : float
        Instrumental FWHM, km/s.

    Returns
    -------
    meta : dict
        fwhm_kms and resolving power R.
    """
    return {"fwhm_kms": fwhm_kms, "R": C_KMS / fwhm_kms}


def _sdss_meta(path, header, specobj):
    """Provenance, observation and pipeline keys of an SDSS spectrum.

    Parameters
    ----------
    path : str or Path
        File the spectrum was read from.
    header : astropy.io.fits.Header
        Primary header (EXPTIME summed over exposures, s).
    specobj : FITS_record
        Single row of the SPECOBJ HDU (PLUG_RA/DEC: fibre position, deg).

    Returns
    -------
    meta : dict
        Keys described in `Spectrum` and `read_sdss_spec`.
    """
    return {
        "instrument": "SDSS",
        "source": Path(path).name,
        "obs_id": f"{specobj['PLATE']}-{specobj['MJD']}-{specobj['FIBERID']:04d}",
        "ra": float(specobj["PLUG_RA"]),
        "dec": float(specobj["PLUG_DEC"]),
        "mjd": int(specobj["MJD"]),
        "exptime": float(header["EXPTIME"]),
        "wave_frame": "vacuum, heliocentric",
        "pixel_dex": SDSS_PIXEL_DEX,
        "flux_unit": SDSS_FLUX_UNIT,
        "snr": float(specobj["SN_MEDIAN_ALL"]),
        "z": float(specobj["Z"]),
        "z_err": float(specobj["Z_ERR"]),
        "class": specobj["CLASS"].strip(),
        "subclass": specobj["SUBCLASS"].strip(),
    }
