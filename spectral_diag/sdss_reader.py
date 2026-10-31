"""Read SDSS spec / spec-lite FITS files into a Spectrum."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .constants import C_KMS, FWHM_PER_SIGMA
from .data_classes import Spectrum

# SDSS pixels are uniform in log10(lambda), 1e-4 dex wide
SDSS_PIXEL_DEX = 1e-4
SDSS_FLUX_UNIT = "1e-17 erg s^-1 cm^-2 Angstrom^-1"


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
    with fits.open(path) as fits_file:
        header = fits_file[0].header
        coadd = Table(fits_file["COADD"].data)
        specobj = fits_file["SPECOBJ"].data[0]

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
