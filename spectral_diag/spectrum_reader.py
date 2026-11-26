"""Read observed spectra into a Spectrum: SDSS spec / spec-lite, HST/COS x1dsum and
HST co-adds of the Hubble Spectroscopic Legacy Archive (HSLA, COS and STIS)."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file

from . import constants
from .data_classes import Spectrum


# --- SDSS -----------------------------------------------------------------------------

# Fixed by the SDSS spec data model.
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
    # COEFF1: step of the pixel grid in log10(lambda)
    fwhm_kms = _sdss_fwhm_kms(coadd["wdisp"][good], header["COEFF1"])
    meta = _sdss_meta(path, header, specobj)
    meta["fwhm_kms"] = fwhm_kms
    meta["R"] = constants.C_KMS / fwhm_kms
    return Spectrum(
        wave=10 ** np.asarray(coadd["loglam"], dtype=float)[good],
        flux=np.asarray(coadd["flux"], dtype=float)[good],
        error=1 / np.sqrt(inverse_variance[good]),
        meta=meta,
    )


def _sdss_fwhm_kms(wdisp, pixel_dex):
    """Median instrumental FWHM from the SDSS `wdisp` column.

    Parameters
    ----------
    wdisp : array
        Gaussian sigma of the LSF, in pixels.
    pixel_dex : float
        Pixel width, log10(Angstrom).

    Returns
    -------
    fwhm_kms : float
        Median over the spectrum, km/s.
    """
    # A step of d(log10 lambda) is a velocity of ln(10) * d(log10 lambda) * c
    sigma_kms = (
        np.asarray(wdisp, dtype=float) * pixel_dex * np.log(10) * constants.C_KMS
    )
    return float(np.median(constants.FWHM_PER_SIGMA * sigma_kms))


def _sdss_meta(path, header, specobj):
    """Provenance, observation and pipeline keys of an SDSS spectrum.

    Parameters
    ----------
    path : str or Path
        File the spectrum was read from.
    header : astropy.io.fits.Header
        Primary header (EXPTIME summed over exposures, s; COEFF1 pixel
        width in log10(Angstrom)).
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
        "pixel_dex": float(header["COEFF1"]),
        "flux_unit": SDSS_FLUX_UNIT,
        "snr": float(specobj["SN_MEDIAN_ALL"]),
        "z": float(specobj["Z"]),
        "z_err": float(specobj["Z_ERR"]),
        "class": specobj["CLASS"].strip(),
        "subclass": specobj["SUBCLASS"].strip(),
    }


# --- HST: shared constants and COS line spread function -------------------------------

# Unit of the FLUX column of HST 1D spectra (COS and STIS x1d, x1dsum, HSLA)
HST_FLUX_UNIT = "erg s^-1 cm^-2 Angstrom^-1"
# COS files do not store the instrumental resolution. Line spread functions
# measured by STScI, one table per grating, central wavelength and lifetime
# position (COS "Spectral Resolution" web page)
COS_LSF_URL = (
    "https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/"
    "cos/performance/spectral-resolution/_documents/"
    "aa_LSFTable_{grating}_{cenwave}_LP{lifetime_position}_cn.dat"
)


def _cos_lsf_fwhm_kms(setup, pixel_width):
    """Median FWHM of the STScI LSF table of a COS observing setup.

    The table is downloaded on the first call and cached by astropy.

    Parameters
    ----------
    setup : mapping
        OPT_ELEM or DISPERSER (grating), CENWAVE (Angstrom) and LIFE_ADJ
        (lifetime position): a primary header or a PROVENANCE row.
    pixel_width : float
        Width of a detector pixel, Angstrom.

    Returns
    -------
    fwhm_kms : float
        Median over the tabulated wavelengths, km/s.
    """
    grating = setup["OPT_ELEM"] if "OPT_ELEM" in setup else setup["DISPERSER"]
    url = COS_LSF_URL.format(
        grating=grating,
        cenwave=setup["CENWAVE"],
        lifetime_position=setup["LIFE_ADJ"],
    )
    return _lsf_table_fwhm_kms(download_file(url, cache=True), pixel_width)


def _lsf_table_fwhm_kms(lsf_file, pixel_width):
    """Median FWHM of the LSFs tabulated in a STScI COS LSF table.

    Parameters
    ----------
    lsf_file : str or Path
        First row: wavelengths (Angstrom); below, one column per
        wavelength with the LSF sampled on detector pixels.
    pixel_width : float
        Width of a detector pixel, Angstrom.

    Returns
    -------
    fwhm_kms : float
        Median over the tabulated wavelengths, km/s.
    """
    table = np.loadtxt(lsf_file)
    lsf_waves, profiles = table[0], table[1:]
    fwhm_kms = []
    for column, lsf_wave in enumerate(lsf_waves):
        fwhm_angstrom = _fwhm_pixels(profiles[:, column]) * pixel_width
        fwhm_kms.append(constants.C_KMS * fwhm_angstrom / lsf_wave)
    return float(np.median(fwhm_kms))


def _fwhm_pixels(profile):
    """Full width at half maximum of a single-peaked sampled profile.

    Parameters
    ----------
    profile : array
        Profile on a grid of unit step, peak away from the edges.

    Returns
    -------
    fwhm : float
        In grid steps, with the half-maximum crossings interpolated
        linearly between samples.
    """
    half_max = profile.max() / 2
    above = np.nonzero(profile >= half_max)[0]
    first, last = above[0], above[-1]
    left = first - (profile[first] - half_max) / (profile[first] - profile[first - 1])
    right = last + (profile[last] - half_max) / (profile[last] - profile[last + 1])
    return right - left


# --- HST/COS x1dsum: one visit --------------------------------------------------------


def read_cos_x1dsum(path):
    """Load an HST/COS FUV x1dsum file (one visit) into a Spectrum.

    Joins the detector segments present in the file (FUVA and/or FUVB,
    separated by the detector gap) and drops pixels with DQ_WGT == 0,
    flagged by the pipeline. COS wavelengths are vacuum, heliocentric.
    The instrumental FWHM comes from the STScI LSF table of the
    observing setup.

    Parameters
    ----------
    path : str or Path
        *_x1dsum.fits file.

    Returns
    -------
    spectrum : Spectrum
        Flux and error in `HST_FLUX_UNIT`, wave sorted. `meta` holds the
        standard keys (see `Spectrum`, fwhm_kms is the median over the
        band) plus the grating central wavelength, cenwave (Angstrom).
    """
    with fits.open(path) as fits_file:
        header = fits_file[0].header
        science_header = fits_file["SCI"].header
        wave, flux, error = _join_cos_segments(fits_file["SCI"].data)

    # Segments do not overlap, so sorting only puts FUVB before FUVA
    order = np.argsort(wave)
    wave, flux, error = wave[order], flux[order], error[order]
    # Steps across the detector gap or across dropped pixels are much wider
    # than one pixel; the median ignores them
    fwhm_kms = _cos_lsf_fwhm_kms(header, np.median(np.diff(wave)))
    meta = _x1dsum_meta(path, header, science_header)
    meta["fwhm_kms"] = fwhm_kms
    meta["R"] = constants.C_KMS / fwhm_kms
    meta["snr"] = float(np.median(flux / error))
    return Spectrum(wave=wave, flux=flux, error=error, meta=meta)


def _join_cos_segments(segments):
    """Concatenate the valid pixels of all COS segments.

    Parameters
    ----------
    segments : FITS_rec
        Rows of the SCI HDU, one per segment.

    Returns
    -------
    wave, flux, error : arrays
        Unsorted, pixels with DQ_WGT > 0 only.
    """
    waves, fluxes, errors = [], [], []
    for segment in segments:
        good = segment["DQ_WGT"] > 0
        waves.append(segment["WAVELENGTH"][good])
        fluxes.append(segment["FLUX"][good])
        errors.append(segment["ERROR"][good])
    return (
        np.concatenate(waves).astype(float),
        np.concatenate(fluxes).astype(float),
        np.concatenate(errors).astype(float),
    )


def _x1dsum_meta(path, header, science_header):
    """Provenance and observation keys of a COS x1dsum spectrum.

    Parameters
    ----------
    path : str or Path
        File the spectrum was read from.
    header : astropy.io.fits.Header
        Primary header (target position, grating).
    science_header : astropy.io.fits.Header
        SCI header (EXPSTART in MJD, EXPTIME summed over exposures, s).

    Returns
    -------
    meta : dict
        Keys described in `Spectrum` and `read_cos_x1dsum`.
    """
    return {
        "instrument": f"HST/COS {header['OPT_ELEM']}",
        "source": Path(path).name,
        "obs_id": Path(path).name.split("_")[0],
        "ra": float(header["RA_TARG"]),
        "dec": float(header["DEC_TARG"]),
        "mjd": float(science_header["EXPSTART"]),
        "exptime": float(science_header["EXPTIME"]),
        "wave_frame": "vacuum, heliocentric",
        "flux_unit": HST_FLUX_UNIT,
        "cenwave": int(header["CENWAVE"]),
    }


# --- HSLA co-adds: all the visits of one grating --------------------------------------


def read_hsla(path):
    """Load an HST co-add (COS or STIS) of the Hubble Spectroscopic Legacy Archive.

    HSLA (Peeples et al. 2017; Debes et al. 2020, RNAAS 4, 205) co-adds
    all the x1d exposures of one target taken with one grating, over
    every central wavelength, on a single wavelength grid. The detector
    gaps and artefacts of each setup then fall at different wavelengths
    and are filled or diluted. Pixels that no exposure covers are
    dropped. Wavelengths are vacuum, heliocentric.

    Parameters
    ----------
    path : str or Path
        hst_cos_*_cspec.fits or hst_stis_*_cspec.fits file.

    Returns
    -------
    spectrum : Spectrum
        Flux and error in `HST_FLUX_UNIT`. `meta` holds the standard keys
        (see `Spectrum`; fwhm_kms is the median over the exposures) plus
        cenwaves, the central wavelengths co-added (Angstrom).
    """
    with fits.open(path) as fits_file:
        header = fits_file[0].header
        science_header = fits_file["SCI"].header
        science = fits_file["SCI"].data[0]
        provenance = Table(fits_file["PROVENANCE"].data)

    covered = science["EFF_EXPTIME"] > 0
    wave = np.asarray(science["WAVELENGTH"], dtype=float)[covered]
    flux = np.asarray(science["FLUX"], dtype=float)[covered]
    error = np.asarray(science["ERROR"], dtype=float)[covered]
    fwhm_kms = _hsla_fwhm_kms(header, provenance, np.median(np.diff(wave)))
    meta = _hsla_meta(path, header, science_header, provenance)
    meta["fwhm_kms"] = fwhm_kms
    meta["R"] = constants.C_KMS / fwhm_kms
    meta["snr"] = float(np.median(flux / error))
    return Spectrum(wave=wave, flux=flux, error=error, meta=meta)


def _hsla_fwhm_kms(header, provenance, pixel_width):
    """Median instrumental FWHM over the exposures of an HSLA co-add.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary header (INSTRUME).
    provenance : astropy.table.Table
        PROVENANCE HDU, one row per exposure (DISPERSER, CENWAVE,
        LIFE_ADJ, SPECRES).
    pixel_width : float
        Width of a pixel of the co-add, Angstrom. The COS LSF tables are
        sampled on detector pixels; the co-add grid is within 0.5% of
        them for G130M.

    Returns
    -------
    fwhm_kms : float
        km/s.
    """
    # STIS has no LSF tables of the COS format; SPECRES is the nominal R
    if header["INSTRUME"] != "COS":
        return float(np.median(constants.C_KMS / provenance["SPECRES"]))
    fwhm_kms = []
    for exposure in provenance:
        fwhm_kms.append(_cos_lsf_fwhm_kms(exposure, pixel_width))
    return float(np.median(fwhm_kms))


def _hsla_meta(path, header, science_header, provenance):
    """Provenance and observation keys of an HSLA co-add.

    Parameters
    ----------
    path : str or Path
        File the spectrum was read from.
    header : astropy.io.fits.Header
        Primary header (instrument, grating, target position, HSLANAME).
    science_header : astropy.io.fits.Header
        SCI header (MJD-BEG of the first exposure, XPOSURE summed over
        exposures, s).
    provenance : astropy.table.Table
        PROVENANCE HDU, one row per exposure.

    Returns
    -------
    meta : dict
        Keys described in `Spectrum` and `read_hsla`.
    """
    cenwaves = sorted(set(int(cenwave) for cenwave in provenance["CENWAVE"]))
    return {
        "instrument": f"HST/{header['INSTRUME']} {header['GRATING']}",
        "source": Path(path).name,
        "obs_id": header["HSLANAME"],
        "ra": float(header["RA_TARG"]),
        "dec": float(header["DEC_TARG"]),
        "mjd": float(science_header["MJD-BEG"]),
        "exptime": float(science_header["XPOSURE"]),
        "wave_frame": "vacuum, heliocentric",
        "flux_unit": HST_FLUX_UNIT,
        "cenwaves": cenwaves,
    }
