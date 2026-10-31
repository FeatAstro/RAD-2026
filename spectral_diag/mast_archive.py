"""Search and download spectra from the MAST archive (astroquery.mast)."""

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

BYTES_PER_MB = 1e6


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
