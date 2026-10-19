"""Continuum fitting and normalisation.

- fit_continuum: cubic spline fitted to the pixels that are not in a
  line, found by iterative asymmetric sigma-clipping
- normalise: divide a Spectrum's flux and error by its continuum

Written for absorption spectra: pixels far below the fit are treated
as lines, pixels far above it as glitches (cosmic rays, bad pixels).
"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.ndimage import binary_dilation

_DEGREE = 3  # cubic spline
# Fewest kept pixels allowed between two knots; below this the cubic in
# that interval is poorly constrained, so the interval is merged.
_MIN_PIXELS_PER_INTERVAL = 10


def _knots(wave, used, knot_spacing):
    """Full knot vector, evenly spaced, with empty intervals merged.

    Interior knots are spaced by about `knot_spacing`; any knot whose
    interval to the next knot holds fewer than _MIN_PIXELS_PER_INTERVAL
    kept pixels is removed, so that interval joins its neighbour.
    """
    lo, hi = wave[0], wave[-1]
    n_intervals = max(1, int(round((hi - lo) / knot_spacing)))
    interior = list(np.linspace(lo, hi, n_intervals + 1)[1:-1])
    x = wave[used]
    while interior:
        edges = np.array([lo, *interior, hi])
        counts = np.histogram(x, bins=edges)[0]
        sparse = np.flatnonzero(counts < _MIN_PIXELS_PER_INTERVAL)
        if sparse.size == 0:
            break
        # Remove the interior knot bounding the first sparse interval
        # (its right edge, or its left edge for the last interval).
        i = sparse[0]
        interior.pop(min(i, len(interior) - 1))
    return np.r_[[lo] * (_DEGREE + 1), interior, [hi] * (_DEGREE + 1)]


def _fit_spline(wave, flux, error, used, knot_spacing):
    """Weighted least-squares cubic spline through the kept pixels."""
    t = _knots(wave, used, knot_spacing)
    return make_lsq_spline(wave[used], flux[used], t, k=_DEGREE, w=1 / error[used])


def _continuum_error(spline, wave, used, error):
    """1-sigma statistical error of the fitted spline at every pixel.

    Cov(c) = (B^T W B)^-1 for the spline coefficients c, with B the
    basis functions at the kept pixels and W = diag(1 / error^2);
    var C(lambda) = B(lambda)^T Cov(c) B(lambda).
    """
    t = spline.t
    B_used = BSpline.design_matrix(wave[used], t, _DEGREE).toarray()
    weighted = B_used / error[used, None]
    cov = np.linalg.pinv(weighted.T @ weighted)
    B_all = BSpline.design_matrix(wave, t, _DEGREE).toarray()
    return np.sqrt(np.einsum("ij,jk,ik->i", B_all, cov, B_all))


def fit_continuum(
    spectrum,
    knot_spacing=250.0,
    k_low=2.0,
    k_high=4.0,
    grow=3,
    exclude=None,
    max_iter=20,
):
    """Fit the continuum of an absorption spectrum.

    Repeats: fit a cubic spline (weights 1/error^2) to the kept pixels,
    compute r = (flux - continuum) / error, keep only the pixels with
    -k_low < r < k_high, then also drop `grow` pixels on each side of
    every rejected one; until the set of kept pixels stops changing
    (or comes back to an earlier state). The upper clip is only switched
    on once the lower clip alone has converged.

    Parameters
    ----------
    spectrum : Spectrum
        wave in Angstrom (increasing), error > 0 at every pixel.
    knot_spacing : float
        Approximate distance between spline knots, Angstrom. The fit
        cannot follow features narrower than this, so it must be wider
        than the widest line (about 100 A for the Balmer lines of an
        A star) and narrower than the real bends of the continuum.
    k_low, k_high : float
        Rejection thresholds below and above the fit, in units of the
        pixel error.
    grow : int
        Number of pixels rejected on each side of a rejected pixel, to
        remove line wings too shallow to be rejected on their own.
    exclude : list of (float, float), optional
        Wavelength ranges (Angstrom) never used for the fit, e.g. known
        broad lines.
    max_iter : int
        Maximum number of fit-and-clip passes, both stages together.

    Returns
    -------
    spectrum : Spectrum
        A copy with `continuum` filled in (same units as flux). Its
        meta also holds:
        continuum_err -- 1-sigma statistical error of the continuum per
            pixel (correlated between neighbouring pixels);
        continuum_mask -- True where the pixel was used in the final fit;
        continuum_extrapolated -- True blueward of the first or redward
            of the last kept pixel, where the spline is not constrained;
        continuum_niter -- number of passes done;
        continuum_status -- "converged" (mask stable), "cycling" (mask
            alternating between states that differ by a few pixels at a
            threshold; harmless) or "max_iter" (still changing).
    """
    wave = np.asarray(spectrum.wave, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    error = np.asarray(spectrum.error, dtype=float)
    if np.any(np.diff(wave) <= 0):
        raise ValueError("wave must be strictly increasing")
    if np.any(~(error > 0)):
        raise ValueError("error must be positive at every pixel (use a noisy spectrum)")

    allowed = np.ones(wave.size, dtype=bool)
    for wmin, wmax in exclude or []:
        allowed &= ~((wave >= wmin) & (wave <= wmax))

    used = allowed.copy()
    n_iter = 0
    status = "max_iter"
    # Stage 1 clips below the fit only. On the first passes the lines
    # drag the fit down, so real continuum pixels can sit more than
    # k_high above it and would be thrown away as glitches. Stage 2
    # adds the upper clip once the fit has risen to the continuum.
    for clip_high in (False, True):
        seen = []
        while n_iter < max_iter:
            n_iter += 1
            r = (flux - _fit_spline(wave, flux, error, used, knot_spacing)(wave)) / error
            rejected = r <= -k_low
            if clip_high:
                rejected |= r >= k_high
            if grow > 0:
                rejected = binary_dilation(rejected, iterations=grow)
            new_used = allowed & ~rejected
            # A few pixels sitting right at a threshold can flip back
            # and forth between two masks; stop when a mask repeats.
            if np.array_equal(new_used, used):
                status = "converged"
                break
            if any(np.array_equal(new_used, m) for m in seen):
                status = "cycling"
                break
            status = "max_iter"
            seen.append(used)
            used = new_used

    # Final fit on the final mask (the loop can stop right after
    # updating it).
    spline = _fit_spline(wave, flux, error, used, knot_spacing)
    continuum = spline(wave)

    meta = dict(spectrum.meta)
    meta["continuum_err"] = _continuum_error(spline, wave, used, error)
    meta["continuum_mask"] = used
    kept = wave[used]
    meta["continuum_extrapolated"] = (wave < kept[0]) | (wave > kept[-1])
    meta["continuum_niter"] = n_iter
    meta["continuum_status"] = status
    return replace(spectrum, continuum=continuum, meta=meta)


def normalise(spectrum):
    """Divide flux and error by the continuum.

    Parameters
    ----------
    spectrum : Spectrum
        With `continuum` filled in (see fit_continuum).

    Returns
    -------
    spectrum : Spectrum
        A copy with flux and error in units of the continuum and
        continuum = 1. The continuum error is scaled the same way.
    """
    if spectrum.continuum is None:
        raise ValueError("spectrum has no continuum; run fit_continuum first")
    C = spectrum.continuum
    meta = dict(spectrum.meta)
    if "continuum_err" in meta:
        meta["continuum_err"] = meta["continuum_err"] / C
    return replace(
        spectrum,
        flux=spectrum.flux / C,
        error=spectrum.error / C,
        continuum=np.ones_like(C),
        meta=meta,
    )
