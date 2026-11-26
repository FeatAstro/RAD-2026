"""Feature detection and line identification.

Interface agreed so both parts can be written in parallel; see
Work_split.pdf.

- find_features(spectrum, threshold=4.0, sign=-1)          [Pablo]
    Find the significant features of a Spectrum whose continuum has
    been fitted (continuum.fit_continuum). sign=-1 looks for absorption
    dips, sign=+1 for emission peaks.
    Returns an astropy Table, one row per feature, with columns:
        wave        observed centre, Angstrom (vacuum)
        wave_err    1-sigma uncertainty on `wave`, Angstrom
        wave_min    start of the feature's window, Angstrom
        wave_max    end of the feature's window, Angstrom
        depth       1 - min(flux / continuum) (absorption),
                    max(flux / continuum) - 1 (emission); always positive
        ew          equivalent width over [wave_min, wave_max], Angstrom;
                    positive for absorption and for emission
        ew_err      1-sigma uncertainty on `ew`, Angstrom; includes the
                    continuum error if `spectrum.continuum_error` is set
        snr         detection significance, ew / ew_err

- identify(features, spectrum, species, settings=CombSettings())   [Gael]
    One "comb" per species (its lines, fixed wavelength ratios), tried
    at the velocity v = cz where any tooth lands on any feature; the comb
    with the most matched teeth sets a system velocity, at which every
    species is then matched (see identify.py). Adds the columns species,
    line (Line.name), wave_rest (Angstrom) and v (km/s), blank for
    unidentified features, and meta["systems"] (velocities, km/s).

Layout (one module per job; `identify` goes in its own module):
    features.py      find_features, the pipeline that builds the table
    significance.py  signal relative to the continuum, matched filter, windows
                     and their split between resolved lines
    measure.py       centre (Gaussian fit), depth and EW of one feature
    identify.py      comb identification of the features [Gael]
"""

from .features import FEATURE_COLUMNS, find_features
from .identify import IDENTIFY_COLUMNS, CombSettings, identify

__all__ = [
    "FEATURE_COLUMNS",
    "IDENTIFY_COLUMNS",
    "CombSettings",
    "find_features",
    "identify",
]
