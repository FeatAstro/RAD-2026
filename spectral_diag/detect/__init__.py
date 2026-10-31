"""Feature detection and line identification.

Interface agreed so both parts can be written in parallel; see
Work_split.pdf.

- find_features(spectrum, threshold=4.0, sign=-1)          [Pablo]
    Find the significant features of a Spectrum whose continuum has
    been fitted (continuum.fit_continuum). sign=-1 looks for absorption
    dips, sign=+1 for emission peaks. A broad line (e.g. a Balmer line
    with its wings) must come out as ONE feature, not several.
    Returns an astropy Table, one row per feature, with columns:
        wave        observed centre, Angstrom (vacuum)
        wave_err    1-sigma uncertainty on `wave`, Angstrom
        wave_min    start of the feature's window, Angstrom
        wave_max    end of the feature's window, Angstrom
        depth       1 - min(flux / continuum) (absorption),
                    max(flux / continuum) - 1 (emission); always positive
        ew          equivalent width over [wave_min, wave_max], Angstrom;
                    positive for absorption and for emission
        ew_err      1-sigma uncertainty on `ew`, Angstrom
        snr         detection significance, ew / ew_err

- identify(features, species, v_range=(-600, 600), tol=None)   [Gael]
    Slide the "comb" of each species (its rest wavelengths, from
    line_list) over the features: for each trial velocity v, place the
    teeth at lambda_rest * (1 + v / c) and score the teeth that land on
    a feature within `tol` (default: about one resolution element),
    weighted by the expected line strength and penalised for strong
    lines that are missing. Species are tried from the most common
    (H first); matched features are removed before the next species.
    Returns the features table with added columns `species`, `line`
    (Line.name), `wave_rest` and `v` (km/s); unmatched features are
    left blank.

Layout (one module per job; `identify` goes in its own module):
    features.py      find_features, the pipeline that builds the table
    significance.py  signal relative to the continuum, matched filter, windows
    measure.py       centre (Gaussian fit), depth and EW of one feature
    identify.py      [Gael] to add, and to re-export here
"""

from .features import FEATURE_COLUMNS, find_features

__all__ = ["FEATURE_COLUMNS", "find_features"]
