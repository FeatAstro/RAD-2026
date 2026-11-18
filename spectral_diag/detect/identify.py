"""Identification of the detected features with one "comb" per species.

The lines of a species (the teeth of its comb) keep their wavelength ratios
at any velocity v = cz: each moves to wave_rest * (1 + v / c). Each feature
is tried as each tooth, which gives a candidate velocity. A tooth matches
a feature if it lands on it and the feature has the strength expected
relative to the other matched teeth. A comb fits if no tooth expected to be
stronger than a matched one is missing. The comb with the most matched
teeth (at least three) sets the velocity of an absorption system, the mean
of its matched teeth; every species is then matched at that velocity, with
a thinner tolerance, including species with a single line in the spectrum.
The search repeats on the features left.
"""

from dataclasses import dataclass

import numpy as np

from .. import constants
from ..line_list import LineSelection, get_lines

IDENTIFY_COLUMNS = ("species", "line", "wave_rest", "v")


@dataclass(frozen=True)
class CombSettings:
    """Parameters of the comb identification.

    Attributes
    ----------
    velocity_min, velocity_max : float
        Range of system velocities (cz) searched, km/s.
    search_tolerance_kms : float
        Largest distance between a tooth and its feature while searching
        for systems, km/s: about the wavelength calibration accuracy
        (~10-15 km/s for COS).
    label_tolerance_kms : float
        The same around a system velocity, once it is known: thinner, as
        the mean of several lines is more accurate than one line. Lines of
        highly ionised species (SiIV, CIV) can be shifted by more than this
        from the low ions in real gas.
    f_min : float
        Weaker lines are left out of the combs: they are rarely detected.
    min_teeth : int
        Fewest matched teeth for a comb to define a system. In 200 sets of
        40 random features over COS G130M, 3 gave no chance system and 2
        gave 4 (tests/test_identify.py). With 3, a system seen only in
        doublets (SiIV, NV, CIV) is not found: use 2 for those.
    strength_tolerance_in_errors : float
        A tooth's W / lambda may leave its expected range by this many
        errors before its match is rejected (see `_wrong_strength`).
    """

    velocity_min: float = -600.0
    velocity_max: float = 600.0
    search_tolerance_kms: float = 15.0
    label_tolerance_kms: float = 10.0
    f_min: float = 1e-3
    min_teeth: int = 3
    strength_tolerance_in_errors: float = 3.0


def identify(features, spectrum, species, settings=CombSettings()):
    """Identify the features with one comb per species.

    Parameters
    ----------
    features : astropy.table.Table
        Output of `find_features` (uses the columns wave, ew, ew_err).
    spectrum : Spectrum
        The spectrum of the features: a tooth with no pixel nearby (gap,
        edge) does not count as missing.
    species : sequence of str
        Species to try, e.g. ["SiII", "SII", "NI", "CII"], each with a
        table saved by `line_list.download_line_table`.
    settings : CombSettings

    Returns
    -------
    identified : astropy.table.Table
        Copy of `features` with the columns species, line (Line.name),
        wave_rest (Angstrom) and v (velocity of the feature for that line,
        km/s); "" and NaN for unidentified features. `meta["systems"]`
        lists the system velocities found, km/s.
    """
    combs = [_comb(name, spectrum, settings) for name in species]
    identified = _with_empty_columns(features)
    velocity = _best_system_velocity(identified, spectrum, combs, settings)
    while velocity is not None:
        for teeth in combs:
            _label_matches(identified, teeth, velocity, settings)
        identified.meta["systems"].append(velocity)
        velocity = _best_system_velocity(identified, spectrum, combs, settings)
    return identified


def _comb(species, spectrum, settings):
    """Ground-level lines of `species` that can fall in the spectrum.

    Parameters
    ----------
    species : str
    spectrum : Spectrum
    settings : CombSettings

    Returns
    -------
    teeth : list of Line
        Empty if none.
    """
    selection = LineSelection(
        wave_min=spectrum.wave[0] / (1 + settings.velocity_max / constants.C_KMS),
        wave_max=spectrum.wave[-1] / (1 + settings.velocity_min / constants.C_KMS),
        f_min=settings.f_min,
        # Hydrogen: one line per n -> n', not its fine-structure components
        multiplet_totals=species == "HI",
    )
    try:
        return get_lines(species, selection)
    except ValueError:
        return []


def _with_empty_columns(features):
    """Copy of `features` with blank identification columns.

    Parameters
    ----------
    features : astropy.table.Table

    Returns
    -------
    identified : astropy.table.Table
    """
    identified = features.copy()
    n_features = len(features)
    identified["species"] = np.full(n_features, "", dtype="U8")
    identified["line"] = np.full(n_features, "", dtype="U20")
    identified["wave_rest"] = np.full(n_features, np.nan)
    identified["v"] = np.full(n_features, np.nan)
    identified.meta["systems"] = []
    return identified


def _best_system_velocity(identified, spectrum, combs, settings):
    """Velocity of the comb that matches the most teeth.

    Parameters
    ----------
    identified : astropy.table.Table
        Features already identified are skipped.
    spectrum : Spectrum
    combs : list of list of Line
    settings : CombSettings

    Returns
    -------
    velocity : float or None
        Mean velocity of the matched features, km/s. None if no comb
        matches `min_teeth` teeth.
    """
    best = []
    for teeth in combs:
        for velocity in _candidate_velocities(identified, teeth, settings):
            matched = _matched_velocities(
                identified, spectrum, teeth, velocity, settings
            )
            if len(matched) > len(best):
                best = matched
    if len(best) < settings.min_teeth:
        return None
    return np.mean(best)


def _candidate_velocities(identified, teeth, settings):
    """Velocities that put one tooth exactly on one unidentified feature.

    Parameters
    ----------
    identified : astropy.table.Table
    teeth : list of Line
    settings : CombSettings

    Returns
    -------
    velocities : list of float
        km/s, inside the searched range.
    """
    velocities = []
    for wave in identified["wave"][identified["line"] == ""]:
        for tooth in teeth:
            velocity = _velocity(wave, tooth)
            if settings.velocity_min <= velocity <= settings.velocity_max:
                velocities.append(velocity)
    return velocities


def _matched_velocities(identified, spectrum, teeth, velocity, settings):
    """Velocities of the features matched by a comb placed at `velocity`.

    Parameters
    ----------
    identified : astropy.table.Table
    spectrum : Spectrum
    teeth : list of Line
    velocity : float
        km/s.
    settings : CombSettings

    Returns
    -------
    velocities : list of float
        km/s. Empty if a tooth stronger than a matched one is missing
        where there is data: it should have been detected.
    """
    matches = _comb_matches(
        identified, teeth, velocity, settings.search_tolerance_kms, settings
    )
    matched_teeth = []
    velocities = []
    for tooth, feature in matches:
        matched_teeth.append(tooth)
        velocities.append(_velocity(identified["wave"][feature], tooth))
    for tooth in teeth:
        if tooth in matched_teeth or not _has_data(spectrum, tooth, velocity, settings):
            continue
        if any(_strength(tooth) > _strength(matched) for matched in matched_teeth):
            return []
    return velocities


def _label_matches(identified, teeth, velocity, settings):
    """Write in `identified` the features matched by a comb at `velocity`.

    Parameters
    ----------
    identified : astropy.table.Table
        Modified in place.
    teeth : list of Line
    velocity : float
        System velocity, km/s.
    settings : CombSettings
    """
    matches = _comb_matches(
        identified, teeth, velocity, settings.label_tolerance_kms, settings
    )
    for tooth, feature in matches:
        identified["species"][feature] = tooth.species
        identified["line"][feature] = tooth.name
        identified["wave_rest"][feature] = tooth.wave_rest
        identified["v"][feature] = _velocity(identified["wave"][feature], tooth)


def _comb_matches(identified, teeth, velocity, tolerance_kms, settings):
    """Teeth that land on a feature with the expected strength.

    Parameters
    ----------
    identified : astropy.table.Table
    teeth : list of Line
    velocity : float
        km/s.
    tolerance_kms : float
    settings : CombSettings

    Returns
    -------
    matches : list of (Line, int)
        Tooth and row index of its feature.
    """
    landed = []
    for tooth in teeth:
        feature = _closest_feature(identified, tooth, velocity, tolerance_kms)
        if feature is not None:
            landed.append((tooth, feature))
    matches = []
    for match in landed:
        if not _wrong_strength(identified, match, landed, settings):
            matches.append(match)
    return matches


def _wrong_strength(identified, match, landed, settings):
    """True if a matched feature has the wrong strength for its tooth.

    Within one species, W / lambda grows with f * lambda: in proportion on
    the linear part of the curve of growth, more slowly once saturated.
    So the ratio of W / lambda of a weaker tooth to that of a stronger one
    lies between their f * lambda ratio and 1. Outside that range (within
    the errors), the weaker tooth landed on an unrelated line by chance.

    Parameters
    ----------
    identified : astropy.table.Table
    match : (Line, int)
        Tooth and row index of its feature.
    landed : list of (Line, int)
        Every tooth of the comb that landed on a feature.
    settings : CombSettings

    Returns
    -------
    wrong : bool
    """
    tooth, feature = match
    strength, strength_err = _measured_strength(identified, feature)
    for stronger_tooth, stronger_feature in landed:
        expected_ratio = _strength(tooth) / _strength(stronger_tooth)
        if expected_ratio >= 1:
            continue
        stronger, stronger_err = _measured_strength(identified, stronger_feature)
        error = settings.strength_tolerance_in_errors * np.hypot(
            strength_err, stronger_err
        )
        too_strong = strength > stronger + error
        too_weak = strength < expected_ratio * stronger - error
        if too_strong or too_weak:
            return True
    return False


def _measured_strength(identified, feature):
    """W / lambda of a feature and its error (the same at any redshift).

    Parameters
    ----------
    identified : astropy.table.Table
    feature : int

    Returns
    -------
    strength, strength_err : float
    """
    wave = identified["wave"][feature]
    return identified["ew"][feature] / wave, identified["ew_err"][feature] / wave


def _closest_feature(identified, tooth, velocity, tolerance_kms):
    """Unidentified feature closest to a tooth placed at `velocity`.

    Parameters
    ----------
    identified : astropy.table.Table
    tooth : Line
    velocity : float
        km/s.
    tolerance_kms : float

    Returns
    -------
    feature : int or None
        Row index, None if no feature is within `tolerance_kms`.
    """
    offsets = np.abs(_velocity(np.asarray(identified["wave"]), tooth) - velocity)
    offsets[identified["line"] != ""] = np.inf
    feature = int(np.argmin(offsets))
    if offsets[feature] > tolerance_kms:
        return None
    return feature


def _has_data(spectrum, tooth, velocity, settings):
    """True if the spectrum has a pixel near a tooth placed at `velocity`.

    Parameters
    ----------
    spectrum : Spectrum
    tooth : Line
    velocity : float
        km/s.
    settings : CombSettings

    Returns
    -------
    has_data : bool
    """
    offsets = np.abs(_velocity(spectrum.wave, tooth) - velocity)
    return np.min(offsets) <= settings.search_tolerance_kms


def _velocity(observed_wave, tooth):
    """Velocity cz that moves a tooth to `observed_wave`.

    Parameters
    ----------
    observed_wave : float or array
        Angstrom.
    tooth : Line

    Returns
    -------
    velocity : float or array
        km/s.
    """
    return constants.C_KMS * (observed_wave / tooth.wave_rest - 1)


def _strength(line):
    """f * lambda: the order of the line strengths within one species.

    Parameters
    ----------
    line : Line

    Returns
    -------
    strength : float
    """
    return line.f * line.wave_rest
