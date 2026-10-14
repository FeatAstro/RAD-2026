"""Atomic data for common ISM/CGM lines.

Rest wavelengths (vacuum) and oscillator strengths are from Verner,
Verner & Ferland 1996, Atomic Data and Nuclear Data Tables 64, 1
(arXiv:atom-ph/9604003), Table 1. Only transitions from the ground
level are listed.

gamma is the sum of the tabulated A_ki over every decay of the upper
level listed in that table (the decays to the ground term). Decays to
other terms are not in the table, so gamma is a slight underestimate
for FeII.
"""

from .core import Line

# species -> (wave_rest [Angstrom], f, gamma [s^-1])
_TABLE = {
    "CIV": [
        (1548.1950, 1.90e-1, 2.64e8),
        (1550.7700, 9.52e-2, 2.64e8),
    ],
    "SiII": [
        (1190.416, 2.93e-1, 4.100e9),
        (1193.290, 5.84e-1, 4.100e9),
        (1260.422, 1.18e0, 2.972e9),
        (1304.370, 8.71e-2, 1.016e9),
        (1526.707, 1.32e-1, 1.121e9),
        (1808.013, 2.56e-3, 3.125e6),
    ],
    "FeII": [
        (1608.4511, 8.92e-2, 3.812e8),
        (2344.2139, 1.26e-1, 2.542e8),
        (2367.5905, 4.25e-3, 3.293e8),
        (2374.4612, 5.27e-2, 3.304e8),
        (2382.7652, 3.43e-1, 3.350e8),
        (2586.6500, 5.48e-2, 2.864e8),
        (2600.1729, 2.40e-1, 2.892e8),
    ],
}


def get_lines(species):
    """Return the tabulated transitions of `species` as a list of Line."""
    if species not in _TABLE:
        raise KeyError(f"no lines tabulated for {species!r}; available: {sorted(_TABLE)}")
    return [Line(species, wave_rest, f, gamma) for wave_rest, f, gamma in _TABLE[species]]
