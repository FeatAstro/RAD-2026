"""Theoretical curves of growth and multi-line fits for N and b.

Planned contents:
- theoretical_cog: W(logN, b) for a given Line, across linear,
  saturated and damped regimes
- fit_cog: given measured EWs for several lines of the same species,
  solve for (logN, b)
"""

import numpy as np

from .core import C_KMS
from .profiles import TAU0_CONST


def ew_linear(line, logN):
    """Rest-frame equivalent width in the linear (optically thin) regime.

    W = sqrt(pi) * TAU0_CONST * N * f * lambda_rest**2 / c, independent of b.

    Parameters
    ----------
    line : Line
        Atomic transition (wave_rest in Angstrom, f).
    logN : float or array
        log10(column density / cm^-2).

    Returns
    -------
    W : float or array
        Equivalent width, Angstrom.
    """
    return TAU0_CONST * np.sqrt(np.pi) * 10**logN * line.f * line.wave_rest**2 / C_KMS
