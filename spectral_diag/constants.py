"""Physical constants and unit conversions shared by the package."""

import numpy as np
from scipy import constants as _const

# Speed of light, km/s
C_KMS = _const.c * 1e-3
ANGSTROM_TO_CM = 1e-8
KMS_TO_CMS = 1e5
# FWHM of a Gaussian in units of its sigma
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
