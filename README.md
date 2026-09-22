# Spectral line diagnostics

Group project for the M2 IRT course "Radiation" (MC, 2026-2027).

A numerical tool to automatically identify lines in a spectrum, measure their equivalent widths above the continuum, and use theoretical curves of growth to estimate the redshift, column density and velocity dispersion of the absorbing gas species.

## Planned features
- Load spectra from files or archive queries (FITS/ASCII), plus a synthetic spectrum generator for testing
- Continuum fitting and normalisation
- Line lists from atomic databases (wavelength, oscillator strength, damping constant)
- Automatic line detection and identification by redshift matching
- Equivalent width measurement with uncertainties
- Voigt line profiles and theoretical curves of growth
- Fit of (z, N, b) per species, with uncertainties (χ² grid / MCMC)
- Interactive visualisation of the spectrum and fitted lines
- Main target: UV/optical quasar absorption spectra; designed to extend to other domains (UV to radio)

## Layout
```
spectral_diag/   Python package (io, continuum, linelist, detect, ew, profiles, cog, uncertainty, viz)
notebooks/       Demonstrations and validation
data/            Small example spectra
Projects.pdf     Project statement
```
(Structure being built; not all directories exist yet.)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy astropy astroquery matplotlib lmfit emcee
```

## Usage
To be written once the first end-to-end pipeline runs.

## Timeline
| Date | Milestone |
|---|---|
| Oct 14 | Check-in 1: end-to-end pipeline on synthetic spectra (profiles, COG, continuum, EW) |
| Nov 18 | Check-in 2: real spectra, line identification, multi-line COG fits |
| Dec 9 | Oral presentation (50% of the course grade) |

## Team
- Student A: data side (I/O, continuum, line lists, detection, visualisation)
- Student B: physics side (profiles, EW, COG, fitting, uncertainties)
