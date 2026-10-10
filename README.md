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
Current structure: `spectral_diag/` has the package skeleton (shared
`Spectrum`/`Line`/`Absorber` classes in `core.py`, one stub module per
planned component). `profiles.py` has the Gaussian (Doppler-only) and
full Voigt optical depth implemented (see Progress log below).

## Installation
```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install the package (editable) and all its dependencies
pip install -e .
```
`pip install -e .` installs `spectral_diag` in editable mode using `pyproject.toml`,
so `from spectral_diag import ...` works from any notebook or script in the repo,
and edits to the package source take effect immediately without reinstalling.

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

## Progress log
- Package skeleton (`spectral_diag/`), shared `Spectrum`/`Line`/`Absorber`
  classes, `pyproject.toml` for `pip install -e .`.
- `profiles.py`: `optical_depth_gaussian(wave, line, logN, b, z)` --
  Doppler-only (no damping wings) optical depth for a single
  transition, as a stepping stone to the full Voigt profile.
  `tau(lambda) = tau0 * exp(-(dv/b)^2)`, with `dv` the velocity offset
  from line center (km/s) and `tau0` set by N, f, lambda_rest, b
  (standard constant 1.4974e-15 for N in cm^-2, lambda_rest in
  Angstrom, b in km/s).
- `notebooks/01_gaussian_optical_depth.ipynb`: plots transmission
  `exp(-tau)` while varying N at fixed b, and b at fixed N, for a
  test line. Shows the line deepening with N until it starts to
  saturate (core stops deepening, only wings keep growing -- this is
  why the curve of growth later is not a straight line), and
  broadening/shallowing with b.
- `profiles.py`: added `voigt(x, a)` (`Re[scipy.special.wofz(x + i*a)]`)
  and `optical_depth(wave, line, logN, b, z)` -- the full Voigt optical
  depth, adding the Lorentzian/damping term (`line.gamma`) on top of
  the Gaussian term, via the damping parameter
  `a = 7.9577e-15 * gamma * lambda_rest / b` (ratio of Lorentzian to
  Doppler width). Validated: matches `optical_depth_gaussian` to
  machine precision when `gamma -> 0`, and for a saturated line
  (logN=17) keeps non-negligible optical depth far in the wings where
  the Gaussian-only version has already dropped to ~0 -- the damping
  wings `optical_depth_gaussian` cannot capture.
- Next: equivalent width measurement (`ew.py`) on top of these
  profiles, then the synthetic spectrum generator (`io.py`) to start
  validating against known injected (z, N, b).
