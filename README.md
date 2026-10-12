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
  from line center (km/s) and `tau0` set by N, f, lambda_rest, b via
  `TAU0_CONST` (N in cm^-2, lambda_rest in Angstrom, b in km/s) --
  computed from the CODATA classical electron radius via
  `scipy.constants`, not a hardcoded literal.
- `notebooks/01_profiles.ipynb`: plots transmission `exp(-tau)` while
  varying N at fixed b, and b at fixed N, for a test line. Shows the
  line deepening with N until it starts to saturate (core stops
  deepening, only wings keep growing -- this is why the curve of
  growth later is not a straight line), and broadening/shallowing
  with b.
- `profiles.py`: added `voigt(x, a)` (`Re[scipy.special.wofz(x + i*a)]`)
  and `optical_depth(wave, line, logN, b, z)` -- the full Voigt optical
  depth, adding the Lorentzian/damping term (`line.gamma`) on top of
  the Gaussian term, via the damping parameter
  `a = A_CONST * gamma * lambda_rest / b` (ratio of Lorentzian to
  Doppler width; `A_CONST` is a pure unit-conversion factor, no
  physical constants needed). Validated: matches `optical_depth_gaussian` to
  machine precision when `gamma -> 0`, and for a saturated line
  (logN=17) keeps non-negligible optical depth far in the wings where
  the Gaussian-only version has already dropped to ~0 -- the damping
  wings `optical_depth_gaussian` cannot capture.
- `ew.py`: `measure_ew(wave, flux, continuum, error=None, wave_min=None,
  wave_max=None)` -- equivalent width via trapezoidal integration of
  `1 - flux/continuum` over an optional wavelength window, with error
  propagation from per-pixel flux uncertainties (assumes the continuum
  itself is error-free). Validated against the closed-form linear-regime
  EW formula (`W = pi * r_e * N * f * lambda_rest^2`, independent of b)
  on a weak synthetic line: numeric integral matches analytic formula
  to 0.08%.
- `io.py`: `make_synthetic(wave, absorbers, continuum=1.0, snr=None, rng=None)`
  -- builds a `Spectrum` from a list of `Absorber`s (summing tau over
  all their lines -- exact, not an approximation, since transmissions
  multiply and optical depths add). `snr=None` gives a noise-free
  spectrum (`error=0`, for checking against closed-form truth);
  `snr=<value>` adds Gaussian noise at a constant level relative to
  the continuum. No LSF convolution yet (`profiles.apply_lsf` doesn't
  exist), so this is the intrinsic model spectrum, not yet a realistic
  stand-in for a specific instrument.
  Validated end-to-end: (1) noise-free weak-line spectrum's EW matches
  the closed-form linear-regime formula to 0.08%; (2) over 400 noisy
  Monte Carlo realizations, the mean *predicted* EW uncertainty from
  `measure_ew`'s error propagation (0.00745) matches the *actual*
  scatter in measured EW (0.00764) to ~2.5% -- confirms the error bars
  are correctly calibrated against real injected noise, not just
  structurally plausible.
- Added validation notebooks for the last three features (each with
  practical, assertion-backed checks, not just ad-hoc scripts): the
  Voigt checks (Voigt reduces to Gaussian as damping -> 0; damping
  wings appear for a saturated line, where Gaussian-only has already
  dropped to 0) ended up in `notebooks/01_profiles.ipynb` alongside
  the Gaussian ones (merged -- see reorganization note below),
  `notebooks/02_equivalent_width.ipynb` (measured EW matches the
  closed-form linear-regime formula to 0.08%; EW is independent of b
  in that regime, to 0.26% spread across b=3-60 km/s),
  `notebooks/03_synthetic_spectrum.ipynb` (same closed-form check
  through the full `Spectrum` object; 400 noisy Monte Carlo
  realizations confirm predicted vs. actual EW uncertainty agree to
  ~2.5%).
- Reorganized `notebooks/`: merged the Gaussian and Voigt notebooks
  into one `01_profiles.ipynb` (they test the same progression,
  Gaussian then full Voigt) and moved the real-CIV reference material
  (atomic data table + observed quasar spectrum images) into its own
  `00_reference_civ_line.ipynb`, linked from the others instead of
  re-embedded in each -- avoids duplicating ~400KB of images per
  notebook. Going forward, notebooks are organized by validation
  theme rather than 1:1 per function.
- Next: continuum fitting (`continuum.py`) and the theoretical curve
  of growth (`cog.py`), to close the loop toward fitting (z, N, b)
  from measured EWs across linear, saturated and damped regimes.
