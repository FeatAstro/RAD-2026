# Spectral line diagnostics

Group project for the M2 IRT course "Radiation" (MC, 2026-2027).

A numerical tool that starts from a real spectrum, fits and removes the
continuum, finds the lines, identifies them and characterises them. We
work on **absorption lines** (decided after check-in 1, Oct 14): first
in the spectrum of a star, with the curve of growth used to derive the
column density N and Doppler parameter b of the species with several
lines. The detection code takes a sign option so that emission spectra
(e.g. planetary nebulae, suggested by the teachers) can be handled
later.

## Pipeline
1. **Read** a real spectrum (MAST / SDSS) -- done (`io`)
2. **Fit and remove the continuum** -- done (`continuum`)
3. **Detect** the significant dips -> table of features -- to do (`detect.find_features`)
4. **Identify** them with a "comb" per species -- to do (`linelist`, `detect.identify`)
5. **Measure** their equivalent widths -- done (`ew`), continuum error still to add
6. **Curve of growth**: N and b per species -- done (`cog`)

### How identification works (the comb)
A Doppler shift or a redshift multiplies every wavelength by the same
factor, `lambda_obs = lambda_rest * (1 + z)`, so the *ratios* between
the wavelengths of one species never change. Each species is a comb
of known rest wavelengths (its teeth); we slide it over the spectrum
(trial velocities, e.g. -600 to +600 km/s for a star, any z for a
quasar absorber) and count how many teeth land on a detected dip. The
best position gives the species' velocity and identifies the dips.
Species are tried from the most common (H, He, Ca, Na, Mg, O, Fe...),
and the matched dips are removed before the next one.
- Hydrogen (and HeII) combs are computed from the Rydberg formula with
  `scipy.constants`; other species come from a line table (NIST).
- Strength ratios guard against chance matches: a species is rejected
  if a line predicted to be strong is missing, and doublets (CaII H&K,
  NaD) must have EW ratios between 2:1 (weak lines) and 1:1 (saturated),
  which is the curve of growth again.
- Earth's atmosphere (O2 at 6870 and 7600 A, H2O) is treated as one
  more "species", so that its bands are not taken for stellar lines.

## Data
- **Main spectrum**: the A0 star SDSS J172637.24+264127.6 (SDSS
  2182-53905-0101; 3810-9210 A, vacuum wavelengths, 69 km/s pixels,
  resolution R ~ 2000, S/N ~ 60). Downloaded with
  `io.query_mast` / `io.download_mast_spectra` into `data/mast/`
  (gitignored: run `notebooks/MAST_query.ipynb` once). Expected lines:
  Balmer and Paschen series, CaII H&K and 8498/8542/8662 triplet,
  MgII 4481, OI 7772-7775, NaD (probably interstellar), many weak Fe
  lines, telluric O2 and H2O bands.
- Planetary nebula spectra found in MAST (HST/STIS, NGC 7662 and
  NGC 7027; `figures/pn_candidates_mast.png`) are kept for a possible
  emission-line extension.

## Layout
```
spectral_diag/   Python package (io, continuum, linelist, detect, ew, profiles, cog, uncertainty, viz)
tests/           pytest suite (synthetic checks), run with `pytest`
notebooks/       Demonstrations and validation
figures/         Reference and illustration images used by the notebooks
data/            Small example spectra
Projects.pdf     Project statement
```
Current structure: `spectral_diag/` has the package skeleton (shared
`Spectrum`/`Line`/`Absorber` classes in `core.py`, one stub module per
planned component). Implemented so far: `profiles` (Gaussian and Voigt
optical depth), `ew`, `io` (synthetic spectra, MAST query/download,
SDSS reader), `continuum`, `linelist` (CIV, SiII, FeII)
and `cog` (see Progress log below).

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
| Oct 14 | Check-in 1: absorption chain validated on synthetic spectra (profiles, EW, COG). Feedback: start from a real spectrum, continuum first, then find and identify the lines |
| Nov 18 | Check-in 2: on the star, continuum + automatic detection + comb identification + EWs; curve of growth for the species with several lines |
| Dec 9 | Oral presentation (50% of the course grade) |

## Team
Details and interface in `Work_split.pdf`.
- **Pablo** (data -> lines): `detect.find_features` (detection of
  significant dips, merging, significance), plot of the detected
  features, telluric masks; later NIST queries for the line list.
- **Gael** (lines -> physics): `linelist` (Rydberg hydrogen comb +
  table of common species with f-values), `detect.identify` (comb scan,
  scoring, strength-ratio checks), continuum error in `measure_ew`,
  curve of growth on the identified lines.
- The two meet at the **features table** (columns in
  `spectral_diag/detect/__init__.py`), so each side can be tested alone: Pablo
  on synthetic spectra with known line positions, Gael on a
  hand-written features table read off the star.

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
- `linelist.py`: `get_lines(species)` returns the tabulated lines of
  CIV (2), SiII (6) or FeII (7) as `Line` objects. Wavelengths and f
  are from Verner, Verner & Ferland 1996 (arXiv:atom-ph/9604003);
  gamma is the sum of the tabulated A values over the listed decays of
  the upper level.
- `cog.py`: `theoretical_ew(line, logN, b, z=0.0)` integrates
  `1 - exp(-tau)` over the Voigt profile in the variable `x = dv/b`,
  so it is valid in all three regimes. `fit_cog(lines, ews, z=0.0)`
  fits (logN, b) by weighted least squares to measured EWs given as a
  dictionary `{line.name: (W, W_err)}`, starting from the best points
  of a coarse grid, and returns the best fit, 1-sigma errors,
  covariance and chi2. Single cloud, no instrumental broadening.
- `notebooks/04_curve_of_growth.ipynb` validates both:
  `theoretical_ew` matches the linear and damped closed forms to
  better than 1e-4; injected (logN, b) are recovered from noise-free
  synthetic FeII spectra to 0.01 dex, and from a damped SiII case when
  the EW window is wide enough (a +-20 A window biases logN by -0.03
  dex); over 100 noisy spectra at S/N = 50 the fit is unbiased and the
  scatter matches the reported errors (0.019 dex in logN, 0.10 km/s in
  b). It also shows the N-b degeneracy: with only the three strongest
  FeII lines, logN is uncertain by 1.8 dex at 68% for a 0.02 A EW
  error, against 0.13 dex with all seven lines.
- Check-in 1 (Oct 14): the teachers asked us to start from a real
  spectrum and to focus on planetary nebulae, so on emission lines. The
  curve of growth does not apply to these (the forbidden lines are
  optically thin, so the flux is simply proportional to the number of
  emitting ions); the absorption code above stays as a validated module.
  Searched MAST for bright nebulae with HST/STIS spectra and selected
  NGC 7662 and NGC 7027 (see Data above).
- First real spectrum: before the planetary nebulae, the absorption
  chain is tested on the A0 star SDSS J172637.24+264127.6 (SDSS
  2182-53905-0101, 3810-9210 A, R ~ 2000, S/N 62). `io.query_mast`,
  `io.download_mast_spectra` and `io.read_sdss_spec` (Pablo) fetch and
  read it (`notebooks/MAST_query.ipynb`); a pytest suite (`tests/`)
  automates the earlier notebook checks.
- `continuum.py`: `fit_continuum(spectrum, knot_spacing=250, k_low=2,
  k_high=4, grow=3, exclude=None)` fits a weighted cubic spline through
  the pixels kept by iterative asymmetric sigma-clipping (reject more
  than 2 sigma below the fit, grow each rejected region by 3 pixels;
  the upper clip is switched on only once the lower clip has converged,
  otherwise the first, line-depressed fit throws away real continuum
  pixels). Returns the continuum, its statistical error from the spline
  covariance, and the mask of kept pixels; `normalise` divides by it.
  Synthetic checks are in `tests/test_continuum.py`: on an SDSS-like
  A-star spectrum (blackbody continuum, S/N 60) the continuum is
  recovered to < 0.5% away from the ends, and a broad (Balmer-like)
  line comes out too weak (about -0.6 A of 15 A at 250 A knot spacing,
  more with closer knots). `notebooks/05_continuum.ipynb` works on the
  real star only: fit, zooms on the difficult regions, residuals of the
  kept pixels (centred on 0, but the 200 A bin around Hbeta sits 0.7%
  below the continuum: its outer wings stay in the fit), and the Balmer
  EWs for knot spacings 150-700 A: e.g. Hbeta = 14.02 +- 0.23 (noise)
  +- 0.84 (continuum) A. The continuum dominates the error budget of
  broad lines.
- Check-in feedback (Oct 14) turned into a plan: absorption lines,
  continuum first, then detection and identification with a comb per
  species (wavelength ratios fixed by atomic physics) checked by
  strength ratios. Split: Pablo on detection, Gael on line list and
  identification (see Team).
- Next: `detect.find_features` (Pablo), `linelist` Rydberg comb and
  `detect.identify` (Gael).
