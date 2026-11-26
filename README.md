# Spectral line diagnostics

Group project for the M2 IRT course "Radiation" (MC, 2026-2027).

A numerical tool that starts from a real spectrum, fits and removes the
continuum, finds the lines, identifies them and characterises them. We
work on **absorption lines** (decided after check-in 1, Oct 14): first
in the spectrum of an A0 star, then, since Nov 18, in the HST/COS
spectrum of the quasar **3C 273**, whose Milky Way absorption lines give
the column density N and Doppler parameter b of the species with several
lines through the curve of growth. The star is run through the whole
chain too, as a counter-example: a photosphere is not a slab of gas, so
its curve of growth fails. The
detection code takes a sign option so that emission spectra can be
handled later.

## Pipeline
1. **Read** a real spectrum (MAST / SDSS) -- done (`mast_archive`, `spectrum_reader`)
2. **Fit and remove the continuum** -- done (`continuum`)
3. **Detect** the significant dips -> table of features -- done (`detect.find_features`)
4. **Identify** them with a "comb" per species -- done (`line_list`, `detect.identify`)
5. **Measure** their equivalent widths -- done (`equivalent_width`, noise and continuum errors,
   statistical and systematic)
6. **Curve of growth**: N and b per species -- done (`curve_of_growth`), on 3C 273 for S II and Si II
7. **Uncertainties**: chi2 map in (log N, b) -- done (`uncertainty`), rescaled by chi2_min/dof

### How identification works (the comb)
A Doppler shift or a redshift multiplies every wavelength by the same
factor, `lambda_obs = lambda_rest * (1 + z)`, so the *ratios* between
the wavelengths of one species never change. Each species is a comb
of known rest wavelengths (its teeth, from NIST). Each detected feature
is tried as each tooth, which gives a candidate velocity (-600 to +600
km/s). A comb defines a system if:
- at least 3 teeth land on features, within 15 km/s;
- the EW ratios are consistent: for a weaker and a stronger line,
  `W/lambda` ratio between the `f*lambda` ratio (linear part of the
  curve of growth) and 1 (saturated), within 3 errors;
- no tooth stronger than a matched one is missing where there is data.

The best comb sets the system velocity; every species is then labelled
at that velocity (within 10 km/s), single-line species included (C II
1334, Si III 1206, O I 1302). The matched features are removed and the
search repeats. A species with a single line in the range (H I in the
COS far UV) cannot define a system on its own. By default the teeth are
ground-level lines (cold gas); `CombSettings.e_low_max` allows excited
levels, e.g. the Balmer lines of a star.

## Data
- **Main spectrum since Nov 18**: the quasar 3C 273, HST/COS G130M
  co-add of the Hubble Spectroscopic Legacy Archive
  (`hst_cos_3c273--0392_g130m-lp01_cspec.fits`; 1134-1470 A, R ~ 18000,
  S/N ~ 39), read with `spectrum_reader.read_hsla` and downloaded by
  `notebooks/full_quasar_tutorial.ipynb` into `data/mast/`. The STIS
  E140M co-add is read too, for comparison.
- **First spectrum**: the A0 star SDSS J172637.24+264127.6 (SDSS
  2182-53905-0101; 3810-9210 A, vacuum wavelengths, 69 km/s pixels,
  resolution R ~ 2000, S/N ~ 60). Downloaded with
  `mast_archive.query_mast` / `mast_archive.download_mast_spectra` into `data/mast/`
  (gitignored: section 2 of `notebooks/full_star_tutorial.ipynb` downloads it). Expected lines:
  Balmer and Paschen series, CaII H&K and 8498/8542/8662 triplet,
  MgII 4481, OI 7772-7775, NaD (probably interstellar), many weak Fe
  lines, telluric O2 and H2O bands.
- Planetary nebula spectra found in MAST (HST/STIS, NGC 7662 and
  NGC 7027; `figures/pn_candidates_mast.png`) are kept for a possible
  emission-line extension.

## Layout
```
spectral_diag/   Python package (constants, data_classes, spectrum_reader, mast_archive, synthetic_spectrum,
                 continuum, detect, line_list, line_profiles, equivalent_width,
                 curve_of_growth, uncertainty, plots)
tests/           pytest suite (synthetic checks), run with `pytest`
notebooks/       Demonstrations and validation
figures/         Reference and illustration images used by the notebooks
data/            Small example spectra
Projects.pdf     Project statement
```
Implemented: `data_classes` (shared `Spectrum`/`Line`/`Absorber`),
`line_profiles` (Gaussian and Voigt optical depth), `equivalent_width`,
`synthetic_spectrum`, `mast_archive`, `spectrum_reader` (SDSS, COS x1dsum,
HSLA co-adds), `continuum`, `detect` (`find_features`, `identify`),
`line_list` (19 species from NIST), `curve_of_growth`, `uncertainty`
(chi2 map) and `plots`. The two tutorials, `full_star_tutorial.ipynb` and
`full_quasar_tutorial.ipynb`, run the whole chain on real data.

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
| Nov 18 | Check-in 2: star done up to detection; scope moved to the quasar 3C 273 |
| Nov 22 | 3C 273: continuum, detection, comb identification, curves of growth of S II and Si II |
| Nov 26 | Star run to the end (counter-example); chi2 map of the curves of growth |
| Dec 9 | Oral presentation (50% of the course grade) |

## Team
Details and interface in `Work_split.pdf`.
- **Pablo** (data -> lines): `detect.find_features` (detection of
  significant dips, merging, significance), plot of the detected
  features, telluric masks; later NIST queries for the line list.
- **Gael** (lines -> physics): `line_list` (Rydberg hydrogen comb +
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
- `line_profiles.py`: `optical_depth_gaussian(wave, line, logN, b, z)` --
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
- `line_profiles.py`: added `voigt(x, a)` (`Re[scipy.special.wofz(x + i*a)]`)
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
- `equivalent_width.py`: `measure_ew(wave, flux, continuum, error=None, wave_min=None,
  wave_max=None)` -- equivalent width via trapezoidal integration of
  `1 - flux/continuum` over an optional wavelength window, with error
  propagation from per-pixel flux uncertainties (assumes the continuum
  itself is error-free). Validated against the closed-form linear-regime
  EW formula (`W = pi * r_e * N * f * lambda_rest^2`, independent of b)
  on a weak synthetic line: numeric integral matches analytic formula
  to 0.08%.
- `synthetic_spectrum.py`: `make_synthetic(wave, absorbers, continuum=1.0, snr=None, rng=None)`
  -- builds a `Spectrum` from a list of `Absorber`s (summing tau over
  all their lines -- exact, not an approximation, since transmissions
  multiply and optical depths add). `snr=None` gives a noise-free
  spectrum (`error=0`, for checking against closed-form truth);
  `snr=<value>` adds Gaussian noise at a constant level relative to
  the continuum. No LSF convolution yet (`line_profiles.apply_lsf` doesn't
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
- `line_list.py`: `get_lines(species, selection)` returns the lines of a
  species as `Line` objects, from the NIST Atomic Spectra Database.
  `download_line_table(species)` queries NIST once and saves the table
  in `data/line_lists/` (committed), so `get_lines` works offline.
  Saved so far: HI, CII, CIV, NI, NV, OI, OVI, NaI, MgII, AlII, SiII,
  SiIII, SiIV, SII, CaII, FeII. gamma sums the decay rates of both
  levels; for hydrogen, `LineSelection(multiplet_totals=True)` gives one
  line per n -> n' instead of its fine-structure components.
- `curve_of_growth.py`: `theoretical_ew(line, logN, b, z=0.0)` integrates
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
  2182-53905-0101, 3810-9210 A, R ~ 2000, S/N 62). `mast_archive.query_mast`,
  `mast_archive.download_mast_spectra` and `spectrum_reader.read_sdss_spec` (Pablo) fetch and
  read it (`notebooks/full_star_tutorial.ipynb`, section 2); a pytest suite (`tests/`)
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
- Next: `detect.find_features` (Pablo), `line_list` Rydberg comb and
  `detect.identify` (Gael).
- Oct 31 (Gael): simpler code. Modules renamed to say what they do:
  `core` -> `constants` + `data_classes`; `io` -> `sdss_reader`,
  `mast_archive`, `synthetic_spectrum`; `linelist` -> `line_list`;
  `profiles` -> `line_profiles`; `ew` -> `equivalent_width`; `cog` ->
  `curve_of_growth`; `viz` -> `plots`. `continuum` keeps only the cut
  below the fit (the upper cut, mask growing and cycle detection
  changed nothing on the star); `curve_of_growth` split into small
  functions with full names. The two continuum notebooks are merged
  into `05_continuum.ipynb`.
- Nov 18 (Gael): scope moved to the quasar 3C 273 (HST/COS G130M). Line
  lists from NIST only, saved in `data/line_lists/`. Continuum error
  propagated into the EWs. `detect.identify`: one comb per species,
  candidate velocities from each feature and each line, at least 3
  matched lines with consistent EW ratios and no strong line missing;
  every species is then labelled at the system velocity.
  `tests/test_identify.py` (features tables) and
  `notebooks/07_identification.ipynb` (synthetic COS spectra with a Milky
  Way absorber, a high-velocity cloud and 30 unrelated lines): over 30
  spectra, every system found within 1 km/s (+0.2 +- 0.7 km/s), no false
  system, 776 correct labels, 7 wrong, 4 missed lines.
- Nov 18-22 (Pablo): `spectrum_reader.read_cos_x1dsum` and `read_hsla`;
  `continuum` option `clip_high`. Quasar tutorial: geocoronal Ly-alpha
  cut (1214-1217 A), continuum with 2 A knots, detection at 12 sigma
  (threshold set from an emission search, where every feature is noise):
  44 features.
- Nov 22 (Gael): `continuum.continuum_systematic_error` (refit with the
  knot spacing +-20 %; ~1.4x the statistical error on 3C 273), added to
  the continuum error of the EWs. `identify` on 3C 273: one system, the
  Milky Way at -12.7 km/s, 19 of 44 features labelled (Fe II, Si II, N I,
  Si III, N V, S II, C I, O I, C II, Si IV); Si II 1190 is only accepted
  with the systematic error. Curves of growth (`measured_ews`,
  `plots.plot_curve_of_growth`): S II log N = 15.33 +- 0.04, b = 21.8
  km/s; Si II 15.08 +- 0.06, b = 31.4 km/s, against 15.46 and 15.10 in
  Sembach et al. 2001 (astro-ph/0108047). Si II is saturated and not
  robust (16.14 with 3 lines instead of 4).
- Nov 26 (Gael): star tutorial run to the end. `CombSettings.e_low_max`
  (default 0, ground level only) lets the Balmer lines form a comb; the
  strength check rejects them (H-alpha has the largest f*lambda but the
  smallest W/lambda), so it is turned off for the star. One system at
  -31 km/s (SDSS: -37); the Balmer curve of growth gives chi2 = 2378 for
  5 dof and b = 103 km/s against 12.6 km/s thermal: Stark wings and a
  photosphere, not a slab, so N and b have no physical meaning there.
- Nov 26 (Gael): `uncertainty.chi2_map` (chi2 on a 241 x 150 grid of
  (log N, b)), `profile` and `profile_interval` (68 and 95 % intervals
  from the profile, Delta chi2 = 1 and 3.84), rescaled by chi2_min/dof;
  `plots.plot_chi2_map`, `plot_profiles`. Validated in
  `notebooks/08_chi2_map.ipynb` and `tests/test_uncertainty.py`. On 3C 273:
  S II log N = 15.33 +0.10/-0.09; Si II has two minima (15.08 and ~16.2),
  14.8-16.5 at 95 %: its saturated lines do not pin N down.
