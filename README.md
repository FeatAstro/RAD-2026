# Spectral line diagnostics

Group project for the M2 IRT course "Radiation" (MC, 2026-2027).

A numerical tool that starts from a real spectrum, fits the underlying
continuum, finds the lines and characterises them. Following the
feedback at check-in 1 (Oct 14), the main target is the optical
emission-line spectrum of a **planetary nebula**: line fluxes and line
ratios give the dust reddening, electron temperature, electron density
and ionic abundances of the nebula.

## Planned features
- Load real spectra (HST/STIS from the MAST archive), including our own
  extraction of a 1D spectrum from the 2D long-slit image
- Continuum fitting with sigma-clipping that rejects emission lines
- Line list of nebular lines (H and He recombination lines, forbidden
  lines of O, N, S, Ne, Ar)
- Automatic line detection and identification against the line list
- Gaussian line fitting (including blends such as Halpha + [NII]) and
  line fluxes with uncertainties
- Diagnostics: reddening from the Balmer decrement, electron temperature
  from [OIII] and [NII], electron density from [SII] and [OII], ionic
  abundances
- Uncertainty estimation (Monte Carlo on the noise and the continuum)
- Interactive visualisation of the spectrum and fitted lines
- Already done for absorption spectra (kept as a validated module):
  Voigt profiles, equivalent widths, theoretical curves of growth and
  the (N, b) fit

## Data
Candidate spectra found in MAST (HST/STIS CCD long slit, G430L +
G750L, together about 2900-10200 A at a few A per pixel):
- **NGC 7662**, programme 12600, slit offset onto the nebula (no central
  star in the slit). Best blue side: [OIII] 4363 separated from Hgamma,
  [NeIII] 3869, HeII 4686, the Balmer series. Main target.
- **NGC 7027**, programme 8285. Richest red side (Halpha + [NII],
  [NII] 5755, [OI], [SII], [ArIII], [OII] 7320/7330, [SIII] 9069/9532),
  but heavily reddened and noisy in the blue. Second target.
- NGC 3242, programme 12600, similar to NGC 7662.

The pipeline 1D products (`_sx1`) extract only 7 pixels along the slit
and are very noisy for an extended nebula; summing the nebular rows of
the 2D rectified image (`_sx2`) gives a much better spectrum (quick
look: `figures/pn_candidates_mast.png`).

## Layout
```
spectral_diag/   Python package (io, continuum, linelist, detect, ew, profiles, cog, uncertainty, viz; diagnostics planned)
notebooks/       Demonstrations and validation
figures/         Reference and illustration images used by the notebooks
data/            Small example spectra
Projects.pdf     Project statement
```
Current structure: `spectral_diag/` has the package skeleton (shared
`Spectrum`/`Line`/`Absorber` classes in `core.py`, one stub module per
planned component). Implemented so far: `profiles` (Gaussian and Voigt
optical depth), `ew`, `io.make_synthetic`, `linelist` (CIV, SiII, FeII)
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
| Oct 14 | Check-in 1: absorption chain validated on synthetic spectra (profiles, EW, COG). New target: planetary nebulae |
| Nov 18 | Check-in 2: real PN spectrum extracted, continuum fitted, lines detected, identified and fitted; reddening and electron temperature |
| Dec 9 | Oral presentation (50% of the course grade) |

## Team
See `Work_split.pdf` for the details.
- Student A: data to lines (MAST download and 1D extraction, continuum,
  line list, detection and identification, visualisation)
- Student B: lines to physics (line profiles and fits, fluxes and
  errors, reddening, temperature and density diagnostics, abundances,
  uncertainties)

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
- Next: `io` reader for STIS `_sx2` files with extraction along the
  slit, then `continuum.py` for an emission-line spectrum.
