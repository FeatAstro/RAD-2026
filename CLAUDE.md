# CLAUDE.md – Project 1: Spectral line diagnostics

M2 IRT, MC "Radiation" 2026-2027. Group project (2 students), described in `Projects.pdf` (section 1).

## Goal
Build a numerical tool that:
1. automatically identifies lines in a given spectrum,
2. measures their equivalent widths (EW) taking into account the underlying continuum,
3. uses theoretical curves of growth (COG) to estimate the physical properties of the absorbing gas per species: redshift z, column density N, velocity dispersion b.

Possible developments listed by the teachers: spectra database queries, continuum fitting, spectral line database queries, EW estimation, theoretical line profiles, theoretical COG, interactive visualization, uncertainty estimation, applicability from UV to radio.

## Deadlines
- Check-ins: **Oct 14** and **Nov 18** (progress monitoring in class)
- Oral presentation: **Dec 9** (50% of final grade, may be individualised per student)

## Scope decision
Updated after check-in 1 (Oct 14): start from a real spectrum, fit and remove the continuum, then detect, identify and characterise the lines. Absorption lines first, on the SDSS spectrum of the A0 star SDSS J172637.24+264127.6; identification by a "comb" per species (fixed wavelength ratios, Rydberg formula for H) checked by line-strength ratios; curve of growth for species with several lines. Keep detection generic (sign option) so emission spectra such as planetary nebulae can be added later. Work split and the detect/identify interface: `Work_split.tex` and `spectral_diag/detect.py`.

## Stack and layout
Python (numpy, scipy, astropy, astroquery, matplotlib; optional lmfit/emcee). Virtualenv in `.venv/` (ignored by git). Package in `spectral_diag/`, notebooks in `notebooks/`, data in `data/` (small files only).

Planned modules in `spectral_diag/`:
- `io` – load spectra (FITS/ASCII), archive queries, synthetic spectra
- `continuum` – iterative sigma-clipped spline/polynomial fit, normalisation
- `linelist` – atomic data (λ, f, Γ) for common ISM/CGM lines
- `detect` – feature detection and line identification via redshift matching
- `ew` – equivalent widths with error propagation
- `profiles` – Voigt profile (`scipy.special.wofz`), optical depth τ(λ; N, b, z), LSF
- `cog` – theoretical COG and multi-line fit for N and b
- `uncertainty` – χ² grid / MCMC / bootstrap
- `viz` – interactive spectrum and fit visualisation

Shared data structures (`Spectrum`, `Line`, `Absorber`) are agreed on early so the two students can work in parallel.

## Conventions
- Validate everything on synthetic spectra with known (z, N, b) before real data; recover injected parameters across linear, saturated and damped regimes.
- Sanity checks for COG: linear regime W ∝ N, damped regime W ∝ √N.
- Units: wavelengths in Å (state explicitly), velocities in km/s, N in cm^-2 (log10 in outputs). Use astropy units where practical.
- Do not commit large data files or the teachers' PDFs; do not commit or push unless asked.
- Course material (lectures, tutorials) lives one level up in `../`; theory reference for curves of growth is in the Lecture 1 summaries there.
