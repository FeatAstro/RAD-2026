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
Main target: UV/optical absorption spectra of quasar absorbers (Voigt profiles, HI + metal lines). Keep the code generic enough to demo a second domain (e.g. 21 cm or stellar) at the end.

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
- Formatting: black, line length 88 (configured in `pyproject.toml`).
