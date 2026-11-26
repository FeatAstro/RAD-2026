# To do — Gaël

Last updated 26 November 2026.

Scope since 18 November: the quasar 3C 273 (HST/COS G130M) is the main target:
identify, comb, EWs and curve of growth. The A0 star stays as done up to detection;
its rotation (v sin i) is an "if time" extra. Separating unresolved blends is not
needed (teacher). Oral on **9 December**.

## Now (git)

- [x] PR #10 (`gael`: NIST line lists, continuum error in EWs) merged into `main`.
- [x] Pablo's PR #11 (`spectrum_reader` with `read_cos_x1dsum`, quasar tutorial) merged;
      `main` pulled into `gael` (18 Nov).
- [x] Pablo's 3 commits (`clip_high`, `read_hsla`, quasar continuum and detection) merged
      into `gael` (22 Nov); they go to `main` with the next `gael` PR.
- [x] CLAUDE.md, `Work_split.tex`, README updated to the quasar focus (22 Nov).
- [ ] Push `gael` and open the PR into `main`; then pull `main` into both branches.

## Quasar 3C 273 (18 November – 8 December)

1. [x] **Continuum** (Pablo): geocoronal Lyα cut (1214–1217 Å), spline with 2 Å knots;
       systematic error from knot spacings ±20 % (`continuum_systematic_error`, Gaël,
       ~1.4× the statistical error).
2. [x] **Detection** (Pablo): HSLA COS co-add, threshold 12σ set from an emission
       (noise-only) search; 44 features.
3. [x] **`detect/identify.py`: comb identification** (Gaël): validated on 30 synthetic
       COS spectra (`07_identification.ipynb`); on 3C 273 one system, the Milky Way at
       −12.7 km/s, 19 of 44 features labelled (quasar tutorial, section 4).
       Not done: Virgo Lyα and the forest (one H I line in range, no comb possible);
       blends not flagged (Si II / Fe II 1260, Si IV 1393.8).
4. [x] **EWs and curve of growth** (quasar tutorial, section 5): S II log N = 15.33
       ± 0.04, b = 21.8 km/s; Si II 15.08 ± 0.06, b = 31.4 km/s (χ² 31 / 2 dof).
       Sembach et al. 2001 (astro-ph/0108047, Table 5): 15.46 and 15.10. S II is at
       the knee of the COG, not linear. Si II is not robust: with 3 lines (no 1190) the
       fit gives 16.14, b = 20.7 km/s.
       - [ ] Check the Table 5 values (Si II, b) in the PDF before the slides.
       - [ ] Convert our heliocentric velocities to the LSR before comparing velocities.
5. [x] **Uncertainties** (26 Nov): χ² map in (log N, b) (`uncertainty.py`, validated in
       `08_chi2_map.ipynb`), rescaled by χ²_min/dof (5.0 and 15.7). S II log N = 15.33
       +0.10/−0.09; Si II has two minima (15.08 and ≈ 16.2), 14.8–16.5 at 95 %.
6. [ ] **Plots and slides** from about 1 December: spectrum with labelled lines,
       curve of growth with measured points and fit; rehearsal.

## If time

- [ ] C I 1277.3 and 1328.9 sit ~+20 km/s from the system (a second cloud group?); the
      C I 1276.5 label is doubtful (C I 1277.2, 25× stronger, not matched).
- [ ] Mark leftover features between 1216 and 1408 Å as probable Lyα.

- [ ] Star: v sin i from the width of Mg II 4481 (measurable only if ≳ 100 km/s at
      SDSS resolution, else an upper limit).
- [ ] Comb on a synthetic spectrum at unknown z (larger z range).
- [x] Continuum systematic: refit with knot spacings ±20 % (`continuum_systematic_error`;
      on 3C 273 ~1.4× the statistical error; on the star, the spread
      between spacings was 2–4× the statistical continuum error).

## Done (star, up to 18 November)

- [x] Star tutorial run to the end (26 Nov): `identify` with `CombSettings.e_low_max = 11`
      (Balmer, n = 2) and the strength check off (it rejects the Balmer series: Halpha has
      the smallest W/lambda); system at -31 km/s (SDSS: -37). Balmer COG: log N = 16.7,
      b = 103 km/s, chi2 2378 / 5: not a column density nor a Doppler parameter (Stark
      wings, photosphere is not a slab). Blends not separated, stated as a limit.

- [x] Line lists: NIST only (Verner removed), saved in `data/line_lists/`; hydrogen
      `multiplet_totals`, checked against the Rydberg formula (offsets 2–3 km/s).
- [x] Continuum error propagated into `measure_ew` and the features' `ew_err`.
- [x] Star: reading, continuum, detection (Pablo), full-star tutorial.

## To discuss with Pablo

- [ ] The SDSS error bars look about 20% too large (continuum χ²_ν = 0.60): rescale?
      (Less important now that the star is not the main target.)
- [ ] Split features (`split_blends`): agree on the `blended` column, so `identify`
      knows which EWs need a joint fit.
- [x] COS reader done by Pablo (`spectrum_reader.read_cos_x1dsum`). Tell him the HSLA
      co-add (`hst_cos_3c273--0392_g130m-lp01_cspec.fits`) has the same columns.
