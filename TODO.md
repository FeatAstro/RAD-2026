# To do — Gaël

Last updated 18 November 2026.

Scope since 18 November: the quasar 3C 273 (HST/COS G130M) is the main target:
identify, comb, EWs and curve of growth. The A0 star stays as done up to detection;
its rotation (v sin i) is an "if time" extra. Separating unresolved blends is not
needed (teacher). Oral on **9 December**.

## Now (git)

- [x] PR #10 (`gael`: NIST line lists, continuum error in EWs) merged into `main`.
- [x] Pablo's PR #11 (`spectrum_reader` with `read_cos_x1dsum`, quasar tutorial) merged;
      `main` pulled into `gael` (18 Nov).
- [ ] Push the updated `gael` (`git push origin gael`).
- [ ] With Pablo's agreement: update CLAUDE.md (scope), `Work_split.tex` (timeline still
      says "curve of growth for CaII / Balmer") to the quasar focus.
- [ ] README line 217 still says `sdss_reader.read_sdss_spec` (now `spectrum_reader`).

## Quasar 3C 273 (18 November – 8 December)

1. [ ] **Continuum** (Pablo): the 250 Å spline cannot follow the Milky Way Lyα trough
       (~1216 Å) nor the quasar Lyα emission (~1408 Å). With 5–10 Å knots the fit does
       not converge and `find_features` returns ~160–200 features at 5σ, many spurious.
       Fit locally (±10–20 Å around each line) or close knots + masked regions.
2. [ ] **Detection** (Pablo): `find_features` on 3C 273 (`meta["fwhm_kms"]` ≈ 16 km/s
       is set by the reader); check by eye that S II, Si II, N I, Si IV are separate.
3. [ ] **`detect/identify.py`: comb identification** (Gaël):
   - [ ] comb per species from `get_lines` (strong lines, inside the data coverage);
   - [ ] anchor velocities (each feature × each tooth), score = matched teeth weighted
         by expected strength, minus strong teeth missing; strength-ratio check;
   - [ ] best system velocity → refine (weighted mean of matched teeth) → apply it to
         every species, single-line ones included (C II 1334, Si III 1206, Lyα);
   - [ ] remove matched features, repeat for the next system (Virgo Lyα at
         cz ≈ 1000–1600 km/s);
   - [ ] adds `species`, `line`, `wave_rest`, `v`; known blends (Si II / Fe II 1260)
         flagged, not separated;
   - [ ] validation: pytest on synthetic spectra (known v, extra spurious features),
         notebook on 3C 273.
4. [x] **EWs and curve of growth** (quasar tutorial, section 5): S II log N = 15.33
       ± 0.04, b = 21.8 km/s; Si II 15.08 ± 0.06, b = 31.4 km/s (χ² 31 / 2 dof).
       Sembach et al. 2001 (astro-ph/0108047, Table 5): 15.46 and 15.10. S II is at
       the knee of the COG, not linear. Check Table 5 values in the PDF for slides.
5. [ ] **Uncertainties**: χ² map in (log N, b) (`uncertainty.py`); Si II is saturated, so
       the curvature error alone is misleading.
6. [ ] **Plots and slides** from about 1 December: spectrum with labelled lines,
       curve of growth with measured points and fit; rehearsal.

## If time

- [ ] Star: v sin i from the width of Mg II 4481 (measurable only if ≳ 100 km/s at
      SDSS resolution, else an upper limit).
- [ ] Comb on a synthetic spectrum at unknown z (larger z range).
- [x] Continuum systematic: refit with knot spacings ±20 % (`continuum_systematic_error`;
      on 3C 273 ~1.4× the statistical error; on the star, the spread
      between spacings was 2–4× the statistical continuum error).

## Done (star, up to 18 November)

- [x] Line lists: NIST only (Verner removed), saved in `data/line_lists/`; hydrogen
      `multiplet_totals`, checked against the Rydberg formula (offsets 2–3 km/s).
- [x] Continuum error propagated into `measure_ew` and the features' `ew_err`.
- [x] Star: reading, continuum, detection (Pablo), full-star tutorial.

## To discuss with Pablo

- [ ] The quasar continuum (step 1 above): who, and local windows or global spline?
- [ ] The SDSS error bars look about 20% too large (continuum χ²_ν = 0.60): rescale?
      (Less important now that the star is not the main target.)
- [ ] Split features (`split_blends`): agree on the `blended` column, so `identify`
      knows which EWs need a joint fit.
- [x] COS reader done by Pablo (`spectrum_reader.read_cos_x1dsum`). Tell him the HSLA
      co-add (`hst_cos_3c273--0392_g130m-lp01_cspec.fits`) has the same columns.
