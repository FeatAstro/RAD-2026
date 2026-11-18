# To do — Gaël

Last updated 18 November 2026.

## Now (git)

- [x] Push `gael` (`git push`) and open the PR into `main`
      ("Simplify continuum and curve-of-growth internals, comment InstrumentGrid").
- [x] After Pablo approves and merges: pull `main` into `gael` (and Pablo into `pablo`).
- [x] Push the updated `gael` (`git push origin gael`).
- [x] Merge `main` (Pablo's PR #9: full-star tutorial, plots, features table) into
      `gael`; `Work_split.tex/.pdf` conflicts resolved with main's version (17 Nov).

## Before check-in 2 (18 November)

- [ ] **Line list for the star** (`line_list`):
  - [x] NIST only (Verner table removed): `download_line_table` saves each species in
        `data/line_lists/` (committed), `get_lines` reads it offline;
  - [x] hydrogen: `LineSelection(multiplet_totals=True)`, one line per n -> n', checked
        against the Rydberg formula (offsets 2–3 km/s);
  - [ ] table of A0 star lines: Ca II K and H, Mg II 4481, O I 7774 triplet, Na I D,
        Ca II triplet near 8500, a few Fe II;
  - [ ] telluric O2 A-band (7594–7621 Å) as one more "species".
- [ ] **`detect/identify.py`, simple version**:
  1. identify the hydrogen lines;
  2. take their mean velocity (expect about −37 km/s);
  3. match the other features to the nearest predicted line within ~3 × `wave_err`.

  Adds the columns `species`, `line`, `wave_rest`, `v`.
- [x] **Continuum error in `measure_ew`**: `ContinuumFit.error` propagated (fully
      correlated over the window), via `Spectrum.continuum_error` into the features' `ew_err`.
- [ ] Optionally a systematic from refitting with knot spacings of 200 and 300 Å: the
      spread between spacings (0.4–1.0 Å on the Balmer lines) is 2–4× the statistical
      continuum error (~0.3 Å).
- [ ] **Validation**: a pytest with a hand-written features table of the star, and a
      notebook showing the identified spectrum (the check-in demo).

## After the check-in (18 November – 8 December)

- [ ] Comb scan identification at an unknown velocity / z (−600…+600 km/s, scored by
      expected strength).
- [ ] Curve of growth on 3C 273 (HST/COS G130M, `lbgl31020`; HSLA co-add has ~1.8x the
      S/N): S II 1250/1253/1259 (linear part) + Si II 1190/1193/1260/1304 (saturated).
      Waits for Pablo's COS reader; continuum fitted locally around each line. (The
      single-slab model does not hold for the star's Balmer lines.)
- [ ] Uncertainties (`uncertainty.py`: bootstrap or χ² grid on N, b).
- [ ] One clear plot of the spectrum with labelled lines (`plots.py`; Pablo's
      `plot_features` already marks the detected features, add the line labels).
- [ ] Slides and rehearsal from about 1 December; **oral on 9 December**.

## To discuss with Pablo

- [ ] The SDSS error bars look about 20% too large (continuum χ²_ν = 0.60), so the 4σ
      detection threshold is really about 5σ: keep it, or rescale the errors?
- [x] `feature_windows` uses `ndimage.label`: Pablo replaced it with a plain loop.
- [x] Tell him: `_kernel_weighted_sum` is back to `np.convolve` (his double loop was
      ~1300x slower, 19 ms vs 0.014 ms), with a comment saying why. Commit on `gael`;
      he matched it on `pablo` in PR #9.
- [ ] COS reader: tell him the HSLA co-add (`hst_cos_3c273--0392_g130m-lp01_cspec.fits`)
      has the same columns as an x1dsum.
- [ ] Split features (`split_blends`): agree on the `blended` column he proposes, so
      `identify` knows which EWs need a joint fit.
