"""Load spectra from files or archives, and generate synthetic spectra.

Planned contents:
- read_fits / read_ascii: load an observed Spectrum from disk
- query_archive: fetch a spectrum from an online archive (astroquery)
- make_synthetic: build a Spectrum with injected absorbers at known
  (z, logN, b), for validating the rest of the pipeline
"""
