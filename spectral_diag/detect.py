"""Feature detection and line identification.

Planned contents:
- find_features: locate candidate absorption dips in a normalised
  Spectrum
- identify: match candidate features to linelist entries via
  redshift matching, producing a line table (species, wave_rest, f,
  z_guess, window) per absorber
"""
