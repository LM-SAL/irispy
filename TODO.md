# Review TODOs

- Fix `_raster_wcs_bad_row_mask()` in `irispy/_spectrograph_wcs.py` so valid helioprojective disk-center coordinates (`crval == [0, 0]`) are not treated as bad WCS rows. Use an all-zero PC table or another stronger invalid-row signal instead of zero CRVAL alone.
- Fix V34 flipped reads in `irispy/io/spectrograph.py` so `data_mask` and computed uncertainty are flipped along the scan axis together with `data`.
