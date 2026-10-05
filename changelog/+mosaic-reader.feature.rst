Add `irispy.io.read_mosaic`, a reader for the IRIS full-disk mosaics (``IRISMosaic_<date>_<window>.fits.gz``), which returns a `~irispy.spectrograph.MosaicCube` with a helioprojective WCS, the time of each mosaic position in ``meta["time"]`` and the positions no raster covered masked; ``wavelength_range`` reads only the wavelengths needed.
`irispy.spectrograph.MosaicCube.to_maps` returns a `sunpy.map.Map` of the mosaic at one wavelength or averaged over a range.
Slicing an `irispy.spectrograph.SpectrogramCube` subclass now keeps the subclass.
