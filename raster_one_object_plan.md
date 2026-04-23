# Raster One-Object UX Plan

Goal: make raster observations feel like `SJICube` by exposing one `SpectrogramCube`
per spectral window as the default public object.

## Public Model

- Multi-file rasters should concatenate along the existing scan axis.
- The public gWCS should expose:
  - wavelength
  - helioprojective longitude
  - helioprojective latitude
  - time
  - global `scan_step`
- Original file grouping should stay out of the gWCS for now.
- Original raster grouping should be exposed on the cube via:
  - `raster_boundaries`
  - `raster_slice(i)`
  - `split_rasters()`

## `basic_wcs`

- Do not invent one fake FITS WCS for a merged multi-file raster.
- Full merged cubes should report `basic_wcs is None`.
- Slices that stay inside one original raster should recover a usable sliced `basic_wcs`.

## `memmap`

- `memmap=False`: eager one-cube path.
- `memmap=True`: lazy one-cube path for multi-file rasters.
- Long-term backend should be:
  - FITS memmap for per-file storage access
  - dask array for the logical merged cube

## Landed On `raster-one-object-ux`

- `read_spectrograph_lvl2(..., memmap=False)` now returns one `SpectrogramCube`
  per raster spectral window, including multi-file rasters.
- Multi-file rasters are merged eagerly along the scan axis into one 3D cube.
- `read_spectrograph_lvl2(..., memmap=True)` now also returns one `SpectrogramCube`
  for multi-file rasters, backed by a dask array of scan-axis chunks sourced from
  file-backed FITS memmaps.
- Combined cubes now expose `raster_boundaries`, `raster_slice(i)`, and `split_rasters()`.
- `SpectrogramCubeSequence` has been removed; per-raster access now goes through
  `split_rasters()` and `raster_slice(i)`.
- `SpectrogramCube.spectrum_at()` now works even when `basic_wcs is None` by
  falling back to gWCS sky matching.
- `memmap=True` now consistently skips uncertainty computation, and lazy combined
  cubes keep a lazy mask while exposing per-raster subcubes through the combined
  cube API.
- Lazy memmap reads now split each raster file into smaller scan-axis dask
  chunks instead of reading one full-file chunk at a time, so interactive
  slices and spectra no longer pull an entire file by default.
- Source docs/examples were swept to the one-cube default, and the explicit
  `SpectrogramCubeSequence` API pages were removed.
- Validation so far:
  - focused raster/user tests updated and passing
  - `pytest --pyargs irispy -q` passed locally after the one-cube and lazy-policy work

## Still Missing

- no major one-object UX items remain on this branch; remaining work is review and
  any follow-on refinements discovered during PR feedback

## Validation Target

- `crop()` still works
- `crop_by_values()` still works
- `spectrum_at()` still works
- one-cube default for `memmap=False`
- one-cube default for `memmap=True`
- single-raster slices preserve `basic_wcs`
- full merged cubes report `basic_wcs is None`
