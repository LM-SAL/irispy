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
  for multi-file rasters, backed by a dask array with one file-sized chunk per raster file.
- Combined cubes now expose `raster_boundaries`, `raster_slice(i)`, and `split_rasters()`.
- `SpectrogramCubeSequence.as_cube()` exists as explicit transition helper.
- `SpectrogramCube.spectrum_at()` now works even when `basic_wcs is None` by
  falling back to gWCS sky matching.
- Focused tests and key examples/docs were updated to the one-cube default.

## Still Missing

- broader doc/example cleanup after full transition
- more granular lazy reads than one full file per dask chunk
- policy for uncertainty/mask behavior on lazy cubes beyond the current first pass
- decision on whether public `SpectrogramCubeSequence` becomes deprecated or internal-only

## Validation Target

- `crop()` still works
- `crop_by_values()` still works
- `spectrum_at()` still works
- one-cube default for `memmap=False`
- one-cube default for `memmap=True`
- single-raster slices preserve `basic_wcs`
- full merged cubes report `basic_wcs is None`
