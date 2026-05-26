# Code Review: gwcs_raster branch

Branch: `gwcs_raster` vs `main`  
Date: 2026-05-26  
Reviewer: Claude (3-angle + verifier, recall-biased)

---

## Confirmed findings

### [BUG-1] crval lat/lon column swap in raster branch
**File:** `irispy/io/spectrograph.py:199`  
**Severity:** HIGH — all helioprojective coordinates silently wrong for every raster observation

```python
xcen = aux.data[:, aux.header["XCENIX"]] - aux.data[:, offset_index] * (SLIT_WIDTH.value / 2)
ycen = aux.data[:, aux.header["YCENIX"]]
crval = np.column_stack((ycen, xcen)) * u.arcsec   # BUG: (lat, lon)
```

`XCENIX` = Tx = helioprojective **lon** (CTYPE3=HPLN-TAN).  
`YCENIX` = Ty = helioprojective **lat** (CTYPE2=HPLT-TAN).  
`np.column_stack((ycen, xcen))` puts `lat` in column-0 and `lon` in column-1.  
`VaryingCelestialTransform` passes `crval_table[..., 0]` as the longitude argument to `RotateNative2Celestial`.  

Sit-and-stare branch (line 189) correctly uses `[CRVAL3, CRVAL2]` = `[lon, lat]`.  
The `SwapHelioprojectiveAxes` mapping (line 473) swaps **outputs**, not the internal crval reference — it does not compensate.

**Fix:** `crval = np.column_stack((xcen, ycen)) * u.arcsec`

---

### [BUG-2] Disk-center observations trigger false "bad row" mask
**File:** `irispy/_spectrograph_wcs.py:288`  
**Severity:** HIGH — valid disk-center rasters fail to read

```python
crval_bad = np.isclose(crval_values, 0).all(axis=1)
```

Any raster step where XCENIX=0 and YCENIX=0 arcsec (valid disk-center pointing) produces `crval = [0.0, 0.0]`.  
`np.isclose([0, 0], 0).all()` → `True` → row flagged bad → interpolated from neighbours or raises.

**Fix:** Use a more specific bad-row criterion (e.g., check for sentinel fill values, or require BOTH pc AND crval to be all-zero simultaneously to be certain it's a fill rather than a genuine pointing).

---

### [BUG-3] First file with all-zero AUX rows raises immediately with no recovery
**File:** `irispy/_spectrograph_wcs.py:348`, triggered from `irispy/io/spectrograph.py:203-219`  
**Severity:** HIGH — any observation whose first file has zeroed AUX data is unreadable

When the first file is processed, `running_wcs_fallbacks[window_name][2] == 0`, so `fallback_pc = None`, `fallback_crval = None`.  
If all rows in that file are bad (e.g., triggered by BUG-2 for disk-center), `_sanitize_raster_wcs_tables` reaches the `good_indices.size == 0` branch and raises:
```
ValueError: All WCS table rows are bad and no fallback values are available.
```
Old code emitted `logger.warning` and continued (graceful degradation). New code raises unconditionally.

BUG-2 and BUG-3 interact: a single-file disk-center raster will always trigger this path.

**Fix:** Either fix BUG-2 (so disk-center isn't misclassified), or degrade gracefully instead of raising when the primary wcs header is still valid (i.e., fall back to the prepared_wcs_header CRVAL values).

---

## Plausible (unconfirmed) findings

### [PLAUSIBLE-1] `.wcs` used for matplotlib projection on gWCS-backed cube
**File:** `examples/calibration/01_remove_spikes.py` (raster section)

```python
axes = [
    fig.add_subplot(1, 2, 1, projection=raster_2796.wcs),
    fig.add_subplot(1, 2, 2, projection=raster_2796_cleaned.wcs),
]
```

`raster_2796 = raster["Mg II k 2796"].raster_slice(10)[4]` — the `.wcs` is a gWCS object.  
WCSAxes accepts APE14-compliant WCS objects (gwcs implements this), but compatibility depends on the gwcs version and whether the sliced gWCS exposes a valid `pixel_shape`. Failure mode: `TypeError` or incorrect axis rendering at runtime.  

Other examples (01_spectral_fitting, 04_spectral_moments, etc.) were updated to use `.basic_wcs` for WCS-dependent operations. This example was not.

**Fix:** Use `projection=raster_2796.basic_wcs` (if it's 2D after slicing), or plot without a WCS projection.

---

### [PLAUSIBLE-2] `uncertainty=True, memmap=True` silently suppresses uncertainty with no warning
**File:** `irispy/io/spectrograph.py:96`

```python
compute_uncertainty = uncertainty and not memmap
```

Callers that pass `uncertainty=True, memmap=True` now receive `uncertainty=None` on every cube with no diagnostic.  
Old code computed uncertainty on unscaled memmap data (incorrect, but at least returned something).

**Fix:** Add `warnings.warn("uncertainty is not computed when memmap=True", UserWarning)` when both are True.

---

## Refuted (false positives)

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| `examples/calibration/02_radiometric_calibration.py:56` `[0]` indexing | REFUTED | Intentional — slices axis-0 to get "first raster step only" as comment says |
| `_raster_combine.py:165` `times[0, 0]` 2D indexing | REFUTED | `cube.time.jd` is always 1-D even for 1-step cubes; list stacks to 2-D |
| `_separate_raster_axis=True` conflict | REFUTED | `raster_slice(index)` correctly returns one full raster scan `(n_steps, slit, wave)` |
| `01_remove_spikes.py` `remove_cosmic_rays` on 3D SJI | REFUTED | `astroscrappy` path iterates `np.ndindex(shape[:-2])` — handles 3D correctly |
| Sit-and-stare flip double-sign-flip | REFUTED | Mirrors old code exactly — not a regression |
| `not spectral_windows` truthy for `[]` | LOW — design nit, not a crash path in practice |
| `aligned_axes` from first window only | LOW — windows always processed identically in current code |

---

## Low severity / design nits

- **`irispy/io/spectrograph.py:104`** — `not spectral_windows` is `True` for `[]`. Consider `if spectral_windows is None:` to distinguish "all" from "empty" intentionally.
- **`irispy/io/_raster_combine.py:109`** — `NAXIS3` only updated in `fits_header` when it already exists; `NAXIS4` always written. Asymmetric guard.
- **`irispy/io/spectrograph.py:162`** — `window_fits_indices` from `filenames[0]` used unvalidated for all subsequent files. No per-file NWIN/TDESC cross-check. (Carried over from old code.)

---

## Summary

| ID | File | Line | Severity | Status |
|----|------|------|----------|--------|
| BUG-1 | `irispy/io/spectrograph.py` | 199 | HIGH | CONFIRMED |
| BUG-2 | `irispy/_spectrograph_wcs.py` | 288 | HIGH | CONFIRMED |
| BUG-3 | `irispy/_spectrograph_wcs.py` | 348 | HIGH | CONFIRMED (depends on BUG-2) |
| PLAUSIBLE-1 | `examples/calibration/01_remove_spikes.py` | ~90 | MEDIUM | PLAUSIBLE |
| PLAUSIBLE-2 | `irispy/io/spectrograph.py` | 96 | LOW | PLAUSIBLE |
