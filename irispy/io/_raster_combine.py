from copy import deepcopy

import dask.array as da
import numpy as np
from dask import delayed

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from irispy.spectrograph import SpectrogramCube
from irispy.utils.constants import BAD_PIXEL_VALUE_UNSCALED

LAZY_RASTER_CHUNK_TARGET_BYTES = 4 * 1024 * 1024


def _concatenate_scan_aligned_values(values):
    first = values[0]
    if isinstance(first, SkyCoord):
        return SkyCoord(np.concatenate(values))
    return np.concatenate(values)


def _stack_scan_aligned_values(values):
    first = values[0]
    if isinstance(first, SkyCoord):
        return SkyCoord(
            Tx=np.stack([value.Tx.to_value(u.arcsec) for value in values]) * u.arcsec,
            Ty=np.stack([value.Ty.to_value(u.arcsec) for value in values]) * u.arcsec,
            frame=first.frame,
        )
    if isinstance(first, u.Quantity):
        return np.stack([value.to_value(first.unit) for value in values]) * first.unit
    return np.stack(values)


def _concatenate_uncertainty(cubes):
    if all(cube.uncertainty is None for cube in cubes):
        return None
    if any(cube.uncertainty is None for cube in cubes):
        msg = "Cannot combine a raster sequence when only some cubes contain uncertainty."
        raise ValueError(msg)
    uncertainty_type = type(cubes[0].uncertainty)
    return uncertainty_type(np.concatenate([cube.uncertainty.array for cube in cubes], axis=0))


def _stack_uncertainty(cubes):
    if all(cube.uncertainty is None for cube in cubes):
        return None
    if any(cube.uncertainty is None for cube in cubes):
        msg = "Cannot combine a raster sequence when only some cubes contain uncertainty."
        raise ValueError(msg)
    uncertainty_type = type(cubes[0].uncertainty)
    return uncertainty_type(np.stack([cube.uncertainty.array for cube in cubes], axis=0))


def _concatenate_mask(cubes):
    if all(cube.mask is None for cube in cubes):
        return None
    return np.concatenate(
        [
            np.zeros(cube.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool)
            for cube in cubes
        ],
        axis=0,
    )


def _stack_mask(cubes):
    if all(cube.mask is None for cube in cubes):
        return None
    return np.stack(
        [
            np.zeros(cube.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool)
            for cube in cubes
        ],
        axis=0,
    )


def _combine_raster_meta(cubes, combined_shape):
    meta = deepcopy(cubes[0].meta)
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    meta["NAXIS3"] = combined_shape[0]
    if "NAXIS3" in meta.fits_header:
        meta.fits_header["NAXIS3"] = combined_shape[0]
    for key in ("DATE_END", "ENDOBS"):
        if cubes[-1].meta.get(key) is not None:
            meta[key] = cubes[-1].meta[key]
            if key in meta.fits_header:
                meta.fits_header[key] = cubes[-1].meta[key]
    for key in ("exposure time", "exposure FOV center", "observer radial velocity", "orbital phase"):
        meta.add(
            key,
            _concatenate_scan_aligned_values([cube.meta[key] for cube in cubes]),
            axes=0,
            overwrite=True,
        )
    return meta


def _combine_raster_sequence_meta(cubes, combined_shape):
    meta = deepcopy(cubes[0].meta)
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    meta["NAXIS4"] = combined_shape[0]
    meta["NAXIS3"] = combined_shape[1]
    if "NAXIS3" in meta.fits_header:
        meta.fits_header["NAXIS3"] = combined_shape[1]
    meta.fits_header["NAXIS4"] = combined_shape[0]
    for key in ("DATE_END", "ENDOBS"):
        if cubes[-1].meta.get(key) is not None:
            meta[key] = cubes[-1].meta[key]
            if key in meta.fits_header:
                meta.fits_header[key] = cubes[-1].meta[key]
    for key in ("exposure time", "exposure FOV center", "observer radial velocity", "orbital phase"):
        meta.add(
            key,
            _stack_scan_aligned_values([cube.meta[key] for cube in cubes]),
            axes=(0, 1),
            overwrite=True,
        )
    return meta


def _validate_combinable_raster_cubes(cubes):
    cubes = tuple(cubes)
    if not cubes:
        msg = "Cannot combine an empty raster cube list."
        raise ValueError(msg)
    if len(cubes) == 1:
        return cubes
    if any(cube.meta.get("OBSID") != cubes[0].meta.get("OBSID") for cube in cubes[1:]):
        msg = "All raster cubes must have the same OBSID."
        raise ValueError(msg)
    start_obs = cubes[0].meta.get("STARTOBS")
    if start_obs is not None and any(cube.meta.get("STARTOBS") != start_obs for cube in cubes[1:]):
        msg = "All raster cubes must have the same STARTOBS."
        raise ValueError(msg)
    if any(cube.shape != cubes[0].shape for cube in cubes[1:]):
        msg = "All raster cubes must have the same step, slit, and wavelength dimensions."
        raise ValueError(msg)
    if any(cube.unit != cubes[0].unit for cube in cubes[1:]):
        msg = "All raster cubes must have the same data unit."
        raise ValueError(msg)
    required_attrs = (
        "_raster_wcs_header",
        "_raster_pc_table",
        "_raster_crval_table",
        "_raster_observer",
    )
    if any(not all(hasattr(cube, attr) for attr in required_attrs) for cube in cubes):
        msg = "Raster cubes do not expose the WCS metadata needed to build a combined cube."
        raise ValueError(msg)
    # Ensure all cubes share the same observer; combined gWCS uses cubes[0].
    first_observer = cubes[0]._raster_observer
    for cube in cubes[1:]:
        if not np.allclose(cube._raster_observer.cartesian.xyz.value, first_observer.cartesian.xyz.value, atol=1e-6):
            msg = "All raster cubes must have the same observer coordinate."
            raise ValueError(msg)
    return cubes


def _build_combined_raster_cube(cubes, data, *, mask, memmap, create_raster_gwcs):
    times = Time([cube.time.jd for cube in cubes], format="jd", scale="utc")
    pc_all = np.stack([cube._raster_pc_table for cube in cubes], axis=0)
    crval_all = np.stack([cube._raster_crval_table for cube in cubes], axis=0)
    return SpectrogramCube(
        data,
        wcs=create_raster_gwcs(
            cubes[0]._raster_wcs_header,
            pc_all,
            crval_all,
            (times - times[0, 0]).to_value(u.s) * u.s,
            times[0, 0],
            cubes[0]._raster_observer,
            sit_and_stare=cubes[0]._sit_and_stare,
        ),
        uncertainty=_stack_uncertainty(cubes),
        unit=cubes[0].unit,
        meta=_combine_raster_sequence_meta(cubes, data.shape),
        mask=mask,
        _basic_wcs_segments=[(index, index + 1, cube.basic_wcs) for index, cube in enumerate(cubes)],
        _raster_boundaries=[(index, index + 1) for index in range(len(cubes))],
        _memmap=memmap,
        _raster_wcs_header=cubes[0]._raster_wcs_header,
        _raster_pc_table=pc_all,
        _raster_crval_table=crval_all,
        _raster_observer=cubes[0]._raster_observer,
        _separate_raster_axis=True,
        _sit_and_stare=cubes[0]._sit_and_stare,
    )


def _lazy_raster_scan_chunk_rows(cube):
    row_bytes = int(np.prod(cube.shape[1:]) * np.dtype(cube.data.dtype).itemsize)
    return max(1, min(cube.shape[0], LAZY_RASTER_CHUNK_TARGET_BYTES // max(row_bytes, 1)))


def _read_memmap_window_chunk(filename, ext, flip, start, stop):
    """
    Read one scan-axis chunk from disk after the public reader has returned.
    """
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as hdulist:
        data = hdulist[ext].data
        if flip:
            original_start = data.shape[0] - stop
            original_stop = data.shape[0] - start
            data = data[original_start:original_stop][::-1]
        else:
            data = data[start:stop]
        return np.array(data, copy=True)


def _cube_to_dask(cube, *, chunk_rows):
    """
    Return a Dask array for one cube, reading from disk if memmap-backed.
    """
    filename = getattr(cube, "_memmap_path", None)
    ext = getattr(cube, "_memmap_ext", None)
    if filename is None or ext is None:
        return da.from_array(
            cube.data,
            chunks=(chunk_rows, *cube.shape[1:]),
            fancy=False,
            asarray=False,
        )

    raster_chunks = []
    flip = getattr(cube, "_flip", False)
    for start in range(0, cube.shape[0], chunk_rows):
        stop = min(start + chunk_rows, cube.shape[0])
        chunk = delayed(_read_memmap_window_chunk)(filename, ext, flip, start, stop)
        raster_chunks.append(da.from_delayed(chunk, shape=(stop - start, *cube.shape[1:]), dtype=cube.data.dtype))
    return da.concatenate(raster_chunks, axis=0)


def _build_lazy_raster_data(cubes):
    dask_chunks = [_cube_to_dask(cube, chunk_rows=_lazy_raster_scan_chunk_rows(cube)) for cube in cubes]
    return da.stack(dask_chunks, axis=0)


def _combine_raster_cubes_lazy(cubes, create_raster_gwcs):
    cubes = _validate_combinable_raster_cubes(cubes)
    if len(cubes) == 1:
        return cubes[0]
    data = _build_lazy_raster_data(cubes)
    mask = data == BAD_PIXEL_VALUE_UNSCALED
    return _build_combined_raster_cube(cubes, data, mask=mask, memmap=True, create_raster_gwcs=create_raster_gwcs)


def _combine_raster_cubes(cubes, create_raster_gwcs):
    cubes = _validate_combinable_raster_cubes(cubes)
    if len(cubes) == 1:
        return cubes[0]
    if any(getattr(cube, "_memmap", False) or isinstance(cube.data, np.memmap) for cube in cubes):
        msg = "Memmap-backed raster cubes must be combined via the lazy reader (memmap=True)."
        raise NotImplementedError(msg)
    data = np.stack([cube.data for cube in cubes], axis=0)
    return _build_combined_raster_cube(
        cubes, data, mask=_stack_mask(cubes), memmap=False, create_raster_gwcs=create_raster_gwcs
    )


def _finalize_window_object(cubes, *, memmap, create_raster_gwcs):
    if len(cubes) == 1:
        cube = cubes[0]
        cube._raster_boundaries = [(0, cube.shape[0])]
        return cube
    if memmap:
        return _combine_raster_cubes_lazy(cubes, create_raster_gwcs)
    return _combine_raster_cubes(cubes, create_raster_gwcs)
