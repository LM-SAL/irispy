import warnings
from copy import deepcopy

import dask.array as da
import numpy as np
from dask import delayed

import astropy.units as u
from astropy.io import fits
from astropy.time import Time

from irispy._spectrograph_wcs import _SPECTROGRAM_CUBE_METADATA_KWARGS
from irispy.spectrograph import SpectrogramCube

LAZY_RASTER_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def _pad_step_aligned_quantity(value, target_steps):
    if value.shape[0] == target_steps:
        return value
    padding = np.repeat(value[-1:], target_steps - value.shape[0], axis=0)
    return np.concatenate([value, padding], axis=0)


def _stack_data(cubes, target_steps):
    ragged = any(cube.shape[0] != target_steps for cube in cubes)
    dtype = np.result_type(*[cube.data.dtype for cube in cubes], float if ragged else cubes[0].data.dtype)
    data = np.empty((len(cubes), target_steps, *cubes[0].shape[1:]), dtype=dtype)
    for index, cube in enumerate(cubes):
        data[index, : cube.shape[0]] = np.asarray(cube.data, dtype=dtype)
        if cube.shape[0] < target_steps:
            data[index, cube.shape[0] :] = np.nan
    return data


def _stack_step_aligned_arrays(arrays, target_steps, *, fill_value):
    ragged = any(array.shape[0] != target_steps for array in arrays)
    dtype_args = [array.dtype for array in arrays]
    if np.isnan(fill_value) and ragged:
        dtype_args.append(float)
    dtype = np.result_type(*dtype_args)
    stacked = np.empty((len(arrays), target_steps, *arrays[0].shape[1:]), dtype=dtype)
    for index, array in enumerate(arrays):
        stacked[index, : array.shape[0]] = np.asarray(array, dtype=dtype)
        if array.shape[0] < target_steps:
            stacked[index, array.shape[0] :] = fill_value
    return stacked


def _stack_uncertainty(cubes, target_steps):
    if all(cube.uncertainty is None for cube in cubes):
        return None
    if any(cube.uncertainty is None for cube in cubes):
        msg = "Cannot combine a raster sequence when only some cubes contain uncertainty."
        raise ValueError(msg)
    uncertainty_type = type(cubes[0].uncertainty)
    return uncertainty_type(
        _stack_step_aligned_arrays(
            [cube.uncertainty.array for cube in cubes],
            target_steps,
            fill_value=np.nan,
        )
    )


def _stack_mask(cubes, target_steps):
    if all(cube.mask is None for cube in cubes) and all(cube.shape[0] == target_steps for cube in cubes):
        return None
    return _stack_step_aligned_arrays(
        [
            np.zeros(cube.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool)
            for cube in cubes
        ],
        target_steps,
        fill_value=True,
    )


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
    if any(cube.shape[1:] != cubes[0].shape[1:] for cube in cubes[1:]):
        msg = "All raster cubes must have the same slit and wavelength dimensions."
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


def _target_step_count(cubes):
    return max(cube.shape[0] for cube in cubes)


def _warn_if_ragged(cubes, target_steps):
    if all(cube.shape[0] == target_steps for cube in cubes):
        return
    warnings.warn(
        "Raster sequence has mismatched step counts; padding shorter rasters with NaN data and masked pixels.",
        UserWarning,
        stacklevel=3,
    )


def _stack_times(cubes, target_steps):
    times = []
    for cube in cubes:
        time = cube.time
        if time.shape[0] == target_steps:
            times.append(time.jd)
            continue
        times.append(np.concatenate([time.jd, np.repeat(time.jd[-1], target_steps - time.shape[0])]))
    return Time(np.stack(times, axis=0), format="jd", scale="utc")


def _single_cube_raster_gwcs(cube):
    from irispy.io.spectrograph import _create_raster_gwcs  # NOQA: PLC0415

    time = cube.time
    return _create_raster_gwcs(
        deepcopy(cube._raster_wcs_header),
        cube._raster_pc_table,
        cube._raster_crval_table,
        (time - time[0]).to_value(u.s) * u.s,
        time[0],
        cube._raster_observer,
        sit_and_stare=cube._sit_and_stare,
    )


def _materialize_deferred_raster_gwcs(cube):
    if not getattr(cube, "_defer_raster_gwcs", False):
        return cube

    kwargs = {attr: getattr(cube, attr) for attr in _SPECTROGRAM_CUBE_METADATA_KWARGS if hasattr(cube, attr)}
    materialized = SpectrogramCube(
        cube.data,
        wcs=_single_cube_raster_gwcs(cube),
        uncertainty=cube.uncertainty,
        unit=cube.unit,
        meta=cube.meta,
        mask=cube.mask,
        **kwargs,
    )
    materialized.extra_coords.add("time", 0, cube.time, physical_types="time")
    return materialized


def _build_combined_raster_cube(cubes, data, *, mask, memmap):
    from irispy.io.spectrograph import _create_raster_gwcs  # NOQA: PLC0415

    target_steps = data.shape[1]
    _warn_if_ragged(cubes, target_steps)
    times = _stack_times(cubes, target_steps)
    pc_all = np.stack([_pad_step_aligned_quantity(cube._raster_pc_table, target_steps) for cube in cubes], axis=0)
    crval_all = np.stack([_pad_step_aligned_quantity(cube._raster_crval_table, target_steps) for cube in cubes], axis=0)
    raster_wcs_header = deepcopy(cubes[0]._raster_wcs_header)
    if "NAXIS3" in raster_wcs_header:
        raster_wcs_header["NAXIS3"] = target_steps
    return SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(
            raster_wcs_header,
            pc_all,
            crval_all,
            (times - times[0, 0]).to_value(u.s) * u.s,
            times[0, 0],
            cubes[0]._raster_observer,
            sit_and_stare=cubes[0]._sit_and_stare,
        ),
        uncertainty=_stack_uncertainty(cubes, target_steps),
        unit=cubes[0].unit,
        meta=cubes[0].meta.combine([cube.meta for cube in cubes], data.shape),
        mask=mask,
        _fits_wcs_segments=[(index, index + 1, cube.fits_wcs) for index, cube in enumerate(cubes)],
        _memmap=memmap,
        _raster_wcs_header=raster_wcs_header,
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


def _pad_dask_data(data, target_steps):
    if data.shape[0] == target_steps:
        return data
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    pad_shape = (target_steps - data.shape[0], *data.shape[1:])
    padding = da.full(pad_shape, np.nan, chunks=pad_shape, dtype=data.dtype)
    return da.concatenate([data, padding], axis=0)


def _build_lazy_raster_data(cubes, target_steps):
    dask_chunks = [
        _pad_dask_data(_cube_to_dask(cube, chunk_rows=_lazy_raster_scan_chunk_rows(cube)), target_steps)
        for cube in cubes
    ]
    return da.stack(dask_chunks, axis=0)


def _combine_raster_cubes(cubes, *, memmap=False):
    cubes = _validate_combinable_raster_cubes(cubes)
    if len(cubes) == 1:
        return cubes[0]
    target_steps = _target_step_count(cubes)
    if memmap:
        data = _build_lazy_raster_data(cubes, target_steps)
        return _build_combined_raster_cube(cubes, data, mask=None, memmap=True)

    data = _stack_data(cubes, target_steps)
    return _build_combined_raster_cube(cubes, data, mask=_stack_mask(cubes, target_steps), memmap=False)


def _finalize_window_object(cubes, *, memmap):
    if len(cubes) == 1:
        cube = _materialize_deferred_raster_gwcs(cubes[0])
        cube._raster_boundaries = [(0, cube.shape[0])]
        return cube
    return _combine_raster_cubes(cubes, memmap=memmap)
