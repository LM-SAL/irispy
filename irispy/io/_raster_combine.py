import warnings
from copy import deepcopy

import dask.array as da
import numpy as np
from dask import delayed

import astropy.units as u
from astropy.io import fits

from irispy._spectrograph_wcs import _create_raster_gwcs
from irispy.spectrograph import SpectrogramCube

LAZY_RASTER_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def _pad_axis0(value, target_length):
    """
    Pad axis 0 to ``target_length`` by repeating the last entry.

    Works on anything supporting numpy fancy indexing along axis 0, including
    `~numpy.ndarray`, `~astropy.units.Quantity`, `~astropy.time.Time` and
    `~astropy.coordinates.SkyCoord`.
    """
    return value[np.minimum(np.arange(target_length), value.shape[0] - 1)]


def _stack_step_aligned_arrays(arrays, target_steps, *, fill_value):
    ragged = any(array.shape[0] != target_steps for array in arrays)
    dtype_args = [array.dtype for array in arrays]
    if np.isnan(fill_value) and ragged:
        dtype_args.append(np.float32)
    dtype = np.result_type(*dtype_args)
    stacked = np.empty((len(arrays), target_steps, *arrays[0].shape[1:]), dtype=dtype)
    for index, array in enumerate(arrays):
        stacked[index, : array.shape[0]] = array
        if array.shape[0] < target_steps:
            stacked[index, array.shape[0] :] = fill_value
    return stacked


def _combine_raster_metas(metas, combined_shape):
    meta = deepcopy(metas[0])
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    if (date_end := metas[-1].get("DATE_END")) is not None:
        meta["DATE_END"] = meta.fits_header["DATE_END"] = date_end
    for key in metas[0].axes:
        values = np.stack([_pad_axis0(source[key], combined_shape[1]) for source in metas])
        meta.add(key, values, axes=(0, 1), overwrite=True)
    return meta


def _lazy_raster_scan_chunk_rows(data):
    row_bytes = int(np.prod(data.shape[1:]) * data.dtype.itemsize)
    return max(1, min(data.shape[0], LAZY_RASTER_CHUNK_TARGET_BYTES // max(row_bytes, 1)))


def _read_memmap_window_chunk(filename, ext, flip, start, stop):
    """
    Read one scan-axis chunk from disk after the public reader has returned.
    """
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as hdulist:
        data = hdulist[ext].data
        return np.array((data[::-1] if flip else data)[start:stop])


def _lazy_window_data(filename, ext, flip, data):
    """
    Return a Dask array of one window's (memmap) ``data`` that reads the file on demand.
    """
    chunk_rows = _lazy_raster_scan_chunk_rows(data)
    chunks = []
    for start in range(0, data.shape[0], chunk_rows):
        stop = min(start + chunk_rows, data.shape[0])
        chunk = delayed(_read_memmap_window_chunk)(filename, ext, flip, start, stop)
        chunks.append(da.from_delayed(chunk, shape=(stop - start, *data.shape[1:]), dtype=data.dtype))
    return da.concatenate(chunks, axis=0)


def _combine_raster_cubes(cubes, window_headers, pc_tables, crval_tables, *, memmap=False):
    for key in ("OBSID", "STARTOBS"):
        if any(cube.meta.get(key) != cubes[0].meta.get(key) for cube in cubes[1:]):
            msg = f"All raster cubes must have the same {key}."
            raise ValueError(msg)
    if any(cube.shape[1:] != cubes[0].shape[1:] for cube in cubes[1:]):
        msg = "All raster cubes must have the same slit and wavelength dimensions."
        raise ValueError(msg)
    target_steps = max(cube.shape[0] for cube in cubes)
    if any(cube.shape[0] != target_steps for cube in cubes):
        if memmap:
            msg = (
                "memmap=True does not support raster files with mismatched step counts; "
                "use memmap=False to pad shorter rasters."
            )
            raise ValueError(msg)
        warnings.warn(
            "Raster sequence has mismatched step counts; padding shorter rasters with NaN data and masked pixels.",
            UserWarning,
            stacklevel=3,
        )
    if memmap:
        data, mask = da.stack([cube.data for cube in cubes]), None
    else:
        data = _stack_step_aligned_arrays([cube.data for cube in cubes], target_steps, fill_value=np.nan)
        mask = _stack_step_aligned_arrays([cube.mask for cube in cubes], target_steps, fill_value=True)
    uncertainty = cubes[0].uncertainty
    if uncertainty is not None:
        uncertainty = type(uncertainty)(
            _stack_step_aligned_arrays([cube.uncertainty.array for cube in cubes], target_steps, fill_value=np.nan)
        )
    times = np.stack([_pad_axis0(cube.time, target_steps) for cube in cubes])
    pc_all = np.stack([_pad_axis0(pc, target_steps) for pc in pc_tables])
    crval_all = np.stack([_pad_axis0(crval, target_steps) for crval in crval_tables])
    meta = _combine_raster_metas([cube.meta for cube in cubes], data.shape)
    meta["fits_wcs"] = [cube.meta["fits_wcs"] for cube in cubes]
    return SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(
            window_headers[0],
            pc_all,
            crval_all,
            (times - times[0, 0]).to_value(u.s) * u.s,
            times[0, 0],
            cubes[0].meta.observer,
            sit_and_stare=cubes[0].meta["sit_and_stare"],
        ),
        uncertainty=uncertainty,
        unit=cubes[0].unit,
        meta=meta,
        mask=mask,
    )
