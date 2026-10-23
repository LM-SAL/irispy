from copy import deepcopy

import dask.array as da
import numpy as np
from dask import delayed

import astropy.units as u
from astropy.io import fits

from irispy._spectrograph_wcs import _create_raster_gwcs
from irispy.spectrograph import SpectrogramCube

LAZY_RASTER_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def _combine_raster_metas(metas, combined_shape):
    # The meta of one file gains the raster axis in place; several files take the first file's
    meta = metas[0] if len(metas) == 1 else deepcopy(metas[0])
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    if len(metas) > 1 and (date_end := metas[-1].get("DATE_END")) is not None:
        meta["DATE_END"] = meta.fits_header["DATE_END"] = date_end
    for key in {key for source in metas for key in source.axes}:
        axes = next(source.axes[key] for source in metas if key in source.axes)
        if key == "saturated" and len(metas) == 1:
            values = meta[key][np.newaxis]  # a view, as large as the data
        elif key == "saturated":
            # Only the files with saturated samples carry them
            values = np.zeros(combined_shape, dtype=bool)
            for index, source in enumerate(metas):
                if key in source:
                    values[index] = source[key]
        else:
            values = np.stack([source[key] for source in metas])
        meta.add(key, values, axes=(0, *(axis + 1 for axis in axes)), overwrite=True)
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
    # Most windows are one chunk, which concatenating would only wrap
    return chunks[0] if len(chunks) == 1 else da.concatenate(chunks, axis=0)


def _stack(arrays):
    # The arrays of one file gain the raster axis as a view, where np.stack would copy them
    return arrays[0][np.newaxis] if len(arrays) == 1 else np.stack(arrays)


def _combine_raster_cubes(cubes, window_headers, pc_tables, crval_tables, *, memmap=False):
    """
    One cube of the rasters of one window, each read from its own file (``cubes``: its
    data, mask, uncertainty, unit, meta, time and shape), with a leading raster axis.
    """
    # The memmap of one file stays one, as when it was read alone
    stack = da.stack if memmap and len(cubes) > 1 else _stack
    data, mask = stack([cube.data for cube in cubes]), stack([cube.mask for cube in cubes])
    uncertainty = cubes[0].uncertainty
    if uncertainty is not None:
        uncertainty = type(uncertainty)(_stack([cube.uncertainty.array for cube in cubes]))
    times = np.stack([cube.time for cube in cubes])
    meta = _combine_raster_metas([cube.meta for cube in cubes], data.shape)
    meta["fits_wcs"] = [cube.meta["fits_wcs"] for cube in cubes]
    return SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(
            window_headers[0],
            np.stack(pc_tables),
            np.stack(crval_tables),
            (times - times[0, 0]).to_value(u.s) * u.s,
            times[0, 0],
            cubes[0].meta.observer,
            sit_and_stare=window_headers[0]["CDELT3"] == 0,
        ),
        uncertainty=uncertainty,
        unit=cubes[0].unit,
        meta=meta,
        mask=mask,
    )
