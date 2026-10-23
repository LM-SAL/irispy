from copy import deepcopy

import dask.array as da
import numpy as np

import astropy.units as u
from astropy.io import fits

from irispy._spectrograph_wcs import _create_raster_gwcs
from irispy.io._mask import _memmap_fill_mask
from irispy.spectrograph import SpectrogramCube


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


def _read_window(source, index):
    if isinstance(source, tuple):
        filename, ext, flip = source
        with fits.open(filename, memmap=True, do_not_scale_image_data=True) as hdulist:
            data = hdulist[ext].data
            return np.array((data[::-1] if flip else data)[index])
    return np.array(source[index])


class _WindowFiles:
    """
    One window of several raster files, stacked along a new first axis, that reads each
    file on indexing.

    Each source is the ``(filename, ext, flip)`` of a file, which stays closed until it
    is read, or the window's array of an HDU list given to the reader. Dask makes one
    graph layer of this array for all the files.
    """

    def __init__(self, sources, shape, dtype):
        self.sources = sources
        self.shape = (len(sources), *shape)
        self.dtype = dtype
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        raster, *rest = item
        if isinstance(raster, slice):
            return np.stack([_read_window(source, tuple(rest)) for source in self.sources[raster]])
        return _read_window(self.sources[raster], tuple(rest))


def _stack(arrays):
    # The arrays of one file gain the raster axis as a view, where np.stack would copy them
    return arrays[0][np.newaxis] if len(arrays) == 1 else np.stack(arrays)


def _combine_raster_cubes(cubes, window_headers, pc_tables, crval_tables, *, memmap=False):
    """
    One cube of the rasters of one window, each read from its own file (``cubes``: its
    data, mask, uncertainty, unit, meta, time, shape and dtype; with ``memmap`` and
    several files, the data are a source of ``_WindowFiles`` and the mask is not used),
    with a leading raster axis.
    """
    if memmap and len(cubes) > 1:
        shape, dtype = cubes[0].shape, cubes[0].dtype
        # Chunks of up to 64 MiB of steps, one file each
        steps = max(1, min(shape[0], 64 * 2**20 // (int(np.prod(shape[1:])) * dtype.itemsize)))
        data = da.from_array(
            _WindowFiles([cube.data for cube in cubes], shape, dtype),
            chunks=(1, steps, *shape[1:]),
            name=False,
            fancy=False,
            meta=np.empty((0,) * (len(shape) + 1), dtype=dtype),
        )
        mask = _memmap_fill_mask(data, window_headers[0])
    else:
        # The memmap of one file stays one, as when it was read alone
        data, mask = _stack([cube.data for cube in cubes]), _stack([cube.mask for cube in cubes])
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
