from copy import deepcopy
from numbers import Integral

import dask.array as da
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty

from irispy._spectrograph_wcs import _create_raster_gwcs
from irispy.io._mask import _memmap_fill_mask
from irispy.spectrograph import SpectrogramCube


def _combine_raster_metas(metas, combined_shape):
    # The meta of one file gains the raster axis in place; several files take the first file's
    meta = metas[0] if len(metas) == 1 else deepcopy(metas[0])
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    if len(metas) > 1 and (date_end := metas[-1].get("DATE_END")) is not None:
        meta["DATE_END"] = meta.fits_header["DATE_END"] = date_end
    for key, axes in list(metas[0].axes.items()):
        values = np.stack([source[key] for source in metas])
        meta.add(key, values, axes=(0, *(axis + 1 for axis in axes)), overwrite=True)
    return meta


class _SaturatedSamples:
    """
    `True` at the saturated samples of one window and `False` elsewhere, as an array
    that holds only their positions.

    A window has few saturated samples, so their positions take a few bytes where an
    array as large as the data takes one byte a sample. Slicing it with integers and
    slices, as `~ndcube.NDMeta` does with the cube, keeps the positions inside the
    slice, and `numpy.asarray` builds the array.
    """

    def __init__(self, positions, shape):
        self.positions = positions  # the array indices of the samples, one row per axis
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(bool)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"<{self.positions.shape[1]} saturated samples in an array of shape {self.shape}>"

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            msg = "The saturated samples are built from their positions, which needs a copy"
            raise ValueError(msg)
        samples = np.zeros(self.shape, dtype=bool)
        samples[tuple(self.positions)] = True
        return samples if dtype is None else samples.astype(dtype, copy=False)

    def __getitem__(self, item):
        item = item if isinstance(item, tuple) else (item,)
        if len(item) > self.ndim or not all(isinstance(index, Integral | slice) for index in item):
            # Any other index, as numpy takes it
            return np.asarray(self)[item]
        item += (slice(None),) * (self.ndim - len(item))
        shape, positions = [], []
        inside = np.ones(self.positions.shape[1], dtype=bool)
        for index, size, axis_positions in zip(item, self.shape, self.positions, strict=True):
            selected = np.arange(size)[index]
            # Where each index of the axis lands in the slice, -1 outside it
            lookup = np.full(size, -1)
            lookup[selected] = np.arange(np.size(selected)).reshape(np.shape(selected))
            inside &= lookup[axis_positions] >= 0
            if np.ndim(selected):  # an integer drops its axis
                shape.append(selected.size)
                positions.append(lookup[axis_positions])
        if not shape:
            return np.bool_(inside.any())
        return _SaturatedSamples(np.stack([axis[inside] for axis in positions]).astype(self.positions.dtype), shape)


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


def _add_raster(rasters, index, count, **arrays):
    """
    Add the arrays of raster ``index`` of ``count``, one window of one file, to
    ``rasters``, the arrays of the window by name with a leading raster axis.

    The arrays of one file gain the axis as a view, so a memmap stays one. Those of
    several files are copied into one array each, made at the first file, as each file
    is read: stacking them once all are read would hold every file's arrays and the
    stack at once.
    """
    for name, array in arrays.items():
        if array is None:
            continue
        if count == 1:
            rasters[name] = array[np.newaxis]
            continue
        if name not in rasters:
            rasters[name] = np.empty((count, *array.shape), array.dtype)
        rasters[name][index] = array


def _add_saturated(rasters, index, count, saturated):
    """
    Add to ``rasters`` the array indices of the ``saturated`` samples of raster
    ``index`` of ``count``, after the raster, in the smallest integers that hold the
    indices of the window.
    """
    positions = np.insert(np.argwhere(saturated), 0, index, axis=1)
    rasters.setdefault("saturated", []).append(positions.astype(np.min_scalar_type(max(count, *saturated.shape))))


def _combine_raster_cubes(cubes, window_headers, pc_tables, crval_tables, rasters, *, memmap=False):
    """
    One cube of the rasters of one window, each read from its own file (``cubes``: its
    unit, meta and time, and with ``memmap`` and several files, its data as a source of
    ``_WindowFiles``, shape and dtype), with a leading raster axis.

    ``rasters`` holds the data, mask and any uncertainty of the window from
    ``_add_raster``, except with ``memmap`` and several files, whose data and mask are
    made here, and any ``"saturated"`` samples from ``_add_saturated``.
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
        data, mask = rasters["data"], rasters["mask"]
    uncertainty = rasters.get("uncertainty")
    times = np.stack([cube.time for cube in cubes])
    meta = _combine_raster_metas([cube.meta for cube in cubes], data.shape)
    meta["fits_wcs"] = [cube.meta["fits_wcs"] for cube in cubes]
    if "saturated" in rasters:
        saturated = _SaturatedSamples(np.concatenate(rasters["saturated"]).T, data.shape)
        if saturated.positions.nbytes >= data.size:
            # So many saturated samples, such as 1 in 8 with 2-byte indices, that an array of them, a byte each, is
            # smaller. The read held their positions until here, so it needed more memory than such an array.
            saturated = np.asarray(saturated)
        meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", tuple(range(data.ndim)))
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
        uncertainty=None if uncertainty is None else StdDevUncertainty(uncertainty, copy=False),
        unit=cubes[0].unit,
        meta=meta,
        mask=mask,
    )
