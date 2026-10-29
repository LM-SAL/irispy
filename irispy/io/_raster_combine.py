from copy import deepcopy
from numbers import Integral

import dask.array as da
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty

from irispy._spectrograph_wcs import _create_raster_gwcs, _create_raster_non_spectral
from irispy.io._mask import _memmap_fill_mask
from irispy.spectrograph import SpectrogramCube


def _combine_raster_metas(metas, combined_shape):
    # One file's meta gets the raster axis in place; for several files, a copy of the first file's meta does
    meta = metas[0] if len(metas) == 1 else deepcopy(metas[0])
    meta._data_shape = np.asarray(combined_shape, dtype=int)
    if len(metas) > 1 and (date_end := metas[-1].get("DATE_END")) is not None:
        meta["DATE_END"] = meta.fits_header["DATE_END"] = date_end
    with_data = next((source for source in metas if source.fits_header["DATE_OBS"]), metas[0])
    if with_data is not metas[0]:
        # A file with no data has no pointing or roll, so take them from the first file with data
        for key in ("XCEN", "YCEN", "SAT_ROT"):
            meta.add(key, with_data[key], with_data.key_comments.get(key), overwrite=True)
            meta.fits_header[key] = with_data.fits_header[key]
    for key, axes in list(metas[0].axes.items()):
        values = metas[0][key][np.newaxis] if len(metas) == 1 else np.stack([source[key] for source in metas])
        meta.add(key, values, axes=(0, *(axis + 1 for axis in axes)), overwrite=True)
    return meta


class _SaturatedSamples:
    """
    A `True`/`False` array of the saturated samples of one window that stores only their
    positions.

    Saturated samples are rare, so a few bytes for each takes far less memory than one
    byte for every sample of the data. Slicing with integers and slices, as
    `~ndcube.NDMeta` does when the cube is sliced, keeps only the positions inside the
    slice. `numpy.asarray` builds the full array.
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
            # Any other index: build the array and let numpy index it
            return np.asarray(self)[item]
        item += (slice(None),) * (self.ndim - len(item))
        shape, positions = [], []
        inside = np.ones(self.positions.shape[1], dtype=bool)
        for index, size, axis_positions in zip(item, self.shape, self.positions, strict=True):
            selected = np.arange(size)[index]
            # The place of each index of this axis in the slice, or -1 if it is outside
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
    One window of several raster files, stacked along a new first axis, that reads a
    file only when it is indexed.

    Each source is either the ``(filename, ext, flip)`` of a file, which stays closed
    until it is read, or the window's array from an HDU list given to the reader. Dask
    wraps this in one graph layer for all the files.
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
    Add the arrays of one window of one file, raster ``index`` of ``count``, to
    ``rasters``: the window's arrays by name, with a leading raster axis.

    For a single file, the arrays get the axis as a view, so a memmap stays a memmap.
    For several files, each array is made at the first file and each file is copied in
    as it is read. Stacking them after all files are read would hold every file's arrays
    and the stack at the same time.
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
    Add the array indices of the ``saturated`` samples of raster ``index`` of ``count``
    to ``rasters``, with the raster index first, in the smallest integer type that holds
    every index of the window.
    """
    positions = np.insert(np.argwhere(saturated), 0, index, axis=1)
    rasters.setdefault("saturated", []).append(positions.astype(np.min_scalar_type(max(count, *saturated.shape))))


def _combine_raster_cubes(reads, rasters, non_spectral, *, lazy):
    """
    Combine the rasters of one window, one read per file, into one cube with a leading
    raster axis.

    Each read has its file's window ``header``, ``pc`` and ``crval`` tables, ``unit``,
    ``meta``, ``time`` and ``observer``; for a ``lazy`` read, also the ``data`` source
    for ``_WindowFiles`` and the ``shape`` and ``dtype`` of one file. ``rasters`` has
    the window's data, mask and any uncertainty from ``_add_raster`` (for a ``lazy``
    read, the data and mask are made here instead), and any saturated samples from
    ``_add_saturated``. ``non_spectral`` maps each detector band to the result of
    ``_create_raster_non_spectral`` for the windows of that band read with these, and
    gains it for the band of this window if missing.
    """
    first = reads[0]
    if lazy:
        shape, dtype = first.shape, first.dtype
        # Chunks of up to 64 MiB of steps, one file each
        steps = max(1, min(shape[0], 64 * 2**20 // (int(np.prod(shape[1:])) * dtype.itemsize)))
        data = da.from_array(
            _WindowFiles([read.data for read in reads], shape, dtype),
            chunks=(1, steps, *shape[1:]),
            name=False,
            fancy=False,
            meta=np.empty((0,) * (len(shape) + 1), dtype=dtype),
        )
        mask = _memmap_fill_mask(data, first.header)
    else:
        data, mask = rasters["data"], rasters["mask"]
    uncertainty = rasters.get("uncertainty")
    band = first.meta.detector_band
    if band not in non_spectral:
        # The windows of a file share its pointing tables, slit reference pixel and plate scales, as its one FITS-TAB
        # WCS does, and those of a band their times, so their gWCSes differ only in the spectral axis
        times = np.stack([read.time for read in reads])
        non_spectral[band] = _create_raster_non_spectral(
            np.stack([read.pc for read in reads]),
            np.stack([read.crval for read in reads]),
            [read.header["CRPIX2"] for read in reads],
            (times - times[0, 0]).to_value(u.s) * u.s,
            times[0, 0],
            first.observer,
            cdelt2=first.header["CDELT2"],
            cdelt3=first.header["CDELT3"],
        )
    meta = _combine_raster_metas([read.meta for read in reads], data.shape)
    meta["fits_wcs"] = [read.meta["fits_wcs"] for read in reads]
    if "saturated" in rasters:
        saturated = _SaturatedSamples(np.concatenate(rasters["saturated"]).T, data.shape)
        if saturated.positions.nbytes >= data.size:
            # With this many saturated samples (more than 1 in 8 for 2-byte indices), an array of one byte per
            # sample is smaller than their positions. The read held the positions until here, so it used more
            # memory than such an array would have.
            saturated = np.asarray(saturated)
        meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", tuple(range(data.ndim)))
    return SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(first.header, *non_spectral[band]),
        uncertainty=None if uncertainty is None else StdDevUncertainty(uncertainty, copy=False),
        unit=first.unit,
        meta=meta,
        mask=mask,
    )
