from copy import deepcopy

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
    For several files, each array is made at the first file that has it and each file
    is copied in as it is read: stacking them after all files are read would hold every
    file's arrays and the stack at the same time. A `None` array is skipped, so the
    rasters of a file without saturated samples stay `False` in the stack of them.
    """
    for name, array in arrays.items():
        if array is None:
            continue
        if count == 1:
            rasters[name] = array[np.newaxis]
            continue
        if name not in rasters:
            rasters[name] = np.zeros((count, *array.shape), array.dtype)
        rasters[name][index] = array


def _sky_time_models(files):
    """
    The part of the raster gWCS after the spectral axis, for each detector band read,
    from the records of the files.

    The windows of a file share its pointing tables, slit reference pixel and plate
    scales, as its one FITS-TAB WCS does, and those of a band their times, so their
    gWCSes differ only in the spectral axis.
    """
    first = files[0]
    models = {}
    for band in first.times:
        times = np.stack([file.times[band] for file in files])
        models[band] = _create_raster_non_spectral(
            np.stack([file.pc for file in files]),
            np.stack([file.crval for file in files]),
            [file.crpix2 for file in files],
            (times - times[0, 0]).to(u.s),
            times[0, 0],
            first.observer,
            cdelt2=first.cdelt2,
            cdelt3=first.cdelt3,
        )
    return models


def _combine_raster_cubes(reads, rasters, non_spectral, *, lazy):
    """
    Combine one window of every file, one read per file, into one cube with a leading
    raster axis.

    ``rasters`` has the window's arrays from ``_add_raster``, unless the read is
    ``lazy``, when the data and mask are made here as Dask arrays that open each file
    when it is read. ``non_spectral`` is the band's model from ``_sky_time_models``.
    """
    first = reads[0]
    if lazy:
        shape, dtype = first.shape, first.dtype
        # Chunks of up to 64 MiB of steps, one file each
        steps = max(1, min(shape[0], 64 * 2**20 // (int(np.prod(shape[1:])) * dtype.itemsize)))
        data = da.from_array(
            _WindowFiles([read.source for read in reads], shape, dtype),
            chunks=(1, steps, *shape[1:]),
            name=False,
            fancy=False,
            meta=np.empty((0,) * (len(shape) + 1), dtype=dtype),
        )
        mask = _memmap_fill_mask(data, first.header)
    else:
        data, mask = rasters["data"], rasters["mask"]
    uncertainty = rasters.get("uncertainty")
    meta = _combine_raster_metas([read.meta for read in reads], data.shape)
    meta["fits_wcs"] = [read.fits_wcs for read in reads]
    if "saturated" in rasters:
        meta.add(
            "saturated", rasters["saturated"], "Samples at the level 2 ceiling, also masked", tuple(range(data.ndim))
        )
    return SpectrogramCube(
        data,
        wcs=_create_raster_gwcs(first.header, *non_spectral),
        uncertainty=None if uncertainty is None else StdDevUncertainty(uncertainty, copy=False),
        unit=first.unit,
        meta=meta,
        mask=mask,
    )
