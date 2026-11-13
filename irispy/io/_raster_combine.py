from copy import deepcopy

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty

from irispy._gwcs import _create_raster_gwcs, _create_raster_non_spectral
from irispy._wcs import _MadeWhenAskedFor
from irispy.io._mask import _memmap_dn, _memmap_fill_mask, _memmap_saturated
from irispy.spectrograph import SpectrogramCube
from irispy.utils import calculate_uncertainty


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
        # astropy stacks sky coordinates into an uninitialized array and wraps every longitude of it on each
        # assignment, which warns when that memory holds an inf
        with np.errstate(invalid="ignore"):
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
    until it is read, or the window's array from an HDU list given to the reader.
    """

    def __init__(self, sources, shape, dtype):
        self.sources = sources
        self.shape = (len(sources), *shape)
        self.dtype = dtype
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        # Dask asks for a chunk, or for the part of one that a slice needs: one raster, or a slice of them
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
    file's arrays and the stack at the same time. A `None` array is skipped, so a
    file without saturated samples stays `False` in their stack.
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

    The windows of a file share its slit lines, and those of a band their times, so
    their gWCS differ only in the spectral axis.
    """
    first = files[0]
    # The times of every band count from the DATE_OBS of the first file
    start = first.observer.obstime
    models = {}
    for band in first.times:
        times = np.stack([file.times[band] for file in files])
        models[band] = _create_raster_non_spectral(
            *(np.stack(table) for table in zip(*(file.slit_lines for file in files), strict=True)),
            (times - start).to(u.s),
            start,
            first.observer,
            sit_and_stare=first.sit_and_stare,
        )
    return models


def _combine_raster_cubes(reads, rasters, non_spectral, *, lazy):
    """
    Combine one window of every file, one read per file, into one cube with a leading
    raster axis.

    ``rasters`` has the window's arrays from ``_add_raster``: every array of a read into
    memory, only the mapped data of one memory-mapped file, and none for a ``lazy``
    read, of several memory-mapped files that are each opened when read. The DN, mask
    and uncertainty of memory-mapped data are made here, as Dask arrays.
    ``non_spectral`` is the band's model from ``_sky_time_models``.
    """
    first = reads[0]
    data = rasters.get("data")
    mask, uncertainty, saturated = rasters.get("mask"), rasters.get("uncertainty"), rasters.get("saturated")
    if mask is None:
        # The raw values of a memory-mapped read, with their raster axis: Dask does not pass a slice through an axis
        # added later, and reads whole chunks
        raw = _WindowFiles([read.source for read in reads], first.shape, first.dtype) if lazy else data
        header = first.header
        data = _memmap_dn(raw, header)
        mask = _memmap_fill_mask(raw, header, saturated=True)
        saturated = _memmap_saturated(raw, header)
        if first.readout_noise is not None:
            # In the byte order of the machine, whatever that of the file
            uncertainty = data.map_blocks(
                calculate_uncertainty, first.readout_noise, first.unit, dtype=data.dtype.newbyteorder("=")
            )
    meta = _combine_raster_metas([read.meta for read in reads], data.shape)
    meta["fits_wcs"] = _MadeWhenAskedFor([read.fits_wcs for read in reads])
    if saturated is not None:
        meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", tuple(range(data.ndim)))
    wcs = _create_raster_gwcs(first.header, *non_spectral)
    wcs.array_shape = data.shape
    return SpectrogramCube(
        data,
        wcs=wcs,
        uncertainty=None if uncertainty is None else StdDevUncertainty(uncertainty, copy=False),
        unit=first.unit,
        meta=meta,
        mask=mask,
    )
