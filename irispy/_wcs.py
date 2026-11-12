"""
Helpers for the WCSs of the cubes.

The maths of each gWCS is in the docstring of the code that builds it: rasters in
``irispy._spectrograph_wcs``, slit-jaw cubes in ``irispy.io.sji._create_gwcs`` and
mosaics in ``irispy.io.mosaic._create_gwcs``. Their transforms are astropy models,
put together with its operators:

``a | b``
    ``b`` of the outputs of ``a``.
``a & b``
    Side by side: ``a`` takes the first inputs and ``b`` the ones after them, and the
    outputs are those of ``a``, then those of ``b``.
``a + b``, ``a - b``, ``a * b``, ``a / b``
    Both take the same inputs, and the arithmetic is on their outputs.
``Mapping((0, 2, 2))``
    Picks, repeats and reorders the inputs by their number: here the first input,
    then the third one twice. It feeds one pixel coordinate to several models.
``Identity(n)``
    Passes ``n`` inputs through unchanged.

A value that changes from exposure to exposure is a lookup table over the pixel
indices (``_table_lookup``). A fractional index interpolates between exposures, or
takes the nearest one in a table made with ``method="nearest"``.
"""

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
from astropy.utils.shapes import simplify_basic_index
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube.utils.wcs import pixel_axis_to_physical_types
from ndcube.wcs.wrappers import ResampledLowLevelWCS
from sunpy.coordinates.frames import Helioprojective


def _table_lookup(values, name=None, method="linear"):
    """
    Lookup of ``values``, such as the time of each exposure, over pixel indices, out to
    the outer pixel edges.

    The table gains nodes at pixel -0.5 and n - 0.5 on each axis, so pixel corners
    have finite values. Pixels further out get NaN, not a plausible but wrong value.

    ``method`` is "linear", which extrapolates to the edges, or "nearest", which gives
    every point in a pixel the value of that pixel.
    """
    if method == "nearest":
        table = np.pad(values, 1, mode="edge")
    else:
        # Odd reflection extrapolates linearly to pixels -1 and n; averaging each with its
        # neighbor moves those nodes to the pixel edges.
        table = np.pad(values, 1, mode="reflect", reflect_type="odd")
        for axis in range(table.ndim):
            nodes = np.moveaxis(table, axis, 0)
            nodes[0] = (nodes[0] + nodes[1]) / 2
            nodes[-1] = (nodes[-1] + nodes[-2]) / 2
    points = tuple(np.r_[-0.5, np.arange(n), n - 0.5] * u.pix for n in values.shape)
    tabular = (m.Tabular1D, m.Tabular2D)[values.ndim - 1]
    return tabular(points, table, method=method, bounds_error=False, fill_value=np.nan, name=name)


def _root_wcs(cube):
    """
    The WCS that the WCS of ``cube`` slices or rebins, or that WCS itself.
    """
    wcs = cube.wcs.low_level_wcs
    while isinstance(wcs, (SlicedLowLevelWCS, ResampledLowLevelWCS)):
        wcs = wcs._wcs
    return wcs


def _root_and_item(cube):
    """
    The WCS that the WCS of ``cube`` slices, with the slice that leads from it to the
    cube's in array order, or that WCS itself with a full slice.
    """
    wcs = cube.wcs.low_level_wcs
    if isinstance(wcs, SlicedLowLevelWCS):
        return wcs._wcs, list(wcs._slices_array)
    return wcs, [slice(None)] * wcs.pixel_n_dim


def _sliced(wcs, item):
    return wcs if all(index == slice(None) for index in item) else wcs.slice(tuple(item), numpy_order=True)


def _fits_wcs_from_cube(cube):
    """
    The FITS WCS of the cube, sliced as the cube is.

    For a raster window, a list with the FITS ``-TAB`` WCS of each raster, which holds
    the pointing of each exposure; for a slit-jaw cube, one per frame, with the pointing
    and time of that frame. For one raster or frame, ``cube[k]``, its FITS WCS. For a
    cube built on a FITS WCS, such as a mosaic, that WCS. `None` when the cube has none,
    such as after rebinning.
    """
    root, item = _root_and_item(cube)
    if isinstance(root, WCS):
        return _sliced(root, item)
    # A rebinned cube has no FITS WCS; a gWCS cube has one for each raster or frame
    roots = cube._fits_wcs_roots(item[0]) if isinstance(root, gwcs.WCS) else None
    if isinstance(roots, list):
        return [_sliced(wcs, item[1:]) for wcs in roots]
    return None if roots is None else _sliced(roots, item[1:])


def _wcs_array_axis_physical_types(cube):
    """
    `~ndcube.NDCube.array_axis_physical_types` from the main WCS alone.

    The combined WCS, with the extra coordinates, raises on a length-1 step axis
    (ndcube: mismatched pixel bounds), such as ``cube[0:1]`` or a one-exposure file.
    """
    wcs = cube.wcs.low_level_wcs
    return [tuple(pixel_axis_to_physical_types(axis, wcs).tolist()) for axis in reversed(range(wcs.pixel_n_dim))]


def _celestial_frame_from_cube(cube):
    """
    The `~sunpy.coordinates.frames.Helioprojective` frame of this observation, as seen
    by the IRIS observer.

    This does not depend on slicing: it works on combined multi-file cubes and on
    cubes sliced along the scan, step, slit, time, or wavelength axes.
    """
    observer = getattr(cube.meta, "observer", None)
    if observer is not None:
        return Helioprojective(observer=observer, obstime=observer.obstime)
    # The FITS WCS of the first frame, whatever the slice, or the FITS WCS the cube is built on
    fits_wcs = cube._fits_wcs_roots(0)
    if fits_wcs is None:
        fits_wcs = _root_wcs(cube)
    if not hasattr(fits_wcs, "celestial"):
        msg = "This cube does not carry the WCS metadata needed to derive a celestial frame."
        raise ValueError(msg)
    return wcs_to_celestial_frame(fits_wcs.celestial)


class _ResolveNegativeIndicesMixin:
    """
    Resolve negative indices before NDCube slicing.

    The sliced WCS keeps negative offsets as they are (astropy#15557), so for example
    ``cube[1:5][-1]`` describes ``cube[0]``, and NDMeta raises on negative slices of
    axis-aware keys.

    TODO: delete this once irispy requires an ndcube that resolves negative indices
    itself (the ndcube ``negative-indices`` branch, which also restores ``meta`` when
    slicing raises).
    """

    def __getitem__(self, item):
        if any(np.iterable(index) for index in (item if isinstance(item, tuple | list) else (item,))):
            # As ndcube, where sanitize_slices rejects an array index, which simplify_basic_index trips over
            msg = "This slice is invalid, only integer or range slices are supported."
            raise IndexError(msg)
        # Against the shape, so that negative indices resolve here
        try:
            item = simplify_basic_index(item, shape=self.shape)
        except np.exceptions.AxisError:
            # An IndexError, as for arrays, not numpy's AxisError, which is a ValueError and speaks of axes
            msg = f"Index {item} is out of range for a cube of shape {self.shape}"
            raise IndexError(msg) from None
        # A sliced WCS cannot describe a step, so dropping the step below loses nothing
        if any(isinstance(index, slice) and index.step not in (None, 1) for index in item):
            msg = "Slicing WCS with a step is not supported."
            raise IndexError(msg)
        item = tuple(slice(index.start, index.stop) if isinstance(index, slice) else index for index in item)
        if 0 not in self.shape and any(isinstance(index, slice) and index.start >= index.stop for index in item):
            msg = "Slicing would give a length-0 axis, which a WCS cannot describe."
            raise IndexError(msg)
        meta = self.meta
        try:
            return super().__getitem__(item)
        finally:
            # NDCube leaves the unsliced cube without its meta when slicing raises.
            self.meta = meta
