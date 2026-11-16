"""
What the cubes need of their WCS, whatever its kind: the WCS that a slice came from, the
FITS WCS, the celestial frame.

``irispy._gwcs`` builds the gWCS of each cube.
"""

from copy import deepcopy
from collections.abc import Sequence

import numpy as np

import gwcs
import gwcs.coordinate_frames as cf
from astropy.utils.shapes import simplify_basic_index
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube.utils.wcs import pixel_axis_to_physical_types
from ndcube.wcs.wrappers import ResampledLowLevelWCS


class _MadeWhenAskedFor(Sequence):
    """
    A sequence whose items are each made by a function of their own when first asked
    for, and kept: the FITS WCS of each raster, which is slow to build and often not
    used.

    A slice of it is a list.
    """

    def __init__(self, makers):
        self._makers = list(makers)
        self._made = {}

    def __deepcopy__(self, memo):
        # A slice of a cube copies its meta. The copy keeps items of its own and shares the functions, which change
        # nothing: copying what they hold, a header for each raster, made a slice take twice as long.
        copied = type(self)(self._makers)
        copied._made = deepcopy(self._made, memo)
        return copied

    def __len__(self):
        return len(self._makers)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        if index not in self._made:
            self._made[index] = self._makers[index]()
        return self._made[index]


def _root_wcs(cube):
    """
    The WCS of ``cube`` before any slicing or rebinning.
    """
    wcs = cube.wcs.low_level_wcs
    while isinstance(wcs, (SlicedLowLevelWCS, ResampledLowLevelWCS)):
        wcs = wcs._wcs
    return wcs


def _root_and_item(cube):
    """
    The WCS of ``cube`` before its slicing, and the slice, in array order, that leads
    from it to the cube's WCS: a full slice for a cube that is not sliced.
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

    For a raster window, a list with the FITS-TAB WCS of each raster, which holds the
    pointing of each exposure; for a slit-jaw cube, one per frame, with the pointing and
    time of that frame. For one raster or frame, ``cube[k]``, its FITS WCS. For a cube
    built on a FITS WCS, that WCS. `None` when the cube has none, such as after
    rebinning, or when it is one pixel of each raster or frame.
    """
    root, item = _root_and_item(cube)
    if isinstance(root, WCS):
        return _sliced(root, item)
    # A rebinned cube has no FITS WCS; a gWCS cube has one for each raster or frame
    roots = cube._fits_wcs_roots(item[0]) if isinstance(root, gwcs.WCS) else None
    if all(isinstance(index, (int, np.integer)) for index in item[1:]):
        # One pixel of each raster or frame, such as a light curve, leaves that WCS no pixel axis
        return None
    if isinstance(roots, list):
        return [_sliced(wcs, item[1:]) for wcs in roots]
    return None if roots is None else _sliced(roots, item[1:])


def _wcs_array_axis_physical_types(cube):
    """
    `~ndcube.NDCube.array_axis_physical_types` from the main WCS alone, without the
    types of the extra coordinates, for the printed summary of a cube.
    """
    wcs = cube.wcs.low_level_wcs
    return [tuple(pixel_axis_to_physical_types(axis, wcs).tolist()) for axis in reversed(range(wcs.pixel_n_dim))]


def _celestial_frame_from_cube(cube):
    """
    The `~sunpy.coordinates.frames.Helioprojective` frame of the sky positions that the
    WCS of the cube gives and takes.

    This does not depend on slicing: it works on combined multi-file cubes and on
    cubes sliced along the raster, step, slit, time, or wavelength axes.
    """
    root = _root_wcs(cube)
    if isinstance(root, gwcs.WCS):
        for frame in getattr(root.output_frame, "frames", [root.output_frame]):
            if isinstance(frame, cf.CelestialFrame):
                return frame.reference_frame
    elif hasattr(root, "celestial"):
        return wcs_to_celestial_frame(root.celestial)
    msg = "This cube does not carry the WCS metadata needed to derive a celestial frame."
    raise ValueError(msg)


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
