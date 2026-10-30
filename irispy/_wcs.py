from functools import reduce, partial

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.modeling.rotations import cartesian2spherical, spherical2cartesian
from astropy.utils.shapes import simplify_basic_index
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube.utils.wcs import pixel_axis_to_physical_types
from ndcube.wcs.wrappers import ResampledLowLevelWCS
from sunpy.coordinates.frames import Helioprojective


def _time_lookup(times, name=None):
    """
    Linear time lookup over pixel indices, extrapolated to the outer pixel edges.

    The table gains nodes at pixel -0.5 and n - 0.5 on each axis, so pixel corners
    have finite times. Pixels further out get NaN, not a plausible but wrong time.
    """
    # Odd reflection extrapolates linearly to pixels -1 and n; averaging each with its
    # neighbor moves those nodes to the pixel edges.
    table = np.pad(times, 1, mode="reflect", reflect_type="odd")
    for axis in range(table.ndim):
        nodes = np.moveaxis(table, axis, 0)
        nodes[0] = (nodes[0] + nodes[1]) / 2
        nodes[-1] = (nodes[-1] + nodes[-2]) / 2
    points = tuple(np.r_[-0.5, np.arange(n), n - 0.5] * u.pix for n in times.shape)
    tabular = (m.Tabular1D, m.Tabular2D)[times.ndim - 1]
    return tabular(points, table, method="linear", bounds_error=False, fill_value=np.nan, name=name)


def _euler(phi, theta, psi):
    # The "zxz" Euler matrices of RotateNative2Celestial and RotateCelestial2Native, one per table row: astropy's
    # rotation models take only scalar angles
    return reduce(
        np.matmul,
        [rotation_matrix(angle, axis, unit=u.rad) for angle, axis in zip((psi, theta, phi), "zxz", strict=True)],
    )


def _rotate(matrix, lon, lat):
    xyz = np.stack(spherical2cartesian(lon, lat), axis=-1)[..., None]
    return cartesian2spherical(*np.moveaxis((matrix @ xyz)[..., 0], -1, 0))


def _value(quantity, unit):
    return quantity.to_value(unit) if isinstance(quantity, u.Quantity) else np.asarray(quantity, dtype=float)


def _map_transform(transform, *arrays, cdelt, lon_pole, inverse=False):
    """
    The ``_map_transform`` of a dkist varying celestial transform, computed in one pass
    over all the points.

    Each point uses the table row that dkist would pick, and points outside the tables
    are NaN, as in dkist.
    """
    arrays = np.broadcast_arrays(*arrays, subok=True)
    has_units = isinstance(arrays[0], u.Quantity)
    x, y, *lookups = (np.asarray(array.value if has_units else array, dtype=float) for array in arrays)
    rows = []
    for lookup, size in zip(lookups, transform.table_shape, strict=True):
        row = transform.sanitize_index(lookup)
        # The outer pixel edge, size - 0.5, stays on the last row, as in dkist's own loop since DKISTDC/dkist#764
        rows.append(np.where(lookup <= size - 0.5, np.minimum(row, size - 1), row))
    valid = np.logical_and.reduce(
        [(row >= 0) & (row < size) for row, size in zip(rows, transform.table_shape, strict=True)]
    )
    rows = tuple(np.where(valid, row, 0) for row in rows)
    # The matrices are built once per table row and then picked for each point
    pc = _value(transform.pc_table, u.pix)
    crpix = _value(transform.crpix_table, u.pix)[rows]
    lon0, lat0 = np.moveaxis(np.deg2rad(_value(transform.crval_table, u.deg)), -1, 0)
    # Modeling passes the parameters with a leading length-1 axis, which dkist's own loop drops with cdelt[0]
    cdelt = _value(cdelt, u.deg / u.pix).reshape(-1, 2)[0]
    pole = np.deg2rad(_value(lon_pole, u.deg).ravel()[0])
    if inverse:
        # RotateCelestial2Native | projection.inverse | cdelt | pc | crpix, as dkist's model for one row
        phi, theta = _rotate(_euler(np.pi / 2 + lon0, np.pi / 2 - lat0, -(pole - np.pi / 2))[rows], x, y)
        phi = np.where(phi < 0, phi + 360, phi)
        intermediate = np.stack(transform.projection.inverse(phi, theta), axis=-1) / cdelt
        offset = (np.linalg.inv(pc)[rows] @ intermediate[..., None])[..., 0]
        out_x, out_y = offset[..., 0] + crpix[..., 0], offset[..., 1] + crpix[..., 1]
    else:
        offset = np.stack([x - crpix[..., 0], y - crpix[..., 1]], axis=-1)[..., None]
        intermediate = (pc[rows] @ offset)[..., 0] * cdelt
        phi, theta = transform.projection(intermediate[..., 0], intermediate[..., 1])
        rotation = _euler(pole - np.pi / 2, -(np.pi / 2 - lat0), -(np.pi / 2 + lon0))[rows]
        out_x, out_y = _rotate(rotation, phi, theta)
        out_x = np.where(out_x < 0, out_x + 360, out_x)
    out_x, out_y = np.where(valid, out_x, np.nan), np.where(valid, out_y, np.nan)
    if has_units:
        unit = u.pix if transform._is_inverse else u.deg
        return out_x << unit, out_y << unit
    return out_x, out_y


def _one_pass(transform):
    """
    ``transform``, a dkist varying celestial transform, evaluated in one pass over its
    points instead of one modeling call per table row.
    """
    # TODO: delete once irispy requires a dkist release that evaluates a VaryingCelestialTransform in one pass over
    # its points (DKISTDC/dkist#762, which did, was closed unmerged). dkist evaluates its model once per table row,
    # ~0.2 ms each, so an exposure-by-slit image of a long sit-and-stare took 6-9 s to plot against 0.05-0.09 s with a
    # FITS WCS. Replaced on these instances only, so dkist's classes and their asdf converters are unchanged; a
    # partial, unlike a bound method, survives pickling and copying. The edge clamp of _map_transform goes with it.
    transform._map_transform = partial(_map_transform, transform)
    return transform


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
