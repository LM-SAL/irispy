"""
The gWCS of a slit-jaw, AIA or SOT cube, whose maths is in the docstring of
``_create_sji_gwcs``.

Its transform is made of astropy models, put together with its operators:

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
takes the nearest one in a table made with ``method="nearest"``. A value that is linear
in other inputs, with such tables as its coefficients, is ``_rows``. The gWCS gives the
helioprojective longitude and latitude in arcsec, from -180 to 180 degrees, and the
time in seconds, in the frames of ``_sky_and_time_frames``.
"""

import operator
from functools import reduce

import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

from sunpy.coordinates.frames import Helioprojective

_FAR = 1e15  # pixels: further out than any index asked of a held lookup table


def _table_lookup(values, name=None, method="linear", *, hold=False):
    """
    Lookup of ``values``, such as the time of each exposure, over pixel indices, out to
    the outer pixel edges.

    The table gains nodes at pixel -0.5 and n - 0.5 on each axis, so pixel corners
    have finite values. Pixels further out get NaN, not a plausible but wrong value.

    ``method`` is ``"linear"``, which extrapolates to the edges, or ``"nearest"``, which
    gives every point in a pixel the value of that pixel.

    ``hold`` gives pixels further out the value at the outer edge, not NaN. It is for
    the tables of a world to pixel transform: a NaN pixel breaks a crop whose bounds
    are outside the cube.
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
    points = [np.r_[-0.5, np.arange(n), n - 0.5] for n in values.shape]
    if hold:
        table = np.pad(table, 1, mode="edge")
        points = [np.r_[-_FAR, nodes, _FAR] for nodes in points]
    tabular = (m.Tabular1D, m.Tabular2D)[values.ndim - 1]
    return tabular(
        tuple(nodes * u.pix for nodes in points), table, method=method, bounds_error=False, fill_value=np.nan, name=name
    )


def _rows(tables, offset=None, *, index=1, names=None, **lookup):
    """
    The model ``offset[row] + a * tables[0][row] + b * tables[1][row] + ...`` of the
    inputs ``(a, b, ..., *row)``, from lookup tables over the last ``index`` inputs,
    such as the frame of a slit-jaw cube.

    ``names`` name the lookups, that of ``offset`` first, and ``lookup`` goes to
    ``_table_lookup``.
    """
    count = len(tables)
    n_inputs = count + index
    # ``row | lookup`` is the value of a table at the row that the last inputs give
    row = m.Mapping(tuple(range(count, n_inputs)), n_inputs=n_inputs)
    names = iter(names or [None] * (count + 1))
    terms = [] if offset is None else [row | _table_lookup(offset, name=next(names), **lookup)]
    terms += [
        m.Mapping((k,), n_inputs=n_inputs) * (row | _table_lookup(table, name=next(names), **lookup))
        for k, table in enumerate(tables)
    ]
    return reduce(operator.add, terms)


def _sky_and_time_frames(observer, obstime, start, sky_axes):
    """
    The output frames of the gWCS: the helioprojective longitude and latitude that
    ``observer`` sees at ``obstime``, in arcsec, on the world axes ``sky_axes``, and on
    the axis after them the time, in seconds from ``start``.
    """
    celestial = cf.CelestialFrame(
        axes_order=sky_axes,
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer=observer, obstime=obstime),
        axis_physical_types=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"],
        axes_names=("Longitude", "Latitude"),
    )
    temporal = cf.TemporalFrame(start, unit=(u.s,), axes_order=(sky_axes[1] + 1,), axes_names=("Time (UTC)",))
    return celestial, temporal


def _create_sji_gwcs(pc_table, crval_table, crpix, cdelt, t_obs):
    """
    The gWCS of a slit-jaw, AIA or SOT cube, from the pixels ``(x, y, frame)`` to the
    helioprojective longitude and latitude and the time.

    Each frame has a FITS WCS of its own: a TAN projection about the pointing of the
    frame (Calabretta & Greisen 2002, A&A 395, 1077). The gWCS gives the sky positions
    of that WCS. The file records the pointing (``XCENIX``, ``YCENIX``) and the ``PC``
    matrix of every frame: ``crval_table``, of shape (frame, 2) in arcsec, and
    ``pc_table``, of shape (frame, 2, 2). So each name below with ``[frame]`` is a lookup
    table with a row for each frame. ``crpix`` is the reference pixel, counted from 0,
    ``cdelt`` the plate scales in arcsec and ``t_obs`` the time of each frame.

    **Pixel to sky**, in three steps:

    1. Where the pixel is on the plane of the projection. A TAN projection draws the
       image on the plane that touches the sky sphere, of radius 1, at the pointing.
       The linear part of the FITS WCS puts a pixel at ``(X, Y)`` on that plane: its
       distances east and north of the pointing (toward greater longitude and
       latitude). ``CDELT`` gives them in arcsec, and in radians they are lengths on
       the plane. FITS scales row i of ``PC`` by ``CDELT`` i::

           (X, Y) = to_plane[frame] @ (pixel - crpix)

           to_plane = | CDELT1 * PC1_1    CDELT1 * PC1_2 |
                      | CDELT2 * PC2_1    CDELT2 * PC2_2 |

    2. Where that point is in space, from the observer at the center of the sphere::

           direction = toward[frame] + X * east[frame] + Y * north[frame]

       ``toward`` is the unit vector from the observer to the pointing, at longitude
       ``lon`` and latitude ``lat``, and ``east`` and ``north`` are the unit vectors
       along the plane. The first axis of the three points to longitude and latitude
       0, the second to longitude 90 degrees and the third to the north pole::

           toward = (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
           east = (-sin(lon), cos(lon), 0)
           north = (-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))

    3. The sky position is the direction to that point, whatever its distance::

           longitude = arctan2(direction[1], direction[0])
           latitude = arctan2(direction[2], hypot(direction[0], direction[1]))

    Steps 1 and 2 together are linear in the pixel, so lookup tables can hold them::

        direction = origin[frame] + x * per_x[frame] + y * per_y[frame]

    ``per_x`` and ``per_y`` are the change of the direction for one pixel along x and
    y, and ``origin`` is the direction of pixel (0, 0). Each has three components, so
    there are nine tables, named ``origin_k``, ``per_x_k`` and ``per_y_k`` for component
    ``k``. Between frames, the tables and the time are interpolated.
    astropy's own TAN and rotation models take one
    pointing, not one for each frame.

    **World to pixel.** The time gives the frame, and the sky position a unit vector
    ``d``, by step 3 backward. The plane of the frame is every point with
    ``point @ toward = 1``, so the line from the observer along ``d`` meets it at
    ``d / (d @ toward)``. Then steps 2 and 1 backward::

        X = (d @ east[frame]) / (d @ toward[frame])
        Y = (d @ north[frame]) / (d @ toward[frame])
        pixel = crpix + inverse(to_plane[frame]) @ (X, Y)

    Between frames this is within 1e-4 pixel of the inverse of the forward transform,
    as the tables of each direction are interpolated on their own. A time outside the
    cube gives a frame outside it, with the tables of the nearest frame, so that a crop
    with such a bound is clipped to the data.
    """
    # Step 2: the unit vectors at the pointing of each frame, each (frame, 3)
    lon, lat = np.deg2rad(crval_table / 3600).T
    toward = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    east_north = np.stack([east, north], axis=1)  # (frame, 2, 3)
    # Step 1: (frame, 2, 2), from arcsec to radians
    to_plane = np.deg2rad(pc_table * cdelt[:, np.newaxis] / 3600)
    # Steps 1 and 2 as one. per_pixel[frame, j] = to_plane[frame, 0, j] * east + to_plane[frame, 1, j] * north is
    # per_x at j = 0 and per_y at j = 1, and (frame, 2, 3) in all.
    per_pixel = np.swapaxes(to_plane, 1, 2) @ east_north
    origin = toward - crpix @ per_pixel
    # Component k of the direction, from its three tables
    direction = [
        _rows(
            list(per_pixel[..., k].T / u.pix),
            origin[:, k] * u.one,
            names=(f"origin_{k}", f"per_x_{k}", f"per_y_{k}"),
        )
        for k in range(3)
    ]
    start_time = t_obs[0]
    cadence = (t_obs - start_time).to_value(u.s) * u.s
    temporal = _table_lookup(cadence)
    # (x, y, frame) to (lon, lat, time): each component of the direction takes (x, y, frame) and the time takes
    # the frame. CartesianToSpherical is step 3.
    forward_transform = (
        m.Mapping((0, 1, 2) * 3 + (2,))
        | (direction[0] & direction[1] & direction[2] | CartesianToSpherical(wrap_lon_at=180)) & temporal
    )
    # World to pixel. ``to_pixel[frame, k]`` is ``inverse(to_plane)[k, 0] * east + inverse(to_plane)[k, 1] * north``,
    # so ``pixel[k]`` is one division of two sums: ``crpix[k] + (d @ to_pixel[frame, k]) / (d @ toward[frame])``
    to_pixel = np.linalg.inv(to_plane) @ east_north
    # A frame outside the cube takes the tables of the nearest frame
    pixel = [
        _rows(list(to_pixel[:, k].T * u.pix), hold=True) / _rows(list(toward.T * u.one), hold=True)
        | m.Shift(crpix[k] * u.pix)
        for k in range(2)
    ]
    if len(t_obs) > 1:
        # A time before the first frame or after the last gives a frame outside the cube, as from a FITS WCS
        to_frame = temporal.inverse
        to_frame.fill_value = None
    else:
        # The times of one frame do not invert: any time is that frame
        to_frame = m.Multiply(0 * u.pix / u.s)
    # (lon, lat, time) to (d, frame), then each pixel coordinate takes (d, frame) and the frame passes through
    forward_transform.inverse = (
        SphericalToCartesian(wrap_lon_at=180) & to_frame
        | m.Mapping((0, 1, 2, 3, 0, 1, 2, 3, 3))
        | pixel[0] & pixel[1] & m.Identity(1)
    )
    output_frame = cf.CompositeFrame(_sky_and_time_frames("earth", start_time, start_time, (0, 1)))
    input_frame = cf.CoordinateFrame(
        axes_order=(0, 1, 2),
        naxes=3,
        axes_names=("image x", "image y", "frame"),
        axes_type=["PIXEL", "PIXEL", "PIXEL"],
        unit=(u.pix, u.pix, u.pix),
    )
    return gwcs.WCS(forward_transform, input_frame=input_frame, output_frame=output_frame)
