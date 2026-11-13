"""
The gWCS of each kind of cube: a raster, a slit-jaw, AIA or SOT cube, and a mosaic.

Their transforms are astropy models, put together with its operators:

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
in other inputs, with such tables as its coefficients, is ``_rows``. All three gWCS
give the helioprojective longitude and latitude in arcsec, from -180 to 180 degrees,
and the time in seconds, in the frames of ``_sky_and_time_frames``.

The maths of the slit-jaw, AIA and SOT cubes is in the docstring of
``_create_sji_gwcs`` and that of mosaics in the one of ``_create_mosaic_gwcs``. This
is the gWCS of a raster.

The pixel axes are (wavelength, slit, step, scan), the reverse of the array axes
(raster, step, slit, wavelength). Forward, from pixels to world, each world coordinate
has a model::

    pixels              model         world
    wavelength          Wavelength    wavelength
    slit, step, scan    Celestial     lon, lat
    step, scan          Time          time

``Wavelength``
    A line, from the ``CRVAL1``, ``CDELT1`` and ``CRPIX1`` of the window::

        wavelength = CRVAL1 + CDELT1 * (pixel - (CRPIX1 - 1))

``Celestial``
    The slit of each exposure as a straight line on the sky, in lookup tables by
    (step, scan)::

        lon = lon0[step, scan] + slit * dlon[step, scan]
        lat = lat0[step, scan] + slit * dlat[step, scan]

    ``dlon`` and ``dlat`` are the change of the sky position for one slit pixel, and
    ``lon0`` and ``lat0`` the position of slit pixel 0. They are the linear part of the
    FITS WCS of the exposure, along its slit. The file records the pointing (``XCENIX``, ``YCENIX``),
    which is at the slit reference pixel ``CRPIX2``, and the ``PC`` matrix of every
    exposure. FITS axis 2 is the slit and the latitude, and axis 3 the step and the
    longitude, and FITS scales row i of ``PC`` by ``CDELT`` i::

        dlon = CDELT3 * PC3_2        lon0 = XCENIX - dlon * (CRPIX2 - 1)
        dlat = CDELT2 * PC2_2        lat0 = YCENIX - dlat * (CRPIX2 - 1)

    With no roll, ``PC3_2 = 0`` and ``PC2_2 = 1``: the slit runs north, ``CDELT2``
    arcsec for each pixel. A sit-and-stare has ``CDELT3 = 0`` and an unscaled ``PC``,
    so its ``dlon`` is ``CDELT2 * PC3_2``.
    ``irispy.io.spectrograph._slit_lines`` works them out once for each file, for this
    gWCS and for the FITS-TAB WCS in ``cube.fits_wcs``, which has the same line. The
    line is within 1e-3 arcsec of the FITS celestial WCS of the exposure, a TAN
    projection.
``Time``
    A lookup table of the seconds from the ``DATE_OBS`` of the first file, by (step,
    scan).

The inverse, from world to pixels, finds the exposure from the time: its step and its
scan, from two lookup tables over the exposures in the order of their times, which
are all different. Between two exposures of a raster the step is interpolated. The
sky position gives only the pixel along the slit, where the position projects onto
the line of that exposure::

    slit = ((lon - lon0) * dlon + (lat - lat0) * dlat) / (dlon**2 + dlat**2)

It takes the longitude as the forward transform gives it. A time outside the cube
gives a step outside it, with the line of the nearest exposure, so that a crop with
such a time is clipped to the data. Finding a step from a sky position takes
``cube.fits_wcs``.

What the models do not show:

- A sit-and-stare (``CDELT3 = 0``) has the same models. The first model of its
  ``Celestial`` is named ``FIXED_SLIT_MODEL``, which ``_is_sit_and_stare`` looks for.
- Other modules find parts of the gWCS by name: the plot layout by the pixel axis names,
  and ``cube.time`` by ``TIME_MODEL``.
- An exposure with no pointing, one that was not taken, is NaN, as in
  ``cube.fits_wcs``, and so is the edge it shares with the next exposure, which a
  nearest lookup gives to the earlier of the two.
- The windows of a detector band share everything after ``Wavelength``
  (``irispy.io._raster_combine._sky_time_models``).
- The pixel scales of the cube come from ``cube.fits_wcs``.
"""

import operator
import itertools
from functools import reduce

import numpy as np
from scipy import ndimage

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

from sunpy.coordinates.frames import Helioprojective
from sunpy.time import parse_time

from irispy._wcs import _root_wcs

# The names other modules look for in a gWCS
WAVELENGTH_AXIS, SLIT_AXIS, STEP_AXIS, SCAN_AXIS = PIXEL_AXIS_NAMES = (
    "dispersion axis",
    "spatial along slit",
    "raster step",
    "raster scan",
)
FIXED_SLIT_MODEL = "FixedSlit"  # the first sky model of a sit-and-stare
TIME_MODEL = "Time"  # the lookup of the exposure times
MOSAIC_Y_AXIS, MOSAIC_X_AXIS = "solar Y", "solar X"  # the pixel axes of a mosaic gWCS after its dispersion axis


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
    inputs ``(a, b, ..., *row)``, from lookup tables over the last ``index`` inputs: the
    frame of a slit-jaw cube, or the step and scan of a raster.

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
    The output frames that the gWCS of every cube shares: the helioprojective longitude
    and latitude that ``observer`` sees at ``obstime``, in arcsec, on the world axes
    ``sky_axes``, and on the axis after them the time, in seconds from ``start``.
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


def _spectral_axis(cdelt, crval, crpix):
    """
    The linear spectral model and frame of a FITS WCS axis with the step ``cdelt`` and
    the wavelength ``crval`` at the 1-based reference pixel ``crpix``, in nm.
    """
    spectral = m.Linear1D(
        slope=cdelt.to(u.nm) / u.pix, intercept=(crval - cdelt * (crpix - 1)).to(u.nm), name="Wavelength"
    )
    frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm, name="wavelength", axes_names=("wavelength",))
    return spectral, frame


def _create_raster_gwcs(window_header, non_spectral, pixel_frame, frames):
    """
    Build the raster gWCS of one window from ``_create_raster_non_spectral``.
    """
    # Level 2 windows give their wavelengths in Å (CUNIT1)
    spectral, spectral_frame = _spectral_axis(
        window_header["CDELT1"] * u.AA, window_header["CRVAL1"] * u.AA, window_header["CRPIX1"]
    )
    # As a gwcs pipeline, not a forward transform, which gwcs would deep-copy: the windows share non_spectral
    return gwcs.WCS([(pixel_frame, spectral & non_spectral), (cf.CompositeFrame([spectral_frame, *frames]), None)])


def _exposures_in_time_order(seconds):
    """
    The nodes of the world to pixel lookup tables: the time, step and scan of each
    exposure, in time order. ``seconds`` is the time of each exposure, by (step, scan).

    Each raster with several steps gains a node half a step beyond its first and its
    last exposure, their outer edges, so that the times of a whole pixel give its step.
    Two rasters closer in time than their cadence meet halfway between the last exposure
    of one and the first of the other. A time at the outer edge of such a step is then
    of the other raster.
    """
    seconds = seconds.to_value(u.s)
    rasters = []
    # The rasters in time order, and the steps of each: a raster that steps backward is read with its steps reversed
    for scan in np.argsort(seconds[0]):
        step = np.argsort(seconds[:, scan])
        time, step = seconds[step, scan], step.astype(float)
        if step.size > 1:
            time, step = (np.r_[(3 * a[0] - a[1]) / 2, a, (3 * a[-1] - a[-2]) / 2] for a in (time, step))
        rasters.append([time, step, np.full(time.size, float(scan))])
    for (time, step, _), (next_time, next_step, _) in itertools.pairwise(rasters):
        if step.size > 1 and time[-1] >= next_time[0]:
            halfway = (time[-2] + next_time[1]) / 2
            # Each edge node moves along the line through it and the exposure beside it
            step[-1] = step[-2] + (step[-1] - step[-2]) * (halfway - time[-2]) / (time[-1] - time[-2])
            next_step[0] = next_step[1] + (next_step[0] - next_step[1]) * (halfway - next_time[1]) / (
                next_time[0] - next_time[1]
            )
            time[-1], next_time[0] = halfway, np.nextafter(halfway, np.inf)
    times, steps, scans = (np.concatenate(nodes) for nodes in zip(*rasters, strict=True))
    # A lookup table needs rising points: the same file read twice has every time twice
    rising = times > np.r_[-np.inf, np.maximum.accumulate(times)[:-1]]
    return times[rising], steps[rising], scans[rising]


def _time_to_step(seconds):
    """
    The step of a time: interpolated between the exposures of a raster, and outside the
    cube for a time outside it, as from a FITS WCS.
    """
    if seconds.size == 1:
        # The one exposure, whatever the time
        return m.Multiply(0 * u.pix / u.s)
    times, steps, _ = _exposures_in_time_order(seconds)
    return m.Tabular1D(times * u.s, steps * u.pix, bounds_error=False, fill_value=None)


def _time_to_scan(seconds):
    """
    The scan of a time: that of the nearest exposure.
    """
    if seconds.size == 1:
        return m.Multiply(0 * u.pix / u.s)
    times, _, scans = _exposures_in_time_order(seconds)
    return m.Tabular1D(times * u.s, scans * u.pix, method="nearest", bounds_error=False, fill_value=None)


def _create_raster_non_spectral(lon0, lat0, dlon, dlat, dt_all, t_ref, observer, *, sit_and_stare):
    """
    Build the part of the raster gWCS after the spectral axis: the transform, the pixel
    frame and the other output frames. The module docstring has the models, their
    formulas and the inverse.

    The tables have a row for each (raster, step): ``lon0``, ``lat0``, ``dlon`` and
    ``dlat`` are the slit lines of ``irispy.io.spectrograph._slit_lines``, in arcsec,
    and ``dt_all`` the times from ``t_ref``.
    """
    # Tables by (step, scan): the pixel inputs are (wavelength, slit, step, scan)
    lon0, lat0, dlon, dlat, dt_all = (np.swapaxes(table, 0, 1) for table in (lon0, lat0, dlon, dlat, dt_all))
    lon0, lat0 = lon0 * u.arcsec, lat0 * u.arcsec
    dlon, dlat = dlon * u.arcsec / u.pix, dlat * u.arcsec / u.pix
    # A linear lookup also gives NaN at the exposure before one that is NaN. So an exposure with no pointing takes
    # the tables of the nearest exposure with one, and gets its NaN from a nearest lookup of its own.
    missing = np.isnan(lon0) | np.isnan(lat0)
    if missing.any() and not missing.all():
        nearest = tuple(ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True))
        lon0, lat0, dlon, dlat = (table[nearest] for table in (lon0, lat0, dlon, dlat))

    lon = _rows([dlon], lon0, index=2, names=("lon0", "dlon"))
    lat = _rows([dlat], lat0, index=2, names=("lat0", "dlat"))
    if missing.any():
        row = m.Mapping((1, 2), n_inputs=3)
        no_pointing = np.where(missing, np.nan, 0) * u.arcsec
        lon = lon + (row | _table_lookup(no_pointing, method="nearest"))
        lat = lat + (row | _table_lookup(no_pointing, method="nearest"))
    name = FIXED_SLIT_MODEL if sit_and_stare else "SlitStepScan"
    sky = m.Mapping((0, 1, 2, 0, 1, 2), name=name) | lon & lat
    temporal = _table_lookup(dt_all, name=TIME_MODEL)
    # (slit, step, scan) to (lon, lat, time)
    non_spectral = m.Mapping((0, 1, 2, 1, 2)) | sky & temporal

    # World to pixel, by the formula of the module docstring with a table for each term: the slit pixel is
    # ``lon * to_slit_lon + lat * to_slit_lat - offset``
    norm = dlon**2 + dlat**2
    to_slit_lon, to_slit_lat = dlon / norm, dlat / norm
    offset = to_slit_lon * lon0 + to_slit_lat * lat0
    to_slit = _rows([to_slit_lon, to_slit_lat], -offset, index=2, hold=True)
    # (lon, lat, time) to (lon, lat, step, scan), then to (slit, step, scan)
    non_spectral.inverse = (
        m.Mapping((0, 1, 2, 2), name="SkyTimeTime")
        | m.Identity(2) & _time_to_step(dt_all) & _time_to_scan(dt_all)
        | m.Mapping((0, 1, 2, 3, 2, 3))
        | to_slit & m.Identity(2, name="step_scan")
    )
    pixel_frame = cf.CoordinateFrame(
        naxes=len(PIXEL_AXIS_NAMES),
        axes_order=tuple(range(len(PIXEL_AXIS_NAMES))),
        axes_names=PIXEL_AXIS_NAMES,
        axes_type=["PIXEL"] * len(PIXEL_AXIS_NAMES),
        unit=(u.pix,) * len(PIXEL_AXIS_NAMES),
    )
    return non_spectral, pixel_frame, _sky_and_time_frames(observer, observer.obstime, parse_time(t_ref), (1, 2))


def _is_sit_and_stare(cube):
    """
    Whether ``cube`` is of a sit-and-stare, a window with ``CDELT3 = 0``: the first sky
    model of its gWCS is named ``FIXED_SLIT_MODEL``.
    """
    transform = getattr(_root_wcs(cube), "forward_transform", None)
    return transform is not None and any(model.name == FIXED_SLIT_MODEL for model in transform)


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
    astropy's own TAN and rotation models, which ``_create_mosaic_gwcs`` uses, take one
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


def _create_mosaic_gwcs(fits_wcs, celestial, time_offsets, date_obs, observer):
    """
    The gWCS of a mosaic: its wavelength, helioprojective longitude and latitude, and
    the time of each position.

    ``fits_wcs`` is the FITS WCS of the cube, with pixel axes (wavelength, solar Y,
    solar X), ``celestial`` the celestial FITS WCS of the file, with pixel axes (solar
    X, solar Y), and ``time_offsets`` the seconds from ``date_obs`` of each (solar Y,
    solar X) position, NaN where no raster covered it.

    A mosaic has one pointing, so its sky positions are the celestial FITS WCS of the
    file, a TAN projection (Calabretta & Greisen 2002, A&A 395, 1077), written as one
    astropy model for each step::

        offset = (x, y) - crpix               Shift
        rotated = PC @ offset                 AffineTransformation2D
        (X, Y) = CDELT * rotated              Multiply
        angles about the pointing of (X, Y)   Pix2Sky_TAN
        (lon, lat) of those angles            EulerAngleRotation

    ``(X, Y)`` is where the pixel is on the plane that touches the sky sphere at the
    pointing ``CRVAL``, east and north of it. ``Pix2Sky_TAN`` gives the angle between
    the pixel and the pointing and the direction around the pointing. The rotation is
    that of ``RotateNative2Celestial``, but gives the longitude from -180 to 180
    degrees, as the other gWCS do, not from 0 to 360.
    """
    unit = u.Unit(fits_wcs.wcs.cunit[0])
    spectral, spectral_frame = _spectral_axis(
        fits_wcs.wcs.cdelt[0] * unit, fits_wcs.wcs.crval[0] * unit, fits_wcs.wcs.crpix[0]
    )
    # FITS CRPIX is 1-based and the transform takes 0-based pixels; wcslib holds the sky keywords in degrees
    crpix = (celestial.wcs.crpix - 1) * u.pix
    cdelt = (celestial.wcs.cdelt * u.deg).to(u.arcsec) / u.pix
    crval = (celestial.wcs.crval * u.deg).to(u.arcsec)
    # The five steps of the docstring. The Euler angles are those of RotateNative2Celestial for the pointing and a
    # LONPOLE of 180 degrees, the default of a TAN projection, with which Y points north.
    sky = (
        m.Shift(-crpix[0]) & m.Shift(-crpix[1])
        | m.AffineTransformation2D(celestial.wcs.get_pc() * u.pix, translation=(0, 0) * u.pix)
        | m.Multiply(cdelt[0]) & m.Multiply(cdelt[1])
        | m.Pix2Sky_TAN()
        | m.EulerAngleRotation(90 * u.deg, crval[1] - 90 * u.deg, -(90 * u.deg + crval[0]), "zxz")
    )
    # The time of the raster that covered each position. A linear lookup weighs in the next position, even at a
    # pixel center, and so is NaN before every uncovered position.
    temporal = _table_lookup(np.asarray(time_offsets, dtype=float) * u.s, name=TIME_MODEL, method="nearest")
    # Pixel inputs are (wavelength, solar Y, solar X): the sky takes (X, Y) and the time lookup (Y, X)
    forward = m.Mapping((0, 2, 1, 1, 2), name="WavelengthXYYX") | (spectral & sky & temporal)
    # The inverse ignores the time, which does not place a pixel on its own
    forward.inverse = (
        m.Mapping((0, 1, 2), n_inputs=4, name="DropTime") | (spectral.inverse & sky.inverse) | m.Mapping((0, 2, 1))
    )
    pixel_frame = cf.CoordinateFrame(
        naxes=3,
        axes_order=(0, 1, 2),
        axes_names=(WAVELENGTH_AXIS, MOSAIC_Y_AXIS, MOSAIC_X_AXIS),
        axes_type=["PIXEL"] * 3,
        unit=(u.pix,) * 3,
    )
    output_frame = cf.CompositeFrame([spectral_frame, *_sky_and_time_frames(observer, date_obs, date_obs, (1, 2))])
    return gwcs.WCS([(pixel_frame, forward), (output_frame, None)])
