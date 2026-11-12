"""
The gWCS of a raster cube.

The pixel axes are (wavelength, slit, step, scan), the reverse of the array axes
(raster, step, slit, wavelength). Forward, from pixels to world::

    pixel    wavelength      slit      step, scan
                 │            │         │  │  │
                 │            └────┬────┘  │  │
                 ▼                 ▼       ▼  ▼
             Wavelength        Celestial  Time  Identity
                 │                 │       │      │
                 ▼                 ▼       ▼      ▼
    world    wavelength        lon, lat   time  raster_step, raster_scan

``Wavelength``
    A line, from the ``CRVAL1``, ``CDELT1`` and ``CRPIX1`` of the window.
``Celestial``
    The slit of each exposure as a straight line on the sky, in lookup tables by
    (step, scan)::

        lon = lon0[step, scan] + slit * dlon[step, scan]
        lat = lat0[step, scan] + slit * dlat[step, scan]

    ``lon0`` and ``lat0`` are the position of slit pixel 0, and ``dlon`` and ``dlat``
    its change for each slit pixel, from the pointing and the ``PC`` matrix of the
    exposure. It is the line of the FITS ``-TAB`` WCS in ``cube.fits_wcs``, within 1e-3
    arcsec of the FITS celestial WCS of the exposure. A fractional step interpolates
    between the two nearest exposures.
``Time``
    A lookup table of the seconds from the first exposure, by (step, scan).
``Identity``
    The step and scan as world coordinates, which the inverse needs.

The inverse, from world to pixels, returns the ``raster_step`` and ``raster_scan`` it is
given as the step and scan pixels, and ignores the time, a 2D table that cannot be
inverted. The sky position gives only the pixel along the slit, the nearest one on
the line of that step and scan. So ``crop`` and ``world_to_pixel`` need the step and
scan, and finding a step from a sky position takes ``cube.fits_wcs``.

What the models do not show:

- A sit-and-stare (``CDELT3 = 0``) has the same models. The first model of its
  ``Celestial`` is named ``FIXED_SLIT_MODEL``, which ``_is_sit_and_stare`` looks for.
- Other modules find parts of the gWCS by name: the plot layout by the pixel axis names,
  and ``cube.time`` by ``TIME_MODEL``.
- An exposure with no pointing is NaN. Its tables hold the values of the nearest
  exposure with a pointing, and a table of its own makes it NaN: a NaN in a linear
  lookup would also make the exposure before it NaN.
- The windows of a detector band share everything after ``Wavelength``
  (``irispy.io._raster_combine._sky_time_models``): gwcs would copy it for each window.
- ``cube.fits_wcs`` holds the FITS ``-TAB`` WCS of each raster, built from the same
  auxiliary table. The pixel scales of the cube come from it.
"""

import warnings

import numpy as np
from scipy import ndimage
from scipy.interpolate import make_interp_spline

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf

from sunpy.coordinates.frames import Helioprojective
from sunpy.time import parse_time

from irispy._wcs import _root_wcs, _table_lookup

# The names other modules look for in a raster gWCS
WAVELENGTH_AXIS, SLIT_AXIS, STEP_AXIS, SCAN_AXIS = PIXEL_AXIS_NAMES = (
    "dispersion axis",
    "spatial along slit",
    "raster step",
    "raster scan",
)
FIXED_SLIT_MODEL = "FixedSlit"  # the first sky model of a sit-and-stare
TIME_MODEL = "Time"  # the lookup of the exposure times
MOSAIC_Y_AXIS, MOSAIC_X_AXIS = "solar Y", "solar X"  # the pixel axes of a mosaic gWCS after its dispersion axis


def _sanitize_raster_wcs_tables(pc, crval):
    """
    Replace, in place, the PC and CRVAL rows of the exposures with no pointing in the
    auxiliary table, from the neighboring exposures. Such an exposure has a PC matrix of
    zeros, which no real rotation has.

    Rows between good rows are interpolated linearly, and rows beyond the first or last
    good row extrapolated from the two nearest. With fewer than two good rows the others
    are NaN, as in the FITS ``-TAB`` WCS.
    """
    bad_rows = np.isclose(pc.to_value(u.pix), 0).all(axis=(1, 2))
    if not bad_rows.any():
        return pc, crval
    good = np.flatnonzero(~bad_rows)
    if good.size < 2:
        # A file with no data, or with one exposure, has no pointing for the other exposures
        pc[bad_rows] = np.eye(2) * u.pix
        crval[bad_rows] = np.nan * u.arcsec
        return pc, crval
    warnings.warn(
        f"Found {bad_rows.sum()} step(s) with no pointing in the auxiliary table. Interpolating from neighboring steps.",
        UserWarning,
        stacklevel=4,  # the caller of read_spectrograph_lvl2, which calls _read_raster_file
    )
    for table in (pc, crval):
        values = np.asarray(table)  # a view, so this writes to the table itself
        values[bad_rows] = make_interp_spline(good, values[good], k=1)(np.flatnonzero(bad_rows))
    return pc, crval


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

    The spectral axis stays a 1D linear transform.
    """
    # Level 2 windows give their wavelengths in Å (CUNIT1)
    spectral, spectral_frame = _spectral_axis(
        window_header["CDELT1"] * u.AA, window_header["CRVAL1"] * u.AA, window_header["CRPIX1"]
    )
    # Steps, not a model, which gwcs would deep-copy: non_spectral is shared by windows and nothing changes it
    return gwcs.WCS([(pixel_frame, spectral & non_spectral), (cf.CompositeFrame([spectral_frame, *frames]), None)])


def _create_raster_non_spectral(pc_all, crval_all, crpix2_all, dt_all, t_ref, observer, *, cdelt2, cdelt3):
    """
    Build the part of the raster gWCS after the spectral axis from per-step AUX sky
    and timing tables: the transform, the pixel frame and the other output frames.

    The scan axis expands into helioprojective sky coordinates, elapsed time from
    ``t_ref``, and an explicit scan-step coordinate so the inverse stays stable for
    crops and other round-trips where sky or time alone are not unique. The module
    docstring has the models.

    Notes
    -----
    The gWCS inverse deliberately ignores the time component (axis 3) and
    uses sky position plus the explicit scan-step coordinate to determine the
    pixel index. This is required because time may not be monotonic across
    flipped or repeated rasters. Consequently, ``world_to_array_index`` will
    return the same pixel regardless of the time value passed in the world
    tuple.
    """
    sit_and_stare = cdelt3 == 0
    # Tables by (step, scan): the pixel inputs are (wavelength, slit, step, scan)
    pc, crval, dt_all = (np.swapaxes(table, 0, 1) for table in (pc_all, crval_all, dt_all))
    pc = pc.to_value(u.pix)
    # The change of the sky position for each slit pixel: PC3_2 and PC2_2 times the plate scales. Sit-and-stare AUX
    # PC is an unscaled rotation, so its step scale is the slit scale.
    dlon = pc[..., 1, 0] * (cdelt2 if sit_and_stare else cdelt3) * u.arcsec / u.pix
    dlat = pc[..., 0, 0] * cdelt2 * u.arcsec / u.pix
    # The position of slit pixel 0, from the pointing at the slit reference pixel. That of each raster comes from
    # its own file: a file with no data has CRPIX2 = 0.
    reference = (np.asarray(crpix2_all) - 1) * u.pix
    lon0 = crval[..., 0] - dlon * reference
    lat0 = crval[..., 1] - dlat * reference
    # A linear lookup also gives NaN at the exposure before one that is NaN. So an exposure with no pointing takes
    # the tables of the nearest exposure with one, and gets its NaN from a nearest lookup of its own.
    missing = np.isnan(lon0) | np.isnan(lat0)
    if missing.any() and not missing.all():
        nearest = tuple(ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True))
        lon0, lat0, dlon, dlat = (table[nearest] for table in (lon0, lat0, dlon, dlat))

    # Pixel inputs are (slit, step, scan)
    slit, row = m.Mapping((0,), n_inputs=3), m.Mapping((1, 2), n_inputs=3)
    lon = (row | _table_lookup(lon0, name="lon0")) + slit * (row | _table_lookup(dlon, name="dlon"))
    lat = (row | _table_lookup(lat0, name="lat0")) + slit * (row | _table_lookup(dlat, name="dlat"))
    if missing.any():
        no_pointing = np.where(missing, np.nan, 0) * u.arcsec
        lon = lon + (row | _table_lookup(no_pointing, method="nearest"))
        lat = lat + (row | _table_lookup(no_pointing, method="nearest"))
    # A sit-and-stare has no model of its own, so the name tells _is_sit_and_stare
    name = FIXED_SLIT_MODEL if sit_and_stare else "SlitStepScan"
    sky = m.Mapping((0, 1, 2, 0, 1, 2), name=name) | lon & lat
    temporal = _table_lookup(dt_all, name=TIME_MODEL)
    # (slit, step, scan) to (lon, lat, time, step, scan)
    non_spectral = m.Mapping((0, 1, 2, 1, 2, 1, 2)) | sky & temporal & m.Identity(2, name="step_scan")

    # World to pixel: the slit pixel nearest the sky position, on the line of the step and scan given
    norm = dlon**2 + dlat**2
    to_slit_lon, to_slit_lat = dlon / norm, dlat / norm
    offset = to_slit_lon * lon0 + to_slit_lat * lat0
    lon_in, lat_in, row = m.Mapping((0,), n_inputs=4), m.Mapping((1,), n_inputs=4), m.Mapping((2, 3), n_inputs=4)
    to_slit = (
        lon_in * (row | _table_lookup(to_slit_lon))
        + lat_in * (row | _table_lookup(to_slit_lat))
        - (row | _table_lookup(offset))
    )
    # (lon, lat, time, step, scan) to (slit, step, scan)
    non_spectral.inverse = m.Mapping((0, 1, 3, 4, 3, 4), n_inputs=5, name="SelectSkyStepScan") | to_slit & m.Identity(
        2, name="step_scan"
    )
    base_time = parse_time(t_ref)
    celestial_frame = cf.CelestialFrame(
        axes_order=(1, 2),
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer=observer, obstime=observer.obstime),
        axis_physical_types=["custom:pos.helioprojective.lon", "custom:pos.helioprojective.lat"],
        axes_names=("helioprojective longitude", "helioprojective latitude"),
    )
    temporal_frame = cf.TemporalFrame(base_time, unit=(u.s,), axes_order=(3,), axes_names=("Time (UTC)",))
    step_frame = cf.CoordinateFrame(
        naxes=1,
        axes_order=(4,),
        axes_names=("raster_step",),
        axes_type=("STEP",),
        unit=(u.pix,),
        name="step",
    )
    scan_frame = cf.CoordinateFrame(
        naxes=1,
        axes_order=(5,),
        axes_names=("raster_scan",),
        axes_type=("SCAN",),
        unit=(u.pix,),
        name="scan",
    )
    pixel_frame = cf.CoordinateFrame(
        naxes=len(PIXEL_AXIS_NAMES),
        axes_order=tuple(range(len(PIXEL_AXIS_NAMES))),
        axes_names=PIXEL_AXIS_NAMES,
        axes_type=["PIXEL"] * len(PIXEL_AXIS_NAMES),
        unit=(u.pix,) * len(PIXEL_AXIS_NAMES),
    )
    return non_spectral, pixel_frame, (celestial_frame, temporal_frame, step_frame, scan_frame)


def _is_sit_and_stare(cube):
    """
    Whether ``cube`` is of a sit-and-stare, whose gWCS holds the slit at one position
    (the fixed-slit model of a window with ``CDELT3 = 0``).
    """
    transform = getattr(_root_wcs(cube), "forward_transform", None)
    return transform is not None and any(model.name == FIXED_SLIT_MODEL for model in transform)
