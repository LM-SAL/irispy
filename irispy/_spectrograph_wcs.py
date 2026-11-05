import warnings

import numpy as np
from scipy.interpolate import make_interp_spline

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf

from dkist.wcs.models import CoupledCompoundModel, VaryingCelestialTransform2D
from sunpy.coordinates.frames import Helioprojective
from sunpy.time import parse_time

from irispy._wcs import _one_pass, _root_wcs, _time_lookup

# The names other modules look for in a raster gWCS
WAVELENGTH_AXIS, SLIT_AXIS, STEP_AXIS, SCAN_AXIS = PIXEL_AXIS_NAMES = (
    "dispersion axis",
    "spatial along slit",
    "raster step",
    "raster scan",
)
FIXED_SLIT_MODEL = "FixedSlit"  # holds the slit of a sit-and-stare in place
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
    crops and other round-trips where sky or time alone are not unique.

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
    # Pixel inputs are (wavelength, slit, step, scan), so lookup tables use (step, scan).
    pc_all, crval_all, dt_all = (np.swapaxes(table, 0, 1) for table in (pc_all, crval_all, dt_all))
    # Each exposure's CRVAL is its own slit position, so its reference pixel is its own step;
    # a sit-and-stare holds the step input at 0 instead (see below).
    crpix = np.zeros((*pc_all.shape[:-2], 2))
    if not sit_and_stare:
        crpix[..., 0] = np.indices(pc_all.shape[:-2])[0]
    # The slit reference pixel of each raster comes from its own file: a file with no data has CRPIX2 = 0
    crpix[..., 1] = np.asarray(crpix2_all) - 1
    # Sit-and-stare AUX PC is an unscaled rotation, so the (virtual) step scale is the slit scale.
    cdelt3 = cdelt2 if sit_and_stare else cdelt3
    celestial = _one_pass(
        VaryingCelestialTransform2D(
            cdelt=[cdelt3, cdelt2] * (u.arcsec / u.pix),
            # Reversing both PC axes turns the FITS (lat, lon) order into the gWCS (lon, lat) one.
            pc_table=pc_all[..., ::-1, ::-1],
            crval_table=crval_all,
            crpix_table=crpix * u.pix,
        )
    )
    celestial_inverse = _one_pass(celestial.inverse)
    sky = celestial
    if sit_and_stare:
        # The slit does not move between exposures, so the step only selects the exposure (dkist's
        # row lookup). It must not shift the pointing by the virtual step scale, whichever way dkist
        # rounds a fractional step.
        sky = (m.Multiply(0, name=FIXED_SLIT_MODEL) & m.Identity(celestial.n_inputs - 1)) | sky
    temporal = _time_lookup(dt_all, name=TIME_MODEL)
    celestial_forward = m.Mapping((1, 0, 1, 2), n_inputs=3, name="StepSlitScanMapping") | sky
    sky_time = CoupledCompoundModel("&", left=celestial_forward, right=temporal, shared_inputs=2)
    non_spectral = CoupledCompoundModel("&", left=sky_time, right=m.Identity(2, name="step_scan"), shared_inputs=2)
    non_spectral.inverse = (
        m.Mapping((0, 1, 3, 4, 3, 4), n_inputs=5, name="SelectSkyStepScan")
        | (celestial_inverse & m.Identity(2, name="step_scan"))
        | m.Mapping((1, 2, 3), n_inputs=4, name="SelectSlitStepScan")
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
