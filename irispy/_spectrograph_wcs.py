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

from irispy._interpolation import _time_lookup
from irispy._wcs import _one_pass


def _sanitize_raster_wcs_tables(pc, crval, *, pc_only=False):
    """
    Replace, in place, the PC and CRVAL rows of the exposures with no pointing in the
    auxiliary table, which are all zero, from the neighboring exposures.

    Rows between good rows are interpolated linearly, and rows beyond the first or last
    good row extrapolated from the two nearest. With fewer than two good rows the others
    are NaN, as in the FITS ``-TAB`` WCS.
    """
    bad_rows = np.isclose(pc.to_value(u.pix), 0).all(axis=(1, 2))
    if not pc_only:
        # A pointing of (0, 0) is disk center, so a row is unfilled only when its PC matrix is zero too
        bad_rows &= np.isclose(crval.to_value(u.arcsec), 0).all(axis=1)
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
        stacklevel=3,
    )
    for table in (pc, crval):
        values = np.asarray(table)  # a view, so this writes to the table itself
        values[bad_rows] = make_interp_spline(good, values[good], k=1)(np.flatnonzero(bad_rows))
    return pc, crval


def _create_raster_gwcs(window_header, pc_all, crval_all, dt_all, t_ref, observer, *, sit_and_stare):
    """
    Build the raster gWCS from per-step AUX sky and timing tables.

    The spectral axis stays a 1D linear transform. The scan axis expands into
    helioprojective sky coordinates, elapsed time from ``t_ref``, and an
    explicit scan-step coordinate so the inverse stays stable for crops and
    other round-trips where sky or time alone are not unique.

    Notes
    -----
    The gWCS inverse deliberately ignores the time component (axis 3) and
    uses sky position plus the explicit scan-step coordinate to determine the
    pixel index. This is required because time may not be monotonic across
    flipped or repeated rasters. Consequently, ``world_to_array_index`` will
    return the same pixel regardless of the time value passed in the world
    tuple.
    """
    # Level 2 windows give their wavelengths in Å (CUNIT1)
    spectral_unit = u.nm
    cdelt1, crval1, crpix1 = window_header["CDELT1"] * 0.1, window_header["CRVAL1"] * 0.1, window_header["CRPIX1"]
    spectral = m.Linear1D(
        slope=cdelt1 * spectral_unit / u.pix,
        intercept=(crval1 - cdelt1 * (crpix1 - 1)) * spectral_unit,
        name="Wavelength",
    )

    # Pixel inputs are (wavelength, slit, step, scan), so lookup tables use (step, scan).
    pc_all, crval_all, dt_all = (np.swapaxes(table, 0, 1) for table in (pc_all, crval_all, dt_all))
    # Each exposure's CRVAL is its own slit position, so its reference pixel is its own step;
    # a sit-and-stare holds the step input at 0 instead (see below).
    crpix = np.zeros((*pc_all.shape[:-2], 2))
    if not sit_and_stare:
        crpix[..., 0] = np.indices(pc_all.shape[:-2])[0]
    crpix[..., 1] = window_header["CRPIX2"] - 1
    # Sit-and-stare AUX PC is an unscaled rotation, so the (virtual) step scale is the slit scale.
    cdelt3 = window_header["CDELT2"] if sit_and_stare else window_header["CDELT3"]
    celestial = _one_pass(
        VaryingCelestialTransform2D(
            cdelt=[cdelt3, window_header["CDELT2"]] * (u.arcsec / u.pix),
            # Reversing both PC axes turns the FITS (lat, lon) order into the gWCS (lon, lat) one.
            pc_table=pc_all[..., ::-1, ::-1],
            crval_table=crval_all,
            crpix_table=crpix * u.pix,
        )
    )
    celestial_inverse = _one_pass(celestial.inverse)
    sky = celestial
    if sit_and_stare:
        # The slit does not move between exposures, so the step only picks the exposure (dkist's
        # row lookup) and must not offset the pointing at the virtual step scale, however dkist
        # rounds a fractional step.
        sky = (m.Multiply(0, name="FixedSlit") & m.Identity(celestial.n_inputs - 1)) | sky
    temporal = _time_lookup(dt_all, name="Time")
    celestial_forward = m.Mapping((1, 0, 1, 2), n_inputs=3, name="StepSlitScanMapping") | sky
    sky_time = CoupledCompoundModel("&", left=celestial_forward, right=temporal, shared_inputs=2)
    non_spectral = CoupledCompoundModel("&", left=sky_time, right=m.Identity(2, name="step_scan"), shared_inputs=2)
    non_spectral.inverse = (
        m.Mapping((0, 1, 3, 4, 3, 4), n_inputs=5, name="SelectSkyStepScan")
        | (celestial_inverse & m.Identity(2, name="step_scan"))
        | m.Mapping((1, 2, 3), n_inputs=4, name="SelectSlitStepScan")
    )
    forward_transform = spectral & non_spectral

    base_time = parse_time(t_ref)
    spectral_frame = cf.SpectralFrame(
        axes_order=(0,), unit=spectral_unit, name="wavelength", axes_names=("wavelength",)
    )
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
    output_frame = cf.CompositeFrame([spectral_frame, celestial_frame, temporal_frame, step_frame, scan_frame])
    pixel_axis_names = ["dispersion axis", "spatial along slit", "raster step", "raster scan"]
    pixel_frame = cf.CoordinateFrame(
        naxes=len(pixel_axis_names),
        axes_order=tuple(range(len(pixel_axis_names))),
        axes_names=pixel_axis_names,
        axes_type=["PIXEL"] * len(pixel_axis_names),
        unit=(u.pix,) * len(pixel_axis_names),
    )
    return gwcs.WCS(
        forward_transform,
        input_frame=pixel_frame,
        output_frame=output_frame,
    )
