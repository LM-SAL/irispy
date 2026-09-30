import warnings

import numpy as np
from scipy.interpolate import make_interp_spline

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.modeling.math_functions import RintUfunc, SubtractUfunc
from astropy.time import Time
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from dkist.wcs.models import (
    AsymmetricMapping,
    BaseVaryingCelestialTransform,
    CoupledCompoundModel,
    VaryingCelestialTransform,
    VaryingCelestialTransform2D,
)
from sunpy.coordinates.frames import Helioprojective
from sunpy.time import parse_time

from irispy._interpolation import _time_lookup
from irispy.utils.constants import SLIT_WIDTH


def _wrap_longitude():
    """
    Wrap a longitude into [-180, 180) degrees, the range of a helioprojective ``Tx``.

    The celestial rotation inside dkist's varying transforms returns longitudes in
    [0, 360) degrees, so without this an exposure east of disk centre is 360 degrees
    too high in the low-level (``*_values``) API; ``SkyCoord`` wraps by itself. This is
    ``lon - 360 * rint(lon / 360)``, from standard models so the gWCS still serialises to
    ASDF, and unit-agnostic because dkist's transform returns degrees as a Quantity.
    """
    turns = m.Multiply(1 / 360) | RintUfunc() | m.Multiply(360)
    return m.Mapping((0, 0)) | (m.Identity(1) & turns) | SubtractUfunc()


def _raster_crop_bounds(cube, points, wcs):
    """
    Bound partial raster coordinates using the measured time and pointing tables.

    Each exposure contributes its own sky bounds. Repeated matches are retained in one
    bounding slice, which can also contain intervening unmatched data.
    """
    if wcs is not cube.wcs.low_level_wcs:
        return NotImplemented
    low = wcs if isinstance(wcs, SlicedLowLevelWCS) else SlicedLowLevelWCS(wcs, Ellipsis)
    root = low._wcs
    if not isinstance(root, gwcs.WCS) or "raster_step" not in root.world_axis_names:
        return NotImplemented
    if not any(value is None for point in points for value in point):
        return NotImplemented

    bounds = [None] * root.world_n_dim
    for index, column in zip(low._world_keep, zip(*points, strict=True), strict=True):
        values = np.array([value for value in column if value is not None])
        if values.size:
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                msg = "Crop coordinates must be finite scalars."
                raise ValueError(msg)
            bounds[index] = values.min(), values.max()
    if all(bound is None for bound in bounds[1:]):
        return NotImplemented
    if (bounds[1] is None) != (bounds[2] is None):
        msg = "Both celestial components are required for a sky crop."
        raise ValueError(msg)

    lengths = iter(cube.shape[::-1])
    pixel_ranges = [
        np.arange(next(lengths)) + (item.start or 0) if isinstance(item, slice) else np.array([item])
        for item in low._slices_pixel
    ]
    wavelength, slit, steps = pixel_ranges[:3]
    scans = pixel_ranges[3] if root.pixel_n_dim == 4 else np.array([0])
    step_grid, scan_grid = np.meshgrid(steps, scans)
    selected = np.ones(step_grid.shape, dtype=bool)
    for index, grid in ((4, step_grid), (5, scan_grid)):
        if index < len(bounds) and bounds[index] is not None:
            start, stop = _raster_crop_pixel_bounds(bounds[index])
            selected &= (grid >= start) & (grid < stop)
    if bounds[3] is not None:
        temporal_inputs = (step_grid * u.pix, scan_grid * u.pix)[: root.pixel_n_dim - 2]
        times = root.forward_transform["Time"](*temporal_inputs).to_value(u.s)
        # Time conversion can introduce sub-microsecond floating-point differences.
        selected &= (times >= bounds[3][0] - 1e-7) & (times <= bounds[3][1] + 1e-7)

    wavelength_bounds = wavelength[0], wavelength[-1] + 1
    if bounds[0] is not None:
        spectral = root.forward_transform["Wavelength"]
        wavelength_bounds = _raster_crop_pixel_bounds(
            spectral.inverse(np.array(bounds[0]) * u.Unit(root.world_axis_units[0])).to_value(u.pix)
        )
    matches = []
    if bounds[1] is not None:
        celestial = next(model for model in root.forward_transform if isinstance(model, BaseVaryingCelestialTransform))
        # Half an exposure's across-slit extent, in step pixels. A sit-and-stare exposure covers the slit
        # width, whatever its (virtual, along-slit) step scale; a raster step covers at least one step.
        step_scale = abs(celestial.cdelt.quantity[0].to_value(u.arcsec / u.pix))
        half_width = SLIT_WIDTH.to_value(u.arcsec) / step_scale / 2
        if not (cube.meta or {}).get("sit_and_stare", False):
            half_width = max(half_width, 0.5)
        lon, lat = np.meshgrid(bounds[1], bounds[2])
        lon = (lon.ravel() * u.Unit(root.world_axis_units[1])).to_value(u.deg)
        lat = (lat.ravel() * u.Unit(root.world_axis_units[2])).to_value(u.deg)
    # ponytail: visit each exposure; batch the transforms if large rasters make this slow.
    for step, scan in zip(step_grid[selected], scan_grid[selected], strict=True):
        slit_bounds = slit[0], slit[-1] + 1
        if bounds[1] is not None:
            lookup = (step, scan)[: root.pixel_n_dim - 2]
            x, y = celestial.transform_at_index(lookup).inverse(lon, lat)
            if not np.all(np.isfinite([x, y])):
                continue
            if not (x.min() < step + half_width and x.max() > step - half_width):
                continue
            start, stop = _raster_crop_pixel_bounds(y)
            slit_bounds = max(start, slit[0]), min(stop, slit[-1] + 1)
            if slit_bounds[0] >= slit_bounds[1]:
                continue
        matches.append((slit_bounds[0], slit_bounds[1], step, scan))
    if not matches:
        msg = "No raster pixels match the crop coordinates."
        raise ValueError(msg)
    matches = np.array(matches)
    starts = [wavelength_bounds[0], matches[:, 0].min(), matches[:, 2].min(), matches[:, 3].min()]
    stops = [wavelength_bounds[1], matches[:, 1].max(), matches[:, 2].max() + 1, matches[:, 3].max() + 1]
    constrained = root.axis_correlation_matrix[[bound is not None for bound in bounds]].any(axis=0)
    item = []
    for axis in low._pixel_keep:
        if not constrained[axis]:
            item.append(None)
            continue
        start = max(starts[axis] - pixel_ranges[axis][0], 0)
        stop = min(stops[axis] - pixel_ranges[axis][0], len(pixel_ranges[axis]))
        if start >= stop:
            msg = "No raster pixels match the crop coordinates."
            raise ValueError(msg)
        item.append((int(start), int(stop) - 1))
    return tuple(item[::-1])


def _raster_crop_pixel_bounds(values):
    """
    Round pixel bounds with the same pixel-edge convention as NDCube.crop.
    """
    start = int(np.floor(np.min(values) + 0.5))
    stop = int(np.ceil(np.max(values) - 0.5)) + 1
    return start, max(start + 1, stop)


def _interpolate_bad_rows(tables, bad_rows, problem):
    """
    Replace the ``bad_rows`` (along axis 0) of each table in place, linearly.

    Rows between good rows are interpolated, and rows beyond the first or last good row
    are extrapolated from the two nearest good rows.
    """
    if not bad_rows.any():
        return
    if bad_rows.all():
        msg = f"Every exposure has {problem} in the AUX data."
        raise ValueError(msg)
    warnings.warn(
        f"Found {bad_rows.sum()} step(s) with {problem} in raster aux data. Interpolating from neighbouring steps.",
        UserWarning,
        stacklevel=4,
    )
    good = np.flatnonzero(~bad_rows)
    for table in tables:
        values = np.asarray(table)  # A view, so this writes to the table itself.
        values[bad_rows] = make_interp_spline(good, values[good], k=min(1, good.size - 1))(np.flatnonzero(bad_rows))


def _raster_wcs_bad_row_mask(pc, crval, *, pc_only=False):
    """
    Return AUX rows whose PC or CRVAL table entries are unusable.
    """
    pc_values, crval_values = pc.to_value(u.pix), crval.to_value(u.arcsec)
    bad_rows = np.isclose(pc_values, 0).all(axis=(1, 2))
    if not pc_only:
        # Require BOTH pc and crval to be all-zero: crval=(0,0) alone is valid for
        # disk-centre pointings.  A truly unfilled row will have an all-zero PC matrix.
        bad_rows &= np.isclose(crval_values, 0).all(axis=1)
    # A non-finite entry is never usable, whatever the rest of the row holds.
    return bad_rows | ~np.isfinite(pc_values).all(axis=(1, 2)) | ~np.isfinite(crval_values).all(axis=1)


def _sanitize_raster_wcs_tables(pc, crval, *, pc_only=False):
    """
    Replace unusable (all-zero or non-finite) PC/crval rows, in place, by interpolating
    from neighbouring rows.
    """
    bad_rows = _raster_wcs_bad_row_mask(pc, crval, pc_only=pc_only)
    _interpolate_bad_rows((pc, crval), bad_rows, "all-zero or non-finite WCS tables")
    return pc, crval


def _sanitize_raster_times(start, offsets, *, fallback_to_start=False):
    """
    Return ``start + offsets``, with the times of non-finite offsets interpolated from
    neighbouring rows.

    With ``fallback_to_start``, an all-non-finite ``offsets`` gives ``start`` with a
    warning, instead of raising.
    """
    bad_rows = ~np.isfinite(offsets)
    times = start + np.where(bad_rows, 0, offsets)
    if fallback_to_start and bad_rows.all():
        warnings.warn(
            "Every exposure time in the AUX data is non-finite. Using the planned start times.",
            UserWarning,
            stacklevel=2,
        )
        return times
    # Interpolating both parts of the two-part Julian date keeps full precision.
    jd = np.stack((times.jd1, times.jd2), axis=-1)
    _interpolate_bad_rows((jd,), bad_rows, "non-finite times")
    times[bad_rows] = Time(jd[bad_rows, 0], jd[bad_rows, 1], format="jd", scale=times.scale)
    return times


def _validate_raster_wcs_inputs(window_header, pc_all, crval_all, dt_all):
    """
    Validate the cheap table/header invariants needed to build the raster gWCS.
    """
    for key in ("CDELT1", "CRVAL1", "CDELT2", "CDELT3", "CRPIX2"):
        try:
            value = float(window_header[key])
        except KeyError as e:
            msg = f"Missing WCS header key {key!r}."
            raise ValueError(msg) from e
        except (TypeError, ValueError) as e:
            msg = f"WCS header key {key!r} must be numeric."
            raise ValueError(msg) from e
        if not np.isfinite(value):
            msg = f"WCS header key {key!r} must be finite."
            raise ValueError(msg)

    # Extrapolating from extreme finite rows can still overflow.
    for name, values in (("PC", pc_all), ("CRVAL", crval_all), ("time-offset", dt_all)):
        if not np.all(np.isfinite(values)):
            msg = f"Raster {name} table must contain only finite values."
            raise ValueError(msg)


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
    cdelt1, crval1 = window_header["CDELT1"], window_header["CRVAL1"]
    if window_header.get("CUNIT1", "").lower() == "angstrom":
        spectral_unit, cdelt1, crval1 = u.nm, cdelt1 * 0.1, crval1 * 0.1
    else:
        spectral_unit = u.Unit(window_header.get("CUNIT1", "nm"))
    crpix1 = window_header.get("CRPIX1", 1.0)
    spectral = m.Linear1D(
        slope=cdelt1 * spectral_unit / u.pix,
        intercept=(crval1 - cdelt1 * (crpix1 - 1)) * spectral_unit,
        name="Wavelength",
    )

    separate_raster_axis = pc_all.ndim == 4
    if separate_raster_axis:
        # Pixel inputs are (wavelength, slit, step, scan), so lookup tables use (step, scan).
        pc_all, crval_all, dt_all = (np.swapaxes(table, 0, 1) for table in (pc_all, crval_all, dt_all))
    # Each exposure's CRVAL is its own slit position, so its reference pixel is its own step.
    crpix = np.empty((*pc_all.shape[:-2], 2))
    crpix[..., 0] = np.indices(pc_all.shape[:-2])[0]
    crpix[..., 1] = window_header["CRPIX2"] - 1
    # dkist picks each exposure's row with np.round, which takes the far edge (pixel n - 0.5)
    # to the missing row n when n is even. Repeating the last row keeps that edge finite.
    lookup_axes = pc_all.ndim - 2
    pc_all, crval_all, crpix = (
        np.pad(table, [(0, 1)] * lookup_axes + [(0, 0)] * (table.ndim - lookup_axes), mode="edge")
        for table in (pc_all, crval_all, crpix)
    )
    # Sit-and-stare AUX PC is an unscaled rotation, so the (virtual) step scale is the slit scale.
    cdelt3 = window_header["CDELT2"] if sit_and_stare else window_header["CDELT3"]
    celestial = (VaryingCelestialTransform2D if separate_raster_axis else VaryingCelestialTransform)(
        cdelt=[cdelt3, window_header["CDELT2"]] * (u.arcsec / u.pix),
        # Reversing both PC axes turns the FITS (lat, lon) order into the gWCS (lon, lat) one.
        pc_table=pc_all[..., ::-1, ::-1],
        crval_table=crval_all,
        crpix_table=crpix * u.pix,
    )
    sky = celestial | (_wrap_longitude() & m.Identity(1))
    if sit_and_stare:
        # The slit does not move between exposures, so a fractional step must not ramp the
        # pointing at the virtual step scale: snap it to the nearest exposure instead.
        sky = (RintUfunc(name="NearestExposure") & m.Identity(celestial.n_inputs - 1)) | sky
    temporal = _time_lookup(dt_all, name="Time")
    if separate_raster_axis:
        celestial_forward = m.Mapping((1, 0, 1, 2), n_inputs=3, name="StepSlitScanMapping") | sky
        sky_time = CoupledCompoundModel("&", left=celestial_forward, right=temporal, shared_inputs=2)
        non_spectral = CoupledCompoundModel("&", left=sky_time, right=m.Identity(2, name="step_scan"), shared_inputs=2)
        non_spectral.inverse = (
            m.Mapping((0, 1, 3, 4, 3, 4), n_inputs=5, name="SelectSkyStepScan")
            | (celestial.inverse & m.Identity(2, name="step_scan"))
            | m.Mapping((1, 2, 3), n_inputs=4, name="SelectSlitStepScan")
        )
    else:
        sky_time = CoupledCompoundModel("&", left=sky, right=temporal, shared_inputs=1)
        non_spectral = AsymmetricMapping([1, 0, 1, 1], [1, 0], name="StepSlitMapping") | (
            sky_time & m.Identity(1, name="step")
        )
        non_spectral.inverse = (
            m.Mapping((0, 1, 3, 3), n_inputs=4, name="SelectSkyAndExplicitStep")
            | (celestial.inverse & m.Identity(1, name="step"))
            | m.Mapping((1, 2), n_inputs=3, name="SelectSlitStep")
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
    temporal_frame = cf.TemporalFrame(base_time, unit=(u.s,), axes_order=(3,), axes_names=("Seconds from Start (s)",))
    step_frame = cf.CoordinateFrame(
        naxes=1,
        axes_order=(4,),
        axes_names=("raster_step",),
        axes_type=("STEP",),
        unit=(u.pix,),
        name="step",
    )
    frames = [spectral_frame, celestial_frame, temporal_frame, step_frame]
    if separate_raster_axis:
        scan_frame = cf.CoordinateFrame(
            naxes=1,
            axes_order=(5,),
            axes_names=("raster_scan",),
            axes_type=("SCAN",),
            unit=(u.pix,),
            name="scan",
        )
        frames.append(scan_frame)
    output_frame = cf.CompositeFrame(frames)
    pixel_axis_names = ["dispersion axis", "spatial along slit", "raster step"]
    if separate_raster_axis:
        pixel_axis_names.append("raster scan")
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
