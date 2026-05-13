"""
Red-blue asymmetry utilities for IRIS spectrogram cubes.
"""

import warnings
from enum import IntEnum
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

from ndcube.wcs.tools import unwrap_wcs_to_fitswcs

from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils._spectral import drop_extra_coords_dependent_on_axis, make_map_cube, make_spatial_template

__all__ = ["RBAQualityFlag", "calculate_red_blue_asymmetry"]


class RBAQualityFlag(IntEnum):
    """
    Quality flags for the per-pixel RBA computation.
    """

    OK = (0, "ok")
    NO_FINITE_DATA = (1, "no finite data")
    PEAK_AT_EDGE = (2, "peak at spectral edge")
    TOO_FEW_POINTS = (3, "too few finite points")
    INTERP_FAILED = (4, "interpolation failed")
    PEAK_IS_ZERO = (5, "peak is zero or non-finite")
    INCOMPLETE_WINGS = (6, "incomplete red or blue wing coverage")
    LOW_SIGNAL = (7, "below min_intensity")
    SATURATED = (8, "above saturation_limit")

    def __new__(cls, value, description):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj


def _make_velocity_wcs(base_wcs, array_shape, velocity_axis, velocity_grid):
    fits_wcs = base_wcs if hasattr(base_wcs, "to_header") else unwrap_wcs_to_fitswcs(base_wcs)[0]
    header = fits_wcs.to_header()
    naxis = len(array_shape)
    wcs_axis = naxis - 1 - velocity_axis
    fits_axis = wcs_axis + 1

    header["NAXIS"] = naxis
    for array_axis, length in enumerate(array_shape):
        header[f"NAXIS{naxis - array_axis}"] = int(length)

    cdelt = float(np.nanmean(np.diff(velocity_grid))) if velocity_grid.size > 1 else 1.0
    header[f"CTYPE{fits_axis}"] = "VELO"
    header[f"CUNIT{fits_axis}"] = "km/s"
    header[f"CRPIX{fits_axis}"] = 1.0
    header[f"CRVAL{fits_axis}"] = float(velocity_grid[0])
    header[f"CDELT{fits_axis}"] = cdelt
    header.pop(f"CNAME{fits_axis}", None)

    for other_axis in range(1, naxis + 1):
        for prefix in ("CD",):
            header.pop(f"{prefix}{fits_axis}_{other_axis}", None)
            header.pop(f"{prefix}{other_axis}_{fits_axis}", None)
        if other_axis != fits_axis:
            header[f"PC{fits_axis}_{other_axis}"] = 0.0
            header[f"PC{other_axis}_{fits_axis}"] = 0.0
    header[f"PC{fits_axis}_{fits_axis}"] = 1.0

    for key in list(header):
        if key.startswith((f"PV{fits_axis}_", f"PS{fits_axis}_")):
            header.pop(key)

    return WCS(header)


def _make_profile_cube(cube, *, data, velocity_grid, wavelength_axis, meta, uncertainty=None, mask=None):
    return SpectrogramCube(
        data,
        wcs=_make_velocity_wcs(cube.wcs, data.shape, wavelength_axis, velocity_grid),
        uncertainty=uncertainty,
        unit=cube.unit,
        meta=meta,
        mask=mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=False),
    )


@dataclass
class _PixelCoreResult:
    quality: RBAQualityFlag
    rba: float = np.nan
    red_wing: float = np.nan
    blue_wing: float = np.nan
    peak_intensity: float = np.nan
    peak_velocity: float = np.nan
    red_blue_error: float = np.nan
    red_wing_error: float = np.nan
    blue_wing_error: float = np.nan


@dataclass
class _PixelProfileDetail:
    interp_profile: np.ndarray
    interp_error: np.ndarray | None = None


@dataclass
class _RBAContext:
    velocity: np.ndarray
    interp_velocity: np.ndarray
    center_on_peak: bool
    fit_window: float
    d_velocity: float
    degree: int
    red_mask: np.ndarray
    blue_mask: np.ndarray
    min_wing_coverage: float


def _prepare_data(cube, wavelengths, wavelength_axis, continuum_windows, uncertainty):
    """
    Extract data/errors, mask negatives, subtract continuum, move spectral axis to -1.
    """
    data = np.asarray(cube.data, dtype=float)
    if cube.mask is not None:
        data = np.where(cube.mask, np.nan, data)
    data = np.where(data < 0, np.nan, data)

    if uncertainty is not None:
        errors = u.Quantity(uncertainty, cube.unit).to_value(cube.unit)
    elif cube.uncertainty is not None:
        errors = np.asarray(cube.uncertainty.array, dtype=float)
    else:
        errors = None
    if errors is not None and cube.mask is not None:
        errors = np.where(cube.mask, np.nan, errors)

    data = np.moveaxis(data, wavelength_axis, -1)
    if errors is not None:
        errors = np.moveaxis(errors, wavelength_axis, -1)

    if continuum_windows is not None:
        windows = u.Quantity(continuum_windows).to(wavelengths.unit)
        if windows.shape == (2,):
            windows = windows[np.newaxis, :]
        if windows.ndim != 2 or windows.shape[1] != 2:
            msg = "continuum_windows must have shape (2,) or (n, 2)"
            raise ValueError(msg)
        continuum_mask = np.zeros(wavelengths.shape, dtype=bool)
        for low, high in windows:
            continuum_mask |= (wavelengths >= low) & (wavelengths <= high)
        if not continuum_mask.any():
            msg = "No wavelength points found within continuum_windows"
            raise ValueError(msg)
        continuum_values = np.nanmean(data[..., continuum_mask], axis=-1)
        data = data - continuum_values[..., np.newaxis]
        if errors is not None:
            n_finite = np.isfinite(errors[..., continuum_mask]).sum(axis=-1)
            continuum_errors = np.sqrt(np.nansum(errors[..., continuum_mask] ** 2, axis=-1))
            continuum_errors = np.where(n_finite > 0, continuum_errors / n_finite, np.nan)
            errors = np.sqrt(errors**2 + continuum_errors[..., np.newaxis] ** 2)

    return data, errors


def _window_profile(profile, profile_error, velocity, peak_index, *, center_on_peak, fit_window, d_velocity):
    if center_on_peak:
        if peak_index == 0 or peak_index == profile.size - 1:
            return None, None, None, RBAQualityFlag.PEAK_AT_EDGE
        window_pixels = profile.size if d_velocity == 0 else int(fit_window / abs(d_velocity))
        low_idx = max(0, peak_index - window_pixels)
        high_idx = min(profile.size, peak_index + window_pixels)
        fit_slice = slice(low_idx, high_idx)
        shifted_velocity = velocity[fit_slice] - velocity[peak_index]
        profile = profile[fit_slice]
        if profile_error is not None:
            profile_error = profile_error[fit_slice]
    else:
        fit_mask = np.abs(velocity) <= fit_window
        shifted_velocity = velocity[fit_mask]
        profile = profile[fit_mask]
        if profile_error is not None:
            profile_error = profile_error[fit_mask]

    return shifted_velocity, profile, profile_error, RBAQualityFlag.OK


def _interpolate_profile_and_error(shifted_velocity, profile, profile_error, interp_velocity, degree):
    finite = np.isfinite(shifted_velocity) & np.isfinite(profile)
    min_points = degree + 1
    if finite.sum() < min_points:
        return None, None, RBAQualityFlag.TOO_FEW_POINTS

    sv, sp = shifted_velocity[finite], profile[finite]
    order = np.argsort(sv)
    ordered_velocity, ordered_profile = sv[order], sp[order]

    try:
        interp_profile = make_interp_spline(ordered_velocity, ordered_profile, k=degree)(
            interp_velocity, extrapolate=False
        )
    except ValueError:
        return None, None, RBAQualityFlag.INTERP_FAILED

    interp_error = None
    if profile_error is not None:
        ordered_error = profile_error[finite][order]
        finite_error = np.isfinite(ordered_error)
        if finite_error.sum() >= min_points:
            try:
                interp_error = make_interp_spline(
                    ordered_velocity[finite_error], ordered_error[finite_error], k=degree
                )(interp_velocity, extrapolate=False)
                interp_error = np.where(interp_error >= 0, interp_error, np.nan)
            except ValueError:
                return None, None, RBAQualityFlag.INTERP_FAILED

    return interp_profile, interp_error, RBAQualityFlag.OK


def _compute_wings_and_peak(interp_profile, red_mask, blue_mask, min_wing_coverage):
    red_finite = red_mask & np.isfinite(interp_profile)
    blue_finite = blue_mask & np.isfinite(interp_profile)
    if red_finite.sum() < min_wing_coverage * red_mask.sum() or blue_finite.sum() < min_wing_coverage * blue_mask.sum():
        return np.nan, np.nan, np.nan, RBAQualityFlag.INCOMPLETE_WINGS

    red_intensity = np.nanmean(interp_profile[red_finite]) if red_finite.any() else np.nan
    blue_intensity = np.nanmean(interp_profile[blue_finite]) if blue_finite.any() else np.nan
    peak = np.nanmax(interp_profile)
    if not np.isfinite(peak) or np.isclose(peak, 0):
        return np.nan, np.nan, np.nan, RBAQualityFlag.PEAK_IS_ZERO

    return red_intensity, blue_intensity, peak, RBAQualityFlag.OK


def _propagate_rba_error(red_intensity, blue_intensity, peak, interp_error, red_mask, blue_mask, interp_profile):
    """Propagate uncertainties through ``(I_R - I_B) / I_p``."""
    if interp_error is None:
        return np.nan, np.nan, np.nan

    n_red = np.isfinite(interp_error[red_mask]).sum()
    n_blue = np.isfinite(interp_error[blue_mask]).sum()
    red_err = np.sqrt(np.nansum(interp_error[red_mask] ** 2)) / n_red if n_red > 0 else np.nan
    blue_err = np.sqrt(np.nansum(interp_error[blue_mask] ** 2)) / n_blue if n_blue > 0 else np.nan
    peak_err = interp_error[np.nanargmax(interp_profile)]
    numerator = red_intensity - blue_intensity
    num_err = np.sqrt(red_err**2 + blue_err**2)

    rba_error = np.nan
    numerator_is_zero = np.isclose(numerator, 0)
    if np.isfinite(num_err) and (numerator_is_zero or np.isfinite(peak_err)):
        variance = (num_err / peak) ** 2
        if not numerator_is_zero:
            variance += (numerator * peak_err / peak**2) ** 2
        rba_error = np.sqrt(variance)

    return rba_error, red_err, blue_err


def _process_pixel(profile, profile_error, ctx: _RBAContext):
    """
    Compute RBA for a single spatial pixel.

    Returns (``_PixelCoreResult``, ``_PixelProfileDetail | None``).
    """
    finite_profile = np.isfinite(profile)
    if not finite_profile.any():
        return _PixelCoreResult(quality=RBAQualityFlag.NO_FINITE_DATA), None

    peak_index = np.nanargmax(np.where(finite_profile, profile, np.nan))
    peak_velocity = ctx.velocity[peak_index]

    shifted_velocity, profile, profile_error, quality = _window_profile(
        profile,
        profile_error,
        ctx.velocity,
        peak_index,
        center_on_peak=ctx.center_on_peak,
        fit_window=ctx.fit_window,
        d_velocity=ctx.d_velocity,
    )
    if quality is not RBAQualityFlag.OK:
        return _PixelCoreResult(quality=quality, peak_velocity=peak_velocity), None

    interp_profile, interp_error, interp_quality = _interpolate_profile_and_error(
        shifted_velocity, profile, profile_error, ctx.interp_velocity, ctx.degree
    )
    detail = _PixelProfileDetail(interp_profile, interp_error) if interp_profile is not None else None

    if interp_quality is not RBAQualityFlag.OK:
        return _PixelCoreResult(quality=interp_quality, peak_velocity=peak_velocity), detail

    red_intensity, blue_intensity, peak, quality = _compute_wings_and_peak(
        interp_profile, ctx.red_mask, ctx.blue_mask, ctx.min_wing_coverage
    )
    if quality is not RBAQualityFlag.OK:
        return _PixelCoreResult(quality=quality, peak_velocity=peak_velocity), detail

    rba = (red_intensity - blue_intensity) / peak
    rba_error, red_err, blue_err = _propagate_rba_error(
        red_intensity, blue_intensity, peak, interp_error, ctx.red_mask, ctx.blue_mask, interp_profile
    )

    core = _PixelCoreResult(
        quality=RBAQualityFlag.OK,
        rba=rba,
        red_wing=red_intensity,
        blue_wing=blue_intensity,
        peak_intensity=peak,
        peak_velocity=peak_velocity,
        red_blue_error=rba_error,
        red_wing_error=red_err,
        blue_wing_error=blue_err,
    )
    return core, detail


def calculate_red_blue_asymmetry(
    cube,
    *,
    rest_wavelength=None,
    velocity_range=(50, 150) * u.km / u.s,
    velocity_window=None,
    fit_window=None,
    dv=10 * u.km / u.s,
    center_on_peak=True,
    continuum_windows=None,
    uncertainty=None,
    degree=3,
    min_intensity=None,
    saturation_limit=None,
    return_profiles=True,
):
    """
    Calculate red-blue asymmetry maps from a spectrogram cube.

    The asymmetry is computed for each spatial pixel as
    :math:`(I_R - I_B) / I_p`, where ``I_R`` and ``I_B`` are mean intensities
    in matching red and blue velocity ranges and ``I_p`` is the interpolated
    peak intensity.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Input spectrogram cube.
    rest_wavelength : `astropy.units.Quantity`, optional
        Rest wavelength used to convert the spectral axis to Doppler velocity.
        If omitted, read from ``cube.meta.rest_wavelength`` (requires an
        `~irispy.meta.SGMeta` instance with a ``TWAVE<n>`` FITS keyword).
    velocity_range : `astropy.units.Quantity`, optional
        Two positive velocities defining the wing range to average.
    velocity_window : `astropy.units.Quantity`, optional
        Symmetric interpolation window about zero velocity. Defaults to the
        high end of ``velocity_range`` plus 50 km/s.
    fit_window : `astropy.units.Quantity`, optional
        Velocity half-width used to crop the source profile before
        interpolation. Defaults to the high end of ``velocity_range`` plus
        100 km/s.
    dv : `astropy.units.Quantity`, optional
        Velocity spacing for the interpolated profile.
    center_on_peak : `bool`, optional
        If `True`, shift each profile so its peak lies at zero velocity before
        sampling the wings.
    continuum_windows : `astropy.units.Quantity`, optional
        One or more wavelength windows used to estimate and subtract a
        continuum.
    uncertainty : `astropy.units.Quantity`, optional
        Per-bin intensity uncertainty. If omitted, ``cube.uncertainty`` is
        used when available.
    degree : `int`, optional
        Spline degree for `scipy.interpolate.make_interp_spline`.
    min_intensity : `float` or `astropy.units.Quantity`, optional
        Minimum peak intensity required for a pixel to be processed.
        Pixels below this threshold are assigned quality flag
        `~irispy.utils.red_blue.RBAQualityFlag.LOW_SIGNAL`.
    saturation_limit : `float` or `astropy.units.Quantity`, optional
        Maximum allowed peak intensity. Pixels above this value are assigned
        quality flag `~irispy.utils.red_blue.RBAQualityFlag.SATURATED`.
    return_profiles : `bool`, optional
        If `True`, include 3D ``"observed_profile"`` and
        ``"interpolated_profile"`` cubes in the output.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
    """
    if rest_wavelength is None:
        try:
            rest_wavelength = cube.meta.rest_wavelength
        except AttributeError as exc:
            pass
        if rest_wavelength is None:
            msg = (
                "rest_wavelength was not provided and could not be read from cube.meta. "
                "Pass rest_wavelength explicitly or provide a cube with SGMeta containing TWAVE keywords."
            )
            raise ValueError(msg)
    rest_wavelength = u.Quantity(rest_wavelength).to(u.nm)

    velocity_range = u.Quantity(velocity_range).to(u.km / u.s)
    if velocity_range.shape != (2,):
        msg = "velocity_range must contain two velocities"
        raise ValueError(msg)
    velocity_low, velocity_high = velocity_range
    if velocity_low < 0 or velocity_high <= velocity_low:
        msg = "velocity_range must be positive and increasing"
        raise ValueError(msg)

    velocity_window = (
        velocity_high + 50 * u.km / u.s if velocity_window is None else u.Quantity(velocity_window)
    ).to_value(u.km / u.s)
    fit_window = (velocity_high + 100 * u.km / u.s if fit_window is None else u.Quantity(fit_window)).to_value(
        u.km / u.s
    )
    dv = u.Quantity(dv).to_value(u.km / u.s)
    if velocity_window <= velocity_high.to_value(u.km / u.s):
        msg = "velocity_window must be larger than velocity_range high end"
        raise ValueError(msg)
    if fit_window <= velocity_window:
        msg = "fit_window must be larger than velocity_window"
        raise ValueError(msg)
    if dv <= 0:
        msg = "dv must be positive"
        raise ValueError(msg)
    degree = int(degree)
    if degree < 0:
        msg = "degree must be a non-negative integer"
        raise ValueError(msg)

    try:
        wavelength_axis = cube.wavelength_axis
    except (AttributeError, StopIteration) as exc:
        msg = "Could not identify a spectral wavelength axis on the input cube"
        raise ValueError(msg) from exc
    wavelengths = cube.axis_world_coords(wavelength_axis)[0].to(u.nm)
    rest_wavelength = rest_wavelength.to(wavelengths.unit)
    velocity = ((wavelengths - rest_wavelength) / rest_wavelength * constants.c).to_value(u.km / u.s)
    interp_velocity = np.arange(-velocity_window, velocity_window + dv, dv)
    velocity_range_kms = (velocity_low.to_value(u.km / u.s), velocity_high.to_value(u.km / u.s))

    data, errors = _prepare_data(cube, wavelengths, wavelength_axis, continuum_windows, uncertainty)

    output_shape = data.shape[:-1]
    d_velocity = float(np.nanmean(np.diff(velocity))) if velocity.size > 1 else 1.0

    low, high = velocity_range_kms
    red_mask = (interp_velocity >= low) & (interp_velocity <= high)
    blue_mask = (interp_velocity >= -high) & (interp_velocity <= -low)

    ctx = _RBAContext(
        velocity=velocity,
        interp_velocity=interp_velocity,
        center_on_peak=center_on_peak,
        fit_window=fit_window,
        d_velocity=d_velocity,
        degree=degree,
        red_mask=red_mask,
        blue_mask=blue_mask,
        min_wing_coverage=0.8,
    )

    red_blue = np.full(output_shape, np.nan)
    red_blue_error = np.full(output_shape, np.nan)
    red_wing = np.full(output_shape, np.nan)
    blue_wing = np.full(output_shape, np.nan)
    red_wing_error = np.full(output_shape, np.nan)
    blue_wing_error = np.full(output_shape, np.nan)
    peak_intensity = np.full(output_shape, np.nan)
    peak_velocity = np.full(output_shape, np.nan)
    quality = np.full(output_shape, RBAQualityFlag.OK, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_peak = np.nanmax(data, axis=-1)
    if min_intensity is not None:
        min_intensity_value = (
            min_intensity.to_value(cube.unit) if isinstance(min_intensity, u.Quantity) else min_intensity
        )
        quality = np.where(raw_peak < min_intensity_value, RBAQualityFlag.LOW_SIGNAL, quality)
    if saturation_limit is not None:
        saturation_limit_value = (
            saturation_limit.to_value(cube.unit) if isinstance(saturation_limit, u.Quantity) else saturation_limit
        )
        quality = np.where(raw_peak > saturation_limit_value, RBAQualityFlag.SATURATED, quality)

    interpolated_profiles = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float) if return_profiles else None
    )
    interpolated_errors = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float)
        if errors is not None and return_profiles
        else None
    )

    for index in np.ndindex(output_shape):
        if quality[index] in (RBAQualityFlag.LOW_SIGNAL, RBAQualityFlag.SATURATED):
            continue

        core, detail = _process_pixel(data[index], None if errors is None else errors[index], ctx)

        red_blue[index] = core.rba
        red_wing[index] = core.red_wing
        blue_wing[index] = core.blue_wing
        peak_intensity[index] = core.peak_intensity
        peak_velocity[index] = core.peak_velocity
        quality[index] = core.quality
        red_blue_error[index] = core.red_blue_error
        red_wing_error[index] = core.red_wing_error
        blue_wing_error[index] = core.blue_wing_error
        if return_profiles and detail is not None:
            interpolated_profiles[index] = detail.interp_profile
            if interpolated_errors is not None and detail.interp_error is not None:
                interpolated_errors[index] = detail.interp_error

    meta = {
        "rba_rest_wavelength": rest_wavelength.to_value(u.nm),
        "rba_rest_wavelength_unit": "nm",
        "rba_velocity_range": velocity_range_kms,
        "rba_velocity_window": velocity_window,
        "rba_fit_window": fit_window,
        "rba_dv": dv,
        "rba_center_on_peak": center_on_peak,
        "rba_interpolation_degree": degree,
    }
    if continuum_windows is not None:
        meta["rba_continuum_windows"] = str(continuum_windows)

    template = make_spatial_template(cube, wavelength_axis)

    def _make_cube(values, unit, *, mask=None):
        c = make_map_cube(template, values, unit, mask=mask)
        c.meta.update(meta)
        return c

    cubes = [
        ("red_blue_asymmetry", red_blue, u.dimensionless_unscaled),
        ("red_wing", red_wing, cube.unit),
        ("blue_wing", blue_wing, cube.unit),
        ("peak_intensity", peak_intensity, cube.unit),
        ("peak_velocity", peak_velocity, u.km / u.s),
        ("quality", quality, u.dimensionless_unscaled),
    ]
    if errors is not None:
        cubes.extend(
            [
                ("red_blue_asymmetry_error", red_blue_error, u.dimensionless_unscaled),
                ("red_wing_error", red_wing_error, cube.unit),
                ("blue_wing_error", blue_wing_error, cube.unit),
            ]
        )

    result_cubes = []
    for name, values, unit in cubes:
        mask = None if name == "quality" else ~np.isfinite(values)
        result_cubes.append((name, _make_cube(values, unit, mask=mask)))

    if return_profiles:
        observed_data = np.moveaxis(data, -1, wavelength_axis)
        interpolated_data = np.moveaxis(interpolated_profiles, -1, wavelength_axis)
        observed_mask = ~np.isfinite(observed_data)
        interpolated_mask = ~np.isfinite(interpolated_data)
        observed_uncertainty = (
            StdDevUncertainty(np.moveaxis(errors, -1, wavelength_axis)) if errors is not None else None
        )
        interpolated_uncertainty = (
            StdDevUncertainty(np.moveaxis(interpolated_errors, -1, wavelength_axis))
            if interpolated_errors is not None
            else None
        )

        result_cubes.extend(
            [
                (
                    "observed_profile",
                    _make_profile_cube(
                        cube,
                        data=observed_data,
                        velocity_grid=velocity,
                        wavelength_axis=wavelength_axis,
                        meta={**cube.meta, **meta, "rba_profile": "observed"},
                        uncertainty=observed_uncertainty,
                        mask=observed_mask,
                    ),
                ),
                (
                    "interpolated_profile",
                    _make_profile_cube(
                        cube,
                        data=interpolated_data,
                        velocity_grid=interp_velocity,
                        wavelength_axis=wavelength_axis,
                        meta={**cube.meta, **meta, "rba_profile": "interpolated_peak_centered"},
                        uncertainty=interpolated_uncertainty,
                        mask=interpolated_mask,
                    ),
                ),
            ]
        )

    return RasterCollection(result_cubes)
