"""
Red-blue asymmetry utilities for IRIS spectrogram cubes.
"""

import warnings
from enum import IntEnum

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

_INTERPOLATION_DEGREES = {"linear": 1, "quadratic": 2, "cubic": 3}


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


def _make_profile_cube(
    cube,
    *,
    data,
    velocity_grid,
    wavelength_axis,
    meta,
    uncertainty=None,
    mask=None,
):
    return SpectrogramCube(
        data,
        wcs=_make_velocity_wcs(cube.wcs, data.shape, wavelength_axis, velocity_grid),
        uncertainty=uncertainty,
        unit=cube.unit,
        meta=meta,
        mask=mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=False),
    )


def calculate_red_blue_asymmetry(
    cube,
    *,
    rest_wavelength,
    velocity_range=(50, 150) * u.km / u.s,
    velocity_window=None,
    fit_window=None,
    dv=10 * u.km / u.s,
    center_on_peak=True,
    continuum_windows=None,
    uncertainty=None,
    interpolation_kind="cubic",
    mask_negative=True,
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
    rest_wavelength : `astropy.units.Quantity`
        Rest wavelength used to convert the spectral axis to Doppler velocity.
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
    interpolation_kind : {"linear", "quadratic", "cubic"} or `int`, optional
        Spline degree used by `scipy.interpolate.make_interp_spline`.
    mask_negative : `bool`, optional
        If `True`, negative intensities are set to NaN before computing
        moments.
    min_intensity : `float` or `astropy.units.Quantity`, optional
        Minimum peak intensity required for a pixel to be processed.
        Pixels with a peak below this threshold are skipped and assigned
        quality flag `~irispy.utils.red_blue.RBAQualityFlag.LOW_SIGNAL`.
    saturation_limit : `float` or `astropy.units.Quantity`, optional
        Maximum allowed peak intensity. Pixels where the peak exceeds this
        value are treated as saturated, skipped, and assigned quality flag
        `~irispy.utils.red_blue.RBAQualityFlag.SATURATED`.
    return_profiles : `bool`, optional
        If `True`, include plot-ready 3D ``"observed_profile"`` and
        ``"interpolated_profile"`` cubes in the output. Set to `False` to
        reduce peak memory use when only the 2D maps are needed.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        Collection with 2D maps. When ``return_profiles=True``, the collection
        also includes plot-ready 3D ``"observed_profile"`` and
        ``"interpolated_profile"`` `~irispy.spectrograph.SpectrogramCube`
        instances. The ``"quality"`` cube stores per-pixel integer flags
        described by
        `irispy.utils.red_blue.RBAQualityFlag`.
    """
    # -- Validate velocity_range ------------------------------------------------
    velocity_range = u.Quantity(velocity_range).to(u.km / u.s)
    if velocity_range.shape != (2,):
        msg = "velocity_range must contain two velocities"
        raise ValueError(msg)
    velocity_low, velocity_high = velocity_range
    if velocity_low < 0 or velocity_high <= velocity_low:
        msg = "velocity_range must be positive and increasing"
        raise ValueError(msg)

    velocity_window = velocity_high + 50 * u.km / u.s if velocity_window is None else u.Quantity(velocity_window)
    velocity_window = velocity_window.to_value(u.km / u.s)
    fit_window = velocity_high + 100 * u.km / u.s if fit_window is None else u.Quantity(fit_window)
    fit_window = fit_window.to_value(u.km / u.s)
    dv = u.Quantity(dv).to_value(u.km / u.s)
    if velocity_window <= velocity_high.to_value(u.km / u.s):
        msg = "velocity_window must be larger than the high end of velocity_range"
        raise ValueError(msg)
    if fit_window <= velocity_window:
        msg = "fit_window must be larger than velocity_window"
        raise ValueError(msg)
    if dv <= 0:
        msg = "dv must be positive"
        raise ValueError(msg)
    interpolation_degree = _INTERPOLATION_DEGREES.get(interpolation_kind, interpolation_kind)
    try:
        interpolation_degree = int(interpolation_degree)
    except (TypeError, ValueError) as exc:
        msg = "interpolation_kind must be 'linear', 'quadratic', 'cubic', or an integer spline degree"
        raise ValueError(msg) from exc
    if interpolation_degree < 0:
        msg = "interpolation_kind must be a non-negative spline degree"
        raise ValueError(msg)

    # -- Locate spectral axis and compute velocities ----------------------------
    try:
        wavelength_axis = next(
            axis
            for axis, physical_types in enumerate(cube.array_axis_physical_types)
            if physical_types and "em.wl" in physical_types
        )
    except StopIteration as exc:
        msg = "Could not identify a spectral wavelength axis on the input cube"
        raise ValueError(msg) from exc
    wavelengths = cube.axis_world_coords(wavelength_axis)[0].to(u.nm)
    rest_wavelength = u.Quantity(rest_wavelength).to(wavelengths.unit)
    velocity = ((wavelengths - rest_wavelength) / rest_wavelength * constants.c).to_value(u.km / u.s)
    interp_velocity = np.arange(-velocity_window, velocity_window + dv, dv)
    velocity_range_value = (velocity_low.to_value(u.km / u.s), velocity_high.to_value(u.km / u.s))

    # -- Prepare data and uncertainty -------------------------------------------
    data = np.asarray(cube.data, dtype=float)
    if cube.mask is not None:
        data = np.where(cube.mask, np.nan, data)
    if mask_negative:
        data = np.where(data < 0, np.nan, data)

    if uncertainty is not None:
        errors = u.Quantity(uncertainty, cube.unit).to_value(cube.unit)
    elif cube.uncertainty is not None:
        errors = np.asarray(cube.uncertainty.array, dtype=float)
    else:
        errors = None
    if errors is not None and cube.mask is not None:
        errors = np.where(cube.mask, np.nan, errors)

    # -- Continuum subtraction --------------------------------------------------
    if continuum_windows is not None:
        windows = u.Quantity(continuum_windows).to(wavelengths.unit)
        if windows.shape == (2,):
            windows = windows[np.newaxis, :]
        if windows.ndim != 2 or windows.shape[1] != 2:
            msg = "continuum_windows must have shape (2,) or (n, 2)"
            raise ValueError(msg)
        continuum = np.zeros(wavelengths.shape, dtype=bool)
        for low, high in windows:
            continuum |= (wavelengths >= low) & (wavelengths <= high)
        if not continuum.any():
            msg = "No wavelength points found within continuum_windows"
            raise ValueError(msg)
        data = np.moveaxis(data, wavelength_axis, -1)
        if errors is not None:
            errors = np.moveaxis(errors, wavelength_axis, -1)
        continuum_values = np.nanmean(data[..., continuum], axis=-1)
        data = data - continuum_values[..., np.newaxis]
        if errors is not None:
            n_finite = np.isfinite(errors[..., continuum]).sum(axis=-1)
            continuum_errors = np.sqrt(np.nansum(errors[..., continuum] ** 2, axis=-1))
            continuum_errors = np.where(n_finite > 0, continuum_errors / n_finite, np.nan)
            errors = np.sqrt(errors**2 + continuum_errors[..., np.newaxis] ** 2)
    else:
        data = np.moveaxis(data, wavelength_axis, -1)
        if errors is not None:
            errors = np.moveaxis(errors, wavelength_axis, -1)

    # -- Per-pixel red-blue computation -----------------------------------------
    output_shape = data.shape[:-1]
    red_blue = np.full(output_shape, np.nan)
    red_blue_error = np.full(output_shape, np.nan)
    red_wing = np.full(output_shape, np.nan)
    blue_wing = np.full(output_shape, np.nan)
    red_wing_error = np.full(output_shape, np.nan)
    blue_wing_error = np.full(output_shape, np.nan)
    peak_intensity = np.full(output_shape, np.nan)
    peak_velocity = np.full(output_shape, np.nan)
    quality = np.full(output_shape, RBAQualityFlag.OK, dtype=np.uint8)

    # Apply min_intensity and saturation_limit masks before the loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_peak = np.nanmax(data, axis=-1)
    if min_intensity is not None:
        if isinstance(min_intensity, u.Quantity):
            min_intensity_value = min_intensity.to_value(cube.unit)
        else:
            min_intensity_value = min_intensity
        low_signal = raw_peak < min_intensity_value
        quality = np.where(low_signal, RBAQualityFlag.LOW_SIGNAL, quality)
    if saturation_limit is not None:
        if isinstance(saturation_limit, u.Quantity):
            saturation_limit_value = saturation_limit.to_value(cube.unit)
        else:
            saturation_limit_value = saturation_limit
        saturated = raw_peak > saturation_limit_value
        quality = np.where(saturated, RBAQualityFlag.SATURATED, quality)

    interpolated_profiles = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float) if return_profiles else None
    )
    interpolated_errors = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float)
        if errors is not None and return_profiles
        else None
    )

    low, high = velocity_range_value
    red_mask = (interp_velocity >= low) & (interp_velocity <= high)
    blue_mask = (interp_velocity >= -high) & (interp_velocity <= -low)

    for index in np.ndindex(output_shape):
        if quality[index] in (RBAQualityFlag.LOW_SIGNAL, RBAQualityFlag.SATURATED):
            continue
        profile = data[index]
        profile_error = None if errors is None else errors[index]
        finite_profile = np.isfinite(profile)
        if not finite_profile.any():
            quality[index] = RBAQualityFlag.NO_FINITE_DATA
            continue
        peak_index = np.nanargmax(np.where(finite_profile, profile, np.nan))
        peak_velocity[index] = velocity[peak_index]

        if center_on_peak:
            # Reject only if the peak itself sits at the very edge of the array
            if peak_index == 0 or peak_index == profile.size - 1:
                quality[index] = RBAQualityFlag.PEAK_AT_EDGE
                continue
            d_velocity = np.nanmean(np.diff(velocity))
            window_pixels = int(fit_window / abs(d_velocity))
            # Clamp the slice to available data (old prototype behaviour)
            low_idx = max(0, peak_index - window_pixels)
            high_idx = min(profile.size, peak_index + window_pixels)
            sl = slice(low_idx, high_idx)
            shifted_velocity = velocity[sl] - velocity[peak_index]
            profile = profile[sl]
            if profile_error is not None:
                profile_error = profile_error[sl]
        else:
            fit_mask = np.abs(velocity) <= fit_window
            shifted_velocity = velocity[fit_mask]
            profile = profile[fit_mask]
            if profile_error is not None:
                profile_error = profile_error[fit_mask]

        # Interpolate onto uniform velocity grid
        finite = np.isfinite(shifted_velocity) & np.isfinite(profile)
        min_points = interpolation_degree + 1
        if finite.sum() < min_points:
            quality[index] = RBAQualityFlag.TOO_FEW_POINTS
            continue
        sv = shifted_velocity[finite]
        sp = profile[finite]
        order = np.argsort(sv)
        ordered_velocity = sv[order]
        ordered_profile = sp[order]
        try:
            interp_profile = make_interp_spline(ordered_velocity, ordered_profile, k=interpolation_degree)(
                interp_velocity,
                extrapolate=False,
            )
        except ValueError:
            quality[index] = RBAQualityFlag.INTERP_FAILED
            continue
        if profile_error is not None:
            ordered_error = profile_error[finite][order]
            finite_error = np.isfinite(ordered_error)
            if finite_error.sum() >= min_points:
                try:
                    interp_error = make_interp_spline(
                        ordered_velocity[finite_error],
                        ordered_error[finite_error],
                        k=interpolation_degree,
                    )(interp_velocity, extrapolate=False)
                    interp_error = np.where(interp_error >= 0, interp_error, np.nan)
                except ValueError:
                    quality[index] = RBAQualityFlag.INTERP_FAILED
                    continue
            else:
                interp_error = None
        else:
            interp_error = None

        if return_profiles:
            interpolated_profiles[index] = interp_profile
        if interp_error is not None and return_profiles:
            interpolated_errors[index] = interp_error

        red_finite = red_mask & np.isfinite(interp_profile)
        blue_finite = blue_mask & np.isfinite(interp_profile)
        if red_finite.sum() < 0.8 * red_mask.sum() or blue_finite.sum() < 0.8 * blue_mask.sum():
            quality[index] = RBAQualityFlag.INCOMPLETE_WINGS
            continue
        red_intensity = np.nanmean(interp_profile[red_finite]) if red_finite.any() else np.nan
        blue_intensity = np.nanmean(interp_profile[blue_finite]) if blue_finite.any() else np.nan
        peak = np.nanmax(interp_profile)
        if not np.isfinite(peak) or peak == 0:
            quality[index] = RBAQualityFlag.PEAK_IS_ZERO
            continue
        rba = (red_intensity - blue_intensity) / peak
        red_blue[index] = rba
        red_wing[index] = red_intensity
        blue_wing[index] = blue_intensity
        peak_intensity[index] = peak

        if interp_error is not None and np.isfinite(rba):
            n_red = np.isfinite(interp_error[red_mask]).sum()
            n_blue = np.isfinite(interp_error[blue_mask]).sum()
            red_err = np.sqrt(np.nansum(interp_error[red_mask] ** 2)) / n_red if n_red > 0 else np.nan
            blue_err = np.sqrt(np.nansum(interp_error[blue_mask] ** 2)) / n_blue if n_blue > 0 else np.nan
            peak_err = interp_error[np.nanargmax(interp_profile)]
            numerator = red_intensity - blue_intensity
            num_err = np.sqrt(red_err**2 + blue_err**2)
            if np.isfinite(num_err) and (numerator == 0 or np.isfinite(peak_err)):
                variance = (num_err / peak) ** 2
                if numerator != 0:
                    variance += (numerator * peak_err / peak**2) ** 2
                red_blue_error[index] = np.sqrt(variance)
            red_wing_error[index] = red_err
            blue_wing_error[index] = blue_err

    # -- Build meta with computation parameters ---------------------------------
    meta = {
        "rba_rest_wavelength": rest_wavelength.to_value(u.nm),
        "rba_rest_wavelength_unit": "nm",
        "rba_velocity_range": velocity_range_value,
        "rba_velocity_window": velocity_window,
        "rba_fit_window": fit_window,
        "rba_dv": dv,
        "rba_center_on_peak": center_on_peak,
        "rba_interpolation_kind": interpolation_kind,
        "rba_mask_negative": mask_negative,
    }
    if continuum_windows is not None:
        meta["rba_continuum_windows"] = str(continuum_windows)

    def _make_cube(values, unit, *, mask=None):
        c = make_map_cube(template, values, unit, mask=mask)
        c.meta.update(meta)
        return c

    # -- Build output RasterCollection ------------------------------------------
    template = make_spatial_template(cube, wavelength_axis)
    cubes = [
        ("red_blue_asymmetry", _make_cube(red_blue, u.dimensionless_unscaled, mask=~np.isfinite(red_blue))),
        ("red_wing", _make_cube(red_wing, cube.unit, mask=~np.isfinite(red_wing))),
        ("blue_wing", _make_cube(blue_wing, cube.unit, mask=~np.isfinite(blue_wing))),
        ("peak_intensity", _make_cube(peak_intensity, cube.unit, mask=~np.isfinite(peak_intensity))),
        ("peak_velocity", _make_cube(peak_velocity, u.km / u.s, mask=~np.isfinite(peak_velocity))),
        ("quality", _make_cube(quality, u.dimensionless_unscaled)),
    ]
    if errors is not None:
        cubes.extend(
            [
                (
                    "red_blue_asymmetry_error",
                    _make_cube(red_blue_error, u.dimensionless_unscaled, mask=~np.isfinite(red_blue_error)),
                ),
                ("red_wing_error", _make_cube(red_wing_error, cube.unit, mask=~np.isfinite(red_wing_error))),
                ("blue_wing_error", _make_cube(blue_wing_error, cube.unit, mask=~np.isfinite(blue_wing_error))),
            ],
        )

    if return_profiles:
        observed_data = np.moveaxis(data, -1, wavelength_axis)
        observed_mask = ~np.isfinite(observed_data)
        observed_uncertainty = None
        if errors is not None:
            observed_uncertainty = StdDevUncertainty(np.moveaxis(errors, -1, wavelength_axis))

        interpolated_data = np.moveaxis(interpolated_profiles, -1, wavelength_axis)
        interpolated_mask = ~np.isfinite(interpolated_data)
        interpolated_uncertainty = None
        if interpolated_errors is not None:
            interpolated_uncertainty = StdDevUncertainty(np.moveaxis(interpolated_errors, -1, wavelength_axis))

        cubes.extend(
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
    return RasterCollection(cubes)
