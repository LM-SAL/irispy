"""
Red-blue asymmetry utilities for IRIS spectrogram cubes.
"""

import warnings

import numpy as np
from scipy.interpolate import make_interp_spline

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils._spectral import (
    _QualityFlag,
    check_scaled,
    drop_extra_coords_dependent_on_axis,
    make_map_cube,
    make_spatial_template,
    standard_deviation,
)

__all__ = ["RBAQualityFlag", "calculate_red_blue_asymmetry"]

# Minimum fraction of wing bins that must have finite interpolated values.
_MIN_WING_COVERAGE = 0.8


class RBAQualityFlag(_QualityFlag):
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
    SATURATED = (8, "saturated")


def _make_velocity_wcs(array_shape, velocity_axis, velocity_grid):
    """
    Return a profile WCS whose only meaningful world axis is Doppler velocity.

    Do not copy the input cube WCS here: profile cubes may have fewer axes than
    the source cube, so stale FITS-WCS axes can warn or fail during construction.
    """
    naxis = len(array_shape)
    wcs_axis = naxis - 1 - velocity_axis
    cdelt = float(np.nanmean(np.diff(velocity_grid))) if velocity_grid.size > 1 else 1.0

    wcs = WCS(naxis=naxis)
    wcs.wcs.ctype[wcs_axis] = "VELO"
    wcs.wcs.cunit[wcs_axis] = "km/s"
    wcs.wcs.crpix[wcs_axis] = 1.0
    wcs.wcs.crval[wcs_axis] = float(velocity_grid[0])
    wcs.wcs.cdelt[wcs_axis] = cdelt
    wcs.array_shape = tuple(array_shape)
    wcs.wcs.set()
    return wcs


def _make_profile_cube(cube, *, data, velocity_grid, wavelength_axis, meta, uncertainty=None, mask=None):
    return SpectrogramCube(
        data,
        wcs=_make_velocity_wcs(data.shape, wavelength_axis, velocity_grid),
        uncertainty=uncertainty,
        unit=cube.unit,
        meta=meta,
        mask=mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=False),
    )


def _prepare_data(cube, wavelength_axis):
    """
    Data and errors, negatives masked, spectral axis last.
    """
    data = np.asarray(cube.data, dtype=float)
    # Negative samples are left out
    dropped = data < 0 if cube.mask is None else (data < 0) | cube.mask
    data = np.where(dropped, np.nan, data)

    errors = standard_deviation(cube)
    if errors is not None and cube.mask is not None:
        errors = np.where(cube.mask, np.nan, errors)

    data = np.moveaxis(data, wavelength_axis, -1)
    if errors is not None:
        errors = np.moveaxis(errors, wavelength_axis, -1)
    return data, errors


def calculate_red_blue_asymmetry(
    cube,
    *,
    rest_wavelength=None,
    velocity_range=(50, 150) * u.km / u.s,
    dv=10 * u.km / u.s,
    spline_degree=3,
    min_intensity=None,
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
    dv : `astropy.units.Quantity`, optional
        Velocity spacing for the interpolated profile.
    spline_degree : `int`, optional
        Spline degree for `scipy.interpolate.make_interp_spline`.
    min_intensity : `float` or `astropy.units.Quantity`, optional
        Minimum peak intensity required for a pixel to be processed.
        Pixels below this threshold are assigned quality flag
        `~irispy.utils.red_blue.RBAQualityFlag.LOW_SIGNAL`.
    return_profiles : `bool`, optional
        If `True`, include 3D ``"observed_profile"`` and
        ``"interpolated_profile"`` cubes in the output.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        Always contains ``"red_blue_asymmetry"`` and ``"quality"`` 2D maps.
        ``"red_blue_asymmetry_error"`` is added when the input cube has uncertainty.
        ``"observed_profile"`` and ``"interpolated_profile"`` 3D cubes are added
        when ``return_profiles=True``. The interpolated profile velocity axis is
        peak-centered; the observed profile velocity axis is relative to
        ``rest_wavelength``.

    Notes
    -----
    Pixels with an unmasked +Inf sample, as the readers set the samples clipped at the level 2
    ceiling, are flagged `~irispy.utils.red_blue.RBAQualityFlag.SATURATED`. Subtract a background
    first with `~irispy.utils.spectrograph.subtract_background` if the line sits on one.
    """
    check_scaled(cube)
    if rest_wavelength is None:
        rest_wavelength = getattr(cube.meta, "rest_wavelength", None)
        if rest_wavelength is None:
            msg = (
                "rest_wavelength was not provided and could not be read from cube.meta. "
                "Pass rest_wavelength explicitly or provide a cube with SGMeta containing TWAVE keywords."
            )
            raise ValueError(msg)
    rest_wavelength = u.Quantity(rest_wavelength).to(u.nm)
    if rest_wavelength.shape != () or not np.isfinite(rest_wavelength.value) or rest_wavelength <= 0 * u.nm:
        msg = "rest_wavelength must be a positive finite scalar wavelength"
        raise ValueError(msg)

    velocity_range = u.Quantity(velocity_range).to(u.km / u.s)
    if velocity_range.shape != (2,):
        msg = "velocity_range must contain two velocities"
        raise ValueError(msg)
    velocity_low, velocity_high = velocity_range
    if velocity_low < 0 or velocity_high <= velocity_low:
        msg = "velocity_range must be positive and increasing"
        raise ValueError(msg)

    dv = u.Quantity(dv).to_value(u.km / u.s)
    if dv <= 0:
        msg = "dv must be positive"
        raise ValueError(msg)
    spline_degree = int(spline_degree)
    if spline_degree < 0:
        msg = "spline_degree must be a non-negative integer"
        raise ValueError(msg)

    wavelength_axis = cube.wavelength_axis
    wavelengths = cube.axis_world_coords(wavelength_axis)[0].to(u.nm)
    rest_wavelength = rest_wavelength.to(wavelengths.unit)
    velocity = ((wavelengths - rest_wavelength) / rest_wavelength * constants.c).to_value(u.km / u.s)
    interp_extent = velocity_high.to_value(u.km / u.s) + dv
    interp_velocity = np.arange(-interp_extent, interp_extent + dv, dv)
    velocity_range_kms = (velocity_low.to_value(u.km / u.s), velocity_high.to_value(u.km / u.s))

    data, errors = _prepare_data(cube, wavelength_axis)

    output_shape = data.shape[:-1]

    low, high = velocity_range_kms
    red_mask = (interp_velocity >= low) & (interp_velocity <= high)
    blue_mask = (interp_velocity >= -high) & (interp_velocity <= -low)

    red_blue = np.full(output_shape, np.nan)
    red_blue_error = np.full(output_shape, np.nan)
    quality = np.full(output_shape, RBAQualityFlag.OK, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_peak = np.nanmax(data, axis=-1)
    if min_intensity is not None:
        low_signal = raw_peak < u.Quantity(min_intensity, cube.unit).value
        quality = np.where(low_signal, RBAQualityFlag.LOW_SIGNAL, quality).astype(np.uint8)
    # Masked samples are NaN here, so they do not saturate, as in calculate_moments
    saturated = np.isposinf(data).any(axis=-1)
    quality = np.where(saturated, RBAQualityFlag.SATURATED, quality).astype(np.uint8)

    interpolated_profiles = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float) if return_profiles else None
    )
    interpolated_errors = (
        np.full((*output_shape, interp_velocity.size), np.nan, dtype=float)
        if errors is not None and return_profiles
        else None
    )

    min_points = spline_degree + 1
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
        if peak_index == 0 or peak_index == profile.size - 1:
            quality[index] = RBAQualityFlag.PEAK_AT_EDGE
            continue

        shifted_velocity = velocity - velocity[peak_index]
        finite = np.isfinite(shifted_velocity) & finite_profile
        if finite.sum() < min_points:
            quality[index] = RBAQualityFlag.TOO_FEW_POINTS
            continue

        order = np.argsort(shifted_velocity[finite])
        ordered_velocity = shifted_velocity[finite][order]
        ordered_profile = profile[finite][order]
        try:
            interp_profile = make_interp_spline(ordered_velocity, ordered_profile, k=spline_degree)(
                interp_velocity, extrapolate=False
            )
        except ValueError:
            quality[index] = RBAQualityFlag.INTERP_FAILED
            continue

        interp_error = None
        if profile_error is not None:
            ordered_error = profile_error[finite][order]
            finite_error = np.isfinite(ordered_error)
            if finite_error.sum() >= min_points:
                try:
                    interp_error = make_interp_spline(
                        ordered_velocity[finite_error], ordered_error[finite_error], k=spline_degree
                    )(interp_velocity, extrapolate=False)
                except ValueError:
                    quality[index] = RBAQualityFlag.INTERP_FAILED
                    continue
                interp_error = np.where(interp_error >= 0, interp_error, np.nan)

        if return_profiles:
            interpolated_profiles[index] = interp_profile
            if interpolated_errors is not None and interp_error is not None:
                interpolated_errors[index] = interp_error

        red_finite = red_mask & np.isfinite(interp_profile)
        blue_finite = blue_mask & np.isfinite(interp_profile)
        if red_finite.sum() < _MIN_WING_COVERAGE * red_mask.sum():
            quality[index] = RBAQualityFlag.INCOMPLETE_WINGS
            continue
        if blue_finite.sum() < _MIN_WING_COVERAGE * blue_mask.sum():
            quality[index] = RBAQualityFlag.INCOMPLETE_WINGS
            continue

        red_intensity = np.nanmean(interp_profile[red_finite]) if red_finite.any() else np.nan
        blue_intensity = np.nanmean(interp_profile[blue_finite]) if blue_finite.any() else np.nan
        peak = np.nanmax(interp_profile)
        if not np.isfinite(peak) or np.isclose(peak, 0):
            quality[index] = RBAQualityFlag.PEAK_IS_ZERO
            continue

        numerator = red_intensity - blue_intensity
        red_blue[index] = numerator / peak

        if interp_error is not None:
            n_red = np.isfinite(interp_error[red_mask]).sum()
            n_blue = np.isfinite(interp_error[blue_mask]).sum()
            red_err = np.sqrt(np.nansum(interp_error[red_mask] ** 2)) / n_red if n_red > 0 else np.nan
            blue_err = np.sqrt(np.nansum(interp_error[blue_mask] ** 2)) / n_blue if n_blue > 0 else np.nan
            peak_err = interp_error[np.nanargmax(interp_profile)]
            num_err = np.sqrt(red_err**2 + blue_err**2)
            numerator_is_zero = np.isclose(numerator, 0)
            if np.isfinite(num_err) and (numerator_is_zero or np.isfinite(peak_err)):
                variance = (num_err / peak) ** 2
                if not numerator_is_zero:
                    variance += (numerator * peak_err / peak**2) ** 2
                red_blue_error[index] = np.sqrt(variance)

    meta = {
        "rba_rest_wavelength": rest_wavelength.to_value(u.nm),
        "rba_rest_wavelength_unit": "nm",
        "rba_velocity_range": velocity_range_kms,
        "rba_dv": dv,
        "rba_spline_degree": spline_degree,
    }

    template = make_spatial_template(cube, wavelength_axis)

    def _make_cube(values, unit, *, mask=None):
        c = make_map_cube(template, values, unit, mask=mask)
        c.meta.update(meta)
        return c

    cubes = [
        ("red_blue_asymmetry", red_blue, u.dimensionless_unscaled),
        ("quality", quality, u.dimensionless_unscaled),
    ]
    if errors is not None:
        cubes.append(("red_blue_asymmetry_error", red_blue_error, u.dimensionless_unscaled))

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
