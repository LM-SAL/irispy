"""
Helpers for IRIS density diagnostics.
"""

from importlib import import_module

import numpy as np
from scipy.interpolate import interp1d

import astropy.units as u

__all__ = ["density_diagnostic", "map_ratio_to_quantity"]


def map_ratio_to_quantity(observed_ratio, quantity, theoretical_ratio, *, bounds_error=False, fill_value=np.nan):
    """
    Map an observed line ratio to the associated physical quantity.

    Parameters
    ----------
    observed_ratio : array-like or `~astropy.units.Quantity`
        Observed line ratio values. Must be dimensionless.
    quantity : `~astropy.units.Quantity`
        Quantity grid associated with ``theoretical_ratio``, e.g. an electron
        density grid.
    theoretical_ratio : array-like or `~astropy.units.Quantity`
        Theoretical line ratio sampled on ``quantity``. Must be dimensionless
        and monotonic.
    bounds_error : `bool`, optional
        If `True`, raise an exception when ``observed_ratio`` falls outside the
        theoretical ratio range. By default, points outside the range are set by
        ``fill_value``.
    fill_value : scalar, tuple, or `~astropy.units.Quantity`, optional
        Value used outside of the interpolation interval. If a quantity is
        given, it must be convertible to the units of ``quantity``.
    """
    observed_ratio = u.Quantity(observed_ratio, u.dimensionless_unscaled)
    quantity = np.ravel(u.Quantity(quantity))
    theoretical_ratio = np.ravel(u.Quantity(theoretical_ratio, u.dimensionless_unscaled).value)

    if quantity.shape != theoretical_ratio.shape:
        msg = "quantity and theoretical_ratio must have the same shape."
        raise ValueError(msg)

    finite = np.isfinite(quantity.value) & np.isfinite(theoretical_ratio)
    quantity = quantity[finite]
    theoretical_ratio = theoretical_ratio[finite]

    if quantity.size < 2:
        msg = "At least two finite samples are required to interpolate a ratio curve."
        raise ValueError(msg)

    diff = np.diff(theoretical_ratio)
    nonzero = diff != 0.0
    if not np.any(nonzero):
        msg = "theoretical_ratio must vary over the provided quantity grid."
        raise ValueError(msg)
    if np.all(diff[nonzero] < 0.0):
        quantity = quantity[::-1]
        theoretical_ratio = theoretical_ratio[::-1]
    elif not np.all(diff[nonzero] > 0.0):
        msg = (
            "theoretical_ratio must be monotonic to map ratios back to a quantity. "
            "Restrict the grid to a monotonic interval first."
        )
        raise ValueError(msg)

    theoretical_ratio, unique_index = np.unique(theoretical_ratio, return_index=True)
    quantity = quantity[unique_index]
    if quantity.size < 2:
        msg = "Theoretical ratio must contain at least two unique samples."
        raise ValueError(msg)

    if isinstance(fill_value, tuple):
        fill_value = tuple(
            value.to_value(quantity.unit) if isinstance(value, u.Quantity) else value for value in fill_value
        )
    elif isinstance(fill_value, u.Quantity):
        fill_value = fill_value.to_value(quantity.unit)

    interp = interp1d(
        theoretical_ratio,
        quantity.value,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return u.Quantity(interp(observed_ratio.to_value(u.dimensionless_unscaled)), quantity.unit)


def density_diagnostic(
    intensity_numerator,
    intensity_denominator,
    density_grid,
    *,
    ion,
    numerator,
    denominator,
    intensity_numerator_uncertainty=None,
    intensity_denominator_uncertainty=None,
    temperature=None,
    bounds_error=False,
    fill_value=np.nan,
    line_ratio_density_kwargs=None,
):
    """
    Map an observed IRIS line ratio onto a theoretical density curve.

    Parameters
    ----------
    intensity_numerator, intensity_denominator : array-like or `~astropy.units.Quantity`
        Measured integrated intensities for the numerator and denominator lines.
    density_grid : `~astropy.units.Quantity`
        Density grid on which to evaluate the theoretical ratio curve.
    ion : `fiasco.Ion`
        Ion used to compute the theoretical ratio curve.
    numerator, denominator : `~astropy.units.Quantity`
        Required wavelengths of the numerator and denominator transitions.
    intensity_numerator_uncertainty, intensity_denominator_uncertainty :
        array-like or `~astropy.units.Quantity`, optional
        Uncertainties on the measured integrated intensities.
    temperature : `~astropy.units.Quantity`, optional
        Temperature at which to evaluate the theoretical ratio curve from
        ``ion``. If omitted and scalar, uses the ion temperature; if omitted
        and non-scalar, falls back to ``ion.formation_temperature``.
    bounds_error : `bool`, optional
        Passed through to `map_ratio_to_quantity`.
    fill_value : scalar or `~astropy.units.Quantity`, optional
        Passed through to `map_ratio_to_quantity`.
    line_ratio_density_kwargs : `dict`, optional
        Extra keyword arguments forwarded to ``fiasco.line_ratio_density``.
    """
    if (intensity_numerator_uncertainty is None) != (intensity_denominator_uncertainty is None):
        msg = "Both numerator and denominator uncertainties must be provided together."
        raise ValueError(msg)
    if any(value is None for value in (ion, numerator, denominator)):
        msg = "ion, numerator, and denominator are required."
        raise ValueError(msg)

    try:
        fiasco = import_module("fiasco")
    except ImportError as exc:
        msg = "IRIS density diagnostics require the optional dependency 'fiasco'."
        raise ImportError(msg) from exc

    if line_ratio_density_kwargs is None:
        line_ratio_density_kwargs = {}

    intensity_numerator = u.Quantity(intensity_numerator)
    intensity_denominator = u.Quantity(intensity_denominator)
    numerator_value, denominator_value = np.broadcast_arrays(intensity_numerator.value, intensity_denominator.value)
    ratio_value = np.divide(
        numerator_value,
        denominator_value,
        out=np.full(numerator_value.shape, np.nan),
        where=denominator_value != 0.0,
    )
    ratio = u.Quantity(ratio_value, intensity_numerator.unit / intensity_denominator.unit).to(u.dimensionless_unscaled)

    ratio_uncertainty = None
    if intensity_numerator_uncertainty is not None:
        numerator_uncertainty = u.Quantity(intensity_numerator_uncertainty)
        denominator_uncertainty = u.Quantity(intensity_denominator_uncertainty)
        numerator_uncertainty_value, numerator_value = np.broadcast_arrays(
            numerator_uncertainty.to_value(intensity_numerator.unit),
            intensity_numerator.value,
        )
        denominator_uncertainty_value, denominator_value = np.broadcast_arrays(
            denominator_uncertainty.to_value(intensity_denominator.unit),
            intensity_denominator.value,
        )
        numerator_fraction = np.divide(
            numerator_uncertainty_value,
            numerator_value,
            out=np.full(numerator_uncertainty_value.shape, np.nan),
            where=numerator_value != 0.0,
        )
        denominator_fraction = np.divide(
            denominator_uncertainty_value,
            denominator_value,
            out=np.full(denominator_uncertainty_value.shape, np.nan),
            where=denominator_value != 0.0,
        )
        ratio_uncertainty = np.abs(ratio) * np.sqrt(numerator_fraction**2 + denominator_fraction**2)
        try:
            ratio_uncertainty = np.broadcast_to(ratio_uncertainty.value, ratio.shape)
        except ValueError as exc:
            msg = (
                f"Uncertainty shapes {numerator_uncertainty.shape} and {denominator_uncertainty.shape} "
                f"are not broadcastable to intensity shape {ratio.shape}."
            )
            raise ValueError(msg) from exc
        ratio_uncertainty = u.Quantity(ratio_uncertainty, ratio.unit)

    theoretical_ratio = fiasco.line_ratio_density(
        ion,
        numerator,
        denominator,
        density_grid,
        **line_ratio_density_kwargs,
    )
    theoretical_ratio = u.Quantity(theoretical_ratio, u.dimensionless_unscaled).squeeze()
    if theoretical_ratio.ndim > 1:
        ion_temperature = u.Quantity(ion.temperature)
        if theoretical_ratio.shape[0] != ion_temperature.size:
            msg = "theoretical_ratio temperature axis does not match ion.temperature"
            raise ValueError(msg)

        if temperature is None:
            ratio_temperature = u.Quantity(ion.temperature)
            if ratio_temperature.size != 1:
                ratio_temperature = u.Quantity(ion.formation_temperature)
        else:
            ratio_temperature = u.Quantity(temperature)
        if ratio_temperature.size == 1:
            ratio_temperature = ratio_temperature.flat[0]

        order = np.argsort(ion_temperature.to_value(u.K))
        temperature_grid = ion_temperature.to_value(u.K)[order]
        ratio_values = theoretical_ratio.value[order]
        interp = interp1d(temperature_grid, ratio_values, axis=0, bounds_error=False, fill_value=np.nan)
        theoretical_ratio = u.Quantity(interp(ratio_temperature.to_value(u.K)), theoretical_ratio.unit).squeeze()

    density_grid = u.Quantity(density_grid)
    theoretical_ratio = u.Quantity(theoretical_ratio, u.dimensionless_unscaled)
    finite = np.isfinite(density_grid.value) & np.isfinite(theoretical_ratio.value)
    density_grid = density_grid[finite]
    theoretical_ratio = theoretical_ratio[finite]

    segments = []
    if density_grid.size < 2:
        segments.append((density_grid, theoretical_ratio))
    else:
        start = 0
        previous_sign = None
        for i, delta in enumerate(np.diff(theoretical_ratio.value)):
            sign = np.sign(delta)
            if sign == 0:
                continue
            if previous_sign is None:
                previous_sign = sign
                continue
            if sign != previous_sign:
                segments.append((density_grid[start : i + 1], theoretical_ratio[start : i + 1]))
                start = i
                previous_sign = sign
        segments.append((density_grid[start:], theoretical_ratio[start:]))
        segments = [(quantity, ratio_segment) for quantity, ratio_segment in segments if quantity.size >= 2]
    if len(segments) == 1:
        density_grid, theoretical_ratio = segments[0]
    elif segments:
        observed_ratio = ratio.value[np.isfinite(ratio.value)]
        if ratio_uncertainty is not None:
            ratio_min_obs = (ratio - ratio_uncertainty).value[np.isfinite((ratio - ratio_uncertainty).value)]
            ratio_max_obs = (ratio + ratio_uncertainty).value[np.isfinite((ratio + ratio_uncertainty).value)]
            observed_ratio_range = np.concatenate([ratio_min_obs, ratio_max_obs])
        else:
            observed_ratio_range = observed_ratio
        best_score = None
        for segment in segments:
            quantity, ratio_segment = segment
            ratio_min = np.nanmin(ratio_segment.value)
            ratio_max = np.nanmax(ratio_segment.value)
            in_range = np.logical_and(observed_ratio_range >= ratio_min, observed_ratio_range <= ratio_max).sum()
            score = (in_range, quantity.size, ratio_max - ratio_min)
            if best_score is None or score > best_score:
                density_grid, theoretical_ratio = segment
                best_score = score

    density = map_ratio_to_quantity(
        ratio,
        density_grid,
        theoretical_ratio,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    density_lower = None
    density_upper = None
    if ratio_uncertainty is not None:
        density_minus = map_ratio_to_quantity(
            ratio - ratio_uncertainty,
            density_grid,
            theoretical_ratio,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
        density_plus = map_ratio_to_quantity(
            ratio + ratio_uncertainty,
            density_grid,
            theoretical_ratio,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
        density_lower = u.Quantity(
            np.fmin(density_minus.to_value(density.unit), density_plus.to_value(density.unit)),
            density.unit,
        )
        density_upper = u.Quantity(
            np.fmax(density_minus.to_value(density.unit), density_plus.to_value(density.unit)),
            density.unit,
        )

    return {
        "ratio": ratio,
        "ratio_uncertainty": ratio_uncertainty,
        "density": density,
        "density_lower": density_lower,
        "density_upper": density_upper,
        "density_grid": u.Quantity(density_grid),
        "theoretical_ratio": u.Quantity(theoretical_ratio, u.dimensionless_unscaled),
    }
