"""
Spectral moment and window maps for IRIS spectrogram cubes.
"""

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.nddata import NDDataArray, StdDevUncertainty

from irispy.spectrograph import RasterCollection, _wavelength_indices
from irispy.utils._spectral import check_scaled, make_map_cube, make_spatial_template, standard_deviation
from irispy.utils.constants import DN_UNIT

__all__ = ["average_window", "calculate_moments"]


def calculate_moments(
    cube, *, rest_wavelength=None, wings=None, integrated=False, min_intensity=None, saturation_limit=None
):
    r"""
    Calculate the 0th, 1st and 2nd spectral moments of a spectrogram cube.

    For each spatial pixel, along the wavelength axis:

    * 0th moment (intensity): :math:`I_0 = \sum I(\lambda_i)`
    * 1st moment (centroid): :math:`\lambda_0 = \sum \lambda_i I(\lambda_i) / I_0`
    * 2nd moment (width, a standard deviation): :math:`\sqrt{\sum (\lambda_i - \lambda_0)^2 I(\lambda_i) / I_0}`

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Input cube with a wavelength axis.
    rest_wavelength : `astropy.units.Quantity`, optional
        Rest wavelength of the line. Defaults to ``cube.meta.rest_wavelength``, if present.
    wings : `astropy.units.Quantity` or `tuple` of `astropy.units.Quantity`, optional
        Wavelength range to use around ``rest_wavelength``: one offset for both sides, or
        ``(lower, upper)`` offsets.
    integrated : `bool`, optional
        If `True`, multiply the 0th moment by the mean wavelength spacing, giving
        :math:`\int I(\lambda) \, d\lambda` in ``cube.unit * nm``; the other moments do not change.
        Defaults to `False`, a sum in ``cube.unit`` as in Gaussian fitting.
    min_intensity : `float` or `astropy.units.Quantity`, optional
        Pixels whose 0th moment is below this get NaN in every map.
    saturation_limit : `float` or `astropy.units.Quantity`, optional
        Pixels with any sample at or above this many DN, or of +Inf, get NaN in every map.
        A `float` is in the DN of ``cube``. For a cube in DN per second, the limit is divided
        by the exposure time of each step. Level 2 data saturate at
        ``irispy.utils.constants.SATURATION_LIMIT``.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        `~irispy.spectrograph.SpectrogramCube` maps with the spatial WCS of ``cube``:

        * ``"intensity"`` — 0th moment
        * ``"centroid"`` — 1st moment, in nm
        * ``"width"`` — 2nd moment, in nm
        * ``"velocity"`` — Doppler velocity of the centroid in km/s, if ``rest_wavelength`` is known
        * ``"velocity_width"`` — width in km/s, if ``rest_wavelength`` is known
        * ``"saturated"`` — `True` for the pixels that tripped ``saturation_limit``, if it is given

        Each moment map has a `~astropy.nddata.StdDevUncertainty` if ``cube`` has an uncertainty
        (e.g. read with ``uncertainty=True``).

    Notes
    -----
    * Negative, non-finite and masked samples are set to zero and add no uncertainty.
    * ``saturation_limit`` is checked before that zeroing, on every sample in the wavelength range,
      so +Inf samples count. Level 2 files cannot hold +Inf and clip saturated samples to
      ``irispy.utils.constants.SATURATION_LIMIT``, which counts as saturated.
    * Uncertainties are propagated to first order, treating an `~astropy.nddata.UnknownUncertainty`
      as a standard deviation. They are NaN where undefined: the intensity error where no sample is
      left, the centroid and velocity errors where fewer than two are left, and the width and velocity
      width errors where the width is 0.
    * The uncertainties are statistical only and unreliable below a signal-to-noise ratio of about 5.
      They leave out the wavelength calibration and orbital drift (several km/s, see
      `#198 <https://github.com/LM-SAL/irispy/pull/198>`__) and the bias from zeroing negative samples,
      which can widen faint lines several times (see ``min_intensity``). Where the background is near
      zero they are conservative, up to about 25% too large.
    * Samples are taken as independent, but the level 2 resampling correlates neighbors, which
      changes the intensity uncertainty of a 15-sample line by 2 to 15%.

    References
    ----------
    * `Spectral-Cube moment maps <https://spectral-cube.readthedocs.io/en/latest/moments.html#moment-map-equations>`__
    * :cite:t:`yu2020`, Section 3.1
    * :cite:t:`cheung2022`, Appendix C
    """
    check_scaled(cube)
    if rest_wavelength is None:
        rest_wavelength = getattr(cube.meta, "rest_wavelength", None)
    wavelength_axis = cube.wavelength_axis
    wavelengths = cube.axis_world_coords(wavelength_axis)[0].to(u.nm)
    data = np.asarray(cube.data)
    mask = None if cube.mask is None else np.asarray(cube.mask, dtype=bool)
    sigma = standard_deviation(cube)
    if wings is not None:
        if rest_wavelength is None:
            msg = "rest_wavelength must be provided (or detectable from cube metadata) when wings is given"
            raise ValueError(msg)
        rest_wavelength = u.Quantity(rest_wavelength)
        wings = u.Quantity(wings)
        if wings.shape not in {(), (2,)}:
            msg = "wings must be one offset or a (lower, upper) pair"
            raise ValueError(msg)
        wing_low, wing_high = (wings, wings) if wings.isscalar else wings
        crop_indices = _wavelength_indices(
            wavelengths, u.Quantity([rest_wavelength - wing_low, rest_wavelength + wing_high])
        )
        slicer = [slice(None)] * data.ndim
        slicer[wavelength_axis] = crop_indices
        data = data[tuple(slicer)]
        if mask is not None:
            mask = mask[tuple(slicer)]
        if sigma is not None:
            sigma = sigma[tuple(slicer)]
        wavelengths = wavelengths[crop_indices]
    data = np.array(data, dtype=float, copy=True)
    if saturation_limit is not None:
        saturated = np.any(data >= _saturation_limit(cube, saturation_limit), axis=wavelength_axis)
    dropped = (data < 0) | ~np.isfinite(data) | (False if mask is None else mask)
    data[dropped] = 0

    dwvl = np.abs(np.mean(np.diff(wavelengths)))
    data_moved = np.moveaxis(data, wavelength_axis, -1)
    wvls = wavelengths.value
    dwvl_value = dwvl.value

    if integrated:
        weights = data_moved * dwvl_value
        intensity_unit = cube.unit * dwvl.unit
    else:
        weights = data_moved
        intensity_unit = cube.unit

    intensity_value = np.nansum(weights, axis=-1)
    intensity_nonzero = intensity_value != 0
    centroid_numerator = np.nansum(weights * wvls, axis=-1)
    with np.errstate(invalid="ignore"):
        centroid_value = np.where(intensity_nonzero, centroid_numerator / intensity_value, np.nan)
    offset_squared = (wvls - centroid_value[..., np.newaxis]) ** 2
    variance_numerator = np.nansum(offset_squared * weights, axis=-1)
    with np.errstate(invalid="ignore"):
        variance_value = np.where(intensity_nonzero, variance_numerator / intensity_value, np.nan)
    variance_value = np.where(variance_value < 0, np.nan, variance_value)
    stddev_value = np.sqrt(variance_value)

    if min_intensity is not None:
        low_intensity = intensity_value < u.Quantity(min_intensity, intensity_unit).value
        intensity_value = np.where(low_intensity, np.nan, intensity_value)
        centroid_value = np.where(low_intensity, np.nan, centroid_value)
        stddev_value = np.where(low_intensity, np.nan, stddev_value)

    if saturation_limit is not None:
        intensity_value = np.where(saturated, np.nan, intensity_value)
        centroid_value = np.where(saturated, np.nan, centroid_value)
        stddev_value = np.where(saturated, np.nan, stddev_value)

    errors = {}
    if sigma is not None:
        # First-order propagation of independent sample errors.
        weight_sigma = np.moveaxis(np.where(dropped, 0, sigma), wavelength_axis, -1) * (dwvl_value if integrated else 1)
        weight_variance = weight_sigma**2
        kept = (~dropped).sum(axis=wavelength_axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            errors["intensity"] = np.where(kept > 0, np.sqrt(weight_variance.sum(axis=-1)), np.nan)
            centroid_error = np.sqrt((offset_squared * weight_variance).sum(axis=-1)) / intensity_value
            errors["centroid"] = np.where(kept > 1, centroid_error, np.nan)
            width_terms = (offset_squared - variance_value[..., np.newaxis]) ** 2 * weight_variance
            width_error = np.sqrt(width_terms.sum(axis=-1)) / intensity_value / (2 * stddev_value)
            # Count the samples, as roundoff can leave a width of 0 at about 1e-14
            errors["width"] = np.where(np.count_nonzero(weights, axis=-1) > 1, width_error, np.nan)

    maps = [
        ("intensity", intensity_value, intensity_unit),
        ("centroid", centroid_value, wavelengths.unit),
        ("width", stddev_value, wavelengths.unit),
    ]
    if rest_wavelength is not None:
        rest_wavelength = u.Quantity(rest_wavelength)

        def to_velocity(delta_wavelength):
            with np.errstate(invalid="ignore"):
                return (delta_wavelength * wavelengths.unit / rest_wavelength * constants.c).to_value(u.km / u.s)

        if sigma is not None:
            errors["velocity"], errors["velocity_width"] = to_velocity(errors["centroid"]), to_velocity(errors["width"])
        velocity = to_velocity(centroid_value - rest_wavelength.to_value(wavelengths.unit))
        maps += [("velocity", velocity, u.km / u.s), ("velocity_width", to_velocity(stddev_value), u.km / u.s)]

    template = make_spatial_template(cube, wavelength_axis)

    def _make_cube(name, values, unit):
        uncertainty = None if sigma is None else StdDevUncertainty(np.where(np.isnan(values), np.nan, errors[name]))
        return make_map_cube(template, values, unit, mask_invalid=True, uncertainty=uncertainty)

    cubes = [(name, _make_cube(name, values, unit)) for name, values, unit in maps]
    if saturation_limit is not None:
        cubes.append(("saturated", make_map_cube(template, saturated, u.dimensionless_unscaled)))
    return RasterCollection(cubes, aligned_axes=tuple(range(len(template.shape))))


def _saturation_limit(cube, saturation_limit):
    """
    ``saturation_limit``, in DN, in ``cube.unit``, broadcastable to ``cube.data``.
    """
    # The same test sunraster's apply_exposure_time_correction uses
    per_second = u.s in cube.unit.decompose().bases
    dn_unit = cube.unit * u.s if per_second else cube.unit
    if not (dn_unit.is_equivalent(u.DN) or dn_unit in DN_UNIT.values()):
        msg = f"saturation_limit needs a cube in DN or DN per second, not {cube.unit}"
        raise ValueError(msg)
    if isinstance(saturation_limit, u.Quantity):
        saturation_limit = saturation_limit.to_value(dn_unit, equivalencies=[(u.DN, dn_unit)])
    if not per_second:
        return saturation_limit
    exposure_time = cube.meta["exposure time"].to_value(u.s)
    if np.ndim(exposure_time):
        shape = [1] * cube.data.ndim
        shape[cube.meta.axes["exposure time"][0]] = -1
        exposure_time = exposure_time.reshape(shape)
    # Steps of 0 s, with no data, get an infinite limit that only +Inf reaches
    with np.errstate(divide="ignore"):
        return saturation_limit / exposure_time


def average_window(cube, wavelength_range, *, method="mean"):
    r"""
    Average or sum a spectrogram cube over a wavelength window, giving an intensity map.

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        Input cube with a wavelength axis.
    wavelength_range : `astropy.units.Quantity`
        The ``(lower, upper)`` wavelengths of the window, ends included.
    method : `str`, optional
        ``"mean"`` (the default) or ``"sum"`` of the samples in the window.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube`
        Map in ``cube.unit`` with the spatial WCS of ``cube``. It has a `~astropy.nddata.StdDevUncertainty`
        if ``cube`` has an uncertainty (e.g. read with ``uncertainty=True``).

    Notes
    -----
    * Masked and non-finite samples are left out, so a sum over a partly masked window is low.
      Pixels with no sample left are NaN and masked.
    * Uncertainties are propagated taking the samples as independent, treating an
      `~astropy.nddata.UnknownUncertainty` as a standard deviation, as in `calculate_moments`.
    * For :math:`\int I(\lambda) \, d\lambda`, multiply a sum by the absolute wavelength step,
      ``abs(cube.spectral_dispersion)``.
    """
    check_scaled(cube)
    if method not in {"mean", "sum"}:
        msg = f'method must be "mean" or "sum", not {method!r}'
        raise ValueError(msg)
    wavelength_axis = cube.wavelength_axis
    window = _wavelength_indices(cube.axis_world_coords(wavelength_axis)[0], wavelength_range)
    data = np.take(cube.data, window, axis=wavelength_axis).astype(float)
    dropped = ~np.isfinite(data)
    if cube.mask is not None:
        mask = np.broadcast_to(np.asarray(cube.mask, dtype=bool), cube.data.shape)
        dropped |= np.take(mask, window, axis=wavelength_axis)
    sigma = standard_deviation(cube)
    uncertainty = None if sigma is None else StdDevUncertainty(np.take(sigma, window, axis=wavelength_axis))
    reduction = getattr(NDDataArray(data, uncertainty=uncertainty, mask=dropped), method)
    result = reduction(axis=wavelength_axis, operation_ignores_mask=True)
    values = np.where(result.mask, np.nan, result.data)
    uncertainty = None if sigma is None else StdDevUncertainty(np.where(result.mask, np.nan, result.uncertainty.array))
    template = make_spatial_template(cube, wavelength_axis)
    return make_map_cube(template, values, cube.unit, mask_invalid=True, uncertainty=uncertainty)
