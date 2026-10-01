"""
Spectral moment calculation utilities for IRIS spectrogram cubes.
"""

import numpy as np

import astropy.units as u
from astropy import constants
from astropy.nddata import StdDevUncertainty

from irispy.spectrograph import RasterCollection
from irispy.utils._spectral import check_scaled, make_map_cube, make_spatial_template, standard_deviation

__all__ = ["calculate_moments"]


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
        Pixels with any sample above this, in ``cube.unit``, get NaN in every map.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        `~irispy.spectrograph.SpectrogramCube` maps with the spatial WCS of ``cube``:

        * ``"intensity"`` — 0th moment
        * ``"centroid"`` — 1st moment, in nm
        * ``"width"`` — 2nd moment, in nm
        * ``"velocity"`` — Doppler velocity of the centroid in km/s, if ``rest_wavelength`` is known
        * ``"velocity_width"`` — width in km/s, if ``rest_wavelength`` is known

        Each map has a `~astropy.nddata.StdDevUncertainty` if ``cube`` has an uncertainty
        (e.g. read with ``uncertainty=True``).

    Notes
    -----
    * Negative, non-finite and masked samples are set to zero and add no uncertainty.
    * Uncertainties are propagated to first order, treating an `~astropy.nddata.UnknownUncertainty`
      as a standard deviation. They are NaN where undefined: the intensity error where no sample is
      left, the centroid and velocity errors where fewer than two are left, and the width and velocity
      width errors where the width is 0.
    * The uncertainties are statistical only and unreliable below a signal-to-noise ratio of about 5.
      They leave out the wavelength calibration and orbital drift (several km/s, see
      `#198 <https://github.com/LM-SAL/irispy/pull/198>`__) and the bias from zeroing negative samples,
      which can widen faint lines several times (see ``min_intensity``). Where the background is near
      zero they are conservative, up to about 25% too large.
    * Samples are taken as independent, but the level 2 resampling correlates neighbours, which
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
        wavelengths_in_rest_unit = wavelengths.to(rest_wavelength.unit)
        wvl_min = rest_wavelength - wing_low.to(rest_wavelength.unit)
        wvl_max = rest_wavelength + wing_high.to(rest_wavelength.unit)
        crop_mask = (wavelengths_in_rest_unit >= wvl_min) & (wavelengths_in_rest_unit <= wvl_max)
        crop_indices = np.where(crop_mask)[0]
        if len(crop_indices) == 0:
            msg = "No wavelength points found within the specified wings"
            raise ValueError(msg)
        slicer = [slice(None)] * data.ndim
        slicer[wavelength_axis] = crop_indices
        data = data[tuple(slicer)]
        if mask is not None:
            mask = mask[tuple(slicer)]
        if sigma is not None:
            sigma = sigma[tuple(slicer)]
        wavelengths = wavelengths[crop_mask]
    data = np.array(data, dtype=float, copy=True)
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
        saturated = np.max(data_moved, axis=-1) > u.Quantity(saturation_limit, cube.unit).value
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
    return RasterCollection(cubes, aligned_axes=tuple(range(len(template.shape))))
