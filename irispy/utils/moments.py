"""
This module provides spectral moment calculation utilities for IRIS spectrogram cubes.
"""

import numpy as np

import astropy.units as u
from astropy import constants

from irispy.spectrograph import RasterCollection, SpectrogramCube

__all__ = ["calculate_moments"]


def _make_moment_cube(template, values, unit):
    new_cube = SpectrogramCube(
        values,
        template.wcs,
        None,
        unit,
        template.meta,
        mask=template.mask,
    )
    # Preserve extra coordinates if they exist on the template
    if hasattr(template, "_extra_coords"):
        new_cube._extra_coords = template._extra_coords
    return new_cube


@u.quantity_input(rest_wavelength=u.nm, wings=u.nm)
def calculate_moments(
    cube, *, rest_wavelength=None, wings=None, integrated=False, min_intensity=None, saturation_limit=None
):
    r"""
    Calculate the 0th, 1st, and 2nd spectral moments of a data cube.

    The moments are computed along the spectral (wavelength) axis for each
    spatial pixel:

    * 0th moment: total intensity, :math:`\sum I(\lambda_i)` (or :math:`\int I(\lambda) \, d\lambda` when ``integrated=True``)
    * 1st moment: centroid wavelength, :math:`\sum \lambda_i I(\lambda_i) / \sum I(\lambda_i)`
    * 2nd moment: standard deviation, :math:`\sqrt{\sum (\lambda_i - \lambda_0)^2 I(\lambda_i) / \sum I(\lambda_i)}`

    Parameters
    ----------
    cube : `irispy.spectrograph.SpectrogramCube`
        The input data cube. Must have a spectral (wavelength) axis.
    rest_wavelength : `astropy.units.Quantity`, optional
        The rest wavelength of the spectral line. Required if ``wings`` is given.
    wings : `astropy.units.Quantity`, optional
        The spectral range around ``rest_wavelength`` to include in the calculation.
        Must be an `~astropy.units.Quantity` with appropriate units (e.g., nm or Angstrom).
        If a scalar Quantity, it is applied symmetrically. If a tuple of two Quantities,
        they are the lower and upper offsets respectively.
    integrated : `bool`, optional
        If `True`, the 0th moment is computed as :math:`\int I(\lambda) \, d\lambda`
        with units of ``DN·nm``. If `False` (default), it is computed as :math:`\sum I(\lambda)`
        with units of ``DN`` (i.e., per-pixel sum, matching the convention used in
        Gaussian fitting).
    intensity_threshold : `float`, optional
        If given, pixels where the 0th moment is below this threshold are masked and are set to
        NaN for those pixels.
    saturation_limit : `float`, optional
        If given, pixels where any spectral bin in the (cropped) profile exceeds
        this value are treated as saturated and are set to NaN for those pixels.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        A collection containing 2D `~irispy.spectrograph.SpectrogramCube`
        objects with the spatial WCS preserved from the input cube.

        Always present:

        * ``"intensity"`` — 0th moment (total intensity)
        * ``"centroid"`` — 1st moment (centroid wavelength)
        * ``"width"`` — 2nd moment (standard deviation)

        Additionally, if ``rest_wavelength`` is provided:

        * ``"velocity"`` — Doppler shift from the centroid in km/s
        * ``"velocity_width"`` — line width converted to velocity units in km/s

    Notes
    -----
    * Negative and non-finite (NaN/inf) data values are set to zero before computing moments.
    * Wavelength coordinates are converted to **nm** internally, so ``centroid`` and ``width``
      are always returned in nm.
    * For a uniform spectral grid, the 1st and 2nd moments are identical regardless of the
      ``integrated`` setting because the pixel spacing cancels out in the ratio.

    References
    ----------
    * `Spectral-Cube moment maps <https://spectral-cube.readthedocs.io/en/latest/moments.html#moment-map-equations>`__
    * `arXiv:2005.02029, Section 3.1 <https://arxiv.org/abs/2005.02029>`__
    * `Færder et al. (2024), ApJ, Appendix C <https://iopscience.iop.org/article/10.3847/1538-4357/ac4223>`__
    """
    wavelength_axis = next(
        axis
        for axis, physical_types in enumerate(cube.array_axis_physical_types)
        if physical_types and "em.wl" in physical_types
    )
    wavelengths = cube.axis_world_coords(wavelength_axis)[0]
    if not isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths * u.one
    wavelengths = wavelengths.to(u.nm)
    data = cube.data.copy()
    data = np.where(data < 0, 0, data)
    data = np.where(np.isfinite(data), data, 0)
    if cube.uncertainty is not None:
        uncertainty = cube.uncertainty.array.copy()
        uncertainty = np.where(np.isfinite(uncertainty), uncertainty, 0)
    else:
        uncertainty = np.abs(data * 0.1)
    if wings is not None:
        if rest_wavelength is None:
            msg = "rest_wavelength must be provided when wings is given"
            raise ValueError(msg)
        rest_wavelength = u.Quantity(rest_wavelength)
        wavelengths_in_rest_unit = wavelengths.to(rest_wavelength.unit)
        if wings.isscalar:
            wing_low = wings
            wing_high = wings
        else:
            wing_low = wings[0]
            wing_high = wings[1]
        wvl_min = rest_wavelength - wing_low
        wvl_max = rest_wavelength + wing_high
        crop_mask = (wavelengths_in_rest_unit >= wvl_min) & (wavelengths_in_rest_unit <= wvl_max)
        crop_indices = np.where(crop_mask)[0]
        if len(crop_indices) == 0:
            msg = "No wavelength points found within the specified wings"
            raise ValueError(msg)
        slicer = [slice(None)] * data.ndim
        slicer[wavelength_axis] = crop_indices
        data = data[tuple(slicer)]
        uncertainty = uncertainty[tuple(slicer)]
        wavelengths = wavelengths[crop_mask]
    # Calculate wavelength step (assumed uniform)
    dwvl = np.mean(np.diff(wavelengths))
    # Move wavelength axis to the end for vectorised computation
    data_moved = np.moveaxis(data, wavelength_axis, -1)
    wvls = wavelengths.value
    # Broadcast wavelengths to match moved data shape
    broadcast_shape = [1] * data_moved.ndim
    broadcast_shape[-1] = -1
    wvls_broadcast = wvls.reshape(broadcast_shape)
    dwvl_value = dwvl.value
    # Weights for moment calculation: integrated (x dλ) or per-pixel
    if integrated:
        weights = data_moved * dwvl_value
        intensity_unit = cube.unit * dwvl.unit
    else:
        weights = data_moved
        intensity_unit = cube.unit
    # 0th moment
    intensity_value = np.nansum(weights, axis=-1)
    # 1st and 2nd moments are ratios where the weighting cancels out,
    # so the result is the same for uniform dλ regardless of ``integrated``.
    intensity_nonzero = intensity_value != 0
    centroid_numerator = np.nansum(weights * wvls_broadcast, axis=-1)
    with np.errstate(invalid="ignore"):
        centroid_value = np.where(intensity_nonzero, centroid_numerator / intensity_value, np.nan)
    # 2nd moment (variance)
    variance_numerator = np.nansum(((wvls_broadcast - centroid_value[..., np.newaxis]) ** 2) * weights, axis=-1)
    with np.errstate(invalid="ignore"):
        variance_value = np.where(intensity_nonzero, variance_numerator / intensity_value, np.nan)
    variance_value = np.where(variance_value < 0, np.nan, variance_value)
    stddev_value = np.sqrt(variance_value)

    if min_intensity is not None:
        low_intensity = intensity_value < min_intensity
        intensity_value = np.where(low_intensity, np.nan, intensity_value)
        centroid_value = np.where(low_intensity, np.nan, centroid_value)
        stddev_value = np.where(low_intensity, np.nan, stddev_value)

    if saturation_limit is not None:
        peak_value = np.max(data_moved, axis=-1)
        saturated = peak_value > saturation_limit
        intensity_value = np.where(saturated, np.nan, intensity_value)
        centroid_value = np.where(saturated, np.nan, centroid_value)
        stddev_value = np.where(saturated, np.nan, stddev_value)

    # Build a 2D spatial template by slicing out the wavelength axis.
    # This preserves the celestial WCS, mask, and meta for the new cubes.
    template_slicer = [slice(None)] * cube.data.ndim
    template_slicer[wavelength_axis] = 0
    template = cube[tuple(template_slicer)]

    intensity_cube = _make_moment_cube(template, intensity_value, intensity_unit)
    centroid_cube = _make_moment_cube(template, centroid_value, wavelengths.unit)
    width_cube = _make_moment_cube(template, stddev_value, wavelengths.unit)
    cubes = [
        ("intensity", intensity_cube),
        ("centroid", centroid_cube),
        ("width", width_cube),
    ]
    # Compute velocity equivalents when rest_wavelength is known
    if rest_wavelength is not None:
        rest_wavelength = u.Quantity(rest_wavelength)
        with np.errstate(invalid="ignore"):
            velocity_value = (
                ((centroid_value * wavelengths.unit).to(rest_wavelength.unit) - rest_wavelength)
                / rest_wavelength
                * constants.c.to(u.km / u.s)
            )
            velocity_width_value = (
                (stddev_value * wavelengths.unit).to(rest_wavelength.unit)
                / rest_wavelength
                * constants.c.to(u.km / u.s)
            )
        velocity_cube = _make_moment_cube(template, velocity_value.value, velocity_value.unit)
        velocity_width_cube = _make_moment_cube(template, velocity_width_value.value, velocity_width_value.unit)
        cubes.extend([("velocity", velocity_cube), ("velocity_width", velocity_width_cube)])
    return RasterCollection(cubes, aligned_axes=tuple(range(len(template.shape))))
