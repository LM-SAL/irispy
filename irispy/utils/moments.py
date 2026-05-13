"""
Spectral moment calculation utilities for IRIS spectrogram cubes.
"""

import contextlib

import numpy as np

import astropy.units as u
from astropy import constants

from irispy.spectrograph import RasterCollection
from irispy.utils._spectral import make_map_cube, make_spatial_template

__all__ = ["calculate_moments"]


def _parse_wings(wings):
    if isinstance(wings, u.Quantity):
        if wings.isscalar:
            return wings, wings
        if len(wings) != 2:
            msg = "wings must be a scalar Quantity or a two-element Quantity"
            raise ValueError(msg)
        return wings[0], wings[1]
    if isinstance(wings, (tuple, list)) and len(wings) == 2:
        if not all(isinstance(wing, u.Quantity) for wing in wings):
            msg = "wings tuple elements must be astropy.units.Quantity"
            raise TypeError(msg)
        return wings[0], wings[1]
    msg = "wings must be an astropy.units.Quantity or a tuple of two Quantities"
    raise TypeError(msg)


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
        The rest wavelength of the spectral line. If omitted, read from
        ``cube.meta.rest_wavelength`` (requires an `~irispy.meta.SGMeta`
        instance with a ``TWAVE<n>`` FITS keyword). Required if ``wings``
        is given and no rest wavelength can be auto-detected.
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
    min_intensity : `float` or `astropy.units.Quantity`, optional
        Minimum integrated (or per-pixel) intensity required for a pixel to be
        considered valid. Pixels below this value have all moments set to NaN.
    saturation_limit : `float` or `astropy.units.Quantity`, optional
        Maximum allowed peak intensity in any spectral bin. Pixels exceeding
        this value have all moments set to NaN.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
        A collection containing 2D `~irispy.spectrograph.SpectrogramCube`
        objects with the spatial WCS preserved from the input cube.

        Always present:

        * ``"intensity"`` — 0th moment (total intensity)
        * ``"centroid"`` — 1st moment (centroid wavelength)
        * ``"width"`` — 2nd moment (standard deviation)

        Additionally, if ``rest_wavelength`` is known:

        * ``"velocity"`` — Doppler shift from the centroid in km/s
        * ``"velocity_width"`` — line width converted to velocity units in km/s

    Notes
    -----
    * Negative and non-finite data values are set to zero before computing moments.
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
    if rest_wavelength is None:
        with contextlib.suppress(AttributeError, TypeError):
            rest_wavelength = cube.meta.rest_wavelength
    try:
        wavelength_axis = cube.wavelength_axis
    except (AttributeError, StopIteration) as exc:
        msg = "Could not identify a spectral wavelength axis on the input cube"
        raise ValueError(msg) from exc
    wavelengths = cube.axis_world_coords(wavelength_axis)[0]
    if not isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths * u.one
    wavelengths = wavelengths.to(u.nm)
    data = np.asarray(cube.data)
    mask = None if cube.mask is None else np.asarray(cube.mask, dtype=bool)
    if wings is not None:
        if rest_wavelength is None:
            msg = "rest_wavelength must be provided (or detectable from cube metadata) when wings is given"
            raise ValueError(msg)
        rest_wavelength = u.Quantity(rest_wavelength)
        wing_low, wing_high = _parse_wings(wings)
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
        wavelengths = wavelengths[crop_mask]
    data = np.array(data, dtype=float, copy=True)
    if mask is not None:
        data[mask] = 0
    data[(data < 0) | ~np.isfinite(data)] = 0

    dwvl = np.mean(np.diff(wavelengths))
    data_moved = np.moveaxis(data, wavelength_axis, -1)
    wvls = wavelengths.value
    broadcast_shape = [1] * data_moved.ndim
    broadcast_shape[-1] = -1
    wvls_broadcast = wvls.reshape(broadcast_shape)
    dwvl_value = dwvl.value

    if integrated:
        weights = data_moved * dwvl_value
        intensity_unit = cube.unit * dwvl.unit
    else:
        weights = data_moved
        intensity_unit = cube.unit

    intensity_value = np.nansum(weights, axis=-1)
    intensity_nonzero = intensity_value != 0
    centroid_numerator = np.nansum(weights * wvls_broadcast, axis=-1)
    with np.errstate(invalid="ignore"):
        centroid_value = np.where(intensity_nonzero, centroid_numerator / intensity_value, np.nan)
    variance_numerator = np.nansum(((wvls_broadcast - centroid_value[..., np.newaxis]) ** 2) * weights, axis=-1)
    with np.errstate(invalid="ignore"):
        variance_value = np.where(intensity_nonzero, variance_numerator / intensity_value, np.nan)
    variance_value = np.where(variance_value < 0, np.nan, variance_value)
    stddev_value = np.sqrt(variance_value)

    if min_intensity is not None:
        min_intensity_value = (
            min_intensity.to_value(intensity_unit) if isinstance(min_intensity, u.Quantity) else min_intensity
        )
        low_intensity = intensity_value < min_intensity_value
        intensity_value = np.where(low_intensity, np.nan, intensity_value)
        centroid_value = np.where(low_intensity, np.nan, centroid_value)
        stddev_value = np.where(low_intensity, np.nan, stddev_value)

    if saturation_limit is not None:
        saturation_limit_value = (
            saturation_limit.to_value(cube.unit) if isinstance(saturation_limit, u.Quantity) else saturation_limit
        )
        saturated = np.max(data_moved, axis=-1) > saturation_limit_value
        intensity_value = np.where(saturated, np.nan, intensity_value)
        centroid_value = np.where(saturated, np.nan, centroid_value)
        stddev_value = np.where(saturated, np.nan, stddev_value)

    template = make_spatial_template(cube, wavelength_axis)
    cubes = [
        ("intensity", make_map_cube(template, intensity_value, intensity_unit, mask_invalid=True)),
        ("centroid", make_map_cube(template, centroid_value, wavelengths.unit, mask_invalid=True)),
        ("width", make_map_cube(template, stddev_value, wavelengths.unit, mask_invalid=True)),
    ]
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
        cubes.extend(
            [
                ("velocity", make_map_cube(template, velocity_value.value, velocity_value.unit, mask_invalid=True)),
                (
                    "velocity_width",
                    make_map_cube(template, velocity_width_value.value, velocity_width_value.unit, mask_invalid=True),
                ),
            ]
        )
    return RasterCollection(cubes, aligned_axes=tuple(range(len(template.shape))))
