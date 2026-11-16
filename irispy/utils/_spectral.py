"""
Shared helpers for spectral map outputs.
"""

from copy import copy, deepcopy
from enum import IntEnum
from numbers import Integral

import numpy as np

import astropy.units as u
from astropy.nddata import StdDevUncertainty, UnknownUncertainty

from ndcube import ExtraCoords

from irispy.spectrograph import SpectrogramCube, _wavelength_indices
from irispy.utils.lines import _BLEND_WIDTH, get_lines
from irispy.utils.utils import _boolean_mask


class _QualityFlag(IntEnum):
    """
    An `~enum.IntEnum` whose members are ``(value, description)`` pairs.
    """

    def __new__(cls, value, description):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj


def check_scaled(cube):
    """
    Raise a `ValueError` if ``cube`` holds unscaled data read with ``memmap=True``.
    """
    # The readers give raw FITS integers the DN_IRIS_*_UNSCALED units, which outlast arithmetic such as an exposure
    # time correction
    if any(base.name.endswith("_UNSCALED") for base in cube.unit.bases):
        msg = "The data are unscaled; read them with memmap=False"
        raise ValueError(msg)


def _to_unit(value, unit, name):
    """
    The `~astropy.units.Quantity` ``value`` in ``unit``, reading ``u.DN`` as the DN of
    ``unit``, such as ``DN_IRIS_FUV``.

    Raise a `TypeError` naming ``name`` if ``value`` is not a Quantity, and a
    `~astropy.units.UnitConversionError` if it does not convert.
    """
    target = unit.to_string() or "dimensionless"
    if not isinstance(value, u.Quantity):
        msg = f"{name} must be an astropy Quantity convertible to {target}, not {value!r}"
        raise TypeError(msg)
    # u.DN stands for the DN_IRIS_* unit of unit, in DN, DN/s or DN Å alike
    dn = next((base for base in unit.bases if base.name.startswith("DN_IRIS")), None)
    equivalencies = [] if dn is None else [(u.DN * unit / dn, unit)]
    try:
        return value.to(unit, equivalencies=equivalencies)
    except u.UnitConversionError:
        msg = f"{name} must be in units convertible to {target}, not {value.unit.to_string() or 'dimensionless'}"
        raise u.UnitConversionError(msg) from None


def check_velocity_range(velocity_range):
    """
    Raise a `ValueError` unless ``velocity_range`` is two increasing velocities.
    """
    if velocity_range.shape != (2,) or not velocity_range[0] < velocity_range[1]:
        msg = f"velocity_range must be two increasing velocities, (lower, upper), not {velocity_range}"
        raise ValueError(msg)


def in_windows(wavelengths, windows):
    """
    `True` for the ``wavelengths`` within any of the ``(lower, upper)`` ``windows``,
    ends included.

    Windows without samples are ignored. Raise a `ValueError` if none contain samples.
    """
    windows = u.Quantity(windows)
    if not windows.unit.is_equivalent(u.AA):
        msg = f"windows must be wavelengths, such as [1401.5, 1402.0] * u.AA, not {windows}"
        raise u.UnitConversionError(msg)
    if windows.shape == (2,):
        windows = windows[np.newaxis]
    if windows.ndim != 2 or windows.shape[1] != 2:
        msg = "The windows must have shape (2,) or (n, 2)"
        raise ValueError(msg)
    inside = np.zeros(wavelengths.shape, dtype=bool)
    for window in windows:
        inside[_wavelength_indices(wavelengths, window, allow_empty=True)] = True
    if not inside.any():
        msg = f"No wavelengths between any of the window bounds: {windows}"
        raise ValueError(msg)
    return inside


def _fit_background(cube, windows, degree):
    """
    The background `~irispy.utils.spectrograph.subtract_background` fits to ``cube``, an
    array of its shape.

    Spectra with no more than ``degree`` samples to fit are NaN.
    """
    wavelength_axis = cube.wavelength_axis
    wavelengths = cube.axis_world_coords(wavelength_axis)[0]
    index = np.flatnonzero(in_windows(wavelengths, windows))
    # Centered wavelengths keep the fit well conditioned
    vander = np.polynomial.polynomial.polyvander((wavelengths - wavelengths.mean()).to_value(u.AA), degree)
    samples = np.moveaxis(cube.data, wavelength_axis, -1)[..., index].astype(float)
    kept = np.isfinite(samples)
    kept &= ~np.moveaxis(_boolean_mask(cube), wavelength_axis, -1)[..., index]
    # Center each spectrum on its first kept sample. A constant the samples hold exactly is then
    # fitted exactly, whatever order the contractions round in, so its residuals stay zero instead
    # of crossing the sign cutoff in calculate_moments by a few ulps.
    anchor = np.take_along_axis(samples, np.argmax(kept, axis=-1)[..., np.newaxis], axis=-1)
    anchor = np.where(kept.any(axis=-1)[..., np.newaxis], anchor, 0.0)
    samples = np.where(kept, samples - anchor, 0)
    # The normal equations of each spectrum, with its own samples
    normal = np.einsum("...k,ki,kj->...ij", kept, vander[index], vander[index], optimize=True)
    enough = kept.sum(axis=-1) > degree
    normal[~enough] = np.eye(degree + 1)
    rhs = np.einsum("...k,ki->...i", samples, vander[index], optimize=True)
    coefficients = np.linalg.solve(normal, rhs[..., np.newaxis])[..., 0]
    # The background is the size of the cube: hold it in the data's floating-point type, evaluated in float64 a
    # block of spectra at a time, as float32 sums of large polynomial terms lose the background at degree 3 and up
    background = np.empty((*coefficients.shape[:-1], len(vander)), np.promote_types(cube.data.dtype, np.float32))
    flat, flat_coefficients, flat_anchor = (
        background.reshape(-1, len(vander)),
        coefficients.reshape(-1, degree + 1),
        anchor.reshape(-1, 1),
    )
    for start in range(0, len(flat), 4096):
        block = slice(start, start + 4096)
        flat[block] = flat_anchor[block] + flat_coefficients[block] @ vander.T
    background[~enough] = np.nan
    return np.moveaxis(background, -1, wavelength_axis)


def default_rest_wavelength(cube):
    """
    ``cube.meta.rest_wavelength``, or the one documented line ``cube`` covers if it does
    not cover ``cube.meta.rest_wavelength``, or its strongest line for a blend.
    """
    # A crop of a window can leave out its line, as a crop to Mg II h leaves out Mg II k. The cube covers
    # its pixels' full width, as a crop at a line keeps the pixel holding it although its center can be past it.
    wavelengths = cube.axis_world_coords(cube.wavelength_axis)[0]
    half = abs(cube.spectral_dispersion) / 2
    low, high = wavelengths.min() - half, wavelengths.max() + half
    rest = getattr(cube.meta, "rest_wavelength", None)
    if rest is not None and low <= rest <= high:
        return rest
    lines = get_lines([low, high], main_only=True)
    if not len(lines) or np.ptp(lines["wavelength"]) > _BLEND_WIDTH:
        return rest
    # Every reference model of the catalog ranks the lines of each blend alike.
    return lines["wavelength"][np.argmax(lines["intensity_quiet_sun_coronal"])]


def standard_deviation(cube):
    """
    The standard deviation of each sample of ``cube``, in its unit, or `None`.

    An `~astropy.nddata.UnknownUncertainty` is taken to be a standard deviation.
    """
    uncertainty = cube.uncertainty
    if uncertainty is None:
        return None
    if not isinstance(uncertainty, StdDevUncertainty | UnknownUncertainty):
        uncertainty = uncertainty.represent_as(StdDevUncertainty)
    sigma = np.asarray(uncertainty.array, dtype=float)
    if uncertainty.unit is not None and cube.unit is not None and uncertainty.unit != cube.unit:
        sigma = sigma * uncertainty.unit.to(cube.unit)
    return np.broadcast_to(sigma, cube.data.shape)


def make_map_cube(template, values, unit, *, mask=None, mask_invalid=False, uncertainty=None):
    combined_mask = None
    for next_mask in (template.mask, mask, ~np.isfinite(values) if mask_invalid else None):
        if next_mask is None:
            continue
        mask_array = np.asarray(next_mask, dtype=bool)
        combined_mask = mask_array.copy() if combined_mask is None else np.logical_or(combined_mask, mask_array)
    # Coordinates point back to their cube; copy them without copying its data.
    coordinate_memo = {id(template): None}
    return type(template)(
        values,
        wcs=template.wcs,
        uncertainty=uncertainty,
        unit=unit,
        meta=deepcopy(template.meta),
        mask=combined_mask,
        extra_coords=deepcopy(template.extra_coords, coordinate_memo),
        global_coords=deepcopy(template.global_coords, coordinate_memo),
    )


def drop_extra_coords_dependent_on_axis(extra_coords, axis, *, reindex):
    if not extra_coords or extra_coords.is_empty:
        return None
    new_extra_coords = ExtraCoords()
    for array_dimension, coord in extra_coords._lookup_tables:
        dimensions = (array_dimension,) if isinstance(array_dimension, Integral) else tuple(array_dimension)
        if axis in dimensions:
            continue
        new_array_dimension = array_dimension
        if reindex:
            dimensions = tuple(dimension - 1 if dimension > axis else dimension for dimension in dimensions)
            new_array_dimension = dimensions[0] if len(dimensions) == 1 else dimensions
        new_extra_coords._lookup_tables.append((new_array_dimension, deepcopy(coord)))
    return new_extra_coords


def make_spatial_template(cube, wavelength_axis):
    if cube.mask is not None:
        # Broadcast before slicing without changing the input cube's mask.
        cube = copy(cube)
        cube.mask = _boolean_mask(cube)
    template_slicer = [slice(None)] * cube.data.ndim
    template_slicer[wavelength_axis] = 0
    sliced_template = super(SpectrogramCube, cube).__getitem__(tuple(template_slicer))
    template_mask = None
    if cube.mask is not None:
        spatial_mask = np.all(cube.mask, axis=wavelength_axis)
        template_mask = spatial_mask if np.any(spatial_mask) else None
    if hasattr(cube.wcs, "dropaxis"):
        template_wcs = cube.wcs.dropaxis(cube.data.ndim - 1 - wavelength_axis)
    else:
        template_wcs = sliced_template.wcs
    return SpectrogramCube(
        sliced_template.data,
        template_wcs,
        uncertainty=sliced_template.uncertainty,
        unit=sliced_template.unit,
        meta=sliced_template.meta,
        mask=template_mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=True),
    )
