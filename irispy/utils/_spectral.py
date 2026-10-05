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
    # The slit-jaw reader records the scaling, as scaled AIA data stay integer
    if not cube.meta.get("scaled", not np.issubdtype(cube.data.dtype, np.integer)):
        msg = "The data are unscaled; read them with memmap=False"
        raise ValueError(msg)


def in_windows(wavelengths, windows):
    """
    `True` for the ``wavelengths`` within any of the ``(lower, upper)`` ``windows``,
    ends included.

    Windows without samples are ignored. Raise a `ValueError` if none contain samples.
    """
    windows = u.Quantity(windows)
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
    return SpectrogramCube(
        values,
        template.wcs,
        uncertainty,
        unit,
        template.meta,
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
        cube.mask = np.broadcast_to(np.asarray(cube.mask, dtype=bool), cube.data.shape)
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
        sliced_template.uncertainty,
        sliced_template.unit,
        sliced_template.meta,
        mask=template_mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=True),
    )
