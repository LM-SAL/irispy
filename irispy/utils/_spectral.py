"""
Shared helpers for spectral map outputs.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np

from ndcube import ExtraCoords

from irispy.spectrograph import SpectrogramCube


def make_map_cube(template, values, unit, *, mask=None, mask_invalid=False):
    if mask is None and mask_invalid:
        mask = ~np.isfinite(values)
    return SpectrogramCube(
        values,
        template.wcs,
        None,
        unit,
        template.meta,
        mask=mask,
        extra_coords=template.extra_coords,
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
    template_slicer = [slice(None)] * cube.data.ndim
    template_slicer[wavelength_axis] = 0
    sliced_template = super(SpectrogramCube, cube).__getitem__(tuple(template_slicer))
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
        mask=sliced_template.mask,
        extra_coords=drop_extra_coords_dependent_on_axis(cube.extra_coords, wavelength_axis, reindex=True),
    )
