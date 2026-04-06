"""
Helpers for reconstructing cube-like objects.
"""

from copy import deepcopy
from typing import Any


def _build_cube_init_kwargs(cube) -> dict[str, Any]:
    kwargs = {
        "uncertainty": deepcopy(cube.uncertainty),
        "unit": cube.unit,
        "meta": deepcopy(cube.meta),
        "mask": deepcopy(cube.mask),
    }
    instrument_axes = getattr(cube, "instrument_axes", None)
    if instrument_axes is not None:
        kwargs["instrument_axes"] = deepcopy(instrument_axes)
    psf = getattr(cube, "psf", None)
    if psf is not None:
        kwargs["psf"] = deepcopy(psf)
    return kwargs
