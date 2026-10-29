"""
Utilities for removing cosmic rays from IRIS data.
"""

from typing import Any
from collections.abc import Mapping

import numpy as np

from irispy.utils.utils import _boolean_mask, _import_optional

__all__ = ["remove_cosmic_rays"]


def _remove_cosmic_rays_rsliding(
    data,
    mask,
    *,
    method_kwargs: dict[str, Any],
) -> np.ndarray:
    slidingsigmaclipping = _import_optional(
        "rsliding", reason="method='rsliding'", extra="cosmic-rays"
    ).SlidingSigmaClipping
    method_kwargs = dict(method_kwargs)
    method_kwargs["masked_array"] = True
    data = np.asarray(data)
    mask = mask | ~np.isfinite(data)
    working_data = data.astype(np.float64, copy=True)
    working_data[mask] = np.nan
    cleaned = np.ma.getdata(slidingsigmaclipping(data=working_data, **method_kwargs).clipped)
    # rsliding fills the NaN it was given with its window's median
    cleaned[mask] = data[mask]
    return cleaned


def _remove_cosmic_rays_astroscrappy(
    data,
    mask,
    *,
    method_kwargs: dict[str, Any],
) -> np.ndarray:
    astroscrappy = _import_optional("astroscrappy", reason="method='astroscrappy'", extra="cosmic-rays")
    method_kwargs = dict(method_kwargs)
    method_kwargs.setdefault("verbose", False)
    inmask = np.broadcast_to(np.asarray(method_kwargs.pop("inmask", False), dtype=bool), data.shape)
    cleaned_data = np.empty(data.shape, dtype=data.dtype)
    for index in np.ndindex(data.shape[:-2]):
        frame = np.asarray(data[index])
        frame_mask = mask[index] | inmask[index] | ~np.isfinite(frame)
        _, cleaned_frame = astroscrappy.detect_cosmics(
            np.where(frame_mask, 0.0, frame), inmask=frame_mask, **method_kwargs
        )
        # astroscrappy returns the zeros it was given at the masked samples
        cleaned_data[index] = np.where(frame_mask, frame, cleaned_frame)
    return cleaned_data


def remove_cosmic_rays(
    cube,
    *,
    method: str = "rsliding",
    method_kwargs: Mapping[str, Any] | None = None,
):
    """
    Remove cosmic rays from a cube and return a cleaned cube.

    Parameters
    ----------
    cube : `~irispy.sji.SJICube` or `~irispy.spectrograph.SpectrogramCube`
        Cube object to clean.
    method : ``{"rsliding", "astroscrappy"}``, optional
        Cosmic ray removal backend.
    method_kwargs : `dict`, optional
        Keyword arguments passed to the selected backend, under its own names: for
        example ``sigma`` and ``max_iters`` for ``rsliding``, ``sigclip`` and ``niter``
        for ``astroscrappy``. An integer ``kernel`` for ``rsliding`` is a window of that
        size along every axis, so on a cube of several rasters it also spans neighboring
        rasters; a tuple, such as ``(1, 5, 5, 5)``, keeps each raster to itself.

    Returns
    -------
    `~irispy.sji.SJICube` or `~irispy.spectrograph.SpectrogramCube`
        A new cube with cleaned data and copied metadata/coordinates. Masked and
        non-finite samples, and those ``inmask`` marks for ``astroscrappy``, keep their
        value. The cleaned data are in memory. The mask and uncertainty are those of ``cube``,
        so a replaced spike is not masked and keeps the uncertainty it had as a spike.
    """
    # Avoid circular imports
    from irispy.utils._spectral import check_scaled, make_map_cube  # NOQA: PLC0415

    check_scaled(cube)
    method = method.lower()
    kwargs = dict(method_kwargs or {})
    backends = {
        "rsliding": _remove_cosmic_rays_rsliding,
        "astroscrappy": _remove_cosmic_rays_astroscrappy,
    }
    if method not in backends:
        msg = f"Unsupported method {method!r}. Supported methods are: {sorted(backends)}."
        raise ValueError(msg)
    cleaned_data = backends[method](cube.data, _boolean_mask(cube), method_kwargs=kwargs)
    return make_map_cube(cube, cleaned_data, cube.unit, uncertainty=cube.uncertainty)
