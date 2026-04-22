"""
Utilities for removing cosmic rays from IRIS data.
"""

from copy import deepcopy
from typing import Any
from importlib import import_module
from collections.abc import Mapping

import numpy as np

__all__ = ["remove_cosmic_rays"]


def _import_optional_backend(module_name: str, *, method: str):
    try:
        return import_module(module_name)
    except ImportError as exc:
        msg = (
            f"{module_name} is an optional dependency required for method='{method}'. "
            f"Install it with `pip install {module_name}` or "
            "`pip install 'irispy-lmsal[cosmic-rays]'`."
        )
        raise ImportError(msg) from exc


def _remove_cosmic_rays_rsliding(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float | None,
    max_iters: int | None,
    method_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    slidingsigmaclipping = _import_optional_backend("rsliding", method="rsliding").SlidingSigmaClipping
    kwargs = {
        "kernel": 3,
        "center_choice": "median",
        "borders": "reflect",
        "threads": 1,
        **method_kwargs,
    }
    kwargs["sigma"] = sigma if sigma is not None else kwargs.get("sigma", 3.0)
    kwargs["max_iters"] = max_iters if max_iters is not None else kwargs.get("max_iters", 5)
    kwargs["masked_array"] = True
    working_data = data.copy()
    working_data[mask] = np.nan
    clipped = slidingsigmaclipping(data=working_data, **kwargs).clipped
    cleaned_data = np.ma.getdata(clipped)
    cosmic_ray_mask = np.asarray(np.ma.getmaskarray(clipped), dtype=bool)
    return cleaned_data, cosmic_ray_mask


def _remove_cosmic_rays_astroscrappy(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float | None,
    max_iters: int | None,
    method_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    astroscrappy = _import_optional_backend("astroscrappy", method="astroscrappy")
    kwargs = {"verbose": False, **method_kwargs}
    kwargs["sigclip"] = sigma if sigma is not None else kwargs.get("sigclip", 4.5)
    kwargs["niter"] = max_iters if max_iters is not None else kwargs.get("niter", 4)
    inmask = kwargs.pop("inmask", None)
    if inmask is not None:
        mask = mask | np.asarray(inmask, dtype=bool)
    working_data = data.copy()
    working_data[mask] = 0.0
    cleaned_data = np.empty_like(working_data)
    cosmic_ray_mask = np.zeros(working_data.shape, dtype=bool)
    if working_data.ndim == 2:
        cosmic_ray_mask, cleaned_data = astroscrappy.detect_cosmics(working_data, inmask=mask, **kwargs)
        return cleaned_data, np.asarray(cosmic_ray_mask, dtype=bool)
    for index in np.ndindex(working_data.shape[:-2]):
        frame_mask, cleaned_frame = astroscrappy.detect_cosmics(working_data[index], inmask=mask[index], **kwargs)
        cosmic_ray_mask[index] = frame_mask
        cleaned_data[index] = cleaned_frame
    return cleaned_data, cosmic_ray_mask


def remove_cosmic_rays(
    cube,
    *,
    method: str = "rsliding",
    sigma: float | None = None,
    max_iters: int | None = None,
    method_kwargs: Mapping[str, Any] | None = None,
):
    """
    Remove cosmic rays from a cube and return a cleaned cube.

    Parameters
    ----------
    cube : irispy.sji.SJICube, irispy.spectrograph.SpectrogramCube, \
           or irispy.spectrograph.SpectrogramCubeSequence
        Cube or sequence object to clean.
    method : ``{"rsliding", "astroscrappy"}``, optional
        Cosmic ray removal backend. ``"rsliding"`` is the default and operates on
        the full array. ``"astroscrappy"`` is applied frame-by-frame over the last
        two axes.
    sigma : `float`, optional
        Shared clipping threshold override. This maps to ``sigma`` for
        ``rsliding`` and ``sigclip`` for ``astroscrappy``.
    max_iters : `int`, optional
        Shared iteration-count override. This maps to ``max_iters`` for
        ``rsliding`` and ``niter`` for ``astroscrappy``.
    method_kwargs : `dict`, optional
        Additional keyword arguments passed to the selected backend.

    Returns
    -------
    irispy.sji.SJICube, irispy.spectrograph.SpectrogramCube, or
    irispy.spectrograph.SpectrogramCubeSequence
        A new cube or sequence with cleaned data and copied metadata/coordinates.
    """
    from irispy.spectrograph import SpectrogramCubeSequence  # NOQA: PLC0415

    if isinstance(cube, SpectrogramCubeSequence):
        return type(cube)(
            [
                remove_cosmic_rays(
                    subcube,
                    method=method,
                    sigma=sigma,
                    max_iters=max_iters,
                    method_kwargs=method_kwargs,
                )
                for subcube in cube
            ],
            meta=deepcopy(cube.meta),
            common_axis=getattr(cube, "_common_axis", None),
        )

    method = method.lower()
    working_mask = np.zeros(cube.data.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool)
    if np.issubdtype(cube.data.dtype, np.floating):
        working_mask = working_mask | np.isnan(cube.data)
    kwargs = dict(method_kwargs or {})
    backends = {
        "rsliding": lambda: _remove_cosmic_rays_rsliding(
            cube.data,
            working_mask,
            sigma=sigma,
            max_iters=max_iters,
            method_kwargs=kwargs,
        ),
        "astroscrappy": lambda: _remove_cosmic_rays_astroscrappy(
            cube.data,
            working_mask,
            sigma=sigma,
            max_iters=max_iters,
            method_kwargs=kwargs,
        ),
    }
    if method not in backends:
        msg = f"Unsupported method {method!r}. Supported methods are: {sorted(backends)}."
        raise ValueError(msg)
    cleaned_data, _ = backends[method]()

    cleaned_cube_kwargs = {
        "data": cleaned_data,
        "mask": "copy",
        "nddata_type": type(cube),
        "extra_coords": "copy",
        "global_coords": "copy",
    }
    if hasattr(cube, "scaled"):
        cleaned_cube_kwargs["scaled"] = "copy"
    if hasattr(cube, "_basic_wcs"):
        cleaned_cube_kwargs["_basic_wcs"] = "copy"
    cleaned_cube = cube.to_nddata(**cleaned_cube_kwargs)
    if hasattr(cleaned_cube, "dust_masked") and hasattr(cube, "dust_masked"):
        cleaned_cube.dust_masked = cube.dust_masked
    return cleaned_cube
