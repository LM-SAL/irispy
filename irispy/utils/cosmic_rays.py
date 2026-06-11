"""
Utilities for removing cosmic rays from IRIS data.
"""

from typing import Any
from importlib import import_module
from collections.abc import Mapping

import dask.array as da
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
    working_data = data.astype(np.float64, copy=True)
    working_data[mask] = np.nan
    clipped = slidingsigmaclipping(data=working_data, **kwargs).clipped
    cleaned_data = np.ma.getdata(clipped)
    cosmic_ray_mask = np.asarray(np.ma.getmaskarray(clipped), dtype=bool)
    return cleaned_data, cosmic_ray_mask


def _remove_cosmic_rays_astroscrappy(
    data,
    mask,
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
    data_shape = data.shape
    cleaned_data = np.empty(data_shape, dtype=data.dtype)
    cosmic_ray_mask = np.zeros(data_shape, dtype=bool)
    frame_indices = [()] if len(data_shape) == 2 else np.ndindex(data_shape[:-2])

    for index in frame_indices:
        frame = np.asarray(data[index]).copy()
        frame_mask = np.zeros(frame.shape, dtype=bool) if mask is None else np.asarray(mask[index], dtype=bool)
        if np.issubdtype(frame.dtype, np.floating):
            frame_mask = frame_mask | np.isnan(frame)
        if inmask is not None:
            frame_mask = frame_mask | np.asarray(inmask[index], dtype=bool)
        frame[frame_mask] = 0.0
        detected_mask, cleaned_frame = astroscrappy.detect_cosmics(frame, inmask=frame_mask, **kwargs)
        if index == ():
            cosmic_ray_mask[...] = detected_mask
            cleaned_data[...] = cleaned_frame
            continue
        cosmic_ray_mask[index] = detected_mask
        cleaned_data[index] = cleaned_frame
    return cleaned_data, cosmic_ray_mask


def _is_dask_array(value):
    return isinstance(value, da.Array)


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
    cube : irispy.sji.SJICube or irispy.spectrograph.SpectrogramCube
        Cube object to clean.
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
    irispy.sji.SJICube or irispy.spectrograph.SpectrogramCube
        A new cube with cleaned data and copied metadata/coordinates.
    """
    method = method.lower()
    kwargs = dict(method_kwargs or {})
    dask_backed = _is_dask_array(cube.data) or _is_dask_array(cube.mask)
    if dask_backed and method == "rsliding":
        msg = (
            "remove_cosmic_rays(method='rsliding') requires the full cube in memory. "
            "Slice the cube first, load it without memmap=True, or use method='astroscrappy'."
        )
        raise ValueError(msg)

    working_mask = None
    if not dask_backed:
        working_mask = np.zeros(cube.data.shape, dtype=bool) if cube.mask is None else np.asarray(cube.mask, dtype=bool)
        if np.issubdtype(cube.data.dtype, np.floating):
            working_mask = working_mask | np.isnan(cube.data)
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
            cube.mask if dask_backed else working_mask,
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
    cleaned_cube = cube.to_nddata(**cleaned_cube_kwargs)
    if hasattr(cleaned_cube, "dust_masked") and hasattr(cube, "dust_masked"):
        cleaned_cube.dust_masked = cube.dust_masked
    return cleaned_cube
