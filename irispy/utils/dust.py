"""
Utilities for repairing dust-darkened pixels in IRIS image cubes.
"""

import warnings

import numpy as np
from scipy import ndimage

import astropy.units as u

from .utils import calculate_dust_mask

__all__ = ["remove_dust"]


def _coerce_dust_mask(data, dust_mask):
    if dust_mask is None:
        return calculate_dust_mask(data)
    dust_mask = np.asarray(dust_mask, dtype=bool)
    if data.ndim == 3 and dust_mask.shape == data.shape[1:]:
        dust_mask = np.broadcast_to(dust_mask, data.shape)
    if dust_mask.shape != data.shape:
        msg = f"dust_mask must have shape {data.shape} or {data.shape[1:]}; got {dust_mask.shape}"
        raise ValueError(msg)
    return dust_mask.copy()


def _local_median_fill(frame, invalid_mask, spatial_box):
    masked_frame = np.where(invalid_mask | ~np.isfinite(frame), np.nan, frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return ndimage.generic_filter(masked_frame, np.nanmedian, size=spatial_box, mode="nearest")


def _resolve_exposure_times(cube, frame_count, *, exposure_normalize):
    """
    Resolve per-frame exposure times for temporal normalization.

    Returns
    -------
    `numpy.ndarray` or None
        Per-frame exposure times in seconds when available and valid.
        Returns None when normalization is disabled or metadata are unusable.
    """
    if not exposure_normalize:
        return None
    try:
        exposure_times = u.Quantity(cube.exposure_time, copy=False).to_value(u.s)
    except (AttributeError, TypeError, ValueError):
        return None
    exposure_times = np.atleast_1d(np.asarray(exposure_times, dtype=float))
    if exposure_times.size == 1:
        return np.repeat(exposure_times, frame_count)
    if exposure_times.size != frame_count:
        warnings.warn(
            "exposure_normalize=True but the number of exposure_time values "
            f"({exposure_times.size}) does not match the number of frames "
            f"in the cube ({frame_count}). Temporal exposure "
            "normalization has been skipped.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return exposure_times


def remove_dust(
    cube,
    *,
    dust_mask=None,
    temporal_window=2,
    exposure_normalize=True,
    fallback="spatial",
    spatial_box=5,
):
    """
    Replace dust-darkened pixels in an IRIS image cube.

    This routine is inspired by SolarSoft's ``IRIS_DUSTBUSTER``. It uses
    neighboring frames at the same pixel location first, and falls back to a local
    spatial median when temporal replacements is not available.

    Parameters
    ----------
    cube : `irispy.sji.SJICube`
        The image cube to clean. Two-dimensional image slices are also supported.
    dust_mask : `numpy.ndarray`, optional
        Boolean mask marking the pixels to repair. If omitted, the mask is derived
        from `irispy.utils.calculate_dust_mask`.
        For a 3D cube, a 2D mask is broadcast over time.
    temporal_window : `int`, optional
        Number of neighboring frames on either side to consider when computing the
        replacement value. Defaults to 2, which mirrors the neighboring frames used
        in the IDL implementation.
    exposure_normalize : `bool`, optional
        If `True`, normalize candidate replacement pixels by the frame exposure time
        before taking the median and scale the result back to the target frame.
        If no usable exposure time metadata are available this step is skipped.
    fallback : {``"spatial"``, None}, optional
        Fallback to use when temporal replacement is unavailable.
        ``"spatial"`` applies a local median filter to the frame.
        None leaves those pixels masked.
    spatial_box : `int`, optional
        Size of the local median filter used by the spatial fallback.
        Defaults to 5.

    Returns
    -------
    `irispy.sji.SJICube`
        A new cube with repaired dust pixels.

    Notes
    -----
    Unlike the IDL routine, this function does not remap instrument bad-pixel tables
    into the image plane. It assumes the dust locations are already represented in
    the data values or supplied via ``dust_mask``.
    """
    if fallback not in {"spatial", None}:
        msg = "fallback must be 'spatial' or None."
        raise ValueError(msg)

    clean_data = np.asarray(cube.data, dtype=float).copy()
    if cube.mask is None:
        original_mask = np.zeros(cube.data.shape, dtype=bool)
    else:
        original_mask = np.asarray(cube.mask, dtype=bool).copy()
    dust_mask = _coerce_dust_mask(clean_data, dust_mask)
    filled_mask = np.zeros_like(dust_mask, dtype=bool)

    if clean_data.ndim == 3 and temporal_window > 0 and np.any(dust_mask):
        exposure_times = _resolve_exposure_times(
            cube,
            clean_data.shape[0],
            exposure_normalize=exposure_normalize,
        )
        for frame_index in range(clean_data.shape[0]):
            target_mask = dust_mask[frame_index]
            if not np.any(target_mask):
                continue
            candidate_frames = []
            for offset in range(1, temporal_window + 1):
                for neighbor_index in (frame_index - offset, frame_index + offset):
                    if 0 <= neighbor_index < clean_data.shape[0]:
                        candidate = clean_data[neighbor_index].copy()
                        invalid = original_mask[neighbor_index] | dust_mask[neighbor_index] | ~np.isfinite(candidate)
                        candidate[invalid] = np.nan
                        if exposure_times is not None:
                            candidate = candidate / exposure_times[neighbor_index] * exposure_times[frame_index]
                        candidate_frames.append(candidate)
            if candidate_frames:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    temporal_fill = np.nanmedian(np.stack(candidate_frames), axis=0)
                can_fill = target_mask & np.isfinite(temporal_fill)
                clean_data[frame_index, can_fill] = temporal_fill[can_fill]
                filled_mask[frame_index, can_fill] = True

    if fallback == "spatial":
        if clean_data.ndim == 2:
            spatial_fill = _local_median_fill(clean_data, original_mask | dust_mask, spatial_box)
            can_fill = dust_mask & np.isfinite(spatial_fill)
            clean_data[can_fill] = spatial_fill[can_fill]
            filled_mask[can_fill] = True
        else:
            for frame_index in range(clean_data.shape[0]):
                target_mask = dust_mask[frame_index] & ~filled_mask[frame_index]
                if not np.any(target_mask):
                    continue
                spatial_fill = _local_median_fill(
                    clean_data[frame_index],
                    original_mask[frame_index] | dust_mask[frame_index],
                    spatial_box,
                )
                can_fill = target_mask & np.isfinite(spatial_fill)
                clean_data[frame_index, can_fill] = spatial_fill[can_fill]
                filled_mask[frame_index, can_fill] = True

    output_mask = original_mask.copy()
    output_mask[dust_mask] = True
    output_mask[filled_mask] = False
    cleaned_cube_kwargs = {
        "data": clean_data,
        "mask": output_mask,
        "nddata_type": type(cube),
        "extra_coords": "copy",
        "global_coords": "copy",
    }
    if hasattr(cube, "scaled"):
        cleaned_cube_kwargs["scaled"] = "copy"
    if hasattr(cube, "_basic_wcs"):
        cleaned_cube_kwargs["_basic_wcs"] = "copy"
    cleaned_cube = cube.to_nddata(**cleaned_cube_kwargs)
    if hasattr(cleaned_cube, "dust_masked"):
        cleaned_cube.dust_masked = bool(np.any(output_mask & dust_mask))
    return cleaned_cube
