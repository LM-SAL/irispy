"""
Utilities for repairing dust-darkened pixels in IRIS image cubes.
"""

import warnings
import functools
from copy import deepcopy

import numpy as np
from scipy import ndimage

import astropy.units as u
import gwcs
from astropy.io import fits

from irispy._wcs import _root_and_item
from irispy.data import ROOTDIR
from ._spectral import check_scaled
from .utils import _boolean_mask

__all__ = ["calculate_dust_mask", "remove_dust"]


@functools.cache
def _badpix_table():
    """
    The detector pixels and pointing constants per channel, and the level 2 plate scale,
    from the newest pipeline bad-pixel table in ``irispy/data``.
    """
    path = max(ROOTDIR.glob("iris_sji_badpix_*.fits"))
    with fits.open(path) as hdul:
        pointing = {
            str(row["CHANNEL"]).strip(): (row["CPX1"], row["CPX2"], row["CDELT"], row["ROLL"])
            for row in hdul["POINTING"].data
        }
        pixels = {
            channel: (hdul[channel].data["X"].astype(int), hdul[channel].data["Y"].astype(int)) for channel in pointing
        }
        return pixels, pointing, hdul[0].header["L2CDELT"]


def calculate_dust_mask(cube):
    """
    The IRIS pipeline's bad pixels (dust and dead pixels) in the frame of a slit-jaw
    cube.

    Parameters
    ----------
    cube : `~irispy.sji.SJICube`
        A slit-jaw cube read from a level 2 file, or a slice of one in time or space.

    Returns
    -------
    `numpy.ndarray` of `bool`
        `True` at the bad pixels and their neighbors, with the shape of the cube's
        data. Pixels without a value (not finite) stay `False`.
    """
    # The fill values of a memmap read are finite raw integers
    check_scaled(cube)
    channel = cube.meta.get("TDESC1")
    pixels, pointing, level2_cdelt = _badpix_table()
    if channel not in pixels:
        msg = f"No bad-pixel table for {channel!r}; the tables cover {sorted(pixels)}."
        raise ValueError(msg)
    root, item = _root_and_item(cube)
    if cube.meta.get("frame_wcs_headers") is None or not isinstance(root, gwcs.WCS):
        msg = "The bad pixels can be placed only in a cube read from a level 2 slit-jaw file and not rebinned."
        raise ValueError(msg)
    spatial = item[1:]  # the cube's slice after the time axis, in array order
    if not all(isinstance(axis, slice) for axis in spatial):
        msg = "The bad pixels can be placed only in an image, not in a 1-D cut of one."
        raise ValueError(msg)
    # IRIS_DUSTBUSTER (SolarSoft): a bad pixel's summed detector pixel relative to the slit center
    # (the pointing database's 1-based FITS pixel, at the center of its summed pixel), rotated by
    # the detector's roll against the slit and scaled from the detector's plate scale to the
    # level 2 one; -0.5 is the routine's default y adjustment.
    detector_x, detector_y = pixels[channel]
    cpx1, cpx2, cdelt, roll = pointing[channel]
    sum_x, sum_y = cube.meta["SUMSPTRL"], cube.meta["SUMSPAT"]
    dx = detector_x // sum_x - ((cpx1 - 1) / sum_x + (sum_x - 1) / (2 * sum_x))
    dy = detector_y // sum_y - 0.5 - ((cpx2 - 1) / sum_y + (sum_y - 1) / (2 * sum_y))
    distance = np.hypot(dx, dy) * cdelt / level2_cdelt
    angle = np.arctan2(dx, dy) - np.deg2rad(roll)
    offset_x = distance * np.sin(angle) - (spatial[1].start or 0)
    offset_y = distance * np.cos(angle) - (spatial[0].start or 0)
    if cube.data.ndim == 2:
        slit_x = np.atleast_1d(cube.global_coords["slit x position"].to_value(u.pix))
        slit_y = np.atleast_1d(cube.global_coords["slit y position"].to_value(u.pix))
    else:
        slit_x = cube.axis_world_coords(wcs=cube.extra_coords["slit x position"])[0].to_value(u.pix)
        slit_y = cube.axis_world_coords(wcs=cube.extra_coords["slit y position"])[0].to_value(u.pix)
    ny, nx = cube.data.shape[-2:]
    # One pixel of margin, so a bad pixel just outside a crop still dilates into it
    mask = np.zeros((len(slit_x), ny + 2, nx + 2), dtype=bool)
    x = offset_x + slit_x[:, np.newaxis] + 1
    y = offset_y + slit_y[:, np.newaxis] + 1
    frame = np.broadcast_to(np.arange(len(slit_x))[:, np.newaxis], x.shape)
    # Floor and ceil in both axes, as the IDL routine does, so the mapped mask has no gaps
    for xi in (np.floor(x), np.ceil(x)):
        for yi in (np.floor(y), np.ceil(y)):
            inside = (xi >= 0) & (xi <= nx + 1) & (yi >= 0) & (yi <= ny + 1)
            mask[frame[inside], yi[inside].astype(int), xi[inside].astype(int)] = True
    # The list is one level 2 pixel tight on the images; IRIS_DUSTBUSTER smooths it over 5 detector pixels
    mask = ndimage.binary_dilation(mask, structure=np.ones((1, 3, 3), dtype=bool))[:, 1:-1, 1:-1]
    mask = mask.reshape(cube.data.shape)
    # The dead columns of a record run through the frame's edge, which holds no values
    return mask & np.asarray(np.isfinite(cube.data))


def _coerce_dust_mask(cube, data, dust_mask):
    dust_mask = calculate_dust_mask(cube) if dust_mask is None else np.asarray(dust_mask, dtype=bool)
    if data.ndim == 3 and dust_mask.shape == data.shape[1:]:
        dust_mask = np.broadcast_to(dust_mask, data.shape)
    if dust_mask.shape != data.shape:
        msg = f"dust_mask must have shape {data.shape} or {data.shape[1:]}; got {dust_mask.shape}"
        raise ValueError(msg)
    # Pixels without a value are not dust: filling them would paint over the frame's edges
    return dust_mask & np.isfinite(data)


def _local_median_fill(frame, invalid_mask, spatial_box, target_mask):
    """
    Local ``nanmedian`` of ``frame`` at ``target_mask`` pixels, NaN elsewhere.

    Matches ``generic_filter`` with ``np.nanmedian`` and ``mode="nearest"`` at the
    target pixels, without filtering the whole frame.
    """
    masked_frame = np.where(invalid_mask | ~np.isfinite(frame), np.nan, frame)
    padded = np.pad(masked_frame, spatial_box // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (spatial_box, spatial_box))
    rows, cols = np.nonzero(target_mask)
    spatial_fill = np.full(frame.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spatial_fill[rows, cols] = np.nanmedian(windows[rows, cols].reshape(rows.size, spatial_box**2), axis=1)
    return spatial_fill


def remove_dust(
    cube,
    *,
    dust_mask=None,
    temporal_window=2,
    fallback="spatial",
    spatial_box=5,
):
    """
    Replace dust-darkened pixels in an IRIS image cube.

    This routine is inspired by SolarSoft's ``IRIS_DUSTBUSTER``. It uses
    neighboring frames at the same pixel location first, and falls back to a local
    spatial median when temporal replacement is not available.
    The dust moves with the slit, so neighboring frames help only when the slit moves between them,
    as in a raster; in a sit-and-stare only the spatial median fills the dust.
    Dust pixels that neither can fill keep their value and stay masked.
    To mask the dust without replacing it, use ``cube.mask | calculate_dust_mask(cube)``.

    Parameters
    ----------
    cube : `~irispy.sji.SJICube`
        The image cube to clean. Two-dimensional image slices are also supported.
    dust_mask : `numpy.ndarray`, optional
        Boolean mask marking the pixels to repair. If omitted, the pipeline's bad
        pixels from `~irispy.utils.dust.calculate_dust_mask` are repaired.
        For a 3D cube, a 2D mask is broadcast over time.
    temporal_window : `int`, optional
        Number of neighboring frames on either side to consider when computing the
        replacement value. Defaults to 2, which mirrors the neighboring frames used
        in the IDL implementation. In a cube of counts, which must carry its exposure
        times, the neighboring frames are scaled to the exposure time of the frame they
        repair; an exposure-corrected or calibrated cube is not scaled.
    fallback : {``"spatial"``, None}, optional
        Fallback to use when temporal replacement is unavailable.
        ``"spatial"`` applies a local median filter to the frame.
        None leaves those pixels masked.
    spatial_box : `int`, optional
        Size of the local median filter used by the spatial fallback.
        Defaults to 5.

    Returns
    -------
    `~irispy.sji.SJICube`
        A new cube with repaired dust pixels, in the floating-point type of the data:
        float32 for Level 2 data.

    Notes
    -----
    Pixels without a value (not finite) are never repaired.
    """
    if fallback not in {"spatial", None}:
        msg = "fallback must be 'spatial' or None."
        raise ValueError(msg)

    check_scaled(cube)
    clean_data = np.array(cube.data, dtype=np.promote_types(cube.data.dtype, np.float32))
    original_mask = _boolean_mask(cube)
    dust_mask = _coerce_dust_mask(cube, clean_data, dust_mask)
    filled_mask = np.zeros_like(dust_mask, dtype=bool)

    if clean_data.ndim == 3 and temporal_window > 0 and np.any(dust_mask):
        # Counts scale with the exposure time and a rate does not, the test sunraster's
        # exposure time correction makes
        exposure_times = None
        if u.s not in cube.unit.decompose().bases:
            exposure_times = np.broadcast_to(cube.exposure_time.to_value(u.s), clean_data.shape[:1])
        n_frames = clean_data.shape[0]
        for frame_index in range(n_frames):
            rows, cols = np.nonzero(dust_mask[frame_index])
            if not rows.size:
                continue
            window = np.arange(max(frame_index - temporal_window, 0), min(frame_index + temporal_window + 1, n_frames))
            # The neighbors' samples at this frame's dust pixels only, one row per neighbor
            neighbors = window[window != frame_index, np.newaxis]
            candidates = clean_data[neighbors, rows, cols]
            invalid = original_mask[neighbors, rows, cols] | dust_mask[neighbors, rows, cols] | ~np.isfinite(candidates)
            candidates[invalid] = np.nan
            if exposure_times is not None:
                candidates = candidates / exposure_times[neighbors] * exposure_times[frame_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temporal_fill = np.nanmedian(candidates, axis=0)
            can_fill = np.isfinite(temporal_fill)
            clean_data[frame_index, rows[can_fill], cols[can_fill]] = temporal_fill[can_fill]
            filled_mask[frame_index, rows[can_fill], cols[can_fill]] = True

    if fallback == "spatial":
        if clean_data.ndim == 2:
            spatial_fill = _local_median_fill(clean_data, original_mask | dust_mask, spatial_box, dust_mask)
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
                    target_mask,
                )
                can_fill = target_mask & np.isfinite(spatial_fill)
                clean_data[frame_index, can_fill] = spatial_fill[can_fill]
                filled_mask[frame_index, can_fill] = True

    output_mask = original_mask.copy()
    output_mask[dust_mask] = True
    output_mask[filled_mask] = False
    # Coordinates point back to their cube; copy them without copying its data.
    memo = {id(cube): None}
    return cube.to_nddata(
        data=clean_data,
        mask=output_mask,
        meta=deepcopy(cube.meta),
        nddata_type=type(cube),
        extra_coords=deepcopy(cube.extra_coords, memo),
        global_coords=deepcopy(cube.global_coords, memo),
    )
