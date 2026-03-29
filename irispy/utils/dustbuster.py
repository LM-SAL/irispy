"""
Dust cleaning for `irispy.sji.SJICube` objects.
"""

from copy import deepcopy
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage
from scipy.io import readsav as read_geny

from astropy.time import Time

from sunpy.data import manager as data_manager
from sunpy.io.special import read_genx

from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, POINTING_INFO, SJI_CHANNEL_SUFFIX

__all__ = ["clean_sji", "clean_sji_regions", "get_sji_dust_params"]

_MASK_SHAPE = (2072, 1096)
_MANUAL_OFFSET = (0.0, -0.5)
_MAX_ALIGNMENT_SHIFT = 7
_MAX_ALIGNMENT_FRAMES = 8
_DISK_RADIUS_LIMIT = 880.0
_TEMPORAL_OFFSETS = np.array((-2, -1, 1, 2))
_SPATIAL_WINDOW = 9
_NEGATIVE_NEIGHBOR_KERNEL = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=np.int8,
)
_MIN_NEGATIVE_NEIGHBORS = 5
_SLIT_EXCLUSION_HALF_WIDTH = 1.0
_LABEL_STRUCTURE = np.ones((3, 3), dtype=np.int8)


def _bin_factor(cube, *, axis: Literal["x", "y"]) -> int:
    return int(cube.meta["SUMSPTRL" if axis == "x" else "SUMSPAT"])


def _align_frame_idx(n_frames: int) -> np.ndarray:
    if n_frames <= _MAX_ALIGNMENT_FRAMES:
        return np.arange(n_frames)
    frame_idx = np.rint(np.linspace(0, n_frames - 1, _MAX_ALIGNMENT_FRAMES)).astype(int)
    return np.unique(frame_idx)


def _fill_negative_islands(data: np.ndarray, slit_x_pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled_data = data.copy()
    filled_mask = np.zeros(data.shape, dtype=bool)
    x_pixels = np.arange(data.shape[2], dtype=float)

    for frame_idx, slit_x in enumerate(slit_x_pix):
        source_frame = data[frame_idx]
        target_frame = filled_data[frame_idx]
        slit_mask = np.abs(x_pixels - slit_x) <= _SLIT_EXCLUSION_HALF_WIDTH
        valid_neighbors = np.isfinite(source_frame) & (source_frame >= 0) & (source_frame != BAD_PIXEL_VALUE_SCALED)
        neighbor_count = ndimage.convolve(valid_neighbors.astype(np.int16), _NEGATIVE_NEIGHBOR_KERNEL, mode="constant")
        candidate_mask = (
            np.isfinite(source_frame)
            & (source_frame < 0)
            & (source_frame != BAD_PIXEL_VALUE_SCALED)
            & ~slit_mask[None, :]
            & (neighbor_count >= _MIN_NEGATIVE_NEIGHBORS)
        )
        if not np.any(candidate_mask):
            continue

        for y_idx, x_idx in np.argwhere(candidate_mask):
            y0 = max(int(y_idx) - 1, 0)
            y1 = min(int(y_idx) + 2, source_frame.shape[0])
            x0 = max(int(x_idx) - 1, 0)
            x1 = min(int(x_idx) + 2, source_frame.shape[1])
            patch = source_frame[y0:y1, x0:x1]
            patch_valid = np.isfinite(patch) & (patch >= 0) & (patch != BAD_PIXEL_VALUE_SCALED)
            patch_valid[int(y_idx) - y0, int(x_idx) - x0] = False
            if np.count_nonzero(patch_valid) < _MIN_NEGATIVE_NEIGHBORS:
                continue
            target_frame[int(y_idx), int(x_idx)] = np.median(patch[patch_valid])
            filled_mask[frame_idx, int(y_idx), int(x_idx)] = True

    return filled_data, filled_mask


def _cube_context(cube) -> dict:
    data = cube.data
    input_ndim = data.ndim
    if input_ndim == 2:
        data = data[np.newaxis, :, :]
    elif input_ndim != 3:
        msg = "cube.data must have shape (nt, ny, nx) or (ny, nx)."
        raise ValueError(msg)
    n_frames, n_y, n_x = data.shape
    if cube.basic_wcs is None:
        msg = "cube.basic_wcs is required."
        raise ValueError(msg)
    wcs_list = cube.basic_wcs if input_ndim == 3 else [cube.basic_wcs]
    if len(wcs_list) != n_frames:
        msg = "cube.basic_wcs must contain one WCS per frame."
        raise ValueError(msg)
    exposure_s = cube.extra_coords["exposure time"].wcs.pixel_to_world(np.arange(n_frames)).value
    slit_x_pix = cube.extra_coords["slit x position"].wcs.pixel_to_world(np.arange(n_frames)).value - 1
    slit_y_pix = cube.extra_coords["slit y position"].wcs.pixel_to_world(np.arange(n_frames)).value - 1
    if exposure_s.shape != (n_frames,) or slit_x_pix.shape != (n_frames,) or slit_y_pix.shape != (n_frames,):
        msg = "The required per-frame extra coordinates must each have the same shape as the number of frames."
        raise ValueError(msg)
    ref_x_pix = np.array([w.wcs.crpix[0] for w in wcs_list])
    ref_y_pix = np.array([w.wcs.crpix[1] for w in wcs_list])
    x_scale = np.array([w.wcs.cdelt[0] for w in wcs_list])
    y_scale = np.array([w.wcs.cdelt[1] for w in wcs_list])
    ref_x_arcsec = np.array([w.wcs.crval[0] for w in wcs_list])
    ref_y_arcsec = np.array([w.wcs.crval[1] for w in wcs_list])
    image_scale = 0.5 * (np.abs(x_scale) + np.abs(y_scale))
    return {
        "data": data,
        "exposure_s": exposure_s,
        "image_scale": image_scale,
        "input_ndim": input_ndim,
        "n_frames": n_frames,
        "n_x": n_x,
        "n_y": n_y,
        "ref_x_arcsec": ref_x_arcsec,
        "ref_x_pix": ref_x_pix,
        "ref_y_arcsec": ref_y_arcsec,
        "ref_y_pix": ref_y_pix,
        "slit_x_pix": slit_x_pix,
        "slit_y_pix": slit_y_pix,
        "x_scale": x_scale,
        "y_scale": y_scale,
    }


def _project_seed_pixels(
    cube,
    context: dict,
    *,
    dust_ids: ArrayLike,
    slit_center: tuple[float, float],
    mask_scale: float,
    roll_deg: float,
    align: bool,
) -> np.ndarray:
    y_bin = _bin_factor(cube, axis="y")
    x_bin = _bin_factor(cube, axis="x")
    mask_nx, mask_ny = _MASK_SHAPE
    detector_mask = np.zeros((mask_nx, mask_ny), dtype=bool)
    dust_ids = np.asarray(dust_ids, dtype=np.int64)
    detector_x = dust_ids % mask_nx
    detector_y = dust_ids // mask_nx
    detector_mask[detector_x, detector_y] = True
    detector_x, detector_y = np.nonzero(detector_mask)
    y_bin_offset = (y_bin - 1.0) / (2.0 * y_bin)
    x_bin_offset = (x_bin - 1.0) / (2.0 * x_bin)
    dust_x_mask = detector_x / x_bin + _MANUAL_OFFSET[0]
    dust_y_mask = detector_y / y_bin + _MANUAL_OFFSET[1]
    slit_x_mask = slit_center[0] / x_bin + x_bin_offset
    slit_y_mask = slit_center[1] / y_bin + y_bin_offset
    dx_mask = dust_x_mask - slit_x_mask
    dy_mask = dust_y_mask - slit_y_mask
    dust_radius_arcsec = np.hypot(dx_mask, dy_mask) * mask_scale
    dust_angle = np.arctan2(dx_mask, dy_mask)
    roll_rad = np.deg2rad(roll_deg)
    dx_pix = (dust_radius_arcsec[:, None] / context["image_scale"][None, :]) * np.sin(dust_angle[:, None] - roll_rad)
    dy_pix = (dust_radius_arcsec[:, None] / context["image_scale"][None, :]) * np.cos(dust_angle[:, None] - roll_rad)
    dust_x = dx_pix + context["slit_x_pix"][None, :]
    dust_y = dy_pix + context["slit_y_pix"][None, :]
    x0 = np.floor(dust_x).astype(np.int64)
    x1 = np.ceil(dust_x).astype(np.int64)
    y0 = np.floor(dust_y).astype(np.int64)
    y1 = np.ceil(dust_y).astype(np.int64)
    dust_x = np.concatenate([x0, x0, x1, x1], axis=0)
    dust_y = np.concatenate([y0, y1, y0, y1], axis=0)
    dust_t = np.broadcast_to(np.arange(context["n_frames"], dtype=np.int64), dust_x.shape)

    if align:
        align_frame_idx = _align_frame_idx(context["n_frames"])
        align_x = dust_x[:, align_frame_idx]
        align_y = dust_y[:, align_frame_idx]
        align_t = np.broadcast_to(align_frame_idx, align_x.shape)
        best_score = np.inf
        best_shift = (0, 0)
        for shift_x in range(-_MAX_ALIGNMENT_SHIFT, _MAX_ALIGNMENT_SHIFT + 1):
            for shift_y in range(-_MAX_ALIGNMENT_SHIFT, _MAX_ALIGNMENT_SHIFT + 1):
                shifted_x = align_x + shift_x
                shifted_y = align_y + shift_y
                x_arcsec = (
                    (shifted_x + 1.0 - context["ref_x_pix"][align_frame_idx][None, :])
                    * context["x_scale"][align_frame_idx][None, :]
                    + context["ref_x_arcsec"][align_frame_idx][None, :]
                )
                y_arcsec = (
                    (shifted_y + 1.0 - context["ref_y_pix"][align_frame_idx][None, :])
                    * context["y_scale"][align_frame_idx][None, :]
                    + context["ref_y_arcsec"][align_frame_idx][None, :]
                )
                valid_hits = (
                    (shifted_x >= 0)
                    & (shifted_x < context["n_x"])
                    & (shifted_y >= 0)
                    & (shifted_y < context["n_y"])
                    & (np.abs(x_arcsec) <= _DISK_RADIUS_LIMIT)
                    & (np.abs(y_arcsec) <= _DISK_RADIUS_LIMIT)
                )
                if not np.any(valid_hits):
                    continue
                hit_t = align_t[valid_hits]
                hit_y = shifted_y[valid_hits]
                hit_x = shifted_x[valid_hits]
                hit_values = context["data"][hit_t, hit_y, hit_x]
                valid_values = (hit_values != BAD_PIXEL_VALUE_SCALED) & np.isfinite(hit_values)
                if not np.any(valid_values):
                    continue
                score = np.mean(hit_values[valid_values])
                if score < best_score:
                    best_score = score
                    best_shift = (shift_x, shift_y)
        dust_x = dust_x + best_shift[0]
        dust_y = dust_y + best_shift[1]

    x_arcsec = (dust_x + 1.0 - context["ref_x_pix"][None, :]) * context["x_scale"][None, :] + context["ref_x_arcsec"][
        None, :
    ]
    y_arcsec = (dust_y + 1.0 - context["ref_y_pix"][None, :]) * context["y_scale"][None, :] + context["ref_y_arcsec"][
        None, :
    ]
    valid_hits = (
        (dust_x >= 0)
        & (dust_x < context["n_x"])
        & (dust_y >= 0)
        & (dust_y < context["n_y"])
        & (np.abs(x_arcsec) <= _DISK_RADIUS_LIMIT)
        & (np.abs(y_arcsec) <= _DISK_RADIUS_LIMIT)
    )
    if not np.any(valid_hits):
        return np.empty((0, 3), dtype=np.int64)
    seed_pixels = np.stack([dust_t[valid_hits], dust_y[valid_hits], dust_x[valid_hits]], axis=1)
    return np.unique(seed_pixels, axis=0)


def _fill_pixels(data: np.ndarray, exposure_s: np.ndarray, pixels: np.ndarray, fill_mode: Literal["blur", "global"]) -> np.ndarray:
    if pixels.size == 0:
        return np.empty(0, dtype=float)

    fill_mask = np.zeros(data.shape, dtype=bool)
    fill_mask[
        pixels[:, 0],
        pixels[:, 1],
        pixels[:, 2],
    ] = True
    fill_t = pixels[:, 0]
    fill_y = pixels[:, 1]
    fill_x = pixels[:, 2]
    neighbor_t = fill_t[:, None] + _TEMPORAL_OFFSETS[None, :]
    neighbor_y = np.broadcast_to(fill_y[:, None], neighbor_t.shape)
    neighbor_x = np.broadcast_to(fill_x[:, None], neighbor_t.shape)
    in_time_range = (neighbor_t >= 0) & (neighbor_t < data.shape[0])
    clipped_t = np.clip(neighbor_t, 0, data.shape[0] - 1)
    valid_neighbors = in_time_range & (~fill_mask[clipped_t, neighbor_y, neighbor_x])
    neighbor_values = np.full(neighbor_t.shape, np.nan, dtype=float)
    neighbor_values[valid_neighbors] = data[
        clipped_t[valid_neighbors],
        neighbor_y[valid_neighbors],
        neighbor_x[valid_neighbors],
    ]
    valid_neighbors &= neighbor_values != BAD_PIXEL_VALUE_SCALED
    valid_neighbors &= np.isfinite(neighbor_values)
    neighbor_values[valid_neighbors] /= exposure_s[clipped_t[valid_neighbors]]
    fill_values = np.full(fill_t.shape, np.nan, dtype=float)
    has_neighbors = np.any(valid_neighbors, axis=1)
    if np.any(has_neighbors):
        with np.errstate(invalid="ignore"):
            fill_values[has_neighbors] = np.nanmedian(
                neighbor_values[has_neighbors],
                axis=1,
            )
    fill_values *= exposure_s[fill_t]
    needs_spatial_fill = ~np.isfinite(fill_values)
    if np.any(needs_spatial_fill):
        if fill_mode == "blur":
            for frame_idx in np.unique(fill_t[needs_spatial_fill]):
                frame_needs = needs_spatial_fill & (fill_t == frame_idx)
                frame_data = data[frame_idx].copy()
                frame_data[frame_data == BAD_PIXEL_VALUE_SCALED] = np.nan
                frame_data[fill_mask[frame_idx]] = np.nan
                finite_mask = np.isfinite(frame_data).astype(float)
                filled_data = np.where(np.isfinite(frame_data), frame_data, 0.0)
                mean_signal = ndimage.uniform_filter(filled_data, size=_SPATIAL_WINDOW, mode="nearest")
                mean_weight = ndimage.uniform_filter(finite_mask, size=_SPATIAL_WINDOW, mode="nearest")
                with np.errstate(invalid="ignore", divide="ignore"):
                    frame_fill = mean_signal / mean_weight
                frame_fill[mean_weight == 0.0] = np.nan
                fill_values[frame_needs] = frame_fill[
                    fill_y[frame_needs],
                    fill_x[frame_needs],
                ]
        needs_global_fill = ~np.isfinite(fill_values)
        if np.any(needs_global_fill):
            good_values = data[np.isfinite(data) & (data != BAD_PIXEL_VALUE_SCALED)]
            fill_values[needs_global_fill] = np.median(good_values)
    return fill_values


def _label_region_pixels(data: np.ndarray, seed_pixels: np.ndarray) -> np.ndarray:
    if seed_pixels.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    region_pixels = []
    for frame_idx in np.unique(seed_pixels[:, 0]):
        frame_seed = seed_pixels[seed_pixels[:, 0] == frame_idx][:, 1:]
        candidate_mask = np.isfinite(data[frame_idx]) & (data[frame_idx] < 0)
        labels, _ = ndimage.label(candidate_mask, structure=_LABEL_STRUCTURE)
        seed_labels = labels[frame_seed[:, 0], frame_seed[:, 1]]
        seed_labels = np.unique(seed_labels[seed_labels > 0])
        if seed_labels.size == 0:
            continue
        region_y, region_x = np.nonzero(np.isin(labels, seed_labels))
        region_t = np.full(region_y.shape, frame_idx, dtype=np.int64)
        region_pixels.append(np.stack([region_t, region_y, region_x], axis=1))

    if not region_pixels:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(region_pixels), axis=0)


def _write_cleaned_cube(cube, context: dict, fill_pixels: np.ndarray, fill_values: np.ndarray):
    cleaned_cube = deepcopy(cube)
    cleaned_cube.data[...] = context["data"][0] if context["input_ndim"] == 2 else context["data"]
    if fill_pixels.size > 0:
        write_index = (
            (fill_pixels[:, 1], fill_pixels[:, 2])
            if context["input_ndim"] == 2
            else (fill_pixels[:, 0], fill_pixels[:, 1], fill_pixels[:, 2])
        )
        cleaned_cube.data[write_index] = fill_values
        if cleaned_cube.mask is not None:
            cleaned_cube.mask[write_index] = False
    cleaned_data = cleaned_cube.data[np.newaxis, :, :] if context["input_ndim"] == 2 else cleaned_cube.data
    cleaned_data, negative_fill_mask = _fill_negative_islands(cleaned_data, context["slit_x_pix"])
    cleaned_cube.data[...] = cleaned_data[0] if context["input_ndim"] == 2 else cleaned_data
    if cleaned_cube.mask is not None:
        if context["input_ndim"] == 2:
            cleaned_cube.mask[negative_fill_mask[0]] = False
        else:
            cleaned_cube.mask[negative_fill_mask] = False
    return cleaned_cube


def _clean_sji_with_params(
    cube,
    *,
    dust_ids: ArrayLike,
    slit_center: tuple[float, float],
    mask_scale: float,
    roll_deg: float,
    align: bool = True,
    fill_mode: Literal["blur", "global"] = "blur",
):
    if fill_mode not in {"blur", "global"}:
        msg = "fill_mode must be 'blur' or 'global'."
        raise ValueError(msg)
    context = _cube_context(cube)
    seed_pixels = _project_seed_pixels(
        cube,
        context,
        dust_ids=dust_ids,
        slit_center=slit_center,
        mask_scale=mask_scale,
        roll_deg=roll_deg,
        align=align,
    )
    if seed_pixels.size == 0:
        fill_pixels = seed_pixels
    else:
        seed_values = context["data"][seed_pixels[:, 0], seed_pixels[:, 1], seed_pixels[:, 2]]
        valid_seed = (seed_values != BAD_PIXEL_VALUE_SCALED) & np.isfinite(seed_values)
        fill_pixels = seed_pixels[valid_seed]
    fill_values = _fill_pixels(context["data"], context["exposure_s"], fill_pixels, fill_mode)
    return _write_cleaned_cube(cube, context, fill_pixels, fill_values)


def _clean_sji_regions_with_params(
    cube,
    *,
    dust_ids: ArrayLike,
    slit_center: tuple[float, float],
    mask_scale: float,
    roll_deg: float,
    align: bool = True,
    fill_mode: Literal["blur", "global"] = "blur",
):
    if fill_mode not in {"blur", "global"}:
        msg = "fill_mode must be 'blur' or 'global'."
        raise ValueError(msg)
    context = _cube_context(cube)
    seed_pixels = _project_seed_pixels(
        cube,
        context,
        dust_ids=dust_ids,
        slit_center=slit_center,
        mask_scale=mask_scale,
        roll_deg=roll_deg,
        align=align,
    )
    region_pixels = _label_region_pixels(context["data"], seed_pixels)
    if region_pixels.size == 0:
        fill_pixels = seed_pixels
    elif seed_pixels.size == 0:
        fill_pixels = region_pixels
    else:
        fill_pixels = np.unique(np.concatenate([seed_pixels, region_pixels]), axis=0)
    fill_values = _fill_pixels(context["data"], context["exposure_s"], fill_pixels, fill_mode)
    return _write_cleaned_cube(cube, context, fill_pixels, fill_values)


def clean_sji(cube, *, align: bool = True, fill_mode: Literal["blur", "global"] = "blur"):
    """
    Remove dust-contaminated pixels from an `irispy.sji.SJICube`.

    Parameters
    ----------
    cube : SJICube-like
        A `~irispy.sji.SJICube`.
    align : bool, optional
        If True (the default), align the projected mask to the darkest valid pixels in the
        cube using a bounded integer search.
    fill_mode : {"blur", "global"}, optional
        Fallback used when temporal filling is not possible. ``"blur"`` uses a
        local smoothed fill before falling back to the overall median.
        ``"global"`` skips the local fill and goes straight to the overall median.
        This can be useful off the limb or in other cases where a local blur
        would smear bright structure into nearby dust pixels.

    Returns
    -------
    cleaned_cube : ~irispy.sji.SJICube`
        Copy of the input cube with dust pixels replaced.
    """
    dust_params = get_sji_dust_params(
        date_obs=cube.meta["DATE_OBS"],
        sji_name=cube.meta["TDESC1"],
    )
    return _clean_sji_with_params(cube, align=align, fill_mode=fill_mode, **dust_params)


def clean_sji_regions(cube, *, align: bool = True, fill_mode: Literal["blur", "global"] = "blur"):
    """
    Remove dust-contaminated pixels from an `irispy.sji.SJICube` by expanding
    calibration-seeded dust locations into connected negative regions.

    Parameters
    ----------
    cube : SJICube-like
        A `~irispy.sji.SJICube`.
    align : bool, optional
        If True (the default), align the projected mask to the darkest valid pixels in the
        cube using a bounded integer search.
    fill_mode : {"blur", "global"}, optional
        Fallback used when temporal filling is not possible. ``"blur"`` uses a
        local smoothed fill before falling back to the overall median.
        ``"global"`` skips the local fill and goes straight to the overall median.

    Returns
    -------
    cleaned_cube : ~irispy.sji.SJICube`
        Copy of the input cube with calibration-seeded negative dust regions replaced.
    """
    dust_params = get_sji_dust_params(
        date_obs=cube.meta["DATE_OBS"],
        sji_name=cube.meta["TDESC1"],
    )
    return _clean_sji_regions_with_params(cube, align=align, fill_mode=fill_mode, **dust_params)


@data_manager.require(
    "iris_sji_flat_index",
    ["https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_flat.genx"],
    "40de195c55b0c5e04acb5f6f55883603c74a71bac6a5d639ec73f9d39d076b24",
    defer_download=True,
)
@data_manager.require(
    "iris_sji_bad_pixel_map",
    ["https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_badpix.geny"],
    "c4d1884fb1a4f09b6ce4fe150a0aadab2664e479d86ae0ae063d8daa559e230d",
    defer_download=True,
)
def get_sji_dust_params(*, date_obs: str, sji_name: str) -> dict:
    """
    Return the detector dust-mask metadata for an IRIS SJI observation.

    Fetches the necessary parameters from the SJI flat-index and bad-pixel map files based on the observation time and SJI descriptor.
    These come from a SSWDB server.

    Parameters
    ----------
    date_obs : str
        Observation start time in FITS format.
    sji_name : str
        SJI descriptor such as ``"SJI_2796"``.

    Returns
    -------
    dict
        Dust-mask metadata used internally by `irispy.utils.dustbuster.clean_sji`.
    """
    if not sji_name.startswith("SJI_"):
        msg = f"Unsupported TDESC1 for SJI dust mask lookup: {sji_name!r}"
        raise ValueError(msg)
    flat_index_path = data_manager.get("iris_sji_flat_index")
    bad_pixel_path = data_manager.get("iris_sji_bad_pixel_map")
    obs_tai = Time(date_obs, format="fits", scale="utc").unix_tai
    channel = sji_name.split("_", 1)[1]
    suffix = SJI_CHANNEL_SUFFIX.get(channel)
    if suffix is None:
        msg = f"Unsupported SJI channel: {channel!r}"
        raise ValueError(msg)
    slit_center = (
        POINTING_INFO[f"CPX1_{suffix}"] - 1.0,
        POINTING_INFO[f"CPX2_{suffix}"] - 1.0,
    )
    mask_scale = POINTING_INFO[f"CDLT_{suffix}"]
    roll_deg = POINTING_INFO[f"BE_{suffix}"]
    flat_index = read_genx(flat_index_path)["SAVEGEN0"]
    bad_pixel_map = read_geny(bad_pixel_path)["p0"]
    matching_rows = [row for row in flat_index if row["IMG_PATH"] == sji_name]
    if not matching_rows:
        msg = f"No flat-index rows matched img_path={sji_name!r}"
        raise ValueError(msg)
    row_tai = np.array([row["FILETAI"] for row in matching_rows])
    record_ids = np.array([row["RECNUM"] for row in matching_rows])
    field_name = f"F{record_ids[np.argmin(np.abs(row_tai - obs_tai))]}"
    field_data = bad_pixel_map[field_name]
    dust_ids = np.concatenate([np.ravel(piece) for piece in field_data.flat])
    return {
        "dust_ids": dust_ids,
        "slit_center": slit_center,
        "mask_scale": mask_scale,
        "roll_deg": roll_deg,
    }
