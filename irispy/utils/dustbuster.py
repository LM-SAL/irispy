"""
Dust cleaning for `irispy.sji.SJICube` objects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
from astropy.time import Time
from numpy.typing import ArrayLike
from scipy import ndimage
from scipy.io import readsav as read_geny
from sunpy.data import manager as data_manager
from sunpy.io.special import read_genx

from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, POINTING_INFO, SJI_CHANNEL_SUFFIX

__all__ = ["clean_sji_dust", "get_sji_dust_params"]

_MASK_SHAPE = (2072, 1096)
_MANUAL_OFFSET = (0.0, -0.5)
_MAX_ALIGNMENT_SHIFT = 7
_MAX_ALIGNMENT_FRAMES = 8
_DISK_RADIUS_LIMIT = 880.0
_TEMPORAL_OFFSETS = np.array((-2, -1, 1, 2))
_SPATIAL_WINDOW = 9
_FLAT_INDEX_URLS = ["https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_flat.genx"]
_FLAT_INDEX_SHA256 = "40de195c55b0c5e04acb5f6f55883603c74a71bac6a5d639ec73f9d39d076b24"
_BAD_PIXEL_URLS = ["https://soho.nascom.nasa.gov/sdb/iris/data/20260326_032515_badpix.geny"]
_BAD_PIXEL_SHA256 = "c4d1884fb1a4f09b6ce4fe150a0aadab2664e479d86ae0ae063d8daa559e230d"


def _coord_values(cube: Any, name: str) -> np.ndarray:
    return np.asarray(cube.extra_coords[name]._lookup_tables[0][1].table[0].value)


def _bin_factor(cube: Any, *, axis: Literal["x", "y"]) -> int:
    return int(cube.meta["SUMSPTRL" if axis == "x" else "SUMSPAT"])


def _align_frame_idx(n_frames: int) -> np.ndarray:
    if n_frames <= _MAX_ALIGNMENT_FRAMES:
        return np.arange(n_frames)
    frame_idx = np.rint(np.linspace(0, n_frames - 1, _MAX_ALIGNMENT_FRAMES)).astype(int)
    return np.unique(frame_idx)


@data_manager.require("iris_sji_flat_index", _FLAT_INDEX_URLS, _FLAT_INDEX_SHA256, defer_download=True)
@data_manager.require("iris_sji_bad_pixel_map", _BAD_PIXEL_URLS, _BAD_PIXEL_SHA256, defer_download=True)
def _sji_dust_calibration_paths() -> tuple[str, str]:
    return (
        str(data_manager.get("iris_sji_flat_index")),
        str(data_manager.get("iris_sji_bad_pixel_map")),
    )


def clean_sji_dust(
    cube: Any,
    *,
    dust_ids: ArrayLike,
    slit_center: tuple[float, float],
    mask_scale: float,
    roll_deg: float,
    align: bool = True,
) -> Any:
    """Remove dust-contaminated pixels from an ``irispy`` ``SJICube``.

    Parameters
    ----------
    cube : Any
        ``irispy.sji.SJICube`` or an object with the same interface.
    dust_ids : array-like of int
        One-dimensional detector-mask bad-pixel addresses using Fortran-style
        linear indexing ``x + nx * y``.
    slit_center : tuple of float
        Slit center in detector-mask coordinates, expressed as zero-based
        detector pixels ``(x, y)`` before summing.
    mask_scale : float
        Detector-mask plate scale in arcsec per detector pixel.
    roll_deg : float
        Rotation angle from detector-mask coordinates into image coordinates,
        in degrees.
    align : bool, default=True
        If True, align the projected mask to the darkest valid pixels in the
        cube using a bounded integer search.

    Returns
    -------
    cleaned_cube : Any
        Copy of the input cube with dust pixels replaced.

    Notes
    -----
    This function intentionally uses the current ``SJICube`` API directly.

    It expects these data to exist on the cube:

    - ``cube.data`` with shape ``(nt, ny, nx)`` or ``(ny, nx)``,
    - ``cube.basic_wcs`` as a list for cubes and a single WCS for 2D slices,
    - ``cube.meta["SUMSPAT"]``,
    - ``cube.meta["SUMSPTRL"]``,
    - extra coordinate ``"exposure time"``,
    - extra coordinate ``"slit x position"``, and
    - extra coordinate ``"slit y position"``.

    No fallbacks are implemented on purpose. If those fields are missing, the
    function raises immediately so the missing metadata can be added to
    ``irispy`` rather than worked around locally.
    """
    data = np.asarray(cube.data, dtype=float)
    input_ndim = data.ndim
    if input_ndim == 2:
        data = data[np.newaxis, :, :]
    elif input_ndim != 3:
        raise ValueError("cube.data must have shape (nt, ny, nx) or (ny, nx).")

    n_frames, n_y, n_x = data.shape

    if cube.basic_wcs is None:
        raise ValueError("cube.basic_wcs is required.")
    wcs_list = cube.basic_wcs if input_ndim == 3 else [cube.basic_wcs]
    if len(wcs_list) != n_frames:
        raise ValueError("cube.basic_wcs must contain one WCS per frame.")

    exposure_s = _coord_values(cube, "exposure time")
    slit_x_pix = _coord_values(cube, "slit x position") - 1.0
    slit_y_pix = _coord_values(cube, "slit y position") - 1.0
    if exposure_s.shape != (n_frames,) or slit_x_pix.shape != (n_frames,) or slit_y_pix.shape != (n_frames,):
        raise ValueError("The required per-frame extra coordinates must each have shape (nt,).")

    y_bin = _bin_factor(cube, axis="y")
    x_bin = _bin_factor(cube, axis="x")

    ref_x_pix = np.array([w.wcs.crpix[0] for w in wcs_list])
    ref_y_pix = np.array([w.wcs.crpix[1] for w in wcs_list])
    x_scale = np.array([w.wcs.cdelt[0] for w in wcs_list])
    y_scale = np.array([w.wcs.cdelt[1] for w in wcs_list])
    ref_x_arcsec = np.array([w.wcs.crval[0] for w in wcs_list])
    ref_y_arcsec = np.array([w.wcs.crval[1] for w in wcs_list])
    image_scale = 0.5 * (np.abs(x_scale) + np.abs(y_scale))

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

    dx_pix = (dust_radius_arcsec[:, None] / image_scale[None, :]) * np.sin(dust_angle[:, None] - roll_rad)
    dy_pix = (dust_radius_arcsec[:, None] / image_scale[None, :]) * np.cos(dust_angle[:, None] - roll_rad)

    dust_x = dx_pix + slit_x_pix[None, :]
    dust_y = dy_pix + slit_y_pix[None, :]

    x0 = np.floor(dust_x).astype(np.int64)
    x1 = np.ceil(dust_x).astype(np.int64)
    y0 = np.floor(dust_y).astype(np.int64)
    y1 = np.ceil(dust_y).astype(np.int64)

    dust_x = np.concatenate([x0, x0, x1, x1], axis=0)
    dust_y = np.concatenate([y0, y1, y0, y1], axis=0)
    dust_t = np.broadcast_to(np.arange(n_frames, dtype=np.int64), dust_x.shape)

    align_shift = (0, 0)
    if align:
        align_frame_idx = _align_frame_idx(n_frames)
        align_x = dust_x[:, align_frame_idx]
        align_y = dust_y[:, align_frame_idx]
        align_t = np.broadcast_to(align_frame_idx, align_x.shape)
        align_ref_x_pix = ref_x_pix[align_frame_idx]
        align_ref_y_pix = ref_y_pix[align_frame_idx]
        align_x_scale = x_scale[align_frame_idx]
        align_y_scale = y_scale[align_frame_idx]
        align_ref_x_arcsec = ref_x_arcsec[align_frame_idx]
        align_ref_y_arcsec = ref_y_arcsec[align_frame_idx]

        best_score = np.inf
        best_shift = (0, 0)
        for shift_x in range(-_MAX_ALIGNMENT_SHIFT, _MAX_ALIGNMENT_SHIFT + 1):
            for shift_y in range(-_MAX_ALIGNMENT_SHIFT, _MAX_ALIGNMENT_SHIFT + 1):
                shifted_x = align_x + shift_x
                shifted_y = align_y + shift_y

                x_arcsec = (
                    (shifted_x + 1.0 - align_ref_x_pix[None, :]) * align_x_scale[None, :]
                    + align_ref_x_arcsec[None, :]
                )
                y_arcsec = (
                    (shifted_y + 1.0 - align_ref_y_pix[None, :]) * align_y_scale[None, :]
                    + align_ref_y_arcsec[None, :]
                )

                valid_hits = (
                    (shifted_x >= 0)
                    & (shifted_x < n_x)
                    & (shifted_y >= 0)
                    & (shifted_y < n_y)
                    & (np.abs(x_arcsec) <= _DISK_RADIUS_LIMIT)
                    & (np.abs(y_arcsec) <= _DISK_RADIUS_LIMIT)
                )
                if not np.any(valid_hits):
                    continue

                hit_t = align_t[valid_hits]
                hit_y = shifted_y[valid_hits]
                hit_x = shifted_x[valid_hits]
                hit_values = data[hit_t, hit_y, hit_x]
                valid_values = (hit_values != BAD_PIXEL_VALUE_SCALED) & np.isfinite(hit_values)
                if not np.any(valid_values):
                    continue

                # Using the sampled values directly keeps the shift ranking stable
                # while avoiding an expensive per-shift deduplication step.
                score = np.mean(hit_values[valid_values])
                if score < best_score:
                    best_score = score
                    best_shift = (shift_x, shift_y)

        align_shift = best_shift
        dust_x = dust_x + align_shift[0]
        dust_y = dust_y + align_shift[1]

    x_arcsec = (dust_x + 1.0 - ref_x_pix[None, :]) * x_scale[None, :] + ref_x_arcsec[None, :]
    y_arcsec = (dust_y + 1.0 - ref_y_pix[None, :]) * y_scale[None, :] + ref_y_arcsec[None, :]

    valid_hits = (
        (dust_x >= 0)
        & (dust_x < n_x)
        & (dust_y >= 0)
        & (dust_y < n_y)
        & (np.abs(x_arcsec) <= _DISK_RADIUS_LIMIT)
        & (np.abs(y_arcsec) <= _DISK_RADIUS_LIMIT)
    )

    dust_pixels = np.empty((0, 3), dtype=np.int64)
    fill_values = np.empty(0, dtype=float)
    if np.any(valid_hits):
        hit_t = dust_t[valid_hits]
        hit_y = dust_y[valid_hits]
        hit_x = dust_x[valid_hits]
        hit_values = data[hit_t, hit_y, hit_x]
        valid_values = (hit_values != BAD_PIXEL_VALUE_SCALED) & np.isfinite(hit_values)
        if np.any(valid_values):
            dust_pixels = np.stack([hit_t[valid_values], hit_y[valid_values], hit_x[valid_values]], axis=1)
            dust_pixels = np.unique(dust_pixels, axis=0)

            fill_mask = np.zeros(data.shape, dtype=bool)
            fill_mask[
                dust_pixels[:, 0],
                dust_pixels[:, 1],
                dust_pixels[:, 2],
            ] = True

            fill_t = dust_pixels[:, 0]
            fill_y = dust_pixels[:, 1]
            fill_x = dust_pixels[:, 2]

            neighbor_t = fill_t[:, None] + _TEMPORAL_OFFSETS[None, :]
            neighbor_y = np.broadcast_to(fill_y[:, None], neighbor_t.shape)
            neighbor_x = np.broadcast_to(fill_x[:, None], neighbor_t.shape)

            in_time_range = (neighbor_t >= 0) & (neighbor_t < n_frames)
            clipped_t = np.clip(neighbor_t, 0, n_frames - 1)

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

    cleaned_cube = deepcopy(cube)
    cleaned_cube.data[...] = data[0] if input_ndim == 2 else data
    if dust_pixels.size > 0:
        write_index = (
            (dust_pixels[:, 1], dust_pixels[:, 2])
            if input_ndim == 2
            else (dust_pixels[:, 0], dust_pixels[:, 1], dust_pixels[:, 2])
        )
        cleaned_cube.data[write_index] = fill_values
        if cleaned_cube.mask is not None:
            cleaned_cube.mask[write_index] = False

    return cleaned_cube


def get_sji_dust_params(*, date_obs: str, sji_name: str) -> dict:
    """
    Return the detector dust-mask arguments needed by ``clean_sji_dust``.

    Parameters
    ----------
    date_obs : str
        Observation start time in FITS format.
    sji_name : str
        SJI descriptor such as ``"SJI_2796"``.
    Returns
    -------
    dict
        Minimal keyword arguments for ``clean_sji_dust``.
    """
    flat_index_path, bad_pixel_path = _sji_dust_calibration_paths()

    obs_tai = Time(date_obs, format="fits", scale="utc").unix_tai

    if not sji_name.startswith("SJI_"):
        raise ValueError(f"Unsupported TDESC1 for SJI dust mask lookup: {sji_name!r}")
    channel = sji_name.split("_", 1)[1]
    suffix = SJI_CHANNEL_SUFFIX.get(channel)
    if suffix is None:
        raise ValueError(f"Unsupported SJI channel: {channel!r}")
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
        raise ValueError(f"No flat-index rows matched img_path={sji_name!r}")
    row_tai = np.array([row["FILETAI"] for row in matching_rows])
    record_ids = np.array([row["RECNUM"] for row in matching_rows])
    field_name = f"F{record_ids[np.argmin(np.abs(row_tai - obs_tai))]}"
    field_data = bad_pixel_map[field_name]
    dust_ids = np.concatenate([np.ravel(piece) for piece in field_data.flat]).astype(np.int64, copy=False)
    return {
        "dust_ids": dust_ids,
        "slit_center": slit_center,
        "mask_scale": mask_scale,
        "roll_deg": roll_deg,
    }
