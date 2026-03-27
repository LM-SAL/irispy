"""Dust cleaning for ``irispy.sji.SJICube`` objects.

This module implements a clean, direct dust-removal algorithm centered on the
current ``irispy`` API. It deliberately avoids metadata wrapper classes and
instead reads what it needs straight from the cube, its ``.meta`` object, its
per-frame ``basic_wcs``, and its extra coordinates.

The detector dust-mask geometry is still passed in explicitly because that is
calibration data, not an intrinsic property of the cube.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
from astropy.time import Time
from numpy.typing import ArrayLike
from scipy import ndimage

from irispy.utils.variables import POINTING_INFO

__all__ = ["dustbuster_sji_cube", "get_sji_dust_metadata_from_ssw"]


def _get_extra_coord_values(cube: Any, name: str) -> np.ndarray:
    extra_coords = getattr(cube, "extra_coords", None)
    if extra_coords is not None and hasattr(extra_coords, "keys") and name in extra_coords.keys():
        named_coord = extra_coords[name]
        lookup_tables = getattr(named_coord, "_lookup_tables", None)
        if lookup_tables:
            table = lookup_tables[0][1].table
            if table:
                values = table[0]
                return np.asarray(getattr(values, "value", values), dtype=float)

    values = cube.axis_world_coords(name, wcs=extra_coords)[0]
    return np.asarray(getattr(values, "value", values), dtype=float)


def _get_sji_summing_factor(cube: Any, *, axis: Literal["x", "y"]) -> int:
    if axis == "x":
        value = getattr(cube.meta, "spectral_summing_factor", None)
        if value is None and hasattr(cube.meta, "get"):
            value = cube.meta.get("SUMSPTRL")
    else:
        value = getattr(cube.meta, "spatial_summing_factor", None)
        if value is None and hasattr(cube.meta, "get"):
            value = cube.meta.get("SUMSPAT")
    return 1 if value is None else int(value)


def dustbuster_sji_cube(
    cube: Any,
    *,
    bad_pixel_addresses: ArrayLike,
    slit_center_mask: tuple[float, float],
    mask_plate_scale: float,
    roll_deg: float,
    mask_shape: tuple[int, int] = (2072, 1096),
    manual_offset: tuple[float, float] = (0.0, -0.5),
    expand_mask: int = 0,
    max_alignment_shift: int = 7,
    disk_radius_limit: float = 880.0,
    invalid_value: float = -200.0,
    align_mask: bool = True,
    temporal_offsets: tuple[int, ...] = (-2, -1, 1, 2),
    spatial_method: Literal["mean", "median"] = "mean",
    spatial_window: int = 9,
) -> tuple[Any, np.ndarray, np.ndarray, tuple[int, int]]:
    """Remove dust-contaminated pixels from an ``irispy`` ``SJICube``.

    Parameters
    ----------
    cube : Any
        ``irispy.sji.SJICube`` or an object with the same interface.
    bad_pixel_addresses : array-like of int
        One-dimensional detector-mask bad-pixel addresses using Fortran-style
        linear indexing ``x + nx * y``.
    slit_center_mask : tuple of float
        Slit center in detector-mask coordinates, expressed as zero-based
        detector pixels ``(x, y)`` before summing.
    mask_plate_scale : float
        Detector-mask plate scale in arcsec per detector pixel.
    roll_deg : float
        Rotation angle from detector-mask coordinates into image coordinates,
        in degrees.
    mask_shape : tuple of int, default=(2072, 1096)
        Shape of the detector dust-mask grid as ``(nx, ny)``.
    manual_offset : tuple of float, default=(0.0, -0.5)
        Empirical x/y offset, in summed image pixels, applied before
        projection.
    expand_mask : int, default=0
        Number of detector-mask pixels by which to dilate the mask before
        projection.
    max_alignment_shift : int, default=7
        Maximum integer x/y shift considered during alignment.
    disk_radius_limit : float, default=880.0
        Ignore candidate pixels farther than this many arcsec from disk center
        when scoring alignment.
    invalid_value : float, default=-200.0
        Sentinel value used for invalid pixels in the scaled SJI cube.
    align_mask : bool, default=True
        If True, align the projected mask to the darkest valid pixels in the
        cube using a bounded integer search.
    temporal_offsets : tuple of int, default=(-2, -1, 1, 2)
        Frame offsets used to seek replacement values at the same spatial
        location in nearby frames.
    spatial_method : {'mean', 'median'}, default='mean'
        Same-frame spatial fallback used when no temporal neighbor is valid.
    spatial_window : int, default=9
        Width of the same-frame spatial fallback window in pixels.

    Returns
    -------
    cleaned_cube : Any
        Copy of the input cube with dust pixels replaced.
    bad_pixel_indices : ndarray of int64, shape (n, 3)
        Replaced pixels as ``(frame, y, x)`` integer triplets.
    replacement_values : ndarray of float, shape (n,)
        Values written into ``bad_pixel_indices``.
    applied_shift : tuple of int
        Integer x/y shift applied to the projected mask.

    Notes
    -----
    This function intentionally uses the current ``SJICube`` API directly.

    It expects these data to exist on the cube:

    - ``cube.data`` with shape ``(nt, ny, nx)`` or ``(ny, nx)``,
    - ``cube.basic_wcs`` containing one FITS-style WCS per frame,
    - ``cube.meta.spatial_summing_factor``,
    - ``cube.meta.spectral_summing_factor``,
    - extra coordinate ``"exposure time"``,
    - extra coordinate ``"slit x position"``, and
    - extra coordinate ``"slit y position"``.

    No fallbacks are implemented on purpose. If those fields are missing, the
    function raises immediately so the missing metadata can be added to
    ``irispy`` rather than worked around locally.
    """
    data = np.asarray(cube.data, dtype=float)
    original_ndim = data.ndim
    if original_ndim == 2:
        data = data[np.newaxis, :, :]
    elif original_ndim != 3:
        raise ValueError("cube.data must have shape (nt, ny, nx) or (ny, nx).")

    nt, ny, nx = data.shape

    basic_wcs = cube.basic_wcs
    if basic_wcs is None:
        raise ValueError("cube.basic_wcs is required.")
    if isinstance(basic_wcs, list):
        wcs_list = basic_wcs
    else:
        wcs_list = [basic_wcs]
    if len(wcs_list) != nt:
        raise ValueError("cube.basic_wcs must contain one WCS per frame.")

    exposure_times = _get_extra_coord_values(cube, "exposure time")
    slit_x = _get_extra_coord_values(cube, "slit x position") - 1.0
    slit_y = _get_extra_coord_values(cube, "slit y position") - 1.0
    if exposure_times.shape != (nt,) or slit_x.shape != (nt,) or slit_y.shape != (nt,):
        raise ValueError("The required per-frame extra coordinates must each have shape (nt,).")

    sumspat = _get_sji_summing_factor(cube, axis="y")
    sumsptrl = _get_sji_summing_factor(cube, axis="x")

    crpix1 = np.asarray([w.wcs.crpix[0] for w in wcs_list], dtype=float)
    crpix2 = np.asarray([w.wcs.crpix[1] for w in wcs_list], dtype=float)
    cdelt1 = np.asarray([w.wcs.cdelt[0] for w in wcs_list], dtype=float)
    cdelt2 = np.asarray([w.wcs.cdelt[1] for w in wcs_list], dtype=float)
    crval1 = np.asarray([w.wcs.crval[0] for w in wcs_list], dtype=float)
    crval2 = np.asarray([w.wcs.crval[1] for w in wcs_list], dtype=float)
    image_plate_scale = 0.5 * (np.abs(cdelt1) + np.abs(cdelt2))

    detector_nx, detector_ny = mask_shape
    detector_mask = np.zeros((detector_nx, detector_ny), dtype=bool)
    detector_addresses = np.asarray(bad_pixel_addresses, dtype=np.int64)
    detector_x = detector_addresses % detector_nx
    detector_y = detector_addresses // detector_nx
    detector_mask[detector_x, detector_y] = True

    if expand_mask > 0:
        structure = np.ones((2 * expand_mask + 1, 2 * expand_mask + 1), dtype=bool)
        detector_mask = ndimage.binary_dilation(detector_mask, structure=structure)

    detector_x, detector_y = np.nonzero(detector_mask)

    offspat = (sumspat - 1.0) / (2.0 * sumspat)
    offsptrl = (sumsptrl - 1.0) / (2.0 * sumsptrl)

    mask_x = detector_x.astype(float) / float(sumsptrl) + manual_offset[0]
    mask_y = detector_y.astype(float) / float(sumspat) + manual_offset[1]

    mask_slit_x = slit_center_mask[0] / float(sumsptrl) + offsptrl
    mask_slit_y = slit_center_mask[1] / float(sumspat) + offspat

    dx_mask = mask_x - mask_slit_x
    dy_mask = mask_y - mask_slit_y
    mask_radius_arcsec = np.hypot(dx_mask, dy_mask) * mask_plate_scale
    mask_angle = np.arctan2(dx_mask, dy_mask)
    roll_rad = np.deg2rad(roll_deg)

    dx_image = (mask_radius_arcsec[:, None] / image_plate_scale[None, :]) * np.sin(mask_angle[:, None] - roll_rad)
    dy_image = (mask_radius_arcsec[:, None] / image_plate_scale[None, :]) * np.cos(mask_angle[:, None] - roll_rad)

    projected_x = dx_image + slit_x[None, :]
    projected_y = dy_image + slit_y[None, :]

    floor_x = np.floor(projected_x).astype(np.int64)
    ceil_x = np.ceil(projected_x).astype(np.int64)
    floor_y = np.floor(projected_y).astype(np.int64)
    ceil_y = np.ceil(projected_y).astype(np.int64)

    mapped_x = np.concatenate([floor_x, floor_x, ceil_x, ceil_x], axis=0)
    mapped_y = np.concatenate([floor_y, ceil_y, floor_y, ceil_y], axis=0)
    mapped_t = np.broadcast_to(np.arange(nt, dtype=np.int64), mapped_x.shape)

    applied_shift = (0, 0)
    if align_mask:
        best_score = np.inf
        best_shift = (0, 0)
        for shift_x in range(-max_alignment_shift, max_alignment_shift + 1):
            for shift_y in range(-max_alignment_shift, max_alignment_shift + 1):
                trial_x = mapped_x + shift_x
                trial_y = mapped_y + shift_y

                x_arcsec = (trial_x + 1.0 - crpix1[None, :]) * cdelt1[None, :] + crval1[None, :]
                y_arcsec = (trial_y + 1.0 - crpix2[None, :]) * cdelt2[None, :] + crval2[None, :]

                valid = (
                    (trial_x >= 0)
                    & (trial_x < nx)
                    & (trial_y >= 0)
                    & (trial_y < ny)
                    & (np.abs(x_arcsec) <= disk_radius_limit)
                    & (np.abs(y_arcsec) <= disk_radius_limit)
                )
                if not np.any(valid):
                    continue

                trial_values = np.full(trial_x.shape, np.nan, dtype=float)
                trial_values[valid] = data[mapped_t[valid], trial_y[valid], trial_x[valid]]
                valid &= trial_values != invalid_value
                valid &= np.isfinite(trial_values)
                if not np.any(valid):
                    continue

                coords = np.stack([mapped_t[valid], trial_y[valid], trial_x[valid]], axis=1)
                coords = np.unique(coords, axis=0)
                score = float(np.mean(data[coords[:, 0], coords[:, 1], coords[:, 2]]))
                if score < best_score:
                    best_score = score
                    best_shift = (shift_x, shift_y)

        applied_shift = best_shift
        mapped_x = mapped_x + applied_shift[0]
        mapped_y = mapped_y + applied_shift[1]

    x_arcsec = (mapped_x + 1.0 - crpix1[None, :]) * cdelt1[None, :] + crval1[None, :]
    y_arcsec = (mapped_y + 1.0 - crpix2[None, :]) * cdelt2[None, :] + crval2[None, :]

    keep = (
        (mapped_x >= 0)
        & (mapped_x < nx)
        & (mapped_y >= 0)
        & (mapped_y < ny)
        & (np.abs(x_arcsec) <= disk_radius_limit)
        & (np.abs(y_arcsec) <= disk_radius_limit)
    )

    mapped_values = np.full(mapped_x.shape, np.nan, dtype=float)
    mapped_values[keep] = data[mapped_t[keep], mapped_y[keep], mapped_x[keep]]
    keep &= mapped_values != invalid_value
    keep &= np.isfinite(mapped_values)

    bad_pixel_indices = np.empty((0, 3), dtype=np.int64)
    replacement_values = np.empty(0, dtype=float)
    if np.any(keep):
        bad_pixel_indices = np.stack([mapped_t[keep], mapped_y[keep], mapped_x[keep]], axis=1)
        bad_pixel_indices = np.unique(bad_pixel_indices, axis=0)

        bad_pixel_mask = np.zeros(data.shape, dtype=bool)
        bad_pixel_mask[
            bad_pixel_indices[:, 0],
            bad_pixel_indices[:, 1],
            bad_pixel_indices[:, 2],
        ] = True

        target_t = bad_pixel_indices[:, 0]
        target_y = bad_pixel_indices[:, 1]
        target_x = bad_pixel_indices[:, 2]

        temporal_offsets_arr = np.asarray(temporal_offsets, dtype=np.int64)
        candidate_t = target_t[:, None] + temporal_offsets_arr[None, :]
        candidate_y = np.broadcast_to(target_y[:, None], candidate_t.shape)
        candidate_x = np.broadcast_to(target_x[:, None], candidate_t.shape)

        in_range = (candidate_t >= 0) & (candidate_t < nt)
        safe_t = np.clip(candidate_t, 0, nt - 1)

        valid_candidates = in_range & (~bad_pixel_mask[safe_t, candidate_y, candidate_x])
        candidate_values = np.full(candidate_t.shape, np.nan, dtype=float)
        candidate_values[valid_candidates] = data[
            safe_t[valid_candidates],
            candidate_y[valid_candidates],
            candidate_x[valid_candidates],
        ]
        valid_candidates &= candidate_values != invalid_value
        valid_candidates &= np.isfinite(candidate_values)

        candidate_values[valid_candidates] /= exposure_times[safe_t[valid_candidates]]
        replacement_values = np.full(target_t.shape, np.nan, dtype=float)
        rows_with_candidates = np.any(valid_candidates, axis=1)
        if np.any(rows_with_candidates):
            with np.errstate(invalid="ignore"):
                replacement_values[rows_with_candidates] = np.nanmedian(
                    candidate_values[rows_with_candidates],
                    axis=1,
                )
        replacement_values *= exposure_times[target_t]

        missing = ~np.isfinite(replacement_values)
        if np.any(missing):
            spatial_fill = np.empty_like(data)
            for frame in range(nt):
                image = data[frame].copy()
                image[image == invalid_value] = np.nan
                image[bad_pixel_mask[frame]] = np.nan

                if spatial_method == "median":
                    smoothed = ndimage.generic_filter(
                        image,
                        np.nanmedian,
                        size=spatial_window,
                        mode="nearest",
                    )
                elif spatial_method == "mean":
                    finite = np.isfinite(image).astype(float)
                    filled = np.where(np.isfinite(image), image, 0.0)
                    numerator = ndimage.uniform_filter(filled, size=spatial_window, mode="nearest")
                    denominator = ndimage.uniform_filter(finite, size=spatial_window, mode="nearest")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        smoothed = numerator / denominator
                    smoothed[denominator == 0.0] = np.nan
                else:
                    raise ValueError("spatial_method must be 'mean' or 'median'.")

                spatial_fill[frame] = smoothed

            replacement_values[missing] = spatial_fill[
                target_t[missing],
                target_y[missing],
                target_x[missing],
            ]

            still_missing = ~np.isfinite(replacement_values)
            if np.any(still_missing):
                valid_data = data[np.isfinite(data) & (data != invalid_value)]
                replacement_values[still_missing] = float(np.median(valid_data))

    cleaned_cube = deepcopy(cube)
    cleaned_cube.data[...] = data[0] if original_ndim == 2 else data
    if bad_pixel_indices.size > 0:
        if original_ndim == 2:
            cleaned_cube.data[
                bad_pixel_indices[:, 1],
                bad_pixel_indices[:, 2],
            ] = replacement_values
            if getattr(cleaned_cube, "mask", None) is not None:
                cleaned_cube.mask[
                    bad_pixel_indices[:, 1],
                    bad_pixel_indices[:, 2],
                ] = False
        else:
            cleaned_cube.data[
                bad_pixel_indices[:, 0],
                bad_pixel_indices[:, 1],
                bad_pixel_indices[:, 2],
            ] = replacement_values
            if getattr(cleaned_cube, "mask", None) is not None:
                cleaned_cube.mask[
                    bad_pixel_indices[:, 0],
                    bad_pixel_indices[:, 1],
                    bad_pixel_indices[:, 2],
                ] = False

    return cleaned_cube, bad_pixel_indices, replacement_values, applied_shift


def get_sji_dust_metadata_from_ssw(
    cube,
    *,
    flat_genx_path: str,
    badpix_geny_path: str,
    read_genx,
    read_geny,
) -> dict:
    """
    Return the detector dust-mask metadata needed by the SJI dustbuster.

    Parameters
    ----------
    cube : irispy.sji.SJICube
        Input SJI cube.
    flat_genx_path : str
        Path to the latest IRIS ``*flat.genx`` file.
    badpix_geny_path : str
        Path to the latest IRIS ``*badpix.geny`` file.
    read_genx : callable
        Reader for the ``.genx`` index file.
    read_geny : callable
        Reader for the ``.geny`` bad-pixel structure.
    Returns
    -------
    dict
        Dictionary with keys:
        ``bad_pixel_addresses``, ``slit_center_mask``, ``mask_plate_scale``,
        ``roll_deg``, ``sumspat``, ``sumsptrl``, ``img_path``, and ``recnum``.
    """
    date_obs = str(cube.meta["DATE_OBS"])
    obs_tai = float(Time(date_obs, format="fits", scale="utc").unix_tai)

    img_path = str(cube.meta["TDESC1"])
    if not img_path.startswith("SJI_"):
        raise ValueError(f"Unsupported TDESC1 for SJI dust mask lookup: {img_path!r}")
    channel = img_path.split("_", 1)[1]

    if channel == "1330":
        suffix = "133"
    elif channel == "1400":
        suffix = "140"
    elif channel == "2796":
        suffix = "279"
    elif channel == "2832":
        suffix = "283"
    elif channel == "1600":
        suffix = "MIR"
    elif channel == "5000":
        suffix = "FSI"
    else:
        raise ValueError(f"Unsupported SJI channel: {channel!r}")

    slit_center_mask = (
        float(POINTING_INFO[f"CPX1_{suffix}"]) - 1.0,
        float(POINTING_INFO[f"CPX2_{suffix}"]) - 1.0,
    )
    mask_plate_scale = float(POINTING_INFO[f"CDLT_{suffix}"])
    roll_deg = float(POINTING_INFO[f"BE_{suffix}"])
    cdlt_1p5 = float(POINTING_INFO["CDLT_1P5"])

    flat_index = read_genx(flat_genx_path)["SAVEGEN0"]
    badpix_struct = read_geny(badpix_geny_path)["p0"]

    # Expect the flat index rows to expose img_path, filetai, recnum.
    matching_rows = [row for row in flat_index if str(row["IMG_PATH"]) == img_path]
    if not matching_rows:
        raise ValueError(f"No flat-index rows matched img_path={img_path!r}")

    filetai = np.array([row["FILETAI"] for row in matching_rows], dtype=float)
    recnums = np.array([row["RECNUM"] for row in matching_rows], dtype=int)

    best = int(np.argmin(np.abs(filetai - obs_tai)))
    recnum = int(recnums[best])

    # Match the IDL field lookup: badpix_str.F<recnum>
    field_name = f"F{recnum}"

    if isinstance(badpix_struct, dict):
        raw = badpix_struct[field_name]
    else:
        raw = getattr(badpix_struct, field_name)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        pieces = [np.asarray(piece, dtype=np.int64).ravel() for piece in raw.flat]
        bad_pixel_addresses = (
            np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        )
    elif isinstance(raw, (list, tuple)):
        pieces = [np.asarray(piece, dtype=np.int64).ravel() for piece in raw]
        bad_pixel_addresses = (
            np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        )
    else:
        bad_pixel_addresses = np.asarray(raw, dtype=np.int64).ravel()

    return {
        "bad_pixel_addresses": bad_pixel_addresses,
        "slit_center_mask": slit_center_mask,
        "mask_plate_scale": mask_plate_scale,
        "roll_deg": roll_deg,
        "cdlt_1p5": cdlt_1p5,
        "sumspat": _get_sji_summing_factor(cube, axis="y"),
        "sumsptrl": _get_sji_summing_factor(cube, axis="x"),
        "img_path": img_path,
        "recnum": recnum,
    }

if __name__ == "__main__":
    from irispy.io import read_files
    from scipy.io import readsav as read_geny
    from sunpy.io.special import read_genx
    from pathlib import Path

    cube = read_files("~/Downloads/iris_l2_20130902_182935_4000005156_SJI_2796_t000.fits")
    meta = get_sji_dust_metadata_from_ssw(
        cube,
        flat_genx_path=Path("~/Downloads/latest_calibration/20260326_032515_flat.genx").expanduser(),
        badpix_geny_path=Path("~/Downloads/latest_calibration/20260326_032515_badpix.geny").expanduser(),
        read_genx=read_genx,
        read_geny=read_geny,
    )

    cleaned_cube, bad_pixel_indices, replacement_values, applied_shift = dustbuster_sji_cube(
        cube,
        bad_pixel_addresses=meta["bad_pixel_addresses"],
        slit_center_mask=meta["slit_center_mask"],
        mask_plate_scale=meta["mask_plate_scale"],
        roll_deg=meta["roll_deg"],
    )
    cube.plot()
    cleaned_cube.plot()
    import matplotlib.pyplot as plt
    plt.show()