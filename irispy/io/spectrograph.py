import warnings
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import TimeDelta
from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord
from sunpy.time import parse_time

from irispy._spectrograph_wcs import (
    _create_raster_gwcs,
    _prepare_raster_wcs_header,
    _raster_wcs_bad_row_mask,
    _sanitize_raster_wcs_tables,
    _slit_offset_column,
    _validate_raster_wcs_inputs,
)
from irispy.io._raster_combine import _combine_raster_cubes, _materialize_deferred_raster_gwcs
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]


def _header_time(header, *keys):
    for key in keys:
        value = header.get(key)
        if value:
            return parse_time(value)
    msg = f"Header is missing all usable time keys: {keys}"
    raise ValueError(msg)


def read_spectrograph_lvl2(
    filenames: str | Path | list[str | Path],
    *,
    spectral_windows: str | list[str] | None = None,
    uncertainty: bool = False,
    memmap: bool = False,
    revert_v34: bool = False,
):
    """
    Reads either a SINGLE IRIS level 2 spectrograph FITS or a list of them.

    .. warning::

        Does not handle tar files.
        That is handled by `irispy.io.read_files`.

    Parameters
    ----------
    filenames: `list` of `str` or `str`
        Filename or list of filenames to be read. They must all be associated with the same
        OBS number; multi-file reads raise a `ValueError` if the OBSID or STARTOBS
        values do not match across files.
    spectral_windows: iterable of `str` or `str`
        Spectral windows to extract from files. Default=None, implies, extract all
        spectral windows.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        the data units will be unscaled, not the usual data numbers (DN).
    revert_v34 : `bool`, optional.
        Will undo the data and WCS flipping made to V34 observations.
        Defaults to `False`.

    Returns
    -------
    `RasterCollection`
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = [str(f) for f in filenames]
    defer_raster_gwcs = len(filenames) > 1
    if uncertainty and memmap:
        warnings.warn(
            "uncertainty is not computed when memmap=True; uncertainty will be None.",
            UserWarning,
            stacklevel=2,
        )
    compute_uncertainty = uncertainty and not memmap
    with fits.open(filenames[0], memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
        v34 = hdulist[0].header["STEPS_AV"] < -0.01
        hdulist.verify("silentfix")
        primary_header = hdulist[0].header.copy()
        windows_in_obs = np.array(
            [primary_header[f"TDESC{i}"] for i in range(1, primary_header["NWIN"] + 1)],
        )
        if not spectral_windows:
            spectral_windows_req = windows_in_obs
            window_fits_indices = range(1, len(hdulist) - 2)
        else:
            spectral_windows_req = [spectral_windows] if isinstance(spectral_windows, str) else spectral_windows
            spectral_windows_req = np.asarray(spectral_windows_req, dtype="U")
            window_is_in_obs = np.asarray([window in windows_in_obs for window in spectral_windows_req])
            if not all(window_is_in_obs):
                missing_windows = spectral_windows_req[~window_is_in_obs]
                msg = f"Spectral windows {missing_windows.tolist()} not in file {filenames[0]}"
                raise ValueError(msg)
            window_indices = []
            for window in spectral_windows_req:
                matches = np.where(windows_in_obs == window)[0]
                if len(matches) > 1:
                    msg = f"Spectral window '{window}' appears multiple times in file {filenames[0]}"
                    raise ValueError(msg)
                window_indices.append(int(matches[0]) + 1)
            window_fits_indices = window_indices
        data_dict = {window_name: [] for window_name in spectral_windows_req}

    # Per-window running means of good WCS table rows across files, used as
    # fallback only when a later file has no good rows for that window.
    running_wcs_fallbacks = {window_name: [np.zeros((2, 2)), np.zeros(2), 0] for window_name in spectral_windows_req}

    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            aux = hdulist[-2]
            file_startobs = _header_time(hdulist[0].header, "STARTOBS", "DATE_OBS")
            times = file_startobs + TimeDelta(aux.data[:, aux.header["TIME"]] * u.s)
            fov_center = SkyCoord(
                Tx=aux.data[:, aux.header["XCENIX"]],
                Ty=aux.data[:, aux.header["YCENIX"]],
                unit=u.arcsec,
                frame=Helioprojective,
            )
            obs_vrix = aux.data[:, aux.header["OBS_VRIX"]] * u.m / u.s
            ophaseix = aux.data[:, aux.header["OPHASEIX"]]
            exposure_times_fuv = aux.data[:, aux.header["EXPTIMEF"]] * u.s
            exposure_times_nuv = aux.data[:, aux.header["EXPTIMEN"]] * u.s
            pc_indices = [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]
            pc = aux.data[:, pc_indices].reshape(-1, 2, 2) * u.pix

            t_ref = times[0]
            flip = v34 and not revert_v34
            if flip:
                times = times[::-1]
                fov_center = fov_center[::-1]
                obs_vrix = obs_vrix[::-1]
                ophaseix = ophaseix[::-1]
                exposure_times_fuv = exposure_times_fuv[::-1]
                exposure_times_nuv = exposure_times_nuv[::-1]
                # Reversing the step axis flips the sign of CDELT3 in the
                # prepared header, so the step-coupled off-diagonal PC terms
                # must change sign too, exactly as _prepare_raster_wcs_header
                # does for the header PC2_3/PC3_2.
                pc = pc[::-1].copy()
                pc[:, 0, 1] *= -1
                pc[:, 1, 0] *= -1
            dt = (times - t_ref).to_value(u.s) * u.s

            for i, window_name in enumerate(spectral_windows_req):
                window_header = hdulist[window_fits_indices[i]].header.copy()
                meta = SGMeta(
                    hdulist[0].header,
                    window_name,
                    data_shape=hdulist[window_fits_indices[i]].data.shape,
                )
                exposure_times = exposure_times_nuv
                dn_unit = DN_UNIT["NUV"]
                readout_noise = READOUT_NOISE["NUV"]
                if "FUV" in meta.detector:
                    exposure_times = exposure_times_fuv
                    dn_unit = DN_UNIT["FUV"]
                    readout_noise = READOUT_NOISE["FUV"]
                meta.add("exposure time", exposure_times, None, 0)
                meta.add("exposure FOV center", fov_center, None, 0)
                meta.add("observer radial velocity", obs_vrix, None, 0)
                meta.add("orbital phase", ophaseix, None, 0)
                observer = meta.observer
                sit_and_stare = window_header["CDELT3"] == 0
                meta["sit_and_stare"] = sit_and_stare
                meta["flipped"] = flip
                meta["memmap_path"] = filename
                meta["memmap_ext"] = window_fits_indices[i]
                prepared_wcs_header = _prepare_raster_wcs_header(
                    window_header,
                    aux.data,
                    meta.spectral_band,
                    sit_and_stare=sit_and_stare,
                    flip=flip,
                )
                if sit_and_stare:
                    # Sit-and-stare: CRVAL comes from the header, only PC may need fallback.
                    crval = (
                        np.repeat(
                            [[prepared_wcs_header["CRVAL3"], prepared_wcs_header["CRVAL2"]]],
                            len(times),
                            axis=0,
                        )
                        * u.arcsec
                    )
                else:
                    offset_index = _slit_offset_column(meta.spectral_band)
                    xcen = aux.data[:, aux.header["XCENIX"]] - aux.data[:, offset_index] * (SLIT_WIDTH.value / 2)
                    ycen = aux.data[:, aux.header["YCENIX"]]
                    crval = np.column_stack((xcen, ycen)) * u.arcsec
                    if flip:
                        crval = crval[::-1]

                running_pc_sum, running_crval_sum, running_count = running_wcs_fallbacks[window_name]
                fallback_pc = (running_pc_sum / running_count * u.pix) if running_count > 0 else None
                fallback_crval = (running_crval_sum / running_count * u.arcsec) if running_count > 0 else None
                bad_rows = _raster_wcs_bad_row_mask(pc, crval, pc_only=sit_and_stare)
                good_mask = ~bad_rows
                # Accumulate raw observed values before sanitization for running mean.
                if good_mask.any():
                    running_wcs_fallbacks[window_name][0] += pc[good_mask].sum(axis=0).to_value(u.pix)
                    running_wcs_fallbacks[window_name][1] += crval[good_mask].sum(axis=0).to_value(u.arcsec)
                    running_wcs_fallbacks[window_name][2] += int(good_mask.sum())
                pc_sanitized, crval = _sanitize_raster_wcs_tables(
                    pc.copy(),
                    crval,
                    fallback_pc,
                    fallback_crval,
                    bad_rows=bad_rows,
                )

                try:
                    fits_wcs = WCS(prepared_wcs_header)
                    _set_wcs_aux_obs_coord(fits_wcs, observer)
                    # Window headers carry no DATE-OBS; without it the frame
                    # derived from this WCS has obstime=None and cannot be
                    # transformed to/from cube.celestial_frame.
                    fits_wcs.wcs.dateobs = observer.obstime.utc.isot
                    _validate_raster_wcs_inputs(prepared_wcs_header, pc_sanitized, crval, dt)
                    if defer_raster_gwcs:
                        cube_wcs = fits_wcs
                    else:
                        cube_wcs = _create_raster_gwcs(
                            prepared_wcs_header,
                            pc_sanitized,
                            crval,
                            dt,
                            t_ref,
                            observer,
                            sit_and_stare=sit_and_stare,
                        )
                except Exception as e:  # NOQA: BLE001
                    logger.warning(
                        f"Skipping spectral window {window_name!r} in {filename}: unable to construct WCS ({e})"
                    )
                    continue

                out_uncertainty = None
                data_mask = None
                if not memmap:
                    data_mask = hdulist[window_fits_indices[i]].data == BAD_PIXEL_VALUE_SCALED
                if compute_uncertainty:
                    out_uncertainty = calculate_uncertainty(
                        hdulist[window_fits_indices[i]].data,
                        readout_noise,
                        dn_unit,
                    )
                if flip:
                    data = np.flip(hdulist[window_fits_indices[i]].data, axis=0)
                    if data_mask is not None:
                        data_mask = np.flip(data_mask, axis=0)
                    if out_uncertainty is not None:
                        out_uncertainty = np.flip(out_uncertainty, axis=0)
                else:
                    data = hdulist[window_fits_indices[i]].data
                cube = SpectrogramCube(
                    data,
                    wcs=cube_wcs,
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    mask=data_mask,
                    _fits_wcs=fits_wcs,
                    _raster_wcs_header=prepared_wcs_header,
                    _raster_pc_table=pc_sanitized,
                    _raster_crval_table=crval,
                )
                cube._defer_raster_gwcs = defer_raster_gwcs
                cube.extra_coords.add("time", 0, times, physical_types="time")
                data_dict[window_name].append(cube)
    window_data_pairs = []
    for _window_name, cubes in data_dict.items():
        if not cubes:
            logger.warning(f"Skipping spectral window {_window_name!r}: no readable cubes were loaded.")
            continue
        if len(cubes) == 1:
            cube = _materialize_deferred_raster_gwcs(cubes[0])
            cube._raster_boundaries = [(0, cube.shape[0])]
        else:
            cube = _combine_raster_cubes(cubes, memmap=memmap)
        window_data_pairs.append((_window_name, cube))
    if not window_data_pairs:
        msg = "No spectral windows could be loaded."
        raise ValueError(msg)
    aligned_axes = tuple(range(window_data_pairs[0][1].data.ndim - 1))
    return RasterCollection(window_data_pairs, aligned_axes=aligned_axes)
