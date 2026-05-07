from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.screens import SphericalScreen
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy._spectrograph_wcs import (
    _create_raster_gwcs,
    _prepare_raster_wcs_header,
    _raster_wcs_bad_row_mask,
    _sanitize_raster_wcs_tables,
)
from irispy.io._raster_combine import _finalize_window_object
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]


def _header_time(header, *keys):
    for key in keys:
        value = header.get(key)
        if value:
            return Time(value, format="isot", scale="utc")
    msg = f"Header is missing all usable time keys: {keys}"
    raise ValueError(msg)


def _make_observer(primary_header):
    base_time = _header_time(primary_header, "DATE_OBS", "STARTOBS")
    location = get_body_heliographic_stonyhurst("Earth", base_time.isot)
    observer = Helioprojective(
        primary_header["XCEN"] * u.arcsec,
        primary_header["YCEN"] * u.arcsec,
        observer=location,
        obstime=base_time,
    )
    with SphericalScreen(observer.observer):
        return observer.transform_to(HeliographicStonyhurst(obstime=base_time))


OBSERVATION_COMPATIBILITY_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "OBSID",
    "STARTOBS",
    "OBS_DESC",
    "NWIN",
)

WINDOW_COMPATIBILITY_KEYS = (
    "NAXIS1",
    "NAXIS2",
    "CTYPE1",
    "CUNIT1",
    "CDELT1",
    "CRPIX1",
    "CUNIT2",
    "CUNIT3",
    "CDELT2",
    "CDELT3",
    "CRPIX2",
    "CRPIX3",
)


def _validate_observation_compatible(reference_header, header, filename):
    """
    Check that observation-level keys match the reference file.
    """
    for key in OBSERVATION_COMPATIBILITY_KEYS:
        expected = reference_header.get(key)
        actual = header.get(key)
        if actual != expected:
            msg = (
                "Spectrograph files must belong to one compatible observation; "
                f"{filename} has {key}={actual!r}, expected {expected!r}."
            )
            raise ValueError(msg)

    reference_windows = np.array(
        [reference_header[f"TDESC{i}"] for i in range(1, reference_header["NWIN"] + 1)], dtype=str
    )
    windows = np.array([header[f"TDESC{i}"] for i in range(1, header["NWIN"] + 1)], dtype=str)
    if not np.array_equal(windows, reference_windows):
        msg = (
            "Spectrograph files must have the same spectral-window order; "
            f"{filename} has {windows.tolist()}, expected {reference_windows.tolist()}."
        )
        raise ValueError(msg)


def _validate_window_compatible(reference_window_headers, hdulist, filename, window_fits_indices, window_names):
    """
    Check that each spectral window header matches the reference.
    """
    for reference_window_header, window_index, window_name in zip(
        reference_window_headers,
        window_fits_indices,
        window_names,
        strict=True,
    ):
        window_header = hdulist[window_index].header
        for key in WINDOW_COMPATIBILITY_KEYS:
            expected = reference_window_header.get(key)
            actual = window_header.get(key)
            if actual != expected:
                msg = (
                    f"Spectral window {window_name!r} in {filename} is not compatible with the first file: "
                    f"{key}={actual!r}, expected {expected!r}."
                )
                raise ValueError(msg)


def read_spectrograph_lvl2(
    filenames,
    *,
    spectral_windows=None,
    uncertainty=False,
    memmap=False,
    revert_v34=False,
):
    """
    Reads either a SINGLE IRIS level 2 spectrograph FITs or a list of them.

    Does not handle tar files.

    Parameters
    ----------
    filenames: `list` of `str` or `str`
        Filename of filenames to be read. They must all be associated with the same
        OBS number.
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
            window_fits_indices = [int(np.where(windows_in_obs == window)[0][0]) + 1 for window in spectral_windows_req]
        data_dict = {window_name: [] for window_name in spectral_windows_req}
        reference_window_headers = [hdulist[index].header.copy() for index in window_fits_indices]
        observer = _make_observer(primary_header)

    # Per-window running means of good WCS table rows across files, used as
    # fallback only when a later file has no good rows for that window.
    running_wcs_fallbacks = {window_name: [np.zeros((2, 2)), np.zeros(2), 0] for window_name in spectral_windows_req}

    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            _validate_observation_compatible(primary_header, hdulist[0].header, filename)
            _validate_window_compatible(
                reference_window_headers,
                hdulist,
                filename,
                window_fits_indices,
                spectral_windows_req,
            )
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

            flip = v34 and not revert_v34
            if flip:
                times = times[::-1]
                fov_center = fov_center[::-1]
                obs_vrix = obs_vrix[::-1]
                ophaseix = ophaseix[::-1]
                exposure_times_fuv = exposure_times_fuv[::-1]
                exposure_times_nuv = exposure_times_nuv[::-1]
                pc = pc[::-1]
            t_ref = times[0]
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
                prepared_wcs_header = _prepare_raster_wcs_header(
                    window_header,
                    aux.data,
                    meta.spectral_band,
                    flip=flip,
                )
                if np.isclose(window_header["CDELT3"], 0):
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
                    offset_index = 34 if meta.spectral_band == "FUV" else 45
                    xcen = aux.data[:, aux.header["XCENIX"]] - aux.data[:, offset_index] * (SLIT_WIDTH.value / 2)
                    ycen = aux.data[:, aux.header["YCENIX"]]
                    crval = np.column_stack((ycen, xcen)) * u.arcsec
                    if flip:
                        crval = crval[::-1]

                running_pc_sum, running_crval_sum, running_count = running_wcs_fallbacks[window_name]
                fallback_pc = (running_pc_sum / running_count * u.pix) if running_count > 0 else None
                fallback_crval = (running_crval_sum / running_count * u.arcsec) if running_count > 0 else None
                bad_rows = _raster_wcs_bad_row_mask(pc, crval)
                good_mask = ~bad_rows
                pc_sanitized, crval = _sanitize_raster_wcs_tables(
                    pc.copy(),
                    crval,
                    fallback_pc,
                    fallback_crval,
                    bad_rows=bad_rows,
                )
                if good_mask.any():
                    running_wcs_fallbacks[window_name][0] += pc[good_mask].sum(axis=0).to_value(u.pix)
                    running_wcs_fallbacks[window_name][1] += crval[good_mask].sum(axis=0).to_value(u.arcsec)
                    running_wcs_fallbacks[window_name][2] += int(good_mask.sum())

                basic_wcs = WCS(prepared_wcs_header)
                _set_wcs_aux_obs_coord(basic_wcs, observer)

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
                else:
                    data = hdulist[window_fits_indices[i]].data
                # For multi-file reads the per-file cubes are only intermediate;
                # the combined cube built in _finalize_window_object gets the
                # full gWCS. Skipping it here avoids (n_files-1) redundant
                # constructions.
                if len(filenames) == 1:
                    cube_wcs = _create_raster_gwcs(
                        prepared_wcs_header,
                        pc_sanitized,
                        crval,
                        dt,
                        t_ref,
                        observer,
                    )
                else:
                    cube_wcs = basic_wcs
                cube = SpectrogramCube(
                    data,
                    wcs=cube_wcs,
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    mask=data_mask,
                    _basic_wcs=basic_wcs,
                    _memmap=memmap,
                    _raster_wcs_header=prepared_wcs_header,
                    _raster_pc_table=pc_sanitized,
                    _raster_crval_table=crval,
                    _raster_observer=observer,
                    _memmap_path=filename,
                    _memmap_ext=window_fits_indices[i],
                    _flip=flip,
                )
                cube.extra_coords.add("time", 0, times, physical_types="time")
                data_dict[window_name].append(cube)
    window_data_pairs = [
        (_window_name, _finalize_window_object(cubes, memmap=memmap, create_raster_gwcs=_create_raster_gwcs))
        for _window_name, cubes in data_dict.items()
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1))
