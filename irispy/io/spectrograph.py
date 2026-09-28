import warnings
from copy import copy
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy._spectrograph_wcs import (
    _create_raster_gwcs,
    _sanitize_raster_times,
    _sanitize_raster_wcs_tables,
    _validate_raster_wcs_inputs,
)
from irispy.io._raster_combine import _combine_raster_cubes, _lazy_window_data
from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE

__all__ = ["read_spectrograph_lvl2"]


def _nuv_t_obs_from_source_filenames(source_data, exposure_times, auxiliary_times, *, filename):
    """
    Return NUV exposure midpoints (T_OBS).

    The midpoints are parsed from the level 1 source filenames, which encode the
    exposure midpoint. Rows without a positive exposure time have no usable source
    filename, so they fall back to their planned ``auxiliary_times``.
    """
    if source_data is None or len(source_data) != len(exposure_times):
        found = 0 if source_data is None else len(source_data)
        msg = f"Expected {len(exposure_times)} NUV source filename rows, found {found}"
        raise ValueError(msg)
    valid_rows = exposure_times > 0 * u.s
    t_obs = auxiliary_times.copy()
    if np.any(valid_rows):
        try:
            timestamps = [
                Path(source_filename).name[4:21] for source_filename in source_data["NUVfilename"][valid_rows]
            ]
            t_obs[valid_rows] = Time.strptime(timestamps, "%Y%m%d_%H%M%S%f")
        except (KeyError, TypeError, ValueError) as error:
            msg = "Invalid timestamp in NUV source filenames"
            raise ValueError(msg) from error
    missing_rows = np.flatnonzero(exposure_times <= 0 * u.s)
    if missing_rows.size:
        logger.warning(
            f"EXPTIMEN is 0 s at row(s) {missing_rows.tolist()} in {filename}; frames retained with auxiliary times."
        )
    return t_obs


def _create_tabular_wcs(header, auxiliary_hdu, *, date_obs, flip=False):
    """
    Create a FITS-TAB WCS from the per-step pointing in the auxiliary table.
    """
    header = copy(header)
    auxiliary_data = auxiliary_hdu.data[::-1] if flip else auxiliary_hdu.data
    # FITS-TAB does not convert table units, and celestial WCS values must be degrees.
    arcsec_to_deg = u.arcsec.to(u.deg)
    spatial_pixels = np.array([1, header["NAXIS2"]], dtype=float)
    spatial_offsets = spatial_pixels - header["CRPIX2"]
    longitude_scale = header["CDELT3"] or header["CDELT2"]

    longitude = auxiliary_data[:, auxiliary_hdu.header["XCENIX"], None] + longitude_scale * (
        auxiliary_data[:, auxiliary_hdu.header["PC3_2IX"], None] * spatial_offsets
    )
    latitude = auxiliary_data[:, auxiliary_hdu.header["YCENIX"], None] + header["CDELT2"] * (
        auxiliary_data[:, auxiliary_hdu.header["PC2_2IX"], None] * spatial_offsets
    )
    coordinates = np.stack((latitude, longitude), axis=-1) * arcsec_to_deg

    spatial_index = (header["CRVAL2"] + header["CDELT2"] * (spatial_pixels - header["CRPIX2"])) * arcsec_to_deg
    table_data = np.array(
        [(coordinates, spatial_index)],
        dtype=[
            ("COORDS", float, coordinates.shape),
            ("SPATIAL", float, spatial_index.shape),
        ],
    )
    table = fits.BinTableHDU(table_data, name="WCS-TABLE")
    table.header["TUNIT1"] = "deg"
    table.header["TUNIT2"] = "deg"

    header["CTYPE2"] = "HPLT-TAB"
    header["CTYPE3"] = "HPLN-TAB"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "deg"
    header["DATE-OBS"] = date_obs
    header["MJD-OBS"] = Time(date_obs).mjd
    header["CRVAL2"] *= arcsec_to_deg
    header["CDELT2"] *= arcsec_to_deg
    header["CRPIX3"] = 1
    header["CRVAL3"] = 1
    header["CDELT3"] = 1
    for row in range(1, 4):
        for column in range(1, 4):
            header[f"PC{row}_{column}"] = float(row == column)
    for axis in (2, 3):
        header[f"PS{axis}_0"] = table.name
        header[f"PS{axis}_1"] = "COORDS"
        header[f"PV{axis}_3"] = axis - 1
    # The step axis has no index vector: FITS-TAB then indexes COORDS directly by the
    # 1-based step. wcslib searches an index vector linearly, so an explicit 1..N index
    # made every lookup cost O(step).
    header["PS2_2"] = "SPATIAL"

    return WCS(header, fits.HDUList([fits.PrimaryHDU(), table]))


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
        Will undo the flipping of the raster step axis made to V34 observations
        (data, mask, uncertainty, WCS, times and per-step metadata).
        Defaults to `False`.

    Returns
    -------
    `irispy.spectrograph.RasterCollection`
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
        windows_in_obs = np.array(
            [hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)],
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
            # Indices must follow the order of the requested windows, not the file order,
            # since they are zipped with ``spectral_windows_req`` below.
            window_fits_indices = [
                int(np.nonzero(windows_in_obs == window)[0][0]) + 1 for window in spectral_windows_req
            ]
        data_dict = {window_name: [] for window_name in spectral_windows_req}

    flip = v34 and not revert_v34
    # V34 rasters are flipped along the step axis: everything indexed by step must follow.
    steps = slice(None, None, -1) if flip else slice(None)
    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            aux = hdulist[-2]
            aux_data = aux.data[steps]
            try:
                aux_times = _sanitize_raster_times(
                    Time(hdulist[0].header["STARTOBS"]), aux_data[:, aux.header["TIME"]] * u.s
                )
            except Exception as error:
                error.add_note(f"While reading the auxiliary times in {filename}.")
                raise
            t_ref = aux_times[steps][0]
            source_data = hdulist[-1].data
            fov_center = SkyCoord(
                Tx=aux_data[:, aux.header["XCENIX"]],
                Ty=aux_data[:, aux.header["YCENIX"]],
                unit=u.arcsec,
                frame=Helioprojective,
            )
            obs_vrix = aux_data[:, aux.header["OBS_VRIX"]] * u.m / u.s
            ophaseix = aux_data[:, aux.header["OPHASEIX"]]
            exposure_times_fuv = aux_data[:, aux.header["EXPTIMEF"]] * u.s
            exposure_times_nuv = aux_data[:, aux.header["EXPTIMEN"]] * u.s
            pc_indices = [aux.header[key] for key in ("PC2_2IX", "PC2_3IX", "PC3_2IX", "PC3_3IX")]
            pc = aux_data[:, pc_indices].reshape(-1, 2, 2) * u.pix
            if flip:
                # Reversing the step axis reverses its PC column.
                pc[:, :, 1] *= -1
            # Every exposure, sit-and-stare included, is pointed at its own FOV centre.
            crval = np.column_stack((aux_data[:, aux.header["XCENIX"]], aux_data[:, aux.header["YCENIX"]])) * u.arcsec
            # Exposure midpoints (T_OBS); NUV ones come from the source filenames.
            t_obs_fuv = t_obs_nuv = None

            for i, window_name in enumerate(spectral_windows_req):
                ext = window_fits_indices[i]
                window_header = hdulist[ext].header
                data = hdulist[ext].data[steps]
                if memmap and defer_raster_gwcs:
                    data = _lazy_window_data(filename, ext, flip, data)
                meta = SGMeta(hdulist[0].header, window_name, data_shape=data.shape)
                meta.add("auxiliary times", aux_times, None, 0)
                try:
                    if "FUV" in meta.detector:
                        exposure_times = exposure_times_fuv
                        dn_unit = DN_UNIT["FUV"]
                        readout_noise = READOUT_NOISE["FUV"]
                        if t_obs_fuv is None:
                            t_obs_fuv = _sanitize_raster_times(
                                aux_times, exposure_times_fuv / 2, fallback_to_start=True
                            )
                        times = t_obs_fuv
                    else:
                        exposure_times = exposure_times_nuv
                        dn_unit = DN_UNIT["NUV"]
                        readout_noise = READOUT_NOISE["NUV"]
                        if t_obs_nuv is None:
                            # The source filename table is in file order, so undo (and redo) any flip.
                            t_obs_nuv = _nuv_t_obs_from_source_filenames(
                                source_data, exposure_times_nuv[steps], aux_times[steps], filename=filename
                            )[steps]
                            # NaN * 0 s is NaN, so only the rows with a non-finite EXPTIMEN are interpolated.
                            t_obs_nuv = _sanitize_raster_times(
                                t_obs_nuv, 0 * exposure_times_nuv, fallback_to_start=True
                            )
                        times = t_obs_nuv
                    dt = (times - t_ref).to_value(u.s) * u.s
                    meta.add("exposure time", exposure_times, None, 0)
                    meta.add("exposure FOV center", fov_center, None, 0)
                    meta.add("observer radial velocity", obs_vrix, None, 0)
                    meta.add("orbital phase", ophaseix, None, 0)
                    observer = meta.observer
                    sit_and_stare = window_header.get("CDELT3") == 0
                    meta["sit_and_stare"] = sit_and_stare
                    pc_sanitized, crval_sanitized = _sanitize_raster_wcs_tables(
                        pc.copy(), crval.copy(), pc_only=sit_and_stare
                    )
                    # Validate before the FITS-TAB build: wcslib can abort the process on a bad header.
                    _validate_raster_wcs_inputs(window_header, pc_sanitized, crval_sanitized, dt)
                    fits_wcs = _create_tabular_wcs(window_header, aux, date_obs=observer.obstime.utc.isot, flip=flip)
                    _set_wcs_aux_obs_coord(fits_wcs, observer)
                    if defer_raster_gwcs:
                        cube_wcs = fits_wcs
                    else:
                        cube_wcs = _create_raster_gwcs(
                            window_header,
                            pc_sanitized,
                            crval_sanitized,
                            dt,
                            t_ref,
                            observer,
                            sit_and_stare=sit_and_stare,
                        )
                except Exception as error:
                    error.add_note(f"While building the WCS of spectral window '{window_name}' in {filename}.")
                    raise

                cube = SpectrogramCube(
                    data,
                    wcs=cube_wcs,
                    uncertainty=StdDevUncertainty(calculate_uncertainty(data, readout_noise, dn_unit))
                    if compute_uncertainty
                    else None,
                    unit=dn_unit,
                    meta=meta,
                    mask=None if memmap else data == BAD_PIXEL_VALUE_SCALED,
                )
                meta.add("fits_wcs", fits_wcs)
                cube.extra_coords.add("time", 0, times, physical_types="time")
                data_dict[window_name].append((cube, window_header, pc_sanitized, crval_sanitized))
    window_data_pairs = [
        (
            window_name,
            reads[0][0] if len(reads) == 1 else _combine_raster_cubes(*zip(*reads, strict=True), memmap=memmap),
        )
        for window_name, reads in data_dict.items()
    ]
    return RasterCollection(window_data_pairs, aligned_axes=tuple(range(window_data_pairs[0][1].data.ndim - 1)))
