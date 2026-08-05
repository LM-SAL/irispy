from copy import copy
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy import log as logger
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube, SpectrogramCubeSequence
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, DN_UNIT, READOUT_NOISE

__all__ = ["read_spectrograph_lvl2"]


def _nuv_exposure_start_times_from_source_filenames(source_data, exposure_times):
    """
    Return NUV exposure starts from level 1 midpoint filenames.
    """
    if source_data is None or len(source_data) != len(exposure_times):
        found = 0 if source_data is None else len(source_data)
        msg = f"Expected {len(exposure_times)} NUV source filename rows, found {found}"
        raise ValueError(msg)
    if np.any(exposure_times <= 0 * u.s):
        msg = "Invalid NUV exposure time"
        raise ValueError(msg)
    try:
        timestamps = [Path(filename).name[4:21] for filename in source_data["NUVfilename"]]
        return Time.strptime(timestamps, "%Y%m%d_%H%M%S%f") - exposure_times / 2
    except (KeyError, TypeError, ValueError) as error:
        msg = "Invalid timestamp in NUV source filenames"
        raise ValueError(msg) from error


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
    raster_index = np.arange(1, header["NAXIS3"] + 1, dtype=float)
    table_data = np.array(
        [(coordinates, spatial_index, raster_index)],
        dtype=[
            ("COORDS", float, coordinates.shape),
            ("SPATIAL", float, spatial_index.shape),
            ("RASTER", float, raster_index.shape),
        ],
    )
    table = fits.BinTableHDU(table_data, name="WCS-TABLE")
    table.header["TUNIT1"] = "deg"
    table.header["TUNIT2"] = "deg"
    table.header["TUNIT3"] = "deg"

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
    for axis, index_column in ((2, "SPATIAL"), (3, "RASTER")):
        header[f"PS{axis}_0"] = table.name
        header[f"PS{axis}_1"] = "COORDS"
        header[f"PS{axis}_2"] = index_column
        header[f"PV{axis}_3"] = axis - 1

    return WCS(header, fits.HDUList([fits.PrimaryHDU(), table]))


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
    # Collecting the window observations
    with fits.open(filenames[0], memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
        # After a discussion with the IRIS team, it was decided that instead of the
        # OBSID, we will use STEPS_AV less than -0.01 to identify V34 observations.
        v34 = hdulist[0].header["STEPS_AV"] < -0.01
        hdulist.verify("silentfix")
        windows_in_obs = np.array(
            [hdulist[0].header[f"TDESC{i}"] for i in range(1, hdulist[0].header["NWIN"] + 1)],
        )
        # If spectral_window is not set then get every window.
        # Else take the appropriate windows
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
            window_fits_indices = np.nonzero(np.isin(windows_in_obs, spectral_windows))[0] + 1
        data_dict = {window_name: [] for window_name in spectral_windows_req}
        base_time = Time(hdulist[0].header["DATE_OBS"])
        observer = get_body_heliographic_stonyhurst("Earth", base_time)
    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            # Extract axis-aligned metadata.
            aux_times = Time(hdulist[0].header["STARTOBS"]) + TimeDelta(
                hdulist[-2].data[:, hdulist[-2].header["TIME"]],
                format="sec",
            )
            source_data = hdulist[-1].data
            fov_center = SkyCoord(
                Tx=hdulist[-2].data[:, hdulist[-2].header["XCENIX"]],
                Ty=hdulist[-2].data[:, hdulist[-2].header["YCENIX"]],
                unit=u.arcsec,
                frame=Helioprojective,
            )
            obs_vrix = hdulist[-2].data[:, hdulist[-2].header["OBS_VRIX"]] * u.m / u.s
            ophaseix = hdulist[-2].data[:, hdulist[-2].header["OPHASEIX"]]
            exposure_times_fuv = hdulist[-2].data[:, hdulist[-2].header["EXPTIMEF"]] * u.s
            exposure_times_nuv = hdulist[-2].data[:, hdulist[-2].header["EXPTIMEN"]] * u.s
            for i, window_name in enumerate(spectral_windows_req):
                meta = SGMeta(
                    hdulist[0].header,
                    window_name,
                    data_shape=hdulist[window_fits_indices[i]].data.shape,
                )
                meta.add("auxiliary times", aux_times, None, 0)
                if "FUV" in meta.detector:
                    exposure_times = exposure_times_fuv
                    dn_unit = DN_UNIT["FUV"]
                    readout_noise = READOUT_NOISE["FUV"]
                    exposure_start_times = aux_times
                else:
                    exposure_times = exposure_times_nuv
                    dn_unit = DN_UNIT["NUV"]
                    readout_noise = READOUT_NOISE["NUV"]
                    exposure_start_times = _nuv_exposure_start_times_from_source_filenames(
                        source_data,
                        exposure_times,
                    )
                meta.add("exposure time", exposure_times, None, 0)
                meta.add("exposure FOV center", fov_center, None, 0)
                meta.add("observer radial velocity", obs_vrix, None, 0)
                meta.add("orbital phase", ophaseix, None, 0)
                header = hdulist[window_fits_indices[i]].header
                try:
                    wcs = _create_tabular_wcs(
                        header,
                        hdulist[-2],
                        date_obs=base_time.utc.isot,
                        flip=v34 and not revert_v34,
                    )
                except Exception as e:  # NOQA: BLE001
                    msg = (
                        f"WCS failed to load while reading one step of the raster due to {e}"
                        "The loading will continue but this will be missing in the final cube. "
                        f"Spectral window: {window_name}, step {i} in file: {filename}"
                    )
                    logger.warning(msg)
                    continue
                out_uncertainty = None
                data_mask = None
                if not memmap:
                    data_mask = hdulist[window_fits_indices[i]].data == BAD_PIXEL_VALUE_SCALED
                if uncertainty:
                    out_uncertainty = calculate_uncertainty(
                        hdulist[window_fits_indices[i]].data,
                        readout_noise,
                        dn_unit,
                    )
                if v34 and not revert_v34:
                    times = exposure_start_times[::-1]
                    data = np.flip(hdulist[window_fits_indices[i]].data, axis=0)
                else:
                    times = exposure_start_times
                    data = hdulist[window_fits_indices[i]].data
                _set_wcs_aux_obs_coord(wcs, observer)
                cube = SpectrogramCube(
                    data,
                    wcs=wcs,
                    uncertainty=out_uncertainty,
                    unit=dn_unit,
                    meta=meta,
                    mask=data_mask,
                )
                cube.extra_coords.add("time", 0, times, physical_types="time")
                data_dict[window_name].append(cube)
    window_data_pairs = [
        (window_name, SpectrogramCubeSequence(data_dict[window_name], common_axis=0, meta=hdulist[0].header))
        for window_name in spectral_windows_req
    ]
    return RasterCollection(window_data_pairs, aligned_axes=(0, 1, 2))
