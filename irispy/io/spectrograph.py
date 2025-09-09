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
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.screens import SphericalScreen
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy.meta import SGMeta
from irispy.spectrograph import RasterCollection, SpectrogramCube, SpectrogramCubeSequence
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import DN_UNIT, READOUT_NOISE, SLIT_WIDTH

__all__ = ["read_spectrograph_lvl2"]


def _pc_matrix(lam, angle_1, angle_2):
    return angle_1, -1 * lam * angle_2, 1 / lam * angle_2, angle_1


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
                missing_windows = window_is_in_obs is False
                msg = f"Spectral windows {spectral_windows[missing_windows]} not in file {filenames[0]}"
                raise ValueError(msg)
            window_fits_indices = np.nonzero(np.isin(windows_in_obs, spectral_windows))[0] + 1
        data_dict = {window_name: [] for window_name in spectral_windows_req}
        # No observer information in the header, so we just assume its at Earth.
        base_time = Time(hdulist[0].header["STARTOBS"])
        location = get_body_heliographic_stonyhurst("Earth", (base_time).isot)
        observer = Helioprojective(
            hdulist[0].header["XCEN"] * u.arcsec,
            hdulist[0].header["YCEN"] * u.arcsec,
            observer=location,
            obstime=base_time,
        )
        with SphericalScreen(observer.observer):
            observer = observer.transform_to(HeliographicStonyhurst(obstime=base_time))
    for filename in filenames:
        with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
            hdulist.verify("silentfix")
            # Extract axis-aligned metadata.
            times = Time(hdulist[0].header["STARTOBS"]) + TimeDelta(
                hdulist[-2].data[:, hdulist[-2].header["TIME"]],
                format="sec",
            )
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
                # Sit-and-stare have a CDELT of 0 which causes issues in WCS.
                # In this case, set CDELT to a small number.
                header = copy(hdulist[window_fits_indices[i]].header)
                # Account for a slit offset (POFFYNUV (45) or POFFYFUV (34))
                idx = 34 if meta.spectral_band == "FUV" else 45
                header["CRVAL3"] -= hdulist[-2].data[:, idx].mean() * (SLIT_WIDTH.value / 2)
                if header["CDELT3"] == 0:
                    header["CDELT3"] = 1e-10
                    ang1, ang2, ang3, ang4 = _pc_matrix(
                        header["CDELT3"] / header["CDELT2"],
                        hdulist[-2].data[:, 20].mean(),
                        hdulist[-2].data[:, 22].mean(),
                    )
                    header["PC2_2"] = ang1
                    header["PC2_3"] = ang2
                    header["PC3_2"] = ang3
                    header["PC3_3"] = ang4
                try:
                    wcs = WCS(header)
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
                    data_mask = hdulist[window_fits_indices[i]].data == -200.0
                if uncertainty:
                    out_uncertainty = calculate_uncertainty(
                        hdulist[window_fits_indices[i]].data,
                        readout_noise,
                        dn_unit,
                    )
                if v34 and not revert_v34:
                    times = times[::-1]
                    data = np.flip(hdulist[window_fits_indices[i]].data, axis=0)
                    header["PC1_3"] = 0
                    header["PC2_3"] = -header["PC2_3"]
                    header["PC3_2"] = -header["PC3_2"]
                    header["CDELT3"] = -header["CDELT3"]
                    header["CRPIX3"] = header["NAXIS3"] - header["CRPIX3"] + 1
                    wcs = WCS(header)
                else:
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
