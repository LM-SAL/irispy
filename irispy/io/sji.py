from contextlib import nullcontext

import numpy as np

import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time

from dkist.wcs.models import CoupledCompoundModel, VaryingCelestialTransform

from irispy._interpolation import _time_lookup
from irispy.io._mask import _memmap_fill_mask
from irispy.meta import SJIMeta
from irispy.sji import AIACube, SJICube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, BAD_PIXEL_VALUES_SCALED, DN_UNIT, READOUT_NOISE

__all__ = ["read_sji_lvl2"]


def _t_obs(hdulist):
    """
    Return the SJI exposure midpoints (T_OBS): ``STARTOBS + AUX TIME + EXPTIMES / 2``.
    """
    frame_count = len(hdulist[0].data)
    auxiliary_count = len(hdulist[1].data)
    if auxiliary_count != frame_count:
        msg = f"Expected {frame_count} SJI auxiliary rows, found {auxiliary_count}"
        raise ValueError(msg)

    return Time(hdulist[0].header["STARTOBS"], format="isot", scale="utc") + (
        (hdulist[1].data[:, hdulist[1].header["TIME"]] + hdulist[1].data[:, hdulist[1].header["EXPTIMES"]] / 2) * u.s
    )


def _fill_dropped_pointing_rows(hdulist) -> None:
    """
    Fill the auxiliary pointing columns of dropped exposures, in place.

    A dropped exposure zeroes its entire pointing row (XCENIX, YCENIX and the PC
    matrix). Only rows where every pointing value is zero are treated as gaps. Gaps are
    filled with the average of the nearest valid neighbor on each side (clamped at the
    ends), so filled values never leave the neighbors' range and an unrotated PC matrix
    stays exactly identity.
    """
    columns = [
        hdulist[1].header["XCENIX"],
        hdulist[1].header["YCENIX"],
        hdulist[1].header["PC1_1IX"],
        hdulist[1].header["PC1_2IX"],
        hdulist[1].header["PC2_1IX"],
        hdulist[1].header["PC2_2IX"],
    ]
    pointing = hdulist[1].data[:, columns]
    dropped_rows = np.flatnonzero(np.all(pointing == 0, axis=1))
    valid_rows = np.flatnonzero(np.any(pointing != 0, axis=1))
    if dropped_rows.size == 0 or valid_rows.size == 0:
        return
    if not hdulist[1].data.flags["W"]:
        hdulist[1].data = hdulist[1].data.copy()
    right = np.clip(np.searchsorted(valid_rows, dropped_rows), 0, valid_rows.size - 1)
    left = np.clip(right - 1, 0, valid_rows.size - 1)
    for column in columns:
        values = hdulist[1].data[:, column]
        values[dropped_rows] = (values[valid_rows[left]] + values[valid_rows[right]]) / 2


def _create_gwcs(hdulist: fits.HDUList, t_obs) -> gwcs.WCS:
    """
    Creates the GWCS object for the SJI file.

    Parameters
    ----------
    hdulist : `astropy.io.fits.HDUList`
        The HDU list of the SJI file.

    Returns
    -------
    `gwcs.WCS`
        GWCS object for the SJI file.
    """
    from sunpy.coordinates.frames import Helioprojective  # NOQA: PLC0415

    pc_table = hdulist[1].data[:, hdulist[1].header["PC1_1IX"] : hdulist[1].header["PC2_2IX"] + 1].reshape(-1, 2, 2)
    crval_table = hdulist[1].data[:, hdulist[1].header["XCENIX"] : hdulist[1].header["YCENIX"] + 1]
    # FITS CRPIX is 1-based and the transform takes 0-based pixels
    crpix_table = [hdulist[0].header["CRPIX1"] - 1, hdulist[0].header["CRPIX2"] - 1]
    cdelt = [hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]]
    celestial = VaryingCelestialTransform(
        cdelt=cdelt * u.arcsec / u.pixel,
        pc_table=pc_table * u.pixel,
        crval_table=crval_table * u.arcsec,
        crpix_table=crpix_table * u.pixel,
    )
    start_time = t_obs[0]
    cadence = (t_obs - start_time).to_value(u.s) * u.s
    temporal = _time_lookup(cadence)
    forward_transform = CoupledCompoundModel("&", left=celestial, right=temporal)
    celestial_frame = cf.CelestialFrame(
        axes_order=(0, 1),
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer="earth", obstime=start_time),
        axis_physical_types=[
            "custom:pos.helioprojective.lon",
            "custom:pos.helioprojective.lat",
        ],
        axes_names=("Longitude", "Latitude"),
    )
    temporal_frame = cf.TemporalFrame(start_time, unit=(u.s,), axes_order=(2,), axes_names=("Time (UTC)",))
    output_frame = cf.CompositeFrame([celestial_frame, temporal_frame])
    input_frame = cf.CoordinateFrame(
        axes_order=(0, 1, 2),
        naxes=3,
        axes_type=["PIXEL", "PIXEL", "PIXEL"],
        unit=(u.pix, u.pix, u.pix),
    )
    return gwcs.WCS(forward_transform, input_frame=input_frame, output_frame=output_frame)


def _create_headers_wcs(hdulist, t_obs):
    """
    This is required as occasionally we need a normal WCS instead of a gWCS due to
    compatibility issues.

    This has been set to have an Earth Observer at the time of the observation.

    However, this only creates the WCS headers, not the full WCS objects. Those are
    created in the SJICube class property fits_wcs.
    """
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst  # NOQA: PLC0415
    from sunpy.coordinates.frames import Helioprojective  # NOQA: PLC0415
    from sunpy.map.header_helper import make_fitswcs_header  # NOQA: PLC0415

    aux = {key: hdulist[1].data[:, hdulist[1].header[key]] for key in ("XCENIX", "YCENIX", "EXPTIMES")}
    pc = [hdulist[1].data[:, hdulist[1].header[key]] for key in ("PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX")]
    # Earth is looked up at millisecond precision, as t_obs[i].isot would for each frame.
    earth = get_body_heliographic_stonyhurst("Earth", Time(t_obs.isot))
    pointing = Helioprojective(aux["XCENIX"] * u.arcsec, aux["YCENIX"] * u.arcsec, observer=earth, obstime=t_obs)
    # make_fitswcs_header takes ~2 ms, so build frame 0 once and swap in the keys that vary per frame.
    template = make_fitswcs_header(
        data=hdulist[0].data.shape[1:],
        coordinate=pointing[0],
        scale=[hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]] * u.arcsec / u.pixel,
        rotation_matrix=np.asanyarray([[pc[0][0], pc[1][0]], [pc[2][0], pc[3][0]]]),
        instrument="SJI",
        telescope="IRIS",
        observatory="IRIS",
        wavelength=int(hdulist[0].header["TWAVE1"]) * u.AA,
        exposure=aux["EXPTIMES"][0] * u.second,
        unit=u.DN,
    )
    per_frame = {
        "crval1": pointing.spherical.lon.to_value(template["cunit1"]),
        "crval2": pointing.spherical.lat.to_value(template["cunit2"]),
        "date-obs": [str(date) for date in t_obs.isot],
        # make_fitswcs_header reads these back from a FITS-WCS header, which keeps 14 significant digits.
        "dsun_obs": [float(f"{value:.14G}") for value in earth.radius.to_value(u.m)],
        "hgln_obs": [float(f"{value:.14G}") for value in earth.lon.to_value(u.deg)],
        "hglt_obs": [float(f"{value:.14G}") for value in earth.lat.to_value(u.deg)],
        "exptime": aux["EXPTIMES"],
        "pc1_1": pc[0],
        "pc1_2": pc[1],
        "pc2_1": pc[2],
        "pc2_2": pc[3],
        "rsun_obs": np.arcsin(pointing.rsun / earth.radius).to_value(u.arcsec),
    }
    wcses = []
    for i in range(hdulist[0].header["NAXIS3"]):
        header = template.copy()
        header.update({key: values[i] for key, values in per_frame.items()})
        wcses.append(header)
    # Object array, so SJICube.fits_wcs can take the frames of any slice.
    headers = np.empty(len(wcses), dtype=object)
    headers[:] = wcses
    return headers


def read_sji_lvl2(filename, *, uncertainty=False, memmap=False):
    """
    Reads a SINGLE level 2 SJI FITS or the IRIS aligned AIA Cube.

    Does not handle multiple files nor tar files.

    Parameters
    ----------
    filename : `str`, `pathlib.Path`, file-like, `bytes` or `astropy.io.fits.HDUList`
        File or decompressed FITS data to read. A supplied HDU list is left open;
        open it with ``do_not_scale_image_data=True`` when using ``memmap=True``.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        the data units will be unscaled, not the usual data numbers (DN).
        When ``memmap=True``, missing pixels retain their original values and are marked in a
        lazy Dask mask, computed only for the slices that are used.
        With ``memmap=False``, missing pixels are marked in the mask and replaced with ``NaN`` for
        floating-point data or ``-200`` for integer data.
        Compressed filenames are decompressed into memory once and cannot be memory-mapped.

    Returns
    -------
    `irispy.sji.SJICube`
        The data cube, using a gWCS.
    """
    if isinstance(filename, fits.HDUList):
        context = nullcontext(filename)
    elif isinstance(filename, bytes):
        context = fits.HDUList.fromstring(filename, do_not_scale_image_data=memmap)
    else:
        context = fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap, decompress_in_memory=True)
    with context as hdulist:
        hdulist.verify("silentfix")
        instrume = hdulist[0].header["INSTRUME"]
        t_obs = _t_obs(hdulist)
        _fill_dropped_pointing_rows(hdulist)
        extra_coords = [
            (
                "exposure time",
                0,
                hdulist[1].data[:, hdulist[1].header["EXPTIMES"]] * u.s,
            ),
            (
                "obs_vrix",
                0,
                hdulist[1].data[:, hdulist[1].header["OBS_VRIX"]] * u.m / u.s,
            ),
            (
                "ophaseix",
                0,
                hdulist[1].data[:, hdulist[1].header["OPHASEIX"]] * u.one,
            ),
            ("pztx", 0, hdulist[1].data[:, hdulist[1].header["PZTX"]] * u.arcsec),
            ("pzty", 0, hdulist[1].data[:, hdulist[1].header["PZTY"]] * u.arcsec),
            (
                "slit x position",
                0,
                hdulist[1].data[:, hdulist[1].header["SLTPX1IX"]] * u.pix,
            ),
            (
                "slit y position",
                0,
                hdulist[1].data[:, hdulist[1].header["SLTPX2IX"]] * u.pix,
            ),
            ("xcenix", 0, hdulist[1].data[:, hdulist[1].header["XCENIX"]] * u.arcsec),
            ("ycenix", 0, hdulist[1].data[:, hdulist[1].header["YCENIX"]] * u.arcsec),
        ]
        data = hdulist[0].data
        data_nan_masked = hdulist[0].data
        out_uncertainty = None
        if memmap:
            mask = _memmap_fill_mask(data, hdulist[0].header)
            scaled = False
            unit = DN_UNIT["SJI_UNSCALED"]
        else:
            # This is a workaround for the AIA cubes being in int and not float
            mask = np.isin(data, BAD_PIXEL_VALUES_SCALED)
            mask_value = BAD_PIXEL_VALUE_SCALED if np.issubdtype(data.dtype, np.integer) else np.nan
            if not data_nan_masked.flags["W"]:
                data_nan_masked = data_nan_masked.copy()
            data_nan_masked[mask] = mask_value
            scaled = True
            unit = DN_UNIT["SJI"]
            if uncertainty and instrume in ["IRIS", "SJI"]:
                out_uncertainty = StdDevUncertainty(
                    calculate_uncertainty(data_nan_masked, READOUT_NOISE["SJI"], DN_UNIT["SJI"])
                )
        cube_class = SJICube if instrume in ["IRIS", "SJI"] else AIACube
        meta = SJIMeta(hdulist[0].header)
        meta["frame_wcs_headers"] = _create_headers_wcs(hdulist, t_obs)  # root-relative, not axis-aware
        meta["scaled"] = scaled
        map_cube = cube_class(
            data_nan_masked,
            _create_gwcs(hdulist, t_obs),
            uncertainty=out_uncertainty,
            unit=unit,
            meta=meta,
            mask=mask,
        )
        [map_cube.extra_coords.add(*extra_coord) for extra_coord in extra_coords]
    return map_cube
