import numpy as np

import astropy.modeling.models as m
import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import dkist
from dkist.wcs.models import CoupledCompoundModel, VaryingCelestialTransform
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import Helioprojective
from sunpy.map.header_helper import make_fitswcs_header

from irispy.meta import SJIMeta
from irispy.sji import AIACube, SJICube
from irispy.utils import calculate_uncertainty
from irispy.utils.constants import BAD_PIXEL_VALUE_SCALED, BAD_PIXEL_VALUE_UNSCALED, DN_UNIT, READOUT_NOISE

__all__ = ["read_sji_lvl2"]


def _create_gwcs(hdulist: fits.HDUList) -> gwcs.WCS:
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
    pc_table = hdulist[1].data[:, hdulist[1].header["PC1_1IX"] : hdulist[1].header["PC2_2IX"] + 1].reshape(-1, 2, 2)
    crval_table = hdulist[1].data[:, hdulist[1].header["XCENIX"] : hdulist[1].header["YCENIX"] + 1]
    crpix_table = [hdulist[0].header["CRPIX1"], hdulist[0].header["CRPIX2"]]
    cdelt = [hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]]
    older_dkist = dkist.__version__ < "1.12.0"
    kwargs = {"crpix": crpix_table * u.pixel} if older_dkist else {"crpix_table": crpix_table * u.pixel}
    celestial = VaryingCelestialTransform(
        cdelt=cdelt * u.arcsec / u.pixel,
        pc_table=pc_table * u.pixel,
        crval_table=crval_table * u.arcsec,
        **kwargs,
    )
    base_time = Time(hdulist[0].header["STARTOBS"], format="isot", scale="utc")
    times = hdulist[1].data[:, hdulist[1].header["TIME"]] * u.s
    # We need to account for a non-zero time delta.
    base_time += times[0]
    times -= times[0]
    temporal = m.Tabular1D(
        np.arange(hdulist[1].data.shape[0]) * u.pix,
        lookup_table=times,
        fill_value=np.nan,
        bounds_error=False,
        method="linear",
    )
    forward_transform = CoupledCompoundModel("&", left=celestial, right=temporal)
    celestial_frame = cf.CelestialFrame(
        axes_order=(0, 1),
        unit=(u.arcsec, u.arcsec),
        reference_frame=Helioprojective(observer="earth", obstime=base_time),
        axis_physical_types=[
            "custom:pos.helioprojective.lon",
            "custom:pos.helioprojective.lat",
        ],
        axes_names=("Longitude", "Latitude"),
    )
    temporal_frame = cf.TemporalFrame(Time(base_time), unit=(u.s,), axes_order=(2,), axes_names=("Time (UTC)",))
    output_frame = cf.CompositeFrame([celestial_frame, temporal_frame])
    input_frame = cf.CoordinateFrame(
        axes_order=(0, 1, 2),
        naxes=3,
        axes_type=["PIXEL", "PIXEL", "PIXEL"],
        unit=(u.pix, u.pix, u.pix),
    )
    return gwcs.WCS(forward_transform, input_frame=input_frame, output_frame=output_frame)


def _create_wcs(hdulist):
    """
    This is required as occasionally we need a normal WCS instead of a gWCS due
    to compatibility issues.

    This has been set to have an Earth Observer at the time of the
    observation.
    """
    wcses = []
    base_time = Time(hdulist[0].header["STARTOBS"], format="isot", scale="utc")
    times = hdulist[1].data[:, hdulist[1].header["TIME"]] * u.s
    # We need to account for a non-zero time delta.
    base_time += times[0]
    times -= times[0]
    xcenix_idx = hdulist[1].header["XCENIX"]
    ycenix_idx = hdulist[1].header["YCENIX"]
    pc1_1ix = hdulist[1].header["PC1_1IX"]
    pc1_2ix = hdulist[1].header["PC1_2IX"]
    pc2_1ix = hdulist[1].header["PC2_1IX"]
    pc2_2ix = hdulist[1].header["PC2_2IX"]
    # Handle if there are 0's in the aux array for XCENIX, YCENIX and the PC_IJ arrays
    xcenix_values = hdulist[1].data[:, xcenix_idx]
    ycenix_values = hdulist[1].data[:, ycenix_idx]
    pc1_1ix_values = hdulist[1].data[:, pc1_1ix]
    pc1_2ix_values = hdulist[1].data[:, pc1_2ix]
    pc2_1ix_values = hdulist[1].data[:, pc2_1ix]
    pc2_2ix_values = hdulist[1].data[:, pc2_2ix]
    for array in [
        xcenix_values,
        ycenix_values,
        pc1_1ix_values,
        pc1_2ix_values,
        pc2_1ix_values,
        pc2_2ix_values,
    ]:
        zero_idx = np.where(array == 0)[0]
        if zero_idx.size > 0:
            nonzero_idx = np.where(array != 0)[0]
            nonzero_vals = array[nonzero_idx]
            array[zero_idx] = np.interp(zero_idx, nonzero_idx, nonzero_vals)
    for i in range(hdulist[0].header["NAXIS3"]):
        location = get_body_heliographic_stonyhurst("Earth", (base_time + times[i]).isot)
        observer = Helioprojective(
            xcenix_values[i] * u.arcsec,
            ycenix_values[i] * u.arcsec,
            observer=location,
            obstime=base_time + times[i],
        )
        rotation_matrix = np.asanyarray(
            [
                [pc1_1ix_values[i], pc1_2ix_values[i]],
                [pc2_1ix_values[i], pc2_2ix_values[i]],
            ]
        )
        new_header = make_fitswcs_header(
            data=hdulist[0].data[i].shape,
            coordinate=observer,
            scale=[hdulist[0].header["CDELT1"], hdulist[0].header["CDELT2"]] * u.arcsec / u.pixel,
            rotation_matrix=rotation_matrix,
            instrument="SJI",
            telescope="IRIS",
            observatory="IRIS",
            wavelength=int(hdulist[0].header["TWAVE1"]) * u.AA,
            exposure=hdulist[1].data[i, hdulist[1].header["EXPTIMES"]] * u.second,
            unit=u.DN,
        )
        wcses.append(WCS(new_header))
    return wcses


def read_sji_lvl2(filename, *, uncertainty=False, memmap=False):
    """
    Reads a SINGLE level 2 SJI FITS or the IRIS aligned AIA Cube.

    Does not handle multiple files nor tar files.

    Parameters
    ----------
    filename: `str`
        Filename to read.
    uncertainty : `bool`, optional
        If `True` (not the default), will compute the uncertainty for the data (slower and
        uses more memory). If ``memmap=True``, the uncertainty is never computed.
    memmap : `bool`, optional
        If `True` (not the default), will not load arrays into memory, and will only read from
        the file into memory when needed. This option is faster and uses a
        lot less memory. However, because FITS scaling is not done on-the-fly,
        the data units will be unscaled, not the usual data numbers (DN).

    Returns
    -------
    `irispy.sji.SJICube`
        The data cube, using a gWCS.
    """
    with fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap) as hdulist:
        hdulist.verify("silentfix")
        instrume = hdulist[0].header["INSTRUME"]
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
                hdulist[1].data[:, hdulist[1].header["OPHASEIX"]] * u.arcsec,
            ),
            ("pztx", 0, hdulist[1].data[:, hdulist[1].header["PZTX"]] * u.arcsec),
            ("pzty", 0, hdulist[1].data[:, hdulist[1].header["PZTY"]] * u.arcsec),
            (
                "slit x position",
                0,
                hdulist[1].data[:, hdulist[1].header["SLTPX1IX"]] * u.arcsec,
            ),
            (
                "slit y position",
                0,
                hdulist[1].data[:, hdulist[1].header["SLTPX2IX"]] * u.arcsec,
            ),
            ("xcenix", 0, hdulist[1].data[:, hdulist[1].header["XCENIX"]] * u.arcsec),
            ("ycenix", 0, hdulist[1].data[:, hdulist[1].header["YCENIX"]] * u.arcsec),
        ]
        data = hdulist[0].data
        data_nan_masked = hdulist[0].data
        out_uncertainty = None
        if memmap:
            data_nan_masked[data == BAD_PIXEL_VALUE_UNSCALED] = 0
            mask = None
            scaled = False
            unit = DN_UNIT["SJI_UNSCALED"]
        else:
            # This is a workaround for the AIA cubes being in int and not float
            mask_value = -200 if np.issubdtype(data.dtype, np.integer) else np.nan
            data_nan_masked[data == BAD_PIXEL_VALUE_SCALED] = mask_value
            mask = data_nan_masked == BAD_PIXEL_VALUE_SCALED
            scaled = True
            unit = DN_UNIT["SJI"]
            if uncertainty and instrume in ["IRIS", "SJI"]:
                out_uncertainty = calculate_uncertainty(data, READOUT_NOISE["SJI"], DN_UNIT["SJI"])
        cube_class = SJICube if instrume in ["IRIS", "SJI"] else AIACube
        map_cube = cube_class(
            data_nan_masked,
            _create_gwcs(hdulist),
            uncertainty=out_uncertainty,
            unit=unit,
            meta=SJIMeta(hdulist[0].header),
            mask=mask,
            scaled=scaled,
            _basic_wcs=_create_wcs(hdulist),
        )
        [map_cube.extra_coords.add(*extra_coord) for extra_coord in extra_coords]
    return map_cube
