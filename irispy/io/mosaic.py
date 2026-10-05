import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy.meta import MosaicMeta
from irispy.spectrograph import MosaicCube
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED, DN_UNIT

__all__ = ["read_mosaic"]


def read_mosaic(filename, *, wavelength_range=None):
    """
    Reads an IRIS full-disk mosaic of one spectral window.

    The mosaics are ``IRISMosaic_<date>_<window>.fits.gz`` files, listed on the
    `IRIS mosaic page <https://iris.lmsal.com/mosaic.html>`__ :cite:p:`irismosaics`.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`
        The mosaic file, gzipped or not.
    wavelength_range : `astropy.units.Quantity`, optional
        Two wavelengths: only the mosaic wavelengths between them (inclusive) are read.
        Defaults to `None`, which reads every wavelength.

    Returns
    -------
    `irispy.spectrograph.MosaicCube`
        The mosaic in DN, with array axes (wavelength, solar Y, solar X).
        Positions no raster covered, and the Level 2 fill values, are masked and NaN.

    Notes
    -----
    The headers name the axes ``'Solar X'``, ``'Solar Y'`` and ``'Wavelength'``; they are
    read as ``HPLN-TAN``, ``HPLT-TAN`` and ``WAVE``. The headers describe a linear grid,
    which differs from the gnomonic (TAN) projection by less than 0.01 arcsec over the disk.
    The observer is Earth at ``DATE_OBS``, the start of the mosaic.

    A mosaic takes about 18 hours to acquire and is not derotated, so it has no single
    observation time: ``meta["time"]`` holds the time of each position, from the time-tag
    extension :cite:p:`irismosaics`, masked where no raster covered it.

    Memory mapping is not possible for gzipped files. Only the selected wavelengths are kept
    in memory, but every read decompresses the file up to the time-tag extension at its end,
    which takes about 10 s for a 550 MB Mg II file. Decompressing the file once, for example
    with ``gunzip``, brings that down to about 1 s at the cost of 1.2 GB of disk.
    """
    with fits.open(filename, memmap=False) as hdulist:
        header = hdulist[0].header
        if [header.get(f"CTYPE{axis}") for axis in (1, 2, 3)] != ["Solar X", "Solar Y", "Wavelength"]:
            msg = f"{filename} is not an IRIS mosaic: its axes are not 'Solar X', 'Solar Y' and 'Wavelength'"
            raise ValueError(msg)
        date_obs = Time(header["DATE_OBS"], format="isot", scale="utc")
        wcs_header = header.copy()
        wcs_header.update(
            {
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CTYPE3": "WAVE",
                "DATE-OBS": date_obs.isot,
                "MJD-OBS": date_obs.mjd,
            }
        )
        wcs = WCS(wcs_header)
        _set_wcs_aux_obs_coord(wcs, get_body_heliographic_stonyhurst("Earth", date_obs))
        wavelengths = wcs.spectral.pixel_to_world(np.arange(header["NAXIS3"]))
        selection = slice(None)
        if wavelength_range is not None:
            wavelength_range = u.Quantity(wavelength_range, u.AA)
            inside = np.flatnonzero((wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1]))
            if not inside.size:
                msg = f"{filename} has no wavelengths between {wavelength_range[0]} and {wavelength_range[1]}"
                raise ValueError(msg)
            selection = slice(inside[0], inside[-1] + 1)
            wcs = wcs[selection]
        # A section reads only the selected wavelengths, scaled by BSCALE and BZERO.
        data = hdulist[0].section[selection]
        time_offset = hdulist[2].data
    # The time tags are NaN where no raster covered the mosaic, and the data are 0 there.
    uncovered = np.isnan(time_offset)
    mask = np.isin(data, BAD_PIXEL_VALUES_SCALED) | uncovered
    data[mask] = np.nan
    times = date_obs + TimeDelta(np.where(uncovered, 0, time_offset), format="sec")
    times[uncovered] = np.ma.masked
    meta = MosaicMeta(header, data_shape=data.shape)
    meta.add("time", times, "Time of each mosaic position, masked where no raster covered it", (1, 2))
    return MosaicCube(data, wcs, unit=DN_UNIT[meta.detector], meta=meta, mask=mask)
