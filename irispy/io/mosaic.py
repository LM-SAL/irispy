import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy.meta import MosaicMeta
from irispy.spectrograph import MosaicCube, _wavelength_indices
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED, DN_UNIT, SATURATION_LIMIT

__all__ = ["read_mosaic"]


def read_mosaic(filename, *, wavelength_range=None):
    """
    Reads an IRIS full-disk mosaic of one spectral window.

    The mosaics are the ``IRISMosaic_<date>_<window>.fits.gz`` files listed on the
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
        The mosaic in DN. Positions no raster covered, and the Level 2 fill values, are
        masked and NaN.

    Notes
    -----
    The WCS is helioprojective, with the observer at Earth at ``DATE_OBS``, the start of
    the mosaic. The mosaic is not derotated and takes about 18 hours to acquire, so
    ``meta["time"]`` holds the time of each position, masked where no raster covered it.

    Gzipped files cannot be memory-mapped, so every read decompresses the file.
    Decompressing it once, for example with ``gunzip``, makes later reads much faster.
    """
    with fits.open(filename, memmap=False) as hdulist:
        header = hdulist[0].header
        axes = [header.get(f"CTYPE{axis}") for axis in (1, 2, 3)]
        if axes != ["Solar X", "Solar Y", "Wavelength"] or "LAMREF" not in header:
            msg = f"{filename} is not an IRIS mosaic: it needs 'Solar X', 'Solar Y' and 'Wavelength' axes and LAMREF"
            raise ValueError(msg)
        date_obs = Time(header["DATE_OBS"], format="isot", scale="utc")
        wcs_header = header.copy()
        # The headers describe a linear grid, within 0.01 arcsec of TAN over the disk.
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
        selection = slice(None)
        if wavelength_range is not None:
            wavelengths = wcs.spectral.pixel_to_world(np.arange(header["NAXIS3"]))
            indices = _wavelength_indices(wavelengths, wavelength_range)
            selection = slice(indices[0], indices[-1] + 1)
            wcs = wcs[selection]
        # A section reads only the selected wavelengths, scaled by BSCALE and BZERO.
        data = hdulist[0].section[selection]
        time_offset = hdulist[2].data
    # The time tags are NaN where no raster covered the mosaic, and the data are 0 there.
    uncovered = np.isnan(time_offset)
    mask = np.isin(data, BAD_PIXEL_VALUES_SCALED) | uncovered
    data[mask] = np.nan
    # Scaled like level 2, which clips at the ceiling where iris_prep's +Inf saturation flags were lost
    data[data >= SATURATION_LIMIT.value] = np.inf
    times = date_obs + np.nan_to_num(time_offset) * u.s
    times[uncovered] = np.ma.masked
    meta = MosaicMeta(header, data_shape=data.shape)
    meta.add("time", times, "Time of each mosaic position, masked where no raster covered it", (1, 2))
    return MosaicCube(data, wcs, unit=DN_UNIT[meta.detector], meta=meta, mask=mask)
