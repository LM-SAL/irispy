import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates.wcs_utils import _set_wcs_aux_obs_coord

from irispy._gwcs import _create_mosaic_gwcs
from irispy.meta import MosaicMeta
from irispy.spectrograph import MosaicCube, _wavelength_indices
from irispy.utils.constants import BAD_PIXEL_VALUES_SCALED, DN_UNIT, SATURATION_LIMIT

__all__ = ["read_mosaic"]


@u.quantity_input
def read_mosaic(filename, *, wavelength_range: u.AA = None):
    """
    Reads an IRIS full-disk mosaic of one spectral window.

    The mosaics are the ``IRISMosaic_<date>_<window>.fits.gz`` files listed on the
    `IRIS mosaic page <https://iris.lmsal.com/mosaic.html>`__ :cite:p:`irismosaics`.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`
        The mosaic file, gzipped or not.
    wavelength_range : `astropy.units.Quantity`, optional
        Two wavelengths, in either order: only the mosaic wavelengths between them (inclusive) are read.
        Defaults to `None`, which reads every wavelength.

    Returns
    -------
    `irispy.spectrograph.MosaicCube`
        The mosaic in DN, with array axes solar X, solar Y and wavelength, as in a Level 2
        raster. Positions no raster covered, and the Level 2 fill values, are masked and NaN.
        Samples at the level 2 ceiling, `irispy.utils.constants.SATURATION_LIMIT`, keep
        their value and are marked in the mask and, if there are any, in ``meta["saturated"]``
        (see :ref:`irispy-tutorial-saturated-data`).

    Notes
    -----
    The gWCS gives the wavelength, the helioprojective longitude and latitude, with the
    observer at Earth at ``DATE_OBS``, the start of the mosaic, and the time of each
    position: the mosaic is not derotated and takes about 18 hours to acquire.
    ``meta["time"]`` holds the same times, masked where no raster covered it, and
    ``fits_wcs`` the FITS WCS the mosaic was read with.

    Gzipped files cannot be memory-mapped, so every read decompresses the file.
    Decompressing it once, for example with ``gunzip``, makes later reads much faster.

    Examples
    --------
    A cut of the Mg II k mosaic of 2015-02-22, in which no raster covered the first solar X position:

    >>> import astropy.units as u
    >>> from irispy.data.test import get_test_filepath
    >>> from irispy.io import read_mosaic
    >>> filename = get_test_filepath("mosaic/IRISMosaic_20150222_MgIIk_test.fits")
    >>> mosaic = read_mosaic(filename)
    >>> mosaic.shape, mosaic.unit, mosaic.wavelength_axis
    ((3, 20, 16), Unit("DN_IRIS_NUV"), 2)
    >>> mosaic.data[:, 0, 7]  # at the Mg II k rest wavelength, 2796.352 Å
    array([ nan, 32.5, 28. ], dtype=float32)
    >>> mosaic.meta["time"].mask[:, 0]
    array([ True, False, False])
    >>> read_mosaic(filename, wavelength_range=[2796.3, 2796.4] * u.AA).shape
    (3, 20, 3)
    """
    with fits.open(filename, memmap=False, do_not_scale_image_data=True) as hdulist:
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
        celestial = wcs.celestial  # (solar X, solar Y), before the axes are reversed
        indices = range(header["NAXIS3"])
        if wavelength_range is not None:
            wavelengths = wcs.spectral.pixel_to_world(indices)
            indices = _wavelength_indices(wavelengths, wavelength_range)
            wcs = wcs[indices[0] : indices[-1] + 1]
        # The file is (wavelength, solar Y, solar X); rasters are the reverse.
        wcs = wcs.sub([3, 2, 1])
        observer = get_body_heliographic_stonyhurst("Earth", date_obs)
        _set_wcs_aux_obs_coord(wcs, observer)
        # The stored integers in one read: astropy seeks back after a read, which restarts a gzip decompression.
        stored = hdulist[0].section[indices[0] : indices[-1] + 1]
        time_offset = hdulist[2].data.T
    # A contiguous copy, scaled in place as astropy does; calculate_moments takes over twice as long on a
    # transposed view.
    data = stored.T.astype(np.float32, order="C")
    del stored
    data *= header["BSCALE"]
    data += header["BZERO"]
    # The time tags are NaN where no raster covered the mosaic, and the data are 0 there.
    uncovered = np.isnan(time_offset)
    mask = np.isin(data, BAD_PIXEL_VALUES_SCALED) | uncovered[..., np.newaxis]
    data[mask] = np.nan
    # Scaled like level 2, which clips at the ceiling where iris_prep's +Inf saturation flags were lost
    saturated = data >= SATURATION_LIMIT.value
    mask |= saturated
    times = date_obs + np.nan_to_num(time_offset) * u.s
    times[uncovered] = np.ma.masked
    meta = MosaicMeta(header, data_shape=data.shape)
    meta.add("time", times, "Time of each mosaic position, masked where no raster covered it", (0, 1))
    if saturated.any():
        meta.add("saturated", saturated, "Samples at the level 2 ceiling, also masked", (0, 1, 2))
    meta["fits_wcs"] = wcs
    gwcs_ = _create_mosaic_gwcs(wcs, celestial, time_offset.T, date_obs, observer)
    gwcs_.array_shape = data.shape
    return MosaicCube(data, gwcs_, unit=DN_UNIT[meta.detector], meta=meta, mask=mask)
