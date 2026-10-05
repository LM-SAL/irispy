import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from sunpy.coordinates import get_earth

from irispy.io import read_mosaic
from irispy.spectrograph import MosaicCube
from irispy.utils.constants import DN_UNIT

# The primary header of IRISMosaic_20150222_MgIIk.fits.gz, with the axes shrunk to 5 x 4 x 3.
HEADER = {
    "DATE": "2025-08-24",
    "DATE_OBS": "2015-02-22T20:47:42.140",
    "DATE_END": "2015-02-23T14:11:37.220",
    "CDELT1": 1.99523,
    "CDELT2": 0.332719,
    "CDELT3": 0.035,
    "CRPIX1": 2,
    "CRPIX2": 3,
    "CRPIX3": 3,
    "CRVAL1": -0.389693,
    "CRVAL2": -0.236448,
    "CRVAL3": 2796.352,
    "CTYPE1": "Solar X",
    "CTYPE2": "Solar Y",
    "CTYPE3": "Wavelength",
    "CUNIT1": "arcsec",
    "CUNIT2": "arcsec",
    "CUNIT3": "Angstrom",
    "OBSID": "3801604195",
    "SUMSPAT": 2,
    "EXPTIME": 1.99956,
    "SUMSPTRN": 2,
    "SUMSPTRF": 4,
    "LAMREF": 2796.352,
}


def _write_mosaic(filename, **keywords):
    """
    Write a mosaic laid out like the archive files: int16 data scaled like SJI files,
    then the integrated spectrum, the time tags and the Level 2 slit rows.
    """
    data = np.arange(60, dtype=np.float32).reshape(5, 4, 3) + 0.25
    data[:, 0, 0] = 0  # no raster covered this position
    data[1, 2, 1] = -199  # Level 2 fill value
    data[:, 3, 2] = 0  # covered, with zero counts
    primary = fits.PrimaryHDU(data, fits.Header({**HEADER, **keywords}))
    primary.scale("int16", bscale=0.25, bzero=7992)
    t_mask = np.arange(12, dtype=np.float32).reshape(4, 3) * 100
    t_mask[0, 0] = np.nan
    y_mask = np.arange(12, dtype=np.int16).reshape(4, 3)
    y_mask[0, 0] = -1
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(data.mean(axis=(1, 2)), fits.Header({"REF_PROF": 0})),
            fits.ImageHDU(t_mask, fits.Header({"T_MASK": 0})),
            fits.ImageHDU(y_mask, fits.Header({"Y_MASK": 0})),
        ]
    ).writeto(filename)
    return filename


@pytest.fixture
def mosaic_file(tmp_path):
    return _write_mosaic(tmp_path / "IRISMosaic_20150222_MgIIk.fits.gz")


def test_read_mosaic(mosaic_file):
    with fits.open(mosaic_file, do_not_scale_image_data=True) as hdulist:
        assert hdulist[0].header["BITPIX"] == 16
        assert hdulist[0].header["BSCALE"] == 0.25
        assert hdulist[0].header["BZERO"] == 7992
    cube = read_mosaic(mosaic_file)
    assert isinstance(cube, MosaicCube)
    assert str(cube)
    assert cube.data.shape == (5, 4, 3)
    assert cube.data.dtype == np.float32
    assert cube.unit == DN_UNIT["NUV"]
    assert list(cube.wcs.wcs.ctype) == ["HPLN-TAN", "HPLT-TAN", "WAVE"]
    assert cube.data[2, 1, 1] == 28.25
    # The uncovered position and the fill value are masked, the covered zeros are not.
    assert cube.mask.sum() == 6
    assert cube.mask[:, 0, 0].all()
    assert cube.mask[1, 2, 1]
    assert np.isnan(cube.data[cube.mask]).all()
    np.testing.assert_array_equal(cube.data[:, 3, 2], 0)

    # The FITS reference pixel (CRPIX, 1-based) is at the reference value (CRVAL).
    coord, wavelength = cube.wcs.pixel_to_world(1, 2, 2)
    assert_quantity_allclose(coord.Tx, -0.389693 * u.arcsec)
    assert_quantity_allclose(coord.Ty, -0.236448 * u.arcsec)
    assert_quantity_allclose(wavelength, 2796.352 * u.AA)
    assert_quantity_allclose(cube.wcs.pixel_to_world(2, 2, 2)[0].Tx, (-0.389693 + 1.99523) * u.arcsec)
    assert coord.obstime == Time("2015-02-22T20:47:42.140")
    assert_quantity_allclose(coord.observer.radius, get_earth("2015-02-22T20:47:42.140").radius)

    meta = cube.meta
    assert str(meta)
    assert meta.detector == "NUV"
    assert meta.spatial_summing_factor == 2
    assert meta.spectral_summing_factor == 2
    assert meta.exposure_time == 1.99956 * u.s
    assert meta.reference_wavelength == 2796.352 * u.AA
    assert meta.observing_mode_id == 3801604195
    assert meta.date_start.isot == "2015-02-22T20:47:42.140"
    assert meta.date_end.isot == "2015-02-23T14:11:37.220"
    times = meta["time"]
    assert times.shape == (4, 3)
    assert times.mask[0, 0]
    assert times.mask.sum() == 1
    assert times[1, 2].isot == "2015-02-22T20:56:02.140"


def test_read_mosaic_wavelength_range(mosaic_file):
    full = read_mosaic(mosaic_file)
    cube = read_mosaic(mosaic_file, wavelength_range=[2796.35, 2796.40] * u.AA)
    assert cube.data.shape == (2, 4, 3)
    np.testing.assert_array_equal(cube.data, full.data[2:4])
    np.testing.assert_array_equal(cube.mask, full.mask[2:4])
    assert_quantity_allclose(cube.axis_world_coords("em.wl")[0], full.axis_world_coords("em.wl")[0][2:4])
    assert cube.meta.data_shape.tolist() == [2, 4, 3]
    with pytest.raises(ValueError, match="no wavelengths between"):
        read_mosaic(mosaic_file, wavelength_range=[2700, 2701] * u.AA)


def test_read_mosaic_rejects_other_files(sns_sji_2832_file):
    with pytest.raises(ValueError, match="is not an IRIS mosaic"):
        read_mosaic(sns_sji_2832_file)


def test_mosaic_to_maps(mosaic_file):
    cube = read_mosaic(mosaic_file)
    core = cube.to_maps(2796.36 * u.AA)
    np.testing.assert_array_equal(core.data, cube.data[2])
    assert_quantity_allclose(core.wavelength, 2796.352 * u.AA)
    assert core.plot_settings["cmap"] == "irissji2796"
    assert core.observatory == "IRIS"
    assert core.exposure_time == 1.99956 * u.s
    assert core.date == Time("2015-02-22T20:47:42.140")
    assert_quantity_allclose(core.observer_coordinate.radius, get_earth("2015-02-22T20:47:42.140").radius)
    map_coord, cube_coord = core.pixel_to_world(1 * u.pix, 2 * u.pix), cube.wcs.pixel_to_world(1, 2, 0)[0]
    assert_quantity_allclose([map_coord.Tx, map_coord.Ty], [cube_coord.Tx, cube_coord.Ty])

    # The mean leaves out masked values; the uncovered position stays NaN.
    wing = cube.to_maps([2796.30, 2796.40] * u.AA)
    assert_quantity_allclose(wing.wavelength, 2796.352 * u.AA)
    assert wing.data[1, 1] == pytest.approx(cube.data[1:4, 1, 1].mean())
    assert wing.data[2, 1] == pytest.approx(cube.data[2:4, 2, 1].mean())
    assert np.isnan(wing.data[0, 0])
    with pytest.raises(ValueError, match="no wavelengths between"):
        cube.to_maps([2700, 2701] * u.AA)

    # Slicing keeps the cube type, the time tags and the map coordinates.
    sub = cube[:, 1:, 1:]
    assert isinstance(sub, MosaicCube)
    assert sub.meta["time"].shape == (3, 2)
    sub_map = sub.to_maps(2796.36 * u.AA)
    sub_coord, core_coord = sub_map.pixel_to_world(0 * u.pix, 0 * u.pix), core.pixel_to_world(1 * u.pix, 1 * u.pix)
    assert_quantity_allclose([sub_coord.Tx, sub_coord.Ty], [core_coord.Tx, core_coord.Ty])


def test_read_mosaic_fuv(tmp_path):
    filename = _write_mosaic(
        tmp_path / "IRISMosaic_20150222_Si1403.fits.gz", CRVAL3=1402.77, CDELT3=0.025, LAMREF=1402.77
    )
    cube = read_mosaic(filename)
    assert cube.meta.detector == "FUV"
    assert cube.meta.spectral_summing_factor == 4
    assert cube.unit == DN_UNIT["FUV"]
    assert cube.to_maps(1402.77 * u.AA).plot_settings["cmap"] == "irissji1400"
