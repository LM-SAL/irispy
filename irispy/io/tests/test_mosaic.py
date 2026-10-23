import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from sunpy.coordinates import get_earth

from irispy.data.test import get_test_filepath
from irispy.io import read_files, read_mosaic
from irispy.spectrograph import MosaicCube
from irispy.utils.bursts import find_bright_spectral_events
from irispy.utils.constants import DN_UNIT
from irispy.utils.mg_features import calculate_mg_features

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
FUV_KEYWORDS = {"CRVAL3": 1402.77, "CDELT3": 0.025, "LAMREF": 1402.77}


def _write_mosaic(filename, **keywords):
    """
    Write a mosaic laid out like the archive files: int16 data scaled like SJI files,
    then the integrated spectrum, the time tags and the Level 2 slit rows.
    """
    data = np.arange(60, dtype=np.float32).reshape(5, 4, 3) + 0.25
    data[:, 0, 0] = 0  # no raster covered this position
    data[1, 2, 1] = -199  # Level 2 fill value
    data[:, 3, 2] = 0  # covered, with zero counts
    data[4, 1, 2] = 16182  # clipped at the level 2 ceiling
    primary = fits.PrimaryHDU(data, fits.Header({**HEADER, **keywords}))
    primary.scale("int16", bscale=0.25, bzero=7992)
    time_offsets = np.arange(12, dtype=np.float32).reshape(4, 3) * 100
    time_offsets[0, 0] = np.nan
    slit_rows = np.arange(12, dtype=np.int16).reshape(4, 3)
    slit_rows[0, 0] = -1
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(data.mean(axis=(1, 2)), fits.Header({"REF_PROF": 0})),
            fits.ImageHDU(time_offsets, fits.Header({"T_MASK": 0})),
            fits.ImageHDU(slit_rows, fits.Header({"Y_MASK": 0})),
        ]
    ).writeto(filename)
    return filename


@pytest.fixture
def mosaic_file(tmp_path):
    return _write_mosaic(tmp_path / "IRISMosaic_20150222_MgIIk.fits.gz")


def test_read_mosaic(mosaic_file):
    cube = read_mosaic(mosaic_file)
    assert isinstance(cube, MosaicCube)
    assert str(cube)
    # Solar X, solar Y and wavelength, as in level 2 rasters
    assert cube.data.shape == (3, 4, 5)
    assert cube.wavelength_axis == 2
    assert cube.data.dtype == np.float32
    assert cube.data[1, 1, 2] == 28.25
    # The uncovered position, the fill value and the clipped sample are masked, the covered zeros are not.
    assert cube.mask.sum() == 7
    assert cube.mask[0, 0].all()
    assert cube.mask[1, 2, 1]
    np.testing.assert_array_equal(cube.data[2, 3], 0)
    # The clipped sample keeps its value, as in level 2 cubes, and is marked saturated; the others are NaN
    saturated = np.zeros(cube.shape, dtype=bool)
    saturated[2, 1, 4] = True
    np.testing.assert_array_equal(cube.meta["saturated"], saturated)
    assert cube.data[2, 1, 4] == 16182
    assert cube.mask[2, 1, 4]
    assert np.isnan(cube.data[cube.mask & ~saturated]).all()

    # The FITS reference pixel (CRPIX, 1-based) is at the reference value (CRVAL).
    wavelength, coord = cube.wcs.pixel_to_world(2, 2, 1)
    assert_quantity_allclose(coord.Tx, -0.389693 * u.arcsec)
    assert_quantity_allclose(coord.Ty, -0.236448 * u.arcsec)
    assert_quantity_allclose(wavelength, 2796.352 * u.AA)
    assert_quantity_allclose(cube.wcs.pixel_to_world(2, 2, 2)[1].Tx, (-0.389693 + 1.99523) * u.arcsec)
    assert coord.obstime == Time("2015-02-22T20:47:42.140")
    assert_quantity_allclose(coord.observer.radius, get_earth("2015-02-22T20:47:42.140").radius)

    meta = cube.meta
    assert str(meta)
    assert_quantity_allclose(meta.rest_wavelength, 2796.352 * u.AA)
    assert meta.fov_center is None
    times = meta["time"]
    assert times.shape == (3, 4)
    assert times.mask[0, 0]
    assert times.mask.sum() == 1
    assert times[2, 1].isot == "2015-02-22T20:56:02.140"


def test_read_mosaic_archive_cut():
    # A cut of the archive file, see data/test/mosaic/overview.txt
    cube = read_mosaic(get_test_filepath("mosaic/IRISMosaic_20150222_MgIIk_test.fits"))
    # The position no raster covered (raw -31968, 0 DN) and the one Level 2 fill value (raw -32764, -199 DN)
    assert cube.mask.sum() == 20 * 16 + 1
    assert cube.mask[0].all()
    assert cube.mask[2, 19, 14]
    assert "saturated" not in cube.meta
    assert cube.meta["time"][2, 19].isot == "2015-02-22T21:17:44.280"
    # A mosaic has no time coordinate table, so its time is that of the meta
    assert cube.time[2, 19] == cube.meta["time"][2, 19]
    # Column 541 and row 5626 (0-based) of the whole file, whose header grid is within 0.01 arcsec of the WCS
    wavelength, coord = cube.wcs.pixel_to_world(7, 0, 1)
    assert_quantity_allclose(wavelength, 2796.352 * u.AA)
    assert_quantity_allclose(
        [coord.Tx, coord.Ty],
        [-0.389693 + (541 + 1 - 502) * 1.99523, -0.236448 + (5626 + 1 - 3006) * 0.332719] * u.arcsec,
        atol=0.01 * u.arcsec,
    )


def test_read_mosaic_wavelength_range(mosaic_file):
    full = read_mosaic(mosaic_file)
    cube = read_mosaic(mosaic_file, wavelength_range=[2796.35, 2796.40] * u.AA)
    assert cube.data.shape == (3, 4, 2)
    np.testing.assert_array_equal(cube.data, full.data[..., 2:4])
    np.testing.assert_array_equal(cube.mask, full.mask[..., 2:4])
    assert_quantity_allclose(cube.axis_world_coords("em.wl")[0], full.axis_world_coords("em.wl")[0][2:4])
    assert cube.meta.data_shape.tolist() == [3, 4, 2]
    # Bounds typed from the grid are kept despite the rounding of the unit conversion.
    assert read_mosaic(mosaic_file, wavelength_range=[2796.317, 2796.387] * u.AA).data.shape == (3, 4, 3)
    with pytest.raises(ValueError, match="No wavelengths between"):
        read_mosaic(mosaic_file, wavelength_range=[2700, 2701] * u.AA)
    with pytest.raises(ValueError, match="two wavelengths"):
        read_mosaic(mosaic_file, wavelength_range=2796.35 * u.AA)
    np.testing.assert_array_equal(read_mosaic(mosaic_file, wavelength_range=[2796.40, 2796.35] * u.AA).data, cube.data)
    with pytest.raises(TypeError, match="wavelength_range"):
        read_mosaic(mosaic_file, wavelength_range=[2796.35, 2796.40])


def test_read_mosaic_rejects_other_files(sns_sji_2832_file, mosaic_file):
    with pytest.raises(ValueError, match="is not an IRIS mosaic"):
        read_mosaic(sns_sji_2832_file)
    with pytest.raises(ValueError, match="full-disk mosaic"):
        read_files(mosaic_file)
    # The name pooch caches a download under.
    cached = mosaic_file.rename(mosaic_file.with_name(f"d41d8cd98f00b204e9800998ecf8427e-{mosaic_file.name}"))
    with pytest.raises(ValueError, match="full-disk mosaic"):
        read_files(cached)


def test_mosaic_to_map(mosaic_file):
    cube = read_mosaic(mosaic_file)
    core = cube.to_map(2796.36 * u.AA)
    np.testing.assert_array_equal(core.data, cube.data[..., 2].T)
    assert_quantity_allclose(core.wavelength, 2796.352 * u.AA)
    assert core.observatory == "IRIS"
    assert core.exposure_time == 1.99956 * u.s
    assert core.date == Time("2015-02-22T20:47:42.140")
    assert_quantity_allclose(core.observer_coordinate.radius, get_earth("2015-02-22T20:47:42.140").radius)
    map_coord, cube_coord = core.wcs.pixel_to_world(1, 2), cube.wcs.pixel_to_world(0, 2, 1)[1]
    assert_quantity_allclose([map_coord.Tx, map_coord.Ty], [cube_coord.Tx, cube_coord.Ty])
    with pytest.raises(ValueError, match="outside the mosaic wavelengths"):
        cube.to_map(3000 * u.AA)

    # The mean leaves out masked values; the uncovered position stays NaN.
    band = cube.to_map([2796.30, 2796.40] * u.AA)
    assert_quantity_allclose(band.wavelength, 2796.352 * u.AA)
    assert band.data[1, 1] == pytest.approx(cube.data[1, 1, 1:4].mean())
    assert band.data[2, 1] == pytest.approx(cube.data[1, 2, 2:4].mean())
    assert np.isnan(band.data[0, 0])
    assert (cube / (2 * u.s)).to_map([2796.30, 2796.40] * u.AA).unit == u.DN / u.s
    np.testing.assert_array_equal(band.mask, np.isnan(band.data))
    # A position with a saturated sample is NaN, as in average_window.
    assert np.isnan(cube.to_map([2796.38, 2796.43] * u.AA).data[1, 2])
    with pytest.raises(ValueError, match="No wavelengths between"):
        cube.to_map([2700, 2701] * u.AA)
    with pytest.raises(TypeError, match="wavelength"):
        cube.to_map(2796.36)

    # Slicing keeps the cube type, the time tags and the map coordinates.
    sub = cube[1:, 1:]
    assert isinstance(sub, MosaicCube)
    assert sub.meta["time"].shape == (2, 3)
    sub_map = sub.to_map(2796.36 * u.AA)
    sub_coord, core_coord = sub_map.wcs.pixel_to_world(0, 0), core.wcs.pixel_to_world(1, 1)
    assert_quantity_allclose([sub_coord.Tx, sub_coord.Ty], [core_coord.Tx, core_coord.Ty])


def test_mosaic_fits_wcs_and_celestial_frame(mosaic_file):
    cube = read_mosaic(mosaic_file)
    # A mosaic is built on a FITS WCS
    assert cube.fits_wcs is cube.wcs
    # Rebinning keeps the pixel scales of the mosaic, as of a raster
    rebinned = cube.rebin((3, 2, 1))
    assert rebinned.spectral_dispersion == cube.spectral_dispersion
    assert rebinned.solid_angle == cube.solid_angle
    assert cube.celestial_frame.obstime == Time("2015-02-22T20:47:42.140")
    assert_quantity_allclose(cube.celestial_frame.observer.radius, get_earth("2015-02-22T20:47:42.140").radius)
    for item in (2, np.s_[:, 1:, 1:], np.s_[:, 2, 1]):
        assert cube[item].celestial_frame == cube.celestial_frame


@pytest.mark.parametrize(
    ("keywords", "detector", "spectral_summing", "cmap"),
    [({}, "NUV", 2, "irissji2796"), (FUV_KEYWORDS, "FUV", 4, "irissji1400")],
)
def test_mosaic_detector(tmp_path, keywords, detector, spectral_summing, cmap):
    cube = read_mosaic(_write_mosaic(tmp_path / "mosaic.fits.gz", **keywords))
    assert cube.meta.detector == detector
    assert cube.meta.spectral_summing_factor == spectral_summing
    assert cube.unit == DN_UNIT[detector]
    assert cube.to_map(cube.meta.rest_wavelength).plot_settings["cmap"] == cmap


def test_mosaic_mg_features(mosaic_file):
    features = calculate_mg_features(read_mosaic(mosaic_file), velocity_range=(-5, 5) * u.km / u.s, lines=("k",))
    assert features["k3_velocity"].data.shape == (3, 4)


def test_mosaic_rejected_by_burst_finders(mosaic_file):
    with pytest.raises(TypeError, match="exposure time"):
        find_bright_spectral_events(read_mosaic(mosaic_file), 100)
