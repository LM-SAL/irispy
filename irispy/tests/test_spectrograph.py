import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from irispy.io.spectrograph import read_spectrograph_lvl2
from irispy.io.utils import read_files
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import SLIT_WIDTH


def test_fits_data_comparison(sns_sg_file):
    """
    Make sure the data is the same in pyfits and irispy.
    """
    iris_l2_test_raster = read_spectrograph_lvl2(sns_sg_file)
    with fits.open(sns_sg_file) as hdulist:
        spectral_window1 = hdulist[0].header["TDESC1"]
        spectral_window2 = hdulist[0].header["TDESC2"]
        spectral_window3 = hdulist[0].header["TDESC3"]
        data1 = copy.deepcopy(hdulist[1].data)
        data2 = copy.deepcopy(hdulist[2].data)
        data3 = copy.deepcopy(hdulist[3].data)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window1].data[0].data, data1)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window2].data[0].data, data2)
        np.testing.assert_array_almost_equal(iris_l2_test_raster[spectral_window3].data[0].data, data3)


def test_spectrogram_cube_slice_preserves_coordinates(sns_sg_file):
    raster = read_spectrograph_lvl2(sns_sg_file)
    cube = raster["C II 1336"][0]

    sliced_cube = cube[0]
    sliced_sequence = cube[:1]

    assert "time" in tuple(sliced_cube.global_coords)
    assert "time" in tuple(sliced_sequence.extra_coords.keys())


def test_spectrogram_cube_remove_cosmic_rays(sns_sg_file, monkeypatch):
    captured = {}

    def fake_remove_cosmic_rays(cube, *, method, sigma, max_iters, method_kwargs):
        captured["method"] = method
        captured["mask"] = cube.mask.copy()
        captured["sigma"] = sigma
        captured["max_iters"] = max_iters
        captured["method_kwargs"] = method_kwargs
        return cube.to_nddata(
            data=cube.data + 2,
            mask="copy",
            nddata_type=type(cube),
            extra_coords="copy",
            global_coords="copy",
        )

    monkeypatch.setattr("irispy.spectrograph.remove_cosmic_rays", fake_remove_cosmic_rays)

    raster = read_spectrograph_lvl2(sns_sg_file)
    key = next(iter(raster.keys()))
    cube = raster[key][0]
    cleaned_cube = cube.remove_cosmic_rays(
        method="astroscrappy",
        sigma=5.0,
        max_iters=5,
        method_kwargs={"batch_size": 16},
    )

    np.testing.assert_array_equal(cleaned_cube.data, cube.data + 2)
    np.testing.assert_array_equal(cleaned_cube.mask, cube.mask)
    assert list(cleaned_cube.extra_coords.keys()) == list(cube.extra_coords.keys())
    assert captured["method"] == "astroscrappy"
    assert captured["sigma"] == 5.0
    assert captured["max_iters"] == 5
    assert captured["method_kwargs"]["batch_size"] == 16
    np.testing.assert_array_equal(captured["mask"], cube.mask)
    # WCS is deep-copied, so check equivalence not identity
    assert dict(cleaned_cube.wcs.to_header()) == dict(cube.wcs.to_header())
    assert list(cleaned_cube.global_coords) == list(cube.global_coords)
    assert cleaned_cube.unit == cube.unit
    # meta is an NDMeta subclass that may contain arrays; compare keys only
    assert set(cleaned_cube.meta.keys()) == set(cube.meta.keys())


def test_spectral_dispersion():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    dispersion = cube.spectral_dispersion
    assert dispersion.unit.is_equivalent(u.nm)
    assert u.isclose(dispersion, 0.02 * u.nm, rtol=0.01)


def test_solid_angle():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    angle = cube.solid_angle
    assert angle.unit.is_equivalent(u.sr)
    expected = 1.0 * u.arcsec * SLIT_WIDTH
    assert u.isclose(angle.to(u.sr), expected.to(u.sr), rtol=0.01)


def test_wavelength_axis():
    wavelengths = np.arange(10) * 0.02 * u.nm + 140 * u.nm
    cube = make_test_spectrogram_cube(np.ones((1, 1, 10)), wavelengths)
    assert cube.wavelength_axis == 2


def test_wavelength_axis_raises_without_wave():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLT-TAN"
    header["CTYPE2"] = "HPLN-TAN"
    header["CDELT1"] = 0.1
    header["CRVAL1"] = 0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "arcsec"
    header["CDELT2"] = 0.1
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "arcsec"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="wavelength axis"):
        _ = cube.wavelength_axis


def test_spectral_dispersion_missing_wave_raises():
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLT-TAN"
    header["CTYPE2"] = "HPLN-TAN"
    header["CDELT1"] = 0.1
    header["CRVAL1"] = 0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "arcsec"
    header["CDELT2"] = 0.1
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "arcsec"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="no WAVE ctype"):
        _ = cube.spectral_dispersion


def test_solid_angle_missing_hplt_raises():
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 5
    header["NAXIS2"] = 2
    header["NAXIS3"] = 1
    header["CTYPE1"] = "WAVE"
    header["CTYPE2"] = "TIME"
    header["CTYPE3"] = "UTC"
    header["CDELT1"] = 0.02
    header["CRVAL1"] = 140.0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "nm"
    header["CDELT2"] = 1.0
    header["CRVAL2"] = 0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "s"
    header["CDELT3"] = 1.0
    header["CRVAL3"] = 0
    header["CRPIX3"] = 1
    header["CUNIT3"] = "s"
    wcs = WCS(header)
    cube = SpectrogramCube(np.ones((1, 2, 5)), wcs=wcs, uncertainty=None, unit=u.DN, meta={}, mask=None)
    with pytest.raises(ValueError, match="no HPLT ctype"):
        _ = cube.solid_angle


def test_spectral_dispersion_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"][0]
    dispersion = cube.spectral_dispersion
    assert dispersion.unit.is_equivalent(u.nm)
    assert dispersion.value > 0


def test_solid_angle_real_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"][0]
    angle = cube.solid_angle
    assert angle.unit.is_equivalent(u.sr)
    assert angle.value > 0


def test_spectrogram_cube_plot_defaults(sns_sg_file):
    cube = read_files(sns_sg_file)["Si IV 1403"][0][0]
    ax = cube.plot()
    # FUV detectors have no irissji colormap, so the default falls back to viridis.
    assert ax.images[0].get_cmap().name == "viridis"
    plt.close(ax.figure)


@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high xlims")
def test_spectrogram_sequence_plot_uses_iris_slider_labels(sns_sg_file):
    sequence = read_files(sns_sg_file)["Si IV 1403"]
    fig = plt.figure()
    animator = sequence.plot(fig=fig)
    assert animator.slider_labels == ["Raster step", "Scan number"]
    plt.close(fig)
