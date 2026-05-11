import numpy as np
import pytest

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from irispy.io.utils import read_files
from irispy.meta import SGMeta, SJIMeta
from irispy.spectrograph import SpectrogramCube
from irispy.tests.helpers import make_test_spectrogram_cube
from irispy.utils.constants import SLIT_WIDTH


def _make_sg_header():
    header = fits.Header()
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["DATA_LEV"] = 2
    header["OBSID"] = 1
    header["OBS_DESC"] = "Test obs"
    header["NWIN"] = 1
    header["TDESC1"] = "Si IV 1403"
    header["TWAVE1"] = 1402.77
    header["TWMIN1"] = 1398.61
    header["TWMAX1"] = 1406.03
    header["TDET1"] = "FUV2"
    return header


def test_sgmeta_rest_wavelength():
    meta = SGMeta(_make_sg_header(), "Si IV 1403")
    assert meta.rest_wavelength.unit == u.nm
    assert u.isclose(meta.rest_wavelength, 140.277 * u.nm, rtol=1e-4)


def test_sgmeta_rest_wavelength_no_twave():
    header = _make_sg_header()
    del header["TWAVE1"]
    meta = SGMeta(header, "Si IV 1403")
    with pytest.raises((KeyError, TypeError)):
        _ = meta.rest_wavelength


def test_sgmeta_detector_band_fuv():
    meta = SGMeta(_make_sg_header(), "Si IV 1403")
    assert meta.detector_band == "FUV"


def test_sgmeta_detector_band_nuv():
    header = _make_sg_header()
    header["TDET1"] = "NUV"
    meta = SGMeta(header, "Si IV 1403")
    assert meta.detector_band == "NUV"


def test_sgmeta_detector_band_sji():
    header = fits.Header()
    header["INSTRUME"] = "IRIS"
    header["TELESCOP"] = "IRIS"
    header["TDET1"] = "SJI"
    header["TDESC1"] = "SJI_1330"
    header["TWAVE1"] = 1330.0
    header["TWMIN1"] = 1330.0
    header["TWMAX1"] = 1330.0
    meta = SJIMeta(header)
    assert meta.detector_band == "SJI"


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


@pytest.mark.parametrize(
    ("date_start", "date_end", "n_frames", "expect_none"),
    [
        (None, "2013-07-24T00:00:09.000", 10, True),
        ("2013-07-24T00:00:00.000", None, 10, True),
        ("2013-07-24T00:00:00.000", "2013-07-24T00:00:09.000", 1, True),
        ("2013-07-24T00:00:00.000", "2013-07-24T00:00:09.000", 10, False),
    ],
)
def test_sgmeta_temporal_cadence(date_start, date_end, n_frames, expect_none):
    header = _make_sg_header()
    if date_start is not None:
        header["DATE_OBS"] = date_start
    if date_end is not None:
        header["DATE_END"] = date_end
    meta = SGMeta(header, "Si IV 1403", data_shape=(n_frames, 2, 2))
    cadence = meta.temporal_cadence
    if expect_none:
        assert cadence is None
    else:
        assert cadence is not None
        start = Time(date_start)
        end = Time(date_end)
        expected = (end - start) / (n_frames - 1)
        assert u.allclose(cadence.to(u.s), expected.to(u.s), rtol=0, atol=1e-6 * u.s)


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


def test_sgmeta_real_sns_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"][0]
    meta = cube.meta

    assert meta.detector_band == "FUV"
    assert meta.rest_wavelength.unit == u.nm
    assert u.isclose(meta.rest_wavelength, 140.277 * u.nm, rtol=1e-3)
    assert meta.temporal_cadence is not None
    assert meta.temporal_cadence.unit.is_equivalent(u.s)


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
