import pytest

import astropy.units as u
from astropy.io import fits

from irispy.io.utils import read_files
from irispy.meta import SGMeta, SJIMeta


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
    assert meta.rest_wavelength is None


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


def test_sgmeta_temporal_cadence():
    header = _make_sg_header()
    header["CADEX_AV"] = 9.264
    meta = SGMeta(header, "Si IV 1403", data_shape=(10, 2, 2))
    cadence = meta.temporal_cadence
    assert cadence is not None
    assert u.isclose(cadence, 9.264 * u.s, rtol=1e-4)


def test_sgmeta_temporal_cadence_missing():
    meta = SGMeta(_make_sg_header(), "Si IV 1403")
    assert meta.temporal_cadence is None


def test_sgmeta_real_sns_data(sns_sg_file):
    raster = read_files(sns_sg_file)
    cube = raster["Si IV 1403"][0]
    meta = cube.meta

    assert meta.detector_band == "FUV"
    assert meta.rest_wavelength.unit == u.nm
    assert u.isclose(meta.rest_wavelength, 140.277 * u.nm, rtol=1e-3)
    assert meta.temporal_cadence is not None
    assert meta.temporal_cadence.unit.is_equivalent(u.s)
