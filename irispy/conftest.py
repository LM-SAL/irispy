import os
import logging
import importlib
from pathlib import Path

import numpy as np
import pooch
import pytest
from scipy.io import readsav

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from sunpy.util import MetaDict

from irispy import SJICube
from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.utils import record_to_dict
from irispy.utils.constants import DN_UNIT

console_logger = logging.getLogger()
console_logger.setLevel("INFO")
_DEFAULT = object()
# Don't actually import pytest_remotedata because that can do things to the
# entrypoints code in pytest.
remotedata_spec = importlib.util.find_spec("pytest_remotedata")
HAVE_REMOTEDATA = remotedata_spec is not None
# Force MPL to use non-gui backends for testing.
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
    mpl.use("Agg")
except ImportError:
    HAVE_MATPLOTLIB = False


def pytest_runtest_setup(item):
    """
    Pytest hook to skip all tests that have the mark 'remotedata' if the
    pytest_remotedata plugin is not installed.
    """
    if isinstance(item, pytest.Function) and "remote_data" in item.keywords and not HAVE_REMOTEDATA:
        pytest.skip("skipping remotedata tests as pytest-remotedata is not installed")

    # Confirm that the pyplot figure stack is empty before the test
    if HAVE_MATPLOTLIB and plt.get_fignums():
        msg = f"There are stale pyplot figures prior to running {item.name}"
        raise UserWarning(msg)


def pytest_runtest_teardown(item):
    # Clear the pyplot figure stack if it is not empty after the test
    # You can see these log messages by passing "-o log_cli=true" to pytest on the command line
    if HAVE_MATPLOTLIB and plt.get_fignums():
        msg = f"Removing {len(plt.get_fignums())} pyplot figure(s) left open by {item.name}"
        console_logger.info(msg)
        plt.close("all")


@pytest.fixture
def idl_response():
    """
    Reads the IDL response file and returns it as a dictionary.

    This file was created from the IDL code calling:
    ``iris_get_response('2025-08-05T22:25:04.723')``
    on the 05/08/2025 using response version 9.
    """
    idl_response = readsav(
        get_test_filepath("iris_response_2025_08_05T22_25_04_723.sav"), python_dict=True, verbose=False
    )
    return record_to_dict(idl_response["iris_response"][0])


@pytest.fixture
def make_sji_cube():
    def _make(
        data,
        *,
        exposure_s=None,
        slit_x=None,
        slit_y=None,
        basic_wcs=_DEFAULT,
        meta=None,
    ):
        data = np.asarray(data, dtype=float)
        n_frames = data.shape[0] if data.ndim == 3 else 1
        ny, nx = data.shape[-2:]
        exposure_s_values = np.ones(n_frames) if exposure_s is None else np.asarray(exposure_s, dtype=float)
        slit_x_values = np.full(n_frames, 2.0) if slit_x is None else np.asarray(slit_x, dtype=float)
        slit_y_values = np.full(n_frames, 2.0) if slit_y is None else np.asarray(slit_y, dtype=float)

        if meta is None:
            meta = {
                "DATE_OBS": "2024-01-01T00:00:00.000",
                "TDESC1": "SJI_2796",
                "SUMSPAT": 1,
                "SUMSPTRL": 1,
            }

        basic_wcs_header = MetaDict(
            {
                "CTYPE1": "HPLN-TAN",
                "CUNIT1": "arcsec",
                "CDELT1": 1.0,
                "CRPIX1": 1.0,
                "CRVAL1": 0.0,
                "NAXIS1": nx,
                "CTYPE2": "HPLT-TAN",
                "CUNIT2": "arcsec",
                "CDELT2": 1.0,
                "CRPIX2": 1.0,
                "CRVAL2": 0.0,
                "NAXIS2": ny,
            }
        )
        if basic_wcs is _DEFAULT:
            basic_wcs_value = [basic_wcs_header.copy() for _ in range(n_frames)] if data.ndim == 3 else basic_wcs_header
        else:
            basic_wcs_value = basic_wcs

        if data.ndim == 3:
            header = {
                "CTYPE1": "HPLN-TAN",
                "CUNIT1": "arcsec",
                "CDELT1": 1.0,
                "CRPIX1": 1.0,
                "CRVAL1": 0.0,
                "NAXIS1": nx,
                "CTYPE2": "HPLT-TAN",
                "CUNIT2": "arcsec",
                "CDELT2": 1.0,
                "CRPIX2": 1.0,
                "CRVAL2": 0.0,
                "NAXIS2": ny,
                "CTYPE3": "Time    ",
                "CUNIT3": "s",
                "CDELT3": 1.0,
                "CRPIX3": 1.0,
                "CRVAL3": 0.0,
                "NAXIS3": n_frames,
            }
            cube = SJICube(
                data,
                WCS(header=header, naxis=3, preserve_units=True),
                unit=DN_UNIT["SJI"],
                meta=meta,
                mask=data < 0.5,
                _basic_wcs=basic_wcs_value,
            )
        else:
            header = {
                "CTYPE1": "HPLN-TAN",
                "CUNIT1": "arcsec",
                "CDELT1": 1.0,
                "CRPIX1": 1.0,
                "CRVAL1": 0.0,
                "NAXIS1": nx,
                "CTYPE2": "HPLT-TAN",
                "CUNIT2": "arcsec",
                "CDELT2": 1.0,
                "CRPIX2": 1.0,
                "CRVAL2": 0.0,
                "NAXIS2": ny,
            }
            cube = SJICube(
                data,
                WCS(header=header, naxis=2, preserve_units=True),
                unit=DN_UNIT["SJI"],
                meta=meta,
                mask=data < 0.5,
            )
            cube._basic_wcs = basic_wcs_value

        cube.extra_coords.add("exposure time", 0, exposure_s_values * u.s)
        cube.extra_coords.add("slit x position", 0, slit_x_values * u.arcsec)
        cube.extra_coords.add("slit y position", 0, slit_y_values * u.arcsec)
        return cube

    return _make


@pytest.fixture
def remote_raster_scanning_tar():
    return pooch.retrieve(
        "https://github.com/LM-SAL/irispy-lmsal-test-data/raw/refs/heads/main/iris_l2_20250613_123658_3620107423/iris_l2_20250613_123658_3620107423_raster.tar.gz",
        known_hash="756ca99cbdfafca2a97c3e357a9e8ab1bc897bca6991f6e0fa42ac2717d5b05a",
    )


@pytest.fixture
def sns_sg_file():
    return get_test_filepath("sns/iris_l2_20210905_001833_3620258102_raster_t000_r00000.fits")


@pytest.fixture
def sns_sji_1330_file():
    return get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1330_t000.fits")


@pytest.fixture
def sns_sji_1400_file():
    return get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1400_t000.fits")


@pytest.fixture
def sns_sji_2796_file():
    return get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2796_t000_deconvolved.fits")


@pytest.fixture
def sns_sji_2832_file():
    return get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2832_t000_deconvolved.fits")


@pytest.fixture
def sns_sjicube_1330():
    return read_sji_lvl2(get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1330_t000.fits"))


@pytest.fixture
def sns_sjicube_1400():
    return read_sji_lvl2(get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1400_t000.fits"))


@pytest.fixture
def sns_sjicube_2796():
    return read_sji_lvl2(get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2796_t000_deconvolved.fits"))


@pytest.fixture
def sns_sjicube_2832():
    return read_sji_lvl2(get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2832_t000_deconvolved.fits"))


@pytest.fixture
def sns_sji_filelist():
    return [
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1330_t000.fits"),
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_1400_t000.fits"),
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2796_t000_deconvolved.fits"),
        get_test_filepath("sns/iris_l2_20210905_001833_3620258102_SJI_2832_t000_deconvolved.fits"),
    ]


@pytest.fixture
def raster_sg_file():
    return get_test_filepath(
        "raster/iris_l2_20140329_140938_3860258481_raster/iris_l2_20140329_140938_3860258481_raster_t000_r00000.fits"
    )


@pytest.fixture
def raster_sg_files():
    from irispy.data.test import ROOTDIR  # NOQA: PLC0415

    files = (Path(ROOTDIR) / "raster/iris_l2_20140329_140938_3860258481_raster").glob(
        "iris_l2_20140329_140938_3860258481_raster_t000_*.fits"
    )
    return sorted([get_test_filepath(file) for file in files])


@pytest.fixture
def raster_sji_filelist():
    return [
        get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_1400_t000.fits"),
        get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2796_t000_deconvolved.fits"),
        get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2832_t000.fits"),
    ]


@pytest.fixture
def raster_sji_1400_file():
    return get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_1400_t000.fits")


@pytest.fixture
def raster_sji_2796_file():
    return get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2796_t000_deconvolved.fits")


@pytest.fixture
def raster_sji_2832_file():
    return get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2832_t000.fits")


@pytest.fixture
def raster_sjicube_1400():
    return read_sji_lvl2(get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_1400_t000.fits"))


@pytest.fixture
def raster_sjicube_2796():
    return read_sji_lvl2(get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2796_t000.fits"))


@pytest.fixture
def raster_sjicube_2832():
    return read_sji_lvl2(get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2832_t000.fits"))


@pytest.fixture(scope="session")
def fake_long_sns_obs(tmp_path_factory):
    header = fits.getheader(get_test_filepath("raster/iris_l2_20140329_140938_3860258481_SJI_2832_t000.fits"))
    header["STARTOBS"] = "2017-05-02T05:25:51.000"
    header["ENDOBS"] = "2017-05-02T08:25:51.000"
    header["NAXIS3"] = 100
    if header["CUNIT3"] == "seconds":
        header["CUNIT3"] = "s"
    rng = np.random.default_rng(12345)
    data = rng.random((header["NAXIS3"], header["NAXIS2"], header["NAXIS1"]))
    temp_dir = tmp_path_factory.mktemp("IRIS")
    hdu = fits.PrimaryHDU(data=data, header=header, do_not_scale_image_data=True, scale_back=True)
    fits_file = os.fspath(temp_dir.joinpath("iris_l2_20140329_140938_3860258481_SJI_2832_t000.fits"))
    hdu.writeto(fits_file)
    return [fits_file]
