import os
import logging
import importlib
from pathlib import Path

import numpy as np
import pooch
import pytest
from scipy.io import readsav

from astropy.io import fits

from irispy.data.test import get_test_filepath
from irispy.io.sji import read_sji_lvl2
from irispy.utils import record_to_dict

console_logger = logging.getLogger()
console_logger.setLevel("INFO")
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
