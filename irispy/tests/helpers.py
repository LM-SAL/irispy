import warnings
from pathlib import Path
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from irispy.meta import SGMeta
from irispy.spectrograph import SpectrogramCube

__all__ = ["make_test_spectrogram_cube", "warnings_as_errors"]


@pytest.fixture
def warnings_as_errors():
    warnings.simplefilter("error")
    yield
    warnings.resetwarnings()


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    """
    ft2_version = f"{mpl.ft2font.__freetype_version__.replace('.', '')}"
    mpl_version = (
        "dev" if (("dev" in mpl.__version__) or ("rc" in mpl.__version__)) else mpl.__version__.replace(".", "")
    )
    astropy_version = (
        "dev"
        if (("dev" in astropy.__version__) or ("rc" in astropy.__version__))
        else astropy.__version__.replace(".", "")
    )
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}_astropy_{astropy_version}.json"


def figure_test(test_function):
    """
    A decorator for a test that verifies the hash of the current figure or the returned
    figure, with the name of the test function as the hash identifier in the library. A
    PNG is also created in the 'result_image' directory, which is created on the current
    path.

    All such decorated tests are marked with `pytest.mark.mpl_image` for convenient filtering.

    Examples
    --------
    @figure_test
    def test_simple_plot():
        plt.plot([0,1])
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / hash_library_name

    @pytest.mark.mpl_image_compare(
        hash_library=hash_library_file, savefig_kwargs={"metadata": {"Software": None}}, style="default"
    )
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        ret = test_function(*args, **kwargs)
        if ret is None:
            ret = plt.gcf()
        return ret

    return test_wrapper


def make_test_spectrogram_cube(data, wavelengths):
    """
    Build a minimal 3-D SpectrogramCube for unit tests.

    N.B. FITS stores axes in reverse order relative to numpy arrays:
    NAXIS1 corresponds to the last numpy axis, NAXIS2 to the penultimate,
    and NAXIS3 to the first.  This helper assumes the wavelength axis is
    the **last** numpy axis.

    Parameters
    ----------
    data : `numpy.ndarray`
        3-D array with shape ``(ny, nx, n_wavelengths)``.
    wavelengths : `astropy.units.Quantity`
        1-D wavelength grid.

    Returns
    -------
    `irispy.spectrograph.SpectrogramCube`
    """
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = data.shape[2]
    header["NAXIS2"] = data.shape[1]
    header["NAXIS3"] = data.shape[0]
    header["CRPIX1"] = 1
    header["CRPIX2"] = 1
    header["CRPIX3"] = 1
    # CTYPE1 / NAXIS1 corresponds to the LAST numpy array axis (wavelength)
    header["CTYPE1"] = "WAVE"
    header["CUNIT1"] = str(wavelengths.unit)
    header["CDELT1"] = float(np.mean(np.diff(wavelengths.value)))
    header["CRVAL1"] = float(wavelengths.value[0])
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT2"] = "arcsec"
    header["CDELT2"] = 1.0
    header["CRVAL2"] = 0.0
    header["CTYPE3"] = "HPLN-TAN"
    header["CUNIT3"] = "arcsec"
    header["CDELT3"] = 1.0
    header["CRVAL3"] = 0.0
    wcs = WCS(header)
    # Build minimal header for SGMeta
    meta_header = fits.Header()
    meta_header["NWIN"] = 1
    meta_header["TDESC1"] = "test"
    meta_header["TWAVE1"] = float(wavelengths.to(u.AA).value[0])
    meta_header["TWMIN1"] = float(wavelengths.to(u.AA).value[0])
    meta_header["TWMAX1"] = float(wavelengths.to(u.AA).value[-1])
    meta_header["TDET1"] = "FUV"
    meta = SGMeta(meta_header, "test")
    return SpectrogramCube(data, wcs=wcs, uncertainty=None, unit=u.DN, meta=meta, mask=None)
